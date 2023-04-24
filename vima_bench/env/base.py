from __future__ import annotations

import os
import tempfile
import time
from typing import Literal

import gym
import numpy as np
import pybullet as p

from ..tasks import ALL_TASKS as _ALL_TASKS
from ..tasks.components.end_effectors import Suction, Spatula
from ..tasks.task_suite.base import BaseTask
from ..tasks.utils import pybullet_utils, misc_utils as utils

PLACE_STEP = 0.0003
PLACE_DELTA_THRESHOLD = 0.005

UR5_URDF_PATH = "ur5/ur5.urdf"
UR5_WORKSPACE_URDF_PATH = "ur5/workspace.urdf"
PLANE_URDF_PATH = "plane/plane.urdf"


class VIMAEnvBase(gym.Env):
    pix_size = 0.003125

    def __init__(
        self,
        modalities: Literal["rgb", "depth", "segm"]
        | list[Literal["rgb", "depth", "segm"]]
        | None = None,
        task: BaseTask | str | None = None,
        task_kwargs: dict | None = None,
        seed: int | None = None,
        hz: int = 240,
        max_sim_steps_to_static: int = 1000,
        debug: bool = False,
        display_debug_window: bool = False,
        hide_arm_rgb: bool = True,
    ):
        assets_root = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "tasks", "assets"
        )
        assert os.path.exists(assets_root), f"Assets root {assets_root} does not exist!"

        self.obj_ids = {"fixed": [], "rigid": []}
        # obj_id_reverse_mapping: a reverse mapping dict that maps object unique id to:
        # 1. object_name appended with color name
        # 2. object_texture entry in TexturePedia
        # 3. object_description entry in ObjPedia
        self.obj_id_reverse_mapping = {}
        self.homej = np.array([-1, -0.5, 0.5, -0.5, -0.5, 0]) * np.pi
        self.step_counter = 0
        self.assets_root = assets_root

        self.episode_counter = 0
        self.all_actions = []

        self._debug = debug
        self.set_task(task, task_kwargs)

        # setup modalities
        modalities = modalities or ["rgb", "segm"]
        if isinstance(modalities, str):
            modalities = [modalities]
        assert set(modalities).issubset(
            {"rgb", "depth", "segm"}
        ), f"Unsupported modalities provided {modalities}"
        assert "depth" not in modalities, "FIXME: fix depth normalization"
        self.modalities = modalities

        # 所要保存的视角List
        viewList = []

        cam_pos1 = np.array([0.0, 0.5, 0.7])
        cam_orientation1 = np.array([0.7, 0.7, 0, 0])

        cam_pos2 = np.array([0.2, 0.3, 0.8])
        cam_orientation2 = np.array([0.7, 0.4, 0.1, 0])

        cam_pos3 = np.array([0.0, 0.5, 0.7])
        cam_orientation3 = np.array([0.3, 0.3, 0.7, 0])

        for [pos, cam_orientation] in [[cam_pos1, cam_orientation1], [cam_pos2, cam_orientation2], [cam_pos3, cam_orientation3]]:
            viewMat, projMatrix = self.get_view(pos, cam_orientation)
            viewList.append([viewMat, projMatrix])
        self.viewList = viewList

        # setup observation space
        obs_space = {}
        for modality in self.modalities:
            if modality == "rgb":
                obs_cam_space = {
                    view: gym.spaces.Box(
                        0, 255, shape=(3, *config["image_size"]), dtype=np.uint8
                    )
                    for view, config in self.agent_cams.items()
                }
                obs_space.update(
                    {
                        "rgb": gym.spaces.Dict(obs_cam_space),
                    }
                )
            elif modality == "depth":
                obs_cam_space = {
                    view: gym.spaces.Box(
                        0.0, 1.0, shape=(1, *config["image_size"]), dtype=np.float32
                    )
                    for view, config in self.agent_cams.items()
                }
                obs_space.update(
                    {
                        "depth": gym.spaces.Dict(obs_cam_space),
                    }
                )
            elif modality == "segm":
                obs_cam_space = {
                    view: gym.spaces.Box(
                        0, 255, shape=config["image_size"], dtype=np.uint8
                    )
                    for view, config in self.agent_cams.items()
                }
                obs_space.update(
                    {
                        "segm": gym.spaces.Dict(obs_cam_space),
                    }
                )
        # observations also include end effector
        obs_space["ee"] = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Dict(obs_space)

        # Start PyBullet.
        client = self.connect_pybullet_hook(display_debug_window)
        self.client_id = client
        file_io = p.loadPlugin("fileIOPlugin", physicsClientId=client)
        if file_io < 0:
            raise RuntimeError("pybullet: cannot load FileIO!")
        if file_io >= 0:
            p.executePluginCommand(
                file_io,
                textArgument=assets_root,
                intArgs=[p.AddFileIOAction],
                physicsClientId=client,
            )

        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=self.client_id)
        p.setPhysicsEngineParameter(enableFileCaching=0, physicsClientId=self.client_id)
        p.setAdditionalSearchPath(assets_root, physicsClientId=self.client_id)
        p.setAdditionalSearchPath(tempfile.gettempdir(), physicsClientId=self.client_id)
        p.setTimeStep(1.0 / hz, physicsClientId=self.client_id)

        # If display debug window, move default camera closer to the scene.
        if display_debug_window:
            target = p.getDebugVisualizerCamera(physicsClientId=self.client_id)[11]
            p.resetDebugVisualizerCamera(
                cameraDistance=1.1,
                cameraYaw=90,
                cameraPitch=-25,
                cameraTargetPosition=target,
                physicsClientId=self.client_id,
            )

        assert max_sim_steps_to_static > 0
        self._max_sim_steps_to_static = max_sim_steps_to_static

        self.seed(seed)

        self.prompt, self.prompt_assets = None, None
        self.meta_info = {}

        self._display_debug_window = display_debug_window

        self._hide_arm_rgb = hide_arm_rgb

    def connect_pybullet_hook(self, display_debug_window: bool):
        return p.connect(p.DIRECT if not display_debug_window else p.GUI)

    def set_task(
        self, task: BaseTask | str | None = None, task_kwargs: dict | None = None
    ):
        # setup task
        ALL_TASKS = _ALL_TASKS.copy()
        ALL_TASKS.update({k.split("/")[1]: v for k, v in ALL_TASKS.items()})
        if isinstance(task, str):
            assert task in ALL_TASKS, f"Invalid task name provided {task}"
            task = ALL_TASKS[task](debug=self._debug, **(task_kwargs or {}))
        elif isinstance(task, BaseTask):
            task = task
        elif task is None:
            task = ALL_TASKS["instruction_following/drag_the_object_into_the_object"](
                debug=self._debug, **(task_kwargs or {})
            )
        task.assets_root = self.assets_root
        task.set_difficulty("easy")
        self._task = task
        # get agent camera config
        self.agent_cams = self.task.agent_cam_config

        # setup action space
        self.position_bounds = gym.spaces.Box(
            low=np.array([0.25, -0.5], dtype=np.float32),
            high=np.array([0.75, 0.50], dtype=np.float32),
            shape=(2,),
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Dict(
            {
                "pose0_position": self.position_bounds,
                "pose0_rotation": gym.spaces.Box(
                    -1.0, 1.0, shape=(4,), dtype=np.float32
                ),
                "pose1_position": self.position_bounds,
                "pose1_rotation": gym.spaces.Box(
                    -1.0, 1.0, shape=(4,), dtype=np.float32
                ),
            }
        )

    @property
    def task(self) -> BaseTask:
        return self._task

    @property
    def task_name(self) -> str:
        return self.task.task_name

    @property
    def is_static(self):
        """Return true if objects are no longer moving."""
        v = [
            np.linalg.norm(p.getBaseVelocity(i, physicsClientId=self.client_id)[0])
            for i in self.obj_ids["rigid"]
        ]
        return all(np.array(v) < 5e-3)

    # ---------------------------------------------------------------------------
    # Standard Gym Functions
    # ---------------------------------------------------------------------------

    def close(self):
        p.disconnect(self.client_id)

    def seed(self, seed=None):
        self._random = np.random.default_rng(seed=seed)
        self._env_seed = seed
        self.task.set_seed(seed)
        return seed

    @property
    def global_seed(self):
        env_seed = self._env_seed
        task_seed = self.task.seed
        return env_seed, task_seed

    @global_seed.setter
    def global_seed(self, seed):
        self.seed(seed)

    def get_prompt_and_assets(self):
        """
        Return prompt and prompt assets.
        Intentionally make this method to preserve Gym API.
        Otherwise something like `initial_obs, prompt, p_assets = env.reset()` breaks Gym API.
        """
        return self.prompt, self.prompt_assets

    def reset(self, task_name="unknow_task", workspace_only=False):
        """Performs common reset functionality for all supported tasks."""
        if not self.task:
            raise ValueError(
                "environment task must be set. Call set_task or pass "
                "the task arg in the environment constructor."
            )
        self.obj_ids = {"fixed": [], "rigid": []}
        self.obj_id_reverse_mapping = {}
        self.meta_info = {}
        self.step_counter = 0
        self.episode_counter += 1
        self.skip_repetition = 0
        # self.task_name = task_name

        p.resetSimulation(physicsClientId=self.client_id)
        p.setGravity(0, 0, -9.8, physicsClientId=self.client_id)

        # Temporarily disable rendering to load scene faster.
        if self._display_debug_window:
            p.configureDebugVisualizer(
                p.COV_ENABLE_RENDERING, 0, physicsClientId=self.client_id
            )

        pybullet_utils.load_urdf(
            p,
            os.path.join(self.assets_root, PLANE_URDF_PATH),
            [0, 0, -0.001],
            physicsClientId=self.client_id,
        )

        pybullet_utils.load_urdf(
            p,
            os.path.join(self.assets_root, UR5_WORKSPACE_URDF_PATH),
            [0.5, 0, 0],
            physicsClientId=self.client_id,
        )

        # Load UR5 robot arm equipped with suction end effector.
        self.ur5 = pybullet_utils.load_urdf(
            p,
            os.path.join(self.assets_root, UR5_URDF_PATH),
            physicsClientId=self.client_id,
        )
        if self._hide_arm_rgb:
            pybullet_utils.set_visibility_bullet(
                self.client_id, self.ur5, pybullet_utils.INVISIBLE_ALPHA
            )
        self.ee = self.task.ee(
            self.assets_root,
            self.ur5,
            9,
            self.obj_ids,
            self.client_id,
        )
        self.ee.is_visible = not self._hide_arm_rgb
        self.ee_tip = 10  # Link ID of suction cup.

        # Get revolute joint indices of robot (skip fixed joints).
        n_joints = p.getNumJoints(self.ur5, physicsClientId=self.client_id)
        joints = [
            p.getJointInfo(self.ur5, i, physicsClientId=self.client_id)
            for i in range(n_joints)
        ]
        self.joints = [j[0] for j in joints if j[2] == p.JOINT_REVOLUTE]
        # Move robot to home joint configuration.
        for i in range(len(self.joints)):
            p.resetJointState(
                self.ur5, self.joints[i], self.homej[i], physicsClientId=self.client_id
            )

        # Reset end effector.
        self.ee.release()

        # Reset task.
        if not workspace_only:
            self.task.reset(self)

        # Re-enable rendering.
        if self._display_debug_window:
            p.configureDebugVisualizer(
                p.COV_ENABLE_RENDERING, 1, physicsClientId=self.client_id
            )

        # generate prompt and corresponding assets
        self.prompt, self.prompt_assets = self.task.generate_prompt()

        # generate meta info dict
        if isinstance(self.ee, Suction):
            self.meta_info["end_effector_type"] = "suction"
        elif isinstance(self.ee, Spatula):
            self.meta_info["end_effector_type"] = "spatula"
        else:
            raise NotImplementedError()
        self.meta_info["n_objects"] = sum(len(v) for v in self.obj_ids.values())
        self.meta_info["difficulty"] = self.task.difficulty_level or "easy"
        self.meta_info["views"] = list(self.agent_cams.keys())
        self.meta_info["modalities"] = self.modalities
        self.meta_info["seed"] = self.task.seed
        self.meta_info["action_bounds"] = {
            "low": self.position_bounds.low,
            "high": self.position_bounds.high,
        }
        self.meta_info["robot_components"] = (
            [self.ur5, self.ee.base_uid, self.ee.body_uid]
            if isinstance(self.ee, Suction)
            else [self.ur5, self.ee.base_uid]
        )
        # check robot components are disjoint with object ids
        assert set(self.meta_info["robot_components"]).isdisjoint(
            set([x for v in self.obj_ids.values() for x in v])
        )
        # add reverse mapping dict that maps object_id in segmentation mask to object information.
        # sanity check of mapping dict
        assert len(self.obj_id_reverse_mapping) > 0, (
            "Please add the mapping into the dict when reset new tasks"
            "Use the method add_object_id_reverse_mapping_info in pybullet_utils.py, "
            "which is similar to the add_any_object method."
            "Hint: You could also refer to the novel_concept_grounding task as an example to see how to implement that"
        )
        # check the completeness of the mapping dict
        assert all(
            [
                (obj_id in self.obj_id_reverse_mapping)
                for each_catagory in self.obj_ids.values()
                for obj_id in each_catagory
            ]
        ), (
            "Incomplete object_id mapping dict. "
            "Please check whether there are missing objects that are not added into the dict"
            f"Currently, we have {len(self.obj_id_reverse_mapping.keys())} in dict, but the total n_objects is {sum(len(v) for v in self.obj_ids.values())}"
            f"[Debug Info] Matching Truth Table: {[(obj_id in self.obj_id_reverse_mapping) for each_catagory in self.obj_ids.values() for obj_id in each_catagory]}"
            f"Mapping Dict {self.obj_id_reverse_mapping}"
            f"Object IDs {self.obj_ids}"
        )
        # check ids from reverse map and ids from obj_ids are the same
        ids_from_reverse_map = list(self.obj_id_reverse_mapping.keys())
        ids_from_obj_ids = [x for v in self.obj_ids.values() for x in v]
        assert set(ids_from_reverse_map) == set(ids_from_obj_ids)

        self.meta_info["obj_id_to_info"] = self.obj_id_reverse_mapping

        obs, _, _, _ = self.step()

        # print("try to print joint info:\n")

        # robot_id = self.ur5
        # # robot_id = 10
        # print("robot_id: ", robot_id)
        # available_joints_indexes = [i for i in range(p.getNumJoints(robot_id)) if
        #                             p.getJointInfo(robot_id, i)[2] != p.JOINT_FIXED]
        # print([p.getJointInfo(robot_id, i)[1] for i in available_joints_indexes])
        # print(p.getJointInfo(robot_id, 4)[1])
        # print([p.getJointInfo(robot_id, i)[1] for i in available_joints_indexes])
        # for i in available_joints_indexes:
        #     tmp = p.getJointInfo(robot_id, i)
        #     # tmp = p.getJointInfo(robot_id, i)
        #     for ind, val in enumerate(tmp):
        #         print(tmp[1],)
        #         print("joint %s, info index: %d, value:"%(tmp[1], ind), val)


        return obs

    def micro_step(self, action=None, skip_oracle=True, episode=0, task_name="notFindName!"):
        """
        Args:
            action: micro action
            skip_oracle:
            episode:
            task_name:

        Returns:
            (obs, reward, done, info) tuple containing MDP step data.
        """
        if action is not None:

            if isinstance(self.ee, Suction):
                timeout, released = self.task.micro_primitive(
                    self.movej, self.movep, self.ee, action, episode, task_name
                )
            elif isinstance(self.ee, Spatula):
                timeout = self.task.micro_primitive(
                    self.movej, self.movep, self.ee, action, episode, task_name
                )
            else:
                raise ValueError("Unknown end effector type")

            # Exit early if action times out. We still return an observation
            # so that we don't break the Gym API contract.
            if timeout:
                obs = self._get_obs()
                return obs, 0.0, True, self._get_info()

        # Step simulator asynchronously until objects settle.
        # print("out of while")
        counter = 0
        while not self.is_static:
            self.step_simulation()
            if counter > self._max_sim_steps_to_static:
                print(
                    f"WARNING: step until static exceeds max {self._max_sim_steps_to_static} steps!"
                )
                break
            counter += 1

        # we don't care about reward in VIMA
        reward = 0
        # update goal sequence accordingly
        # self.task.update_goals(skip_oracle=skip_oracle)
        # check if done
        if isinstance(self.ee, Suction):
            if action is not None:
                result_tuple = self.task.check_success(release_obj=released)
            else:
                result_tuple = self.task.check_success(release_obj=False)
        elif isinstance(self.ee, Spatula):
            result_tuple = self.task.check_success()
        else:
            raise NotImplementedError()

        done = result_tuple.success or result_tuple.failure
        obs = self._get_obs()

        return obs, reward, done, self._get_info()

    def step(self, action=None, skip_oracle=True, episode=0, task_name="notFindName!"):
        """Execute action with specified primitive.

        Args:
          action: action to execute.
          skip_oracle: boolean variable that indicates whether to update oracle-only goals

        Returns:
          (obs, reward, done, info) tuple containing MDP step data.
        """
        if action is not None:
            assert self.action_space.contains(
                action
            ), f"got {action} instead, action space {self.action_space}"

            pose0 = (action["pose0_position"], action["pose0_rotation"])
            pose1 = (action["pose1_position"], action["pose1_rotation"])

            if isinstance(self.ee, Suction):
                timeout, released = self.task.primitive(
                    self.movej, self.movep, self.ee, pose0, pose1, episode, task_name
                )
            elif isinstance(self.ee, Spatula):
                timeout = self.task.primitive(
                    self.movej, self.movep, self.ee, pose0, pose1, episode, task_name
                )
            else:
                raise ValueError("Unknown end effector type")

            # Exit early if action times out. We still return an observation
            # so that we don't break the Gym API contract.
            if timeout:
                obs = self._get_obs()
                return obs, 0.0, True, self._get_info()

        # Step simulator asynchronously until objects settle.
        # print("out of while")
        counter = 0
        while not self.is_static:
            self.step_simulation()
            if counter > self._max_sim_steps_to_static:
                print(
                    f"WARNING: step until static exceeds max {self._max_sim_steps_to_static} steps!"
                )
                break
            counter += 1

        # we don't care about reward in VIMA
        reward = 0
        # update goal sequence accordingly
        self.task.update_goals(skip_oracle=skip_oracle)
        # check if done
        if isinstance(self.ee, Suction):
            if action is not None:
                result_tuple = self.task.check_success(release_obj=released)
            else:
                result_tuple = self.task.check_success(release_obj=False)
        elif isinstance(self.ee, Spatula):
            result_tuple = self.task.check_success()
        else:
            raise NotImplementedError()

        done = result_tuple.success or result_tuple.failure
        obs = self._get_obs()

        return obs, reward, done, self._get_info()

    def oracle_action_to_env_actions(self, oracle_action: dict):
        if isinstance(self.ee, Suction):  # we use MoveEndEffector Primitive
            pick_pose, place_pose = oracle_action["pose0"], oracle_action["pose1"]
            pick_position, pick_rotation = pick_pose[0], pick_pose[1]
            place_position, place_rotation = place_pose[0], place_pose[1]
            # "push" is not used in suction, we still include it for action space completeness
            sub_actions = [
                # sub-action 1, move the arm to pick the object
                {
                    "position": np.array(
                        [pick_position[0], pick_position[1], 0.0], dtype=np.float32
                    ),
                    "rotation": pick_rotation.astype(np.float32),
                    "release": False,
                    "push": False,
                },
                # sub-action 2, vertically raise the arm to a safe height before move to the place position
                {
                    "position": np.array(
                        [
                            pick_position[0],
                            pick_position[1],
                            self.task.primitive.height,
                        ],
                        dtype=np.float32,
                    ),
                    "rotation": np.array(
                        utils.eulerXYZ_to_quatXYZW((0, 0, 0)), dtype=np.float32
                    ),
                    "release": False,
                    "push": False,
                },
                # sub-action 3, move the arm to place position and release the end effector
                {
                    "position": np.array(
                        [place_position[0], place_position[1], 0.0], dtype=np.float32
                    ),
                    "rotation": place_rotation.astype(np.float32),
                    "release": True,
                    "push": False,
                },
                # sub-action 4, vertically raise the arm to the post-place height
                {
                    "position": np.array(
                        [
                            place_position[0],
                            place_position[1],
                            self.task.primitive.height,
                        ],
                        dtype=np.float32,
                    ),
                    "rotation": np.array(
                        utils.eulerXYZ_to_quatXYZW((0, 0, 0)), dtype=np.float32
                    ),
                    "release": False,
                    "push": False,
                },
            ]
        elif isinstance(self.ee, Spatula):
            start_pose, end_pose = oracle_action["pose0"], oracle_action["pose1"]
            start_position, start_rotation = start_pose[0], start_pose[1]
            end_position, end_rotation = end_pose[0], end_pose[1]
            # "release" is not used in spatula, we still include it for action space completeness
            sub_actions = [
                # sub-action 1, move the arm to the start position of pushing in the air
                {
                    "position": np.array(
                        [
                            start_position[0],
                            start_position[1],
                            self.task.primitive.height,
                        ],
                        dtype=np.float32,
                    ),
                    "rotation": start_rotation.astype(np.float32),
                    "push": False,
                    "release": False,
                },
                # sub-action 2, vertically descend the arm and push to the end position
                {
                    "position": end_position.astype(np.float32),
                    "rotation": end_rotation.astype(np.float32),
                    "push": True,
                    "release": False,
                },
                # sub-action 3, vertically raise the arm to the post-push height
                {
                    "position": np.array(
                        [end_position[0], end_position[1], self.task.primitive.height],
                        dtype=np.float32,
                    ),
                    "rotation": end_rotation.astype(np.float32),
                    "push": False,
                    "release": False,
                },
            ]
        else:
            raise NotImplementedError()
        assert len(sub_actions) == self.task.oracle_step_to_env_step_ratio
        return sub_actions

    def oracle_step(self, oracle_action: dict):
        """Execute oracle action.
        Note that the original "Pick and Place" primitive can be divided into
        several sub-actions using "Move End Effector" primitive.

        Args:
          oracle_action: action to execute.
        """
        sub_actions = self.oracle_action_to_env_actions(oracle_action)
        # execute these sub-actions sequentially and buffer corresponding returns
        obs_list, reward_list, done_list, info_list = [], [], [], []
        for idx, sub_action in enumerate(sub_actions):
            obs, reward, done, info = self.step(sub_action, skip_oracle=False)
            obs_list.append(obs)
            reward_list.append(reward)
            done_list.append(done)
            info_list.append(info)
            if done:
                # only keep those subactions (from oracle) executed
                sub_actions = sub_actions[: idx + 1]
                break
        return obs_list, sub_actions, reward_list, done_list, info_list

    def step_simulation(self):
        p.stepSimulation(physicsClientId=self.client_id)
        self.step_counter += 1
        if (
            self.task.constraint_checking["enabled"]
            and self.step_counter
            % self.task.constraint_checking["checking_sim_interval"]
            == 0
        ):
            self.task.check_constraint()

    def render_camera(self, config, image_size=None):
        """Render RGB-D image with specified camera configuration."""
        if not image_size:
            image_size = config["image_size"]

        # OpenGL camera settings.
        lookdir = np.float32([0, 0, 1]).reshape(3, 1)
        updir = np.float32([0, -1, 0]).reshape(3, 1)
        rotation = p.getMatrixFromQuaternion(
            config["rotation"], physicsClientId=self.client_id
        )
        rotm = np.float32(rotation).reshape(3, 3)
        lookdir = (rotm @ lookdir).reshape(-1)
        updir = (rotm @ updir).reshape(-1)
        lookat = config["position"] + lookdir
        focal_len = config["intrinsics"][0]
        znear, zfar = config["zrange"]
        viewm = p.computeViewMatrix(
            config["position"], lookat, updir, physicsClientId=self.client_id
        )
        fovh = (image_size[0] / 2) / focal_len
        fovh = 180 * np.arctan(fovh) * 2 / np.pi

        # Notes: 1) FOV is vertical FOV 2) aspect must be float
        aspect_ratio = image_size[1] / image_size[0]
        projm = p.computeProjectionMatrixFOV(
            fovh, aspect_ratio, znear, zfar, physicsClientId=self.client_id
        )

        # Render with OpenGL camera settings.
        _, _, color, depth, segm = p.getCameraImage(
            width=image_size[1],
            height=image_size[0],
            viewMatrix=viewm,
            projectionMatrix=projm,
            flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
            renderer=p.ER_BULLET_HARDWARE_OPENGL,
            physicsClientId=self.client_id,
        )

        # Get color image.
        color_image_size = (image_size[0], image_size[1], 4)
        color = np.array(color, dtype=np.uint8).reshape(color_image_size)
        color = color[:, :, :3]  # remove alpha channel
        if config["noise"]:
            color = np.int32(color)
            color += np.int32(self._random.normal(0, 3, image_size))
            color = np.uint8(np.clip(color, 0, 255))
        # transpose from HWC to CHW
        color = np.transpose(color, (2, 0, 1))

        # Get depth image.
        depth_image_size = (image_size[0], image_size[1])
        zbuffer = np.array(depth).reshape(depth_image_size)
        depth = zfar + znear - (2.0 * zbuffer - 1.0) * (zfar - znear)
        depth = (2.0 * znear * zfar) / depth
        if config["noise"]:
            depth += self._random.normal(0, 0.003, depth_image_size)
        # normalize depth to be within range [0, 1]
        depth /= 20.0
        # add 'C' dimension
        depth = depth[np.newaxis, ...]

        # Get segmentation image.
        segm = np.uint8(segm).reshape(depth_image_size)

        # images = p.getCameraImage(300,
        #                           300,
        #                           # viewm,
        #                           # projm,
        #                           shadow=False,
        #                           renderer=p.ER_BULLET_HARDWARE_OPENGL)
        # print(images[2])
        # import cv2
        # # import numpy as np
        # cv2.imwrite("%f.jpg"%(np.random.rand()), images[2])  # [2]是rpg图像

        return color, depth, segm

    def _get_obs(self):
        obs = {f"{modality}": {} for modality in self.modalities}
        # print("self.agent_cams:\n ", self.agent_cams)
        for view, config in self.agent_cams.items():
            color, depth, segm = self.render_camera(config)
            render_result = {"rgb": color, "depth": depth, "segm": segm}
            for modality in self.modalities:
                obs[modality][view] = render_result[modality]

        # add end effector into observation dict
        obs["ee"] = 0 if isinstance(self.ee, Suction) else 1

        assert self.observation_space.contains(obs)
        return obs

    def _get_info(self):
        result_tuple = self.task.check_success()
        info = {
            "prompt": self.prompt,
            "success": result_tuple.success,
            "failure": result_tuple.failure,
        }
        return info

    def render(self, mode="human"):
        # functionality filled by subclass
        pass

    # ---------------------------------------------------------------------------
    # Robot Movement Functions
    # ---------------------------------------------------------------------------

    def movej(self, targj, speed=0.01, timeout=50, is_act=0, is_end=0,task_stage="un_specified"):
        """Move UR5 to target joint configuration."""
        t0 = time.time()
        # print("timeout: ", timeout)
        timeout = 500
        speed = 0.01
        while (time.time() - t0) < timeout:
            # if task_stage!="control":
            #     time.sleep(0.02)
            currj = [
                p.getJointState(self.ur5, i, physicsClientId=self.client_id)[0]
                for i in self.joints
            ]

            currj = np.array(currj)
            diffj = targj - currj
            if all(np.abs(diffj) < 1e-2):
                "当tar j与curr j差异够小时，停止移动"
                return False

            # Move with constant velocity
            norm = np.linalg.norm(diffj)
            v = diffj / norm if norm > 0 else 0
            stepj = currj + v * speed
            gains = np.ones(len(self.joints))
            p.setJointMotorControlArray(
                bodyIndex=self.ur5,
                jointIndices=self.joints,
                controlMode=p.POSITION_CONTROL,
                targetPositions=stepj,
                positionGains=gains,
                physicsClientId=self.client_id,
            )
            # self.step_counter += 1
            # print(self.step_counter)

            # if (self.step_counter + 1) % 10 == 0:
            #     self.view_image_save(episode=self.episode_counter, viewList=self.viewList, freq_save=self.step_counter, is_act=is_act, is_end=is_end,
            #                          stage=task_stage)
            #     self.skip_repetition = self.step_counter + 1
            # print("self.step_counter: ", self.step_counter)
            self.step_simulation()

        print(f"Warning: movej exceeded {timeout} second timeout. Skipping.")
        return True

    def movep(self, pose, speed=0.01, meetParaOfMovej=None):
        """Move UR5 to target end effector pose."""
        targj = self.solve_ik(pose)

        self.all_actions.append([meetParaOfMovej[0]] + [i for i in pose[0]] + list(utils.quatXYZW_to_eulerXYZ(pose[1])) + [meetParaOfMovej[1]])
        # def view_image_save(self, episode, viewList, freq_save, is_end, is_act, stage):
        if (self.step_counter + 1) % 100 == 0 and meetParaOfMovej[-1]!="control":
            self.view_image_save(self.episode_counter, pose, viewList=self.viewList, freq_save=self.step_counter, is_act=meetParaOfMovej[0], is_end=meetParaOfMovej[1],
                                         stage=meetParaOfMovej[2])

        # if self.step_counter%10==0 and meetParaOfMovej[-1]!="control":
        #     frontPath = "/Users/liushaofan/code/VIMA"
        #     np.save(
        #         file=frontPath + r"/save_data/%s/traj_%d/action_all.npy" % (self.task_name, self.episode_counter),
        #         arr=self.all_actions)
        return self.movej(targj, speed, is_act=meetParaOfMovej[0], is_end=meetParaOfMovej[1], task_stage=meetParaOfMovej[2])

    def solve_ik(self, pose):
        """Calculate joint configuration with inverse kinematics."""
        joints = p.calculateInverseKinematics(
            bodyUniqueId=self.ur5,  # self.ur5: 2
            endEffectorLinkIndex=self.ee_tip,
            targetPosition=pose[0],
            targetOrientation=pose[1],
            lowerLimits=[-3 * np.pi / 2, -2.3562, -17, -17, -17, -17],
            upperLimits=[-np.pi / 2, 0, 17, 17, 17, 17],
            jointRanges=[np.pi, 2.3562, 34, 34, 34, 34],  # * 6,
            restPoses=np.float32(self.homej).tolist(),
            maxNumIterations=100,
            residualThreshold=1e-5,
            physicsClientId=self.client_id,
        )
        joints = np.float32(joints)
        joints[2:] = (joints[2:] + np.pi) % (2 * np.pi) - np.pi
        return joints

    def get_view(self, cam_pos=0, cam_orientation=0):
        """
        Arguments
            cam_pos: camera position
            cam_orientation: camera orientation in quaternion
        """
        width = 300
        height = 300
        fov = 90
        aspect = width / height
        near = 0.001
        far = 5

        # cam_pos = np.array([0.0, 0.5, 0.7])
        # cam_orientation = np.array([0.7, 0.7, 0, 0])

        use_maximal_coordinates = False
        if use_maximal_coordinates:
            # cam_orientation has problem when enable bt_rigid_body,
            # looking at 0.0, 0.0, 0.0 instead
            # this does not affect performance
            cam_pos_offset = cam_pos + np.array([0.0, 0.0, 0.3])
            target_pos = np.array([0.0, 0.0, 0.0])
        else:
            # camera pos, look at, camera up direction
            rot_matrix = p.getMatrixFromQuaternion(cam_orientation)
            # offset to base pos
            cam_pos_offset = cam_pos + np.dot(
                np.array(rot_matrix).reshape(3, 3), np.array([0.1, 0.0, 0.3]))
            target_pos = cam_pos_offset + np.dot(
                np.array(rot_matrix).reshape(3, 3), np.array([-1.0, 0.0, 0.0]))
        # compute view matrix
        view_matrix = p.computeViewMatrix(cam_pos_offset, target_pos, [0, 0, 1])
        projection_matrix = p.computeProjectionMatrixFOV(fov, aspect, near, far)

        return view_matrix, projection_matrix

    def view_image_save(self, episode, pose, viewList, freq_save, is_end, is_act, stage):
        '''
        Args:
            episode:
            viewList:
            freq_save:
            stage:  [move_to_pick, picking_success, goal_state]
        Returns:

        '''
        for ind, view in enumerate(viewList):
            images = p.getCameraImage(300,
                                      300,
                                      view[0],
                                      view[1],
                                      shadow=True,
                                      renderer=p.ER_BULLET_HARDWARE_OPENGL)
            # print(images[2])
            # import cv2
            from PIL import Image
            # import numpy as np
            '''
            暂存代码
            info = p.getLinkState(4, 0)
            info = list(info)[:2]  # 只保留坐标与转角
            info.insert(0, is_act)  # 夹了东西
            info.insert(-1, is_end)  # 没有到终点            
            '''
            # 实验代码
            info = p.getLinkState(3, 0)
            info = list(info)[:2]  # 只保留坐标与转角
            # print("pose for save:", list(info[0]),list(info[1]))
            # print(list(info[0]),list(info[1]))
            # print("3_0 info: ", info)
            # print(list(info[0]),',', list(utils.quatXYZW_to_eulerXYZ(info[-1])))

            info = p.getLinkState(4, 0)
            info = list(info)[:2]  # 只保留坐标与转角
            # print("4_0 info: ", info)
            # print("4_0 423:",list(utils.quatXYZW_to_eulerXYZ(info[-1])))


            # cv2.imwrite("traj_%d_view_%d_%s_%d.jpg" % (episode, ind, stage, freq_save),
            #             images[2])  # [2]是rpg图像
            im = Image.fromarray(images[2])
            # rgb_im = rgb_im.convert('RGB')

            if im.mode in ("RGBA", "P"): im = im.convert("RGB")
            frontPath = "/Users/liushaofan/code/VIMA"
            if not os.path.exists(frontPath+r"/save_data/%s/traj_%d/view_%d/" % (self.task_name,episode, ind)):
                os.makedirs(frontPath+r"/save_data/%s/traj_%d/view_%d/" % (self.task_name,episode, ind))
            im.save(frontPath+r"/save_data/%s/traj_%d/view_%d/%s_%d.jpg" % (self.task_name,episode, ind, stage, freq_save))
            # scipy.misc.imsave("traj_%d_view_%d_%s_%d.jpg" % (episode, ind, stage, freq_save),
            #             images[2])
        # np.save(file="traj_%d_%s_%d.npy" % (episode, stage, freq_save), arr=info)
        if not os.path.exists(frontPath+"/save_data/%s/traj_%d/action/" % (self.task_name,episode)):
            os.makedirs(frontPath+
            r"/save_data/%s/traj_%d/action/" % (self.task_name,episode))
        self.all_actions.append([is_act]+[i for i in pose[0]]+list(utils.quatXYZW_to_eulerXYZ(pose[1])) +[is_end])
        # np.save(file=frontPath+r"/save_data/%s/traj_%d/action/%s_%d.npy" % (self.task_name,episode, stage, freq_save), arr=[is_act]+[i for i in pose[0]]+list(utils.quatXYZW_to_eulerXYZ(pose[1])) +[is_end])
