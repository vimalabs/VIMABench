from __future__ import annotations

import os
import random
import string
import tempfile
from collections import namedtuple
from copy import deepcopy
from typing import NamedTuple, Any

import cv2
import gym
import numpy as np
import pybullet as p

from ..components import PickPlace, Oracle, get_agent_cam_config, Suction
from ..components.encyclopedia.definitions import SizeRange, ObjEntry, TextureEntry
from ..components.placeholders import Placeholder
from ..utils import misc_utils as utils
from ..utils.pybullet_utils import (
    add_any_object,
    add_object_id_reverse_mapping_info,
    p_change_texture,
)

OracleAgent = namedtuple("OracleAgent", ["act"])


class ResultTuple(NamedTuple):
    success: bool
    failure: bool


class BaseTask:
    """
    Base class for VIMA task.

    `task_name`: name of the task

    `prompt_template` is a string object storing the template of the multimodal prompt for a certain task.
    Placeholders should be wrapped with curly brackets, e.g., {object}, {color}, {number}, etc.

    `task_meta` can be any objects storing task-relevant information, configuration, and so on.
    E.g., it stores how many objects to be spawned.

    `placeholder_expression` maps name of a placeholder to its expression types.

    `oracle_step_to_env_step_ratio` specifies the number of env steps corresponding to one oracle step.

    `oracle_max_steps` specifies the maximum number of steps allowed for the oracle agents.

    `difficulties` specifies all possible difficulties of evaluation schemes.
    """

    task_name: str
    REJECT_SAMPLING_MAX_TIMES = 10

    def __init__(
        self,
        prompt_template: str,
        task_meta: Any,
        placeholder_expression: dict[str, Any],
        oracle_max_steps: int,
        oracle_step_to_env_step_ratio: int = 4,
        difficulties: list[str] | None = None,
        obs_img_views: str | list[str] | None = None,
        obs_img_size: tuple[int, int] = (128, 256),
        placeholder_img_size: tuple[int, int] = (128, 256),
        seed: int | None = None,
        debug: bool = False,
    ):
        self.prompt_template = prompt_template
        self.task_meta = task_meta
        self.placeholder_expression = placeholder_expression
        self.oracle_max_steps = oracle_max_steps
        self.oracle_step_to_env_step_ratio = oracle_step_to_env_step_ratio
        self.difficulties = difficulties or ["easy", "medium", "hard"]

        self.difficulty_level = None
        self.ee = Suction
        self.sixdof = False
        self.primitive = PickPlace()
        self.oracle_cams = Oracle.CONFIG
        self.constraint_checking = {"enabled": False}

        # Evaluation epsilons (for pose evaluation metric).
        self.pos_eps = 0.01
        self.rot_eps = np.deg2rad(15)
        self.pos_eps_original = 0.05
        self.pos_eps_bowl_multiplier = 1.75

        # Workspace bounds.
        self.pix_size = 0.003125
        self.bounds = np.array([[0.25, 0.75], [-0.5, 0.5], [0, 0.3]])
        self.zone_bounds = np.copy(self.bounds)
        self._valid_workspace = gym.spaces.Box(
            low=np.array([0.25, -0.5]),
            high=np.array([0.75, 0.5]),
            shape=(2,),
            dtype=np.float32,
        )

        # self.goals -- List of elements: (objs, matches, targs, replace, rotations, metric, params, max_reward)
        self.goals = []
        self.progress = 0
        self.placeholders: dict[str, Placeholder] = {}

        self.assets_root = None

        self.rng = np.random.default_rng(seed=seed)
        self.seed = seed

        self._placeholder_img_size = placeholder_img_size
        obs_img_views = obs_img_views or ["front", "top"]
        if isinstance(obs_img_views, str):
            obs_img_views = [obs_img_views]
        all_cam_config = get_agent_cam_config(obs_img_size)
        self.agent_cam_config = {view: all_cam_config[view] for view in obs_img_views}

        self._once_set_difficulty = Once()
        self.client_id = None

        # for checking whether the distractors at goals
        self._all_goals = None
        self.distractors = None

    def set_difficulty(self, difficulty: str):
        """
        Set difficulty for different evaluation schemes.
        Note that this method is supposed to be only called ONCE, because implementations of most subclasses involve
        reuse "states" that will be changed by this method.

        All subclasses should super call this parent method for sanity check.
        """
        assert difficulty in self.difficulties
        self.difficulty_level = difficulty
        assert (
            self._once_set_difficulty()
        ), "You may try to call `set_difficulty` multiple times, but this is prohibitive"

    def set_seed(self, seed):
        self.rng = np.random.default_rng(seed=seed)
        self.seed = seed

    def reset(self, env):
        self.client_id = env.client_id

        if not self.assets_root:
            raise ValueError("assets_root must be set")
        self.goals = []
        self.progress = 0  # Task progression metric in range [0, 1].
        self.placeholders = {}

    @staticmethod
    def _scale_size(size, scalar):
        return (
            tuple([scalar * s for s in size])
            if isinstance(scalar, float)
            else tuple([sc * s for sc, s in zip(scalar, size)])
        )

    def add_object_to_env(
        self,
        env,
        obj_entry: ObjEntry,
        color: TextureEntry,
        size: tuple[float, float, float],
        scalar: float | list[float] = 1.0,
        pose: tuple[tuple, tuple] = None,
        category: str = "rigid",
        retain_temp: bool = True,
        **kwargs,
    ):
        """helper function for adding object to env."""
        scaled_size = self._scale_size(size, scalar)
        if pose is None:
            pose = self.get_random_pose(env, scaled_size)
        if pose[0] is None or pose[1] is None:
            # reject sample because of no extra space to use (obj type & size) sampled outside this helper function
            return None, None, None
        obj_id, urdf_full_path = add_any_object(
            env=env,
            obj_entry=obj_entry,
            pose=pose,
            size=scaled_size,
            scaling=scalar,
            retain_temp=retain_temp,
            category=category,
            **kwargs,
        )
        if obj_id is None:  # pybullet loaded error.
            return None, urdf_full_path, pose
        # change texture
        p_change_texture(obj_id, color, env.client_id)
        # add mapping info
        add_object_id_reverse_mapping_info(
            mapping_dict=env.obj_id_reverse_mapping,
            obj_id=obj_id,
            object_entry=obj_entry,
            texture_entry=color,
        )

        return obj_id, urdf_full_path, pose

    def oracle(self, env):
        # the goals of oracle could be different from the agent (containing additional steps)
        # e.g., in the rearrangement task, we allow the agent to move away the distractors to anywhere.
        def act(obs):
            """Calculate action."""

            # Oracle uses perfect RGB-D orthographic images and segmentation masks.
            _, hmap, obj_mask = self.get_true_image(env)

            # Unpack next goal step.
            objs, matches, targs, replace, rotations, _, _, _ = self.goals[0]

            # Match objects to targets without replacement.
            if not replace:

                # Modify a copy of the match matrix.
                matches = matches.copy()

                # Ignore already matched objects.
                for i in range(len(objs)):
                    object_id, (symmetry, _) = objs[i]
                    pose = p.getBasePositionAndOrientation(
                        object_id, physicsClientId=self.client_id
                    )
                    targets_i = np.argwhere(matches[i, :]).reshape(-1)
                    for j in targets_i:
                        if self.is_match(pose, targs[j], symmetry):
                            matches[i, :] = 0
                            matches[:, j] = 0

            # Get objects to be picked (prioritize farthest from nearest neighbor).
            nn_dists = []
            nn_targets = []
            for i in range(len(objs)):
                object_id, (symmetry, _) = objs[i]
                xyz, _ = p.getBasePositionAndOrientation(
                    object_id, physicsClientId=self.client_id
                )
                targets_i = np.argwhere(matches[i, :]).reshape(-1)
                if len(targets_i) > 0:
                    targets_xyz = np.float32([targs[j][0] for j in targets_i])
                    dists = np.linalg.norm(
                        targets_xyz - np.float32(xyz).reshape(1, 3), axis=1
                    )
                    nn = np.argmin(dists)
                    nn_dists.append(dists[nn])
                    nn_targets.append(targets_i[nn])

                # Handle ignored objects.
                else:
                    # change nn_dists from 0 to -1 to support rotation case where distance to travel is zero. Originally: nn_dists.append(0)
                    nn_dists.append(-1)
                    nn_targets.append(-1)
            order = np.argsort(nn_dists)[::-1]

            # Filter out matched objects.
            order = [i for i in order if nn_dists[i] >= 0]

            pick_mask = None
            for pick_i in order:
                pick_mask = np.uint8(obj_mask == objs[pick_i][0])

                # Erode to avoid picking on edges.
                # pick_mask = cv2.erode(pick_mask, np.ones((3, 3), np.uint8))

                if np.sum(pick_mask) > 0:
                    break

            # Trigger task reset if no object is visible.
            if pick_mask is None or np.sum(pick_mask) == 0:
                self.goals = []
                print("Object for pick is not visible. Skipping demonstration.")
                return

            # Get picking pose.
            pick_prob = np.float32(pick_mask)
            pick_pix = utils.sample_distribution(prob=pick_prob, rng=self.rng)
            pick_pos = utils.pix_to_xyz(pick_pix, hmap, self.bounds, self.pix_size)
            pick_pose = (np.asarray(pick_pos), np.asarray((0, 0, 0, 1)))

            # Get placing pose.
            targ_pose = targs[nn_targets[pick_i]]
            obj_pose = p.getBasePositionAndOrientation(
                objs[pick_i][0], physicsClientId=self.client_id
            )
            if not self.sixdof:
                obj_euler = utils.quatXYZW_to_eulerXYZ(obj_pose[1])
                obj_quat = utils.eulerXYZ_to_quatXYZW((0, 0, obj_euler[2]))
                obj_pose = (obj_pose[0], obj_quat)
            world_to_pick = utils.invert(pick_pose)
            obj_to_pick = utils.multiply(world_to_pick, obj_pose)
            pick_to_obj = utils.invert(obj_to_pick)
            place_pose = utils.multiply(targ_pose, pick_to_obj)

            # Rotate end effector?
            if not rotations:
                place_pose = (place_pose[0], (0, 0, 0, 1))

            place_pose = (np.asarray(place_pose[0]), np.asarray(place_pose[1]))
            return {
                "pose0_position": pick_pose[0].astype(np.float32)[:-1],
                "pose0_rotation": pick_pose[1].astype(np.float32),
                "pose1_position": place_pose[0].astype(np.float32)[:-1],
                "pose1_rotation": place_pose[1].astype(np.float32),
            }

        return OracleAgent(act)

    def update_goals(self, skip_oracle=True):
        """
        Update the sequence of goals.
        One goal each time, E.g., original goal sequence (A, B), after successfully achieving goal A, the sequence will be updated to (B).
        """
        # check len of goals, if zero, all goals have been achieved
        if len(self.goals) == 0:
            return

        goal_to_check = None
        goal_index = None
        if skip_oracle:
            for index, goal in enumerate(self.goals):
                # skip goals that should be ignored for agents
                # (one example use could see task rearrangement)
                _, _, _, _, _, _, params, _ = goal
                if isinstance(params, dict) and params["oracle_only"]:
                    continue
                else:
                    goal_to_check = goal
                    goal_index = index
                    break
            if goal_to_check is None:  # there is no non-oracle goal
                return
        else:
            goal_to_check = self.goals[0]
            goal_index = 0

        # Unpack goal step.
        objs, matches, targs, _, _, metric, params, max_progress = goal_to_check

        # Evaluate by matching object poses.
        progress_per_goal = 0
        if metric == "pose":
            for i in range(len(objs)):
                object_id, (symmetry, _) = objs[i]
                pose = p.getBasePositionAndOrientation(
                    object_id, physicsClientId=self.client_id
                )
                targets_i = np.argwhere(matches[i, :]).reshape(-1)
                for j in targets_i:
                    target_pose = targs[j]
                    if self.is_match(pose, target_pose, symmetry):
                        progress_per_goal += max_progress / len(objs)
                        break

        # Evaluate by measuring object intersection with zone.
        elif metric == "zone":
            zone_pts, total_pts = 0, 0
            obj_pts, zones = params
            for zone_idx, (zone_pose, zone_size) in enumerate(zones):

                # Count valid points in zone.
                for obj_idx, obj_id in enumerate(obj_pts):
                    pts = obj_pts[obj_id]
                    obj_pose = p.getBasePositionAndOrientation(
                        obj_id, physicsClientId=self.client_id
                    )
                    world_to_zone = utils.invert(zone_pose)
                    obj_to_zone = utils.multiply(world_to_zone, obj_pose)
                    pts = np.float32(utils.apply(obj_to_zone, pts))
                    if len(zone_size) > 1:
                        valid_pts = np.logical_and.reduce(
                            [
                                pts[0, :] > -zone_size[0] / 2,
                                pts[0, :] < zone_size[0] / 2,
                                pts[1, :] > -zone_size[1] / 2,
                                pts[1, :] < zone_size[1] / 2,
                                pts[2, :] < self.zone_bounds[2, 1],
                            ]
                        )

                    # if zone_idx == matches[obj_idx].argmax():
                    zone_pts += np.sum(np.float32(valid_pts))
                    total_pts += pts.shape[1]
            progress_per_goal = max_progress * (zone_pts / total_pts)
        # Move to next goal step if current goal step is complete
        # OR
        # no reward (we will use 0.01) is given for this operation,
        # but this operation is actually necessary!!! (e.g. in task rearrangement we move away the distractors)
        if np.abs(max_progress - progress_per_goal) < 0.001:
            self.goals.pop(goal_index)

    def check_distractors(self, distractor_obj: tuple[int, tuple]) -> bool:
        """check whether distractor is mis-considered as manipulated obj or target"""
        assert (
            self._all_goals is not None
        ), "_all_goals should be initialized once goals initialized"

        distractor_obj_id, (distractor_symmetry, _) = distractor_obj
        distractor_pose = p.getBasePositionAndOrientation(
            distractor_obj_id, physicsClientId=self.client_id
        )
        for goal in self._all_goals:
            objs, _, targs, _, _, metric, params, _ = goal
            if isinstance(params, dict) and params["oracle_only"]:
                continue
            if metric == "pose":
                # incorrectly consider the distractor as manipulate objs:
                if any(
                    [
                        self.is_match(distractor_pose, target_pose, distractor_symmetry)
                        for target_pose in targs
                    ]
                ):
                    return False
                # incorrectly consider the distractor as targets
                for i in range(len(objs)):
                    object_id, (symmetry, _) = objs[i]
                    pose = p.getBasePositionAndOrientation(
                        object_id, physicsClientId=self.client_id
                    )
                    if self.is_match(pose, distractor_pose, symmetry):
                        return False
            else:
                raise NotImplementedError("class should implement for the zone metric")
        return True

    def check_success(self, *args, **kwargs) -> NamedTuple:
        """
        Check success. It should return a tuple of two boolean values, (success, failure).
        A trajectory will be terminated if fails.
        This function may be invoked at the final step.
        It may also be invoked every step for "constraint satisfaction" tasks.
        """
        raise NotImplementedError

    def check_constraint(self, *args, **kwargs):
        """
        Check constraint. This should be implemented by constraint satisfaction tasks.
        The result of constraint checking should be stored and used to determine
        episode success in check_success() later
        """
        return

    def generate_prompt(self, *args, **kwargs) -> Any:
        """
        Generate prompt from `self.prompt_template`, 'self.task_meta', and `self.placeholders`.
        This method may be invoked in `env.reset()`.
        Implementation of this method may vary in different tasks.
        """
        expressions = {}
        # for each placeholder items, generate required expressions
        for name, placeholder in self.placeholders.items():
            args = self.placeholder_expression[name]
            expressions[name] = placeholder.get_expression(**args)
        # now assemble the prompt
        prompt = deepcopy(self.prompt_template)
        assets = {}
        for name in self.placeholders:
            replacement = ""
            for expression_type in self.placeholder_expression[name]["types"]:
                if expression_type == "image":
                    replacement = replacement + "{" + name + "} "
                    assets[name] = expressions[name]["image"]
                else:
                    # text expression, e.g., name, novel_name, alias, etc
                    replacement = replacement + expressions[name][expression_type] + " "
            # stripe the last white space
            replacement = replacement[:-1]
            prompt = prompt.replace("{" + f"{name}" + "}", replacement)
        return prompt, assets

    def is_match(self, pose0, pose1, symmetry, position_only=False):
        """Check if pose0 and pose1 match within a threshold."""

        # Get translational error.
        diff_pos = np.float32(pose0[0][:2]) - np.float32(pose1[0][:2])
        dist_pos = np.linalg.norm(diff_pos)

        is_pos_match = dist_pos < self.pos_eps

        if position_only:
            is_rot_match = True
        else:
            # Get rotational error around z-axis (account for symmetries).
            diff_rot = 0
            if symmetry > 0:
                rot0 = np.array(utils.quatXYZW_to_eulerXYZ(pose0[1]))[2]
                rot1 = np.array(utils.quatXYZW_to_eulerXYZ(pose1[1]))[2]
                diff_rot = np.abs(rot0 - rot1) % symmetry
                if diff_rot > (symmetry / 2):
                    diff_rot = symmetry - diff_rot
            is_rot_match = diff_rot < self.rot_eps

        return is_pos_match and is_rot_match

    def get_true_image(self, env):
        """Get RGB-D orthographic heightmaps and segmentation masks."""

        # Capture near-orthographic RGB-D images and segmentation masks.
        # color: (C, H, W), depth: (1, H, W) within [0, 1]
        color, depth, segm = env.render_camera(self.oracle_cams[0])
        # process images to be compatible with oracle input
        # rgb image, from CHW to HWC
        color = np.transpose(color, (1, 2, 0))
        # depth image, from (1, H, W) to (H, W) within [0, 20]
        depth = 20.0 * np.squeeze(depth, axis=0)

        # Combine color with masks for faster processing.
        color = np.concatenate((color, segm[Ellipsis, None]), axis=2)

        # Reconstruct real orthographic projection from point clouds.
        hmaps, cmaps = utils.reconstruct_heightmaps(
            [color], [depth], self.oracle_cams, self.bounds, self.pix_size
        )

        # Split color back into color and masks.
        cmap = np.uint8(cmaps)[0, Ellipsis, :3]
        hmap = np.float32(hmaps)[0, Ellipsis]
        mask = np.int32(cmaps)[0, Ellipsis, 3:].squeeze()
        return cmap, hmap, mask

    def get_random_pose(self, env, obj_size):
        """Get random collision-free object pose within workspace bounds."""

        # Get erosion size of object in pixels.
        max_size = np.sqrt(obj_size[0] ** 2 + obj_size[1] ** 2)
        erode_size = int(np.round(max_size / self.pix_size))

        _, hmap, obj_mask = self.get_true_image(env)

        # Randomly sample an object pose within free-space pixels.
        free = np.ones(obj_mask.shape, dtype=np.uint8)
        for obj_ids in env.obj_ids.values():
            for obj_id in obj_ids:
                free[obj_mask == obj_id] = 0
        free[0, :], free[:, 0], free[-1, :], free[:, -1] = 0, 0, 0, 0
        free = cv2.erode(free, np.ones((erode_size, erode_size), np.uint8))
        if np.sum(free) == 0:
            return None, None
        pix = utils.sample_distribution(prob=np.float32(free), rng=self.rng)
        pos = utils.pix_to_xyz(pix, hmap, self.bounds, self.pix_size)
        pos = (pos[0], pos[1], obj_size[2] / 2)
        theta = self.rng.random() * 2 * np.pi
        rot = utils.eulerXYZ_to_quatXYZW((0, 0, theta))
        return pos, rot

    def fill_template(self, template, replace):
        """Read a file and replace key strings."""
        full_template_path = os.path.join(self.assets_root, template)
        with open(full_template_path, "r") as file:
            fdata = file.read()
        for field in replace:
            for i in range(len(replace[field])):
                fdata = fdata.replace(f"{field}{i}", str(replace[field][i]))
        alphabet = string.ascii_lowercase + string.digits
        rname = "".join(random.choices(alphabet, k=16))
        tmpdir = tempfile.gettempdir()
        template_filename = os.path.split(template)[-1]
        fname = os.path.join(tmpdir, f"{template_filename}.{rname}")
        with open(fname, "w") as file:
            file.write(fdata)
        return fname

    def get_random_size(self, size_range: SizeRange):
        random_size = self.rng.uniform(low=size_range.low, high=size_range.high)
        return tuple(random_size)

    def get_box_object_points(self, obj):
        obj_shape = p.getVisualShapeData(obj, physicsClientId=self.client_id)
        obj_dim = obj_shape[0][3]
        obj_dim = tuple(d for d in obj_dim)
        xv, yv, zv = np.meshgrid(
            np.arange(-obj_dim[0] / 2, obj_dim[0] / 2, 0.02),
            np.arange(-obj_dim[1] / 2, obj_dim[1] / 2, 0.02),
            np.arange(-obj_dim[2] / 2, obj_dim[2] / 2, 0.02),
            sparse=False,
            indexing="xy",
        )
        return np.vstack((xv.reshape(1, -1), yv.reshape(1, -1), zv.reshape(1, -1)))

    def get_mesh_object_points(self, obj):
        mesh = p.getMeshData(obj, physicsClientId=self.client_id)
        mesh_points = np.array(mesh[1])
        mesh_dim = np.vstack((mesh_points.min(axis=0), mesh_points.max(axis=0)))
        xv, yv, zv = np.meshgrid(
            np.arange(mesh_dim[0][0], mesh_dim[1][0], 0.02),
            np.arange(mesh_dim[0][1], mesh_dim[1][1], 0.02),
            np.arange(mesh_dim[0][2], mesh_dim[1][2], 0.02),
            sparse=False,
            indexing="xy",
        )
        return np.vstack((xv.reshape(1, -1), yv.reshape(1, -1), zv.reshape(1, -1)))

    def color_random_brown(self, obj):
        shade = np.random.rand() + 0.5
        color = np.float32([shade * 156, shade * 117, shade * 95, 255]) / 255
        p.changeVisualShape(obj, -1, rgbaColor=color, physicsClientId=self.client_id)


class Once:
    def __init__(self):
        self._triggered = False

    def __call__(self):
        if not self._triggered:
            self._triggered = True
            return True
        else:
            return False

    def __bool__(self):
        raise RuntimeError("`Once` objects should be used by calling ()")
