from __future__ import annotations

import math
from functools import partial
from typing import NamedTuple, Literal

import numpy as np
import pybullet as p

from ..base import BaseTask
from ...components.encyclopedia import ObjPedia, TexturePedia
from ...components.encyclopedia.definitions import ObjEntry, TextureEntry
from ...components.placeholders import PlaceholderScene, PlaceholderObj
from ...components.placeholders.placeholder_scene import SceneRenderEnv


class ResultTuple(NamedTuple):
    success: bool
    failure: bool
    distance: float


class FollowMotion(BaseTask):
    """One shot imitation task: Follow the motion of the object {video frames}
    In this task, the agent is required to imitate the moves in the given one shot video trajectory.
    Here we define video trajectory by variable length of frames of image.
    This task is one of the tasks that have strict requirements on the manipulation sequence.
    The agent is required to precisely follow a leader object's motion, which is presented in the prompt frames.
    And here we define the motion as a random walk along a polygon that approximates a circle.
    """

    task_name = "follow_motion"

    def __init__(
        self,
        # ====== task specific ======
        num_dragged_obj: int = 1,
        num_frames: int = 3,
        circle_radius: float = 0.19,
        num_possible_motion_points: int = 5,
        scene_express_types: Literal["image", "name"] = "image",
        dragged_obj_express_types: Literal["image", "name"] = "image",
        placeholder_scene_views: str | list[str] | None = None,
        prepend_color_to_name: bool = True,
        oracle_max_steps: int = 4,
        oracle_step_to_env_step_ratio: int = 4,
        possible_dragged_obj: str | list[str] | ObjEntry | list[ObjEntry] | None = None,
        possible_dragged_obj_texture: str
        | list[str]
        | TextureEntry
        | list[TextureEntry]
        | None = None,
        # ====== general ======
        obs_img_views: str | list[str] | None = None,
        obs_img_size: tuple[int, int] = (128, 256),
        placeholder_img_size: tuple[int, int] = (128, 256),
        seed: int | None = None,
        debug: bool = False,
    ):
        task_meta = {
            # 1 for now
            "num_dragged_obj": num_dragged_obj,
            "num_frames": num_frames,
        }
        placeholder_expression = {
            f"frame_{i}": {
                "types": [scene_express_types],
            }
            for i in range(num_frames)
        }
        self._scene_express_types = scene_express_types
        placeholder_expression["dragged_obj"] = {
            "types": [dragged_obj_express_types],
            "prepend_color": prepend_color_to_name,
        }

        assert (
            placeholder_scene_views is None
            or str
            or (
                isinstance(placeholder_scene_views, list)
                and all(isinstance(views, str) for views in placeholder_scene_views)
            )
        ), (
            f"`placeholder_scene_views` must be a str, list of str, or None, "
            f"but get {type(placeholder_scene_views)}"
        )
        self._placeholder_scene_views = placeholder_scene_views

        if possible_dragged_obj is None:
            self.possible_dragged_obj = [
                ObjPedia.BLOCK,
                ObjPedia.L_BLOCK,
                ObjPedia.CAPITAL_LETTER_A,
                ObjPedia.CAPITAL_LETTER_E,
                ObjPedia.CAPITAL_LETTER_G,
                ObjPedia.CAPITAL_LETTER_M,
                ObjPedia.CAPITAL_LETTER_R,
                ObjPedia.CAPITAL_LETTER_T,
                ObjPedia.CAPITAL_LETTER_V,
                ObjPedia.CROSS,
                ObjPedia.DIAMOND,
                ObjPedia.TRIANGLE,
                ObjPedia.FLOWER,
                ObjPedia.HEART,
                ObjPedia.HEXAGON,
                ObjPedia.PENTAGON,
                ObjPedia.RING,
                ObjPedia.ROUND,
                ObjPedia.STAR,
            ]
        elif isinstance(possible_dragged_obj, str):
            self.possible_dragged_obj = [
                ObjPedia.lookup_object_by_name(possible_dragged_obj)
            ]
        elif isinstance(possible_dragged_obj, ObjEntry):
            self.possible_dragged_obj = [possible_dragged_obj]
        elif isinstance(possible_dragged_obj, list):
            if isinstance(possible_dragged_obj[0], str):
                self.possible_dragged_obj = [
                    ObjPedia.lookup_object_by_name(obj) for obj in possible_dragged_obj
                ]
            elif isinstance(possible_dragged_obj[0], ObjEntry):
                self.possible_dragged_obj = possible_dragged_obj
            else:
                raise ValueError(
                    "possible_dragged_obj must be a list of str or ObjEntry"
                )
        else:
            raise ValueError(
                "possible_dragged_obj must be a str or list of str or ObjEntry"
            )

        if possible_dragged_obj_texture is None:
            self.possible_dragged_obj_texture = TexturePedia.all_entries()
        elif isinstance(possible_dragged_obj_texture, str):
            self.possible_dragged_obj_texture = [
                TexturePedia.lookup_color_by_name(possible_dragged_obj_texture)
            ]
        elif isinstance(possible_dragged_obj_texture, TextureEntry):
            self.possible_dragged_obj_texture = [possible_dragged_obj_texture]
        elif isinstance(possible_dragged_obj_texture, list):
            if isinstance(possible_dragged_obj_texture[0], str):
                self.possible_dragged_obj_texture = [
                    TexturePedia.lookup_color_by_name(obj)
                    for obj in possible_dragged_obj_texture
                ]
            elif isinstance(possible_dragged_obj_texture[0], TextureEntry):
                self.possible_dragged_obj_texture = possible_dragged_obj_texture
            else:
                raise ValueError(
                    "possible_dragged_obj_texture must be a list of str or TextureEntry"
                )
        else:
            raise ValueError(
                "possible_dragged_obj_texture must be a str or list of str or TextureEntry"
            )

        super().__init__(
            prompt_template=(
                "Follow this motion for {dragged_obj}: "
                + " ".join(["{{frame_{i}}}".format(i=i) for i in range(num_frames)])
                + "."
            ),
            task_meta=task_meta,
            placeholder_expression=placeholder_expression,
            oracle_max_steps=oracle_max_steps,
            oracle_step_to_env_step_ratio=oracle_step_to_env_step_ratio,
            obs_img_views=obs_img_views,
            obs_img_size=obs_img_size,
            placeholder_img_size=placeholder_img_size,
            seed=seed,
            debug=debug,
        )
        self.pos_eps = 0.1
        self.circle_radius = circle_radius
        self.num_possible_motion_points = num_possible_motion_points

        # --- get the center of motion  ----
        # calculate bounds
        x_bound, y_bound, _ = self.bounds
        pos_x_center = (x_bound[1] + x_bound[0]) / 2
        pos_y_center = (y_bound[1] + y_bound[0]) / 2
        self.motion_center_pos = (pos_x_center, pos_y_center)

        # get the motion sequence
        self.possible_motion_points = self.all_motion_point_coordinates()

        self._n_goals_before, self._n_goals_after = None, None

    def all_motion_point_coordinates(self) -> list:
        """
        Get all valid motion points (points that approximates a circle)
        """
        points = []
        for i in range(self.num_possible_motion_points):
            new_x = (
                math.cos(i * 2 * math.pi / self.num_possible_motion_points)
                * self.circle_radius
            )
            new_y = (
                math.sin(i * 2 * math.pi / self.num_possible_motion_points)
                * self.circle_radius
            )
            points.append((new_x, new_y))
        return points

    def get_one_random_motion_sequence(self):
        # random walk motion sequence
        sequence = []
        start_point_idx = self.rng.choice(range(len(self.possible_motion_points)))
        sequence.append(self.possible_motion_points[start_point_idx])

        curr_point_idx = start_point_idx
        for i in range(self.num_operations):
            if self.rng.random() < 0.5:  # turn left
                curr_point_idx = (curr_point_idx - 1) % self.num_possible_motion_points
            else:
                curr_point_idx = (curr_point_idx + 1) % self.num_possible_motion_points
            sequence.append(self.possible_motion_points[curr_point_idx])
        return sequence

    def _reset_prompt(self):
        self.num_operations = self.task_meta["num_frames"] - 1
        self.possible_motion_points = self.all_motion_point_coordinates()

        self.prompt_template = (
            "Follow this motion for {dragged_obj}: "
            + " ".join(
                [
                    "{{frame_{i}}}".format(i=i)
                    for i in range(self.task_meta["num_frames"])
                ]
            )
            + "."
        )
        self.placeholder_expression = {
            "dragged_obj": self.placeholder_expression["dragged_obj"],
            **{
                f"frame_{i}": {
                    "types": [self._scene_express_types],
                }
                for i in range(self.task_meta["num_frames"])
            },
        }

        # NO mistakes are allowed for tasks with strict manipulation sequence
        self.oracle_max_steps = self.num_operations
        # For tasks with strict requirement on manipulation sequence,
        # use this variable to maintain the total number of performed op
        self.num_op_performed = 0

    def reset(self, env):
        super().reset(env)
        self._reset_prompt()

        # add dragged obj
        sampled_dragged_obj = self.rng.choice(self.possible_dragged_obj).value
        sampled_dragged_obj_texture = self.rng.choice(
            self.possible_dragged_obj_texture
        ).value

        moving_seq = self.get_one_random_motion_sequence()
        zero_rot = (0, 0, 0, 1)
        # put the dragged obj at the start point of the moving seq
        start_point = moving_seq[0]
        not_reach_max_times = False
        dragged_entries_archive = []

        sampled_dragged_obj_size = self.rng.uniform(
            low=sampled_dragged_obj.size_range.low,
            high=sampled_dragged_obj.size_range.high,
        )
        height_dragged_obj = sampled_dragged_obj_size[2]

        dragged_obj_start_pos = (
            self.motion_center_pos[0] + start_point[0],
            self.motion_center_pos[1] + start_point[1],
            height_dragged_obj,
        )
        dragged_obj_start_pose = dragged_obj_start_pos, zero_rot

        dragged_obj_id, dragged_obj_urdf, dragged_obj_pose = self.add_object_to_env(
            env,
            sampled_dragged_obj,
            sampled_dragged_obj_texture,
            sampled_dragged_obj_size,
            pose=dragged_obj_start_pose,
            category="rigid",
        )
        dragged_entries_archive.append(
            (sampled_dragged_obj, sampled_dragged_obj_texture)
        )

        self.placeholders["dragged_obj"] = PlaceholderObj(
            name=sampled_dragged_obj.name,
            obj_id=dragged_obj_id,
            urdf=dragged_obj_urdf,
            novel_name=sampled_dragged_obj.novel_name,
            alias=sampled_dragged_obj.alias,
            color=sampled_dragged_obj_texture,
            image_size=self._placeholder_img_size,
            seed=self.seed,
        )

        # set goals
        dragged_obj = (dragged_obj_id, (0, None))
        for point in moving_seq[1:]:
            dragged_obj_tar_pos = (
                self.motion_center_pos[0] + point[0],
                self.motion_center_pos[1] + point[1],
                height_dragged_obj,
            )
            dragged_obj_tar_pose = dragged_obj_tar_pos, zero_rot

            self.goals.append(
                (
                    [dragged_obj],
                    np.ones((1, 1)),
                    [dragged_obj_tar_pose],
                    False,
                    True,
                    "pose",
                    None,
                    1,
                )
            )
        assert len(self.goals) == len(moving_seq) - 1

        def _valid_distractor(entry: ObjEntry, texture: TextureEntry) -> bool:
            for dragged_entry in dragged_entries_archive:
                if entry == dragged_entry[0] and texture == dragged_entry[1]:
                    return False
            return True

        # only one distractor in the center (hardly ever reject sampling)
        not_reach_max_times = False
        for i in range(self.REJECT_SAMPLING_MAX_TIMES):
            sampled_distractor_obj = self.rng.choice(self.possible_dragged_obj).value
            sampled_distractor_obj_texture = self.rng.choice(
                self.possible_dragged_obj_texture
            ).value
            # check the validity of the samples
            if _valid_distractor(
                sampled_distractor_obj, sampled_distractor_obj_texture
            ):
                sampled_distractor_obj_size = self.rng.uniform(
                    low=sampled_distractor_obj.size_range.low,
                    high=sampled_distractor_obj.size_range.high,
                )

                distractor_pos = (
                    self.motion_center_pos[0],
                    self.motion_center_pos[1],
                    sampled_distractor_obj_size[2],
                )
                distractor_pose = distractor_pos, zero_rot
                obj_id, urdf, pose = self.add_object_to_env(
                    env=env,
                    obj_entry=sampled_distractor_obj,
                    color=sampled_distractor_obj_texture,
                    size=sampled_distractor_obj_size,
                    pose=distractor_pose,
                    category="rigid",
                )
                if obj_id is not None:
                    not_reach_max_times = True
                    break
        if not not_reach_max_times:
            raise ValueError("Error in adding object to env.")

        # sample distractor in prompt scene (hardly ever reject sampling)
        for i in range(self.REJECT_SAMPLING_MAX_TIMES):
            sampled_distractor_obj_in_scene = self.rng.choice(
                self.possible_dragged_obj
            ).value
            sampled_distractor_obj_texture_in_scene = self.rng.choice(
                self.possible_dragged_obj_texture
            ).value
            # check the validity of the samples
            if _valid_distractor(
                sampled_distractor_obj_in_scene, sampled_distractor_obj_texture_in_scene
            ):
                break

        # create scene placeholder in the prompt
        def create_scene_fn(scene_render_env: SceneRenderEnv, _, frame_idx: int):
            point = moving_seq[frame_idx]
            dragged_obj_pos_at_frame_i = (
                self.motion_center_pos[0] + point[0],
                self.motion_center_pos[1] + point[1],
                height_dragged_obj,
            )
            dragged_obj_pose_at_frame_i = dragged_obj_pos_at_frame_i, zero_rot

            self.add_object_to_env(
                scene_render_env,
                sampled_dragged_obj,
                sampled_dragged_obj_texture,
                sampled_dragged_obj_size,
                pose=dragged_obj_pose_at_frame_i,
                category="rigid",
            )
            # add only one distractor in scene
            sampled_distractor_obj_size_in_scene = self.rng.uniform(
                low=sampled_distractor_obj.size_range.low,
                high=sampled_distractor_obj.size_range.high,
            )
            distractor_pos = (
                self.motion_center_pos[0],
                self.motion_center_pos[1],
                sampled_distractor_obj_size[2],
            )
            distractor_pose = distractor_pos, zero_rot
            self.add_object_to_env(
                scene_render_env,
                sampled_distractor_obj_in_scene,
                sampled_distractor_obj_texture_in_scene,
                sampled_distractor_obj_size_in_scene,
                pose=distractor_pose,
                category="rigid",
            )

        for i in range(self.task_meta["num_frames"]):
            self.placeholders[f"frame_{i}"] = PlaceholderScene(
                create_scene_fn=partial(create_scene_fn, frame_idx=i),
                assets_root=self.assets_root,
                views=self._placeholder_scene_views,
                image_size=self._placeholder_img_size,
                seed=self.seed,
            )

    def check_success(self, *args, **kwargs) -> ResultTuple:
        # check success by checking the number of goals to be achieved
        all_achieved = len(self.goals) == 0
        if all_achieved:
            # when all goals are achieved, distance is simply the threshold
            return ResultTuple(success=True, failure=False, distance=self.pos_eps)
        else:
            # iterative over all objects to be manipulated to check failure
            failures = []
            distances = []
            # check not in_workspace failure
            for goal in self.goals:
                obj, _, bases, _, _, _, _, _ = goal
                obj_id, _ = obj[0]
                obj_pose = p.getBasePositionAndOrientation(
                    obj_id, physicsClientId=self.client_id
                )
                failures.append(
                    not self._valid_workspace.contains(
                        np.array(obj_pose[0][:2], dtype=self._valid_workspace.dtype)
                    )
                )
                distances.append(
                    np.linalg.norm(
                        np.float32(obj_pose[0][:2]) - np.float32(bases[0][0][:2])
                    )
                )
            is_failure = any(failures) or (
                self._n_goals_before - self._n_goals_after == 0
            )
            return ResultTuple(
                success=False,
                failure=is_failure,
                distance=sum(distances) / len(distances),
            )

    def set_difficulty(self, difficulty: str):
        super().set_difficulty(difficulty)
        if difficulty == "easy":
            self.task_meta["num_frames"] = 3
        elif difficulty == "medium":
            self.task_meta["num_frames"] = 4
        else:
            self.task_meta["num_frames"] = 5
        self.oracle_max_steps = self.task_meta["num_frames"] + 1

    def update_goals(self, skip_oracle=True):
        self._n_goals_before = len(self.goals)
        rtn = super().update_goals(skip_oracle=skip_oracle)
        self._n_goals_after = len(self.goals)
        return rtn

    def is_match(self, pose0, pose1, symmetry):
        return super().is_match(pose0, pose1, symmetry, position_only=True)
