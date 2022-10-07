from __future__ import annotations

import math
from abc import ABC
from typing import NamedTuple

import numpy as np
import pybullet as p

from ..base import BaseTask
from ...components.encyclopedia import ObjPedia, TexturePedia
from ...components.encyclopedia.definitions import ObjEntry, TextureEntry
from ...components.placeholders import PlaceholderObj
from ...utils.misc_utils import quatXYZW_to_eulerXYZ, eulerXYZ_to_quatXYZW


class ResultTuple(NamedTuple):
    success: bool
    failure: bool
    distance: float
    rot_error: float


class RotateTheObjBase(BaseTask, ABC):
    """
    rotating objects base class
    """

    def __init__(
        self,
        # ====== task specific ======
        prompt_template: str,
        task_meta: dict[str:int],
        placeholder_expression: dict[str:dict],
        oracle_max_steps: int = 2,  # one mistake is allowed
        oracle_step_to_env_step_ratio: int = 4,
        possible_angles_of_rotation: list[float | int] | float | int | None = None,
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
        is_subclassed_by_twist_task: bool = False,
    ):
        if possible_dragged_obj is None:
            self.possible_dragged_obj = ObjPedia.all_entries_no_rotational_symmetry()
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

        if possible_angles_of_rotation is None:
            # [start = 1/6 * pi, end = 5/6 * pi], step_size = 1/6 * pi
            # to try the best for keeping all rotations clockwise
            self.possible_angles_of_rotation = [
                1 / 6 * math.pi * i for i in range(1, 6)
            ]
        elif isinstance(possible_angles_of_rotation, (float, int)):
            self.possible_angles_of_rotation = [possible_angles_of_rotation]
        elif isinstance(possible_angles_of_rotation, list) and all(
            isinstance(angle, (float, int)) for angle in possible_angles_of_rotation
        ):
            self.possible_angles_of_rotation = [
                float(angle) for angle in possible_angles_of_rotation
            ]
        else:
            raise ValueError(
                "possible_angles_of_rotation must be None or float or a list of floats"
            )

        super().__init__(
            prompt_template=prompt_template,
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
        assert all(
            self.rot_eps < abs(e) for e in self.possible_angles_of_rotation
        ), f"Angle(s) of rotation {self.possible_angles_of_rotation} is too small."
        self.pos_eps = 0.1
        # Sampled in reset
        self.sampled_dragged_obj_texture = None

        self._is_subclassed_by_twist_task = is_subclassed_by_twist_task

    def _reset_prompt(self, *args, **kwargs):
        self.sampled_angle_of_rotation = self.rng.choice(
            self.possible_angles_of_rotation
        )
        self.sampled_dragged_obj_texture = self.rng.choice(
            self.possible_dragged_obj_texture
        ).value

    def reset(
        self, env, same_dragged_obj: bool = True, same_dragged_color: bool = True
    ):
        super().reset(env)
        self._reset_prompt(same_dragged_obj)

        # add dragged obj
        not_reach_max_times = False
        num_added_dragged_obj = 0
        dragged = []
        dragged_poses = []
        dragged_entries_archive = []
        for i in range(
            self.task_meta["num_dragged_obj"] + self.REJECT_SAMPLING_MAX_TIMES
        ):
            if same_dragged_obj:
                sampled_dragged_obj = self.rng.choice(self.possible_dragged_obj).value
                sampled_dragged_obj_size = self.rng.uniform(
                    low=sampled_dragged_obj.size_range.low,
                    high=sampled_dragged_obj.size_range.high,
                )

            for _ in range(self.task_meta["num_dragged_obj"]):
                if not same_dragged_obj:
                    sampled_dragged_obj = self.rng.choice(
                        self.possible_dragged_obj
                    ).value
                    sampled_dragged_obj_size = self.rng.uniform(
                        low=sampled_dragged_obj.size_range.low,
                        high=sampled_dragged_obj.size_range.high,
                    )
                    if not same_dragged_color:
                        self.sampled_dragged_obj_texture = self.rng.choice(
                            self.possible_dragged_obj_texture
                        ).value

                # noinspection PyUnboundLocalVariable
                obj_id, urdf, pose = self.add_object_to_env(
                    env,
                    sampled_dragged_obj,
                    self.sampled_dragged_obj_texture,
                    sampled_dragged_obj_size,
                    category="rigid",
                )

                if obj_id is None:
                    print(
                        f"Warning: {i + 1} repeated sampling when try to spawn dragged object"
                    )
                    break
                else:
                    dragged.append((obj_id, (sampled_dragged_obj.symmetry, None)))
                    dragged_poses.append(pose)
                    dragged_entries_archive.append(
                        (sampled_dragged_obj, self.sampled_dragged_obj_texture, urdf)
                    )
                    num_added_dragged_obj += 1

                    if not self._is_subclassed_by_twist_task:
                        placeholder = "dragged_obj"
                        placeholder += (
                            ""
                            if placeholder in self.placeholder_expression.keys()
                            or same_dragged_obj
                            else f"_{num_added_dragged_obj}"
                        )
                        self.placeholders[placeholder] = PlaceholderObj(
                            name=sampled_dragged_obj.name,
                            obj_id=obj_id,
                            urdf=urdf,
                            novel_name=sampled_dragged_obj.novel_name,
                            alias=sampled_dragged_obj.alias,
                            color=self.sampled_dragged_obj_texture,
                            image_size=self._placeholder_img_size,
                            seed=self.seed,
                        )
            if num_added_dragged_obj == self.task_meta["num_dragged_obj"]:
                not_reach_max_times = True
                break
        if not not_reach_max_times:
            raise ValueError("Error in adding object to env.")

        # set goals
        for dragged_obj, dragged_pose in zip(dragged, dragged_poses):
            dragged_obj_euler = quatXYZW_to_eulerXYZ(dragged_pose[1])
            #  rotate direction as clockwise
            target_rot_quat = eulerXYZ_to_quatXYZW(
                (
                    *dragged_obj_euler[:2],
                    dragged_obj_euler[2] - self.sampled_angle_of_rotation,
                )
            )

            target_pose = dragged_pose[0], target_rot_quat
            self.goals.append(
                (
                    [dragged_obj],
                    np.ones((1, 1)),
                    [target_pose],
                    False,
                    True,
                    "pose",
                    None,
                    1,
                )
            )
        self._all_goals = self.goals.copy()

        # add distractors
        self.distractors = []
        not_reach_max_times = False
        num_added_distractors = 0

        for i in range(
            self.task_meta["num_distractors"] + self.REJECT_SAMPLING_MAX_TIMES
        ):
            sampled_distractor_obj = self.rng.choice(self.possible_dragged_obj).value
            sampled_distractor_obj_texture = self.rng.choice(
                self.possible_dragged_obj_texture
            ).value
            # different from for dragged obj for simplicity
            while sampled_distractor_obj_texture == self.sampled_dragged_obj_texture:
                sampled_distractor_obj_texture = self.rng.choice(
                    self.possible_dragged_obj_texture
                ).value
            # check the validity of the samples
            is_valid_sample = True
            for dragged_entry in dragged_entries_archive:
                if (
                    sampled_distractor_obj == dragged_entry[0]
                    and sampled_distractor_obj_texture == dragged_entry[1]
                ):
                    is_valid_sample = False
                    break
            if not is_valid_sample:
                continue
            sampled_distractor_obj_size = self.rng.uniform(
                low=sampled_distractor_obj.size_range.low,
                high=sampled_distractor_obj.size_range.high,
            )

            obj_id, urdf, pose = self.add_object_to_env(
                env,
                sampled_distractor_obj,
                sampled_distractor_obj_texture,
                sampled_distractor_obj_size,
                category="rigid",
            )

            if obj_id is None:
                print(
                    f"Warning: {i + 1} repeated sampling when try to spawn distractor object"
                )
            else:
                num_added_distractors += 1
                self.distractors.append((obj_id, (0, None)))
            if num_added_distractors == self.task_meta["num_distractors"]:
                not_reach_max_times = True
                break
        if not not_reach_max_times:
            raise ValueError("Error in adding object to env.")

        # for twist_object_by_grounding_new_verbs task
        return dragged_entries_archive

    def check_success(self, *args, **kwargs) -> ResultTuple:
        # check for distractors:
        for distractor in self.distractors:
            if not self.check_distractors(distractor):
                return ResultTuple(
                    success=False,
                    failure=True,
                    distance=self.pos_eps,
                    rot_error=self.rot_eps,
                )
        # check success by checking the number of goals to be achieved
        all_achieved = len(self.goals) == 0
        if all_achieved:
            # when all goals are achieved, distance is simply the threshold
            return ResultTuple(
                success=True,
                failure=False,
                distance=self.pos_eps,
                rot_error=self.rot_eps,
            )
        else:
            # iterative over all objects to be manipulated to check failure
            failures = []
            distances = []
            rot_errors = []
            for goal in self.goals:
                obj, _, bases, _, _, _, _, _ = goal
                obj_id, (symmetry, _) = obj[0]
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
                # calculate rotational error
                rot0 = np.array(quatXYZW_to_eulerXYZ(obj_pose[1]))[2]
                rot1 = np.array(quatXYZW_to_eulerXYZ(bases[0][1]))[2]
                diff_rot = np.abs(rot0 - rot1) % symmetry
                if diff_rot > (symmetry / 2):
                    diff_rot = symmetry - diff_rot
                rot_errors.append(diff_rot)

            return ResultTuple(
                success=False,
                failure=any(failures),
                distance=sum(distances) / len(distances),
                rot_error=sum(rot_errors) / len(rot_errors),
            )

    def update_goals(self, skip_oracle=True):
        """
        Need to check all goals every time step
        """
        # check len of goals, if zero, all goals have been achieved
        if len(self.goals) == 0:
            return

        # each step of checking needs to enumerate all goals
        indices_to_pop = []
        for goal_index in range(len(self.goals)):
            goal_to_check = self.goals[goal_index]
            # Unpack goal step.
            objs, matches, targs, _, _, metric, params, max_progress = goal_to_check

            # Evaluate by matching object poses.
            progress_per_goal = 0
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
            if np.abs(max_progress - progress_per_goal) < 0.001:
                indices_to_pop.append(goal_index)
        # remove completed goals
        for index in sorted(indices_to_pop, reverse=True):
            del self.goals[index]
