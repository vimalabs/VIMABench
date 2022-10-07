from __future__ import annotations

from typing import Literal, NamedTuple

import numpy as np
import pybullet as p
from pybullet import getBasePositionAndOrientation

from ..base import BaseTask
from ...components import Spatula, Push
from ...components.encyclopedia import ObjPedia, TexturePedia
from ...components.encyclopedia.definitions import TextureEntry
from ...utils import misc_utils as utils


class ResultTuple(NamedTuple):
    success: bool
    failure: bool
    distance: float | None


det_to_integer = {"any": 1, "one": 1, "two": 2, "three": 3}


class SweepObjectsToZoneBase(BaseTask):
    """
    Base class for tasks that sweep objects into a zone
    Shared helper functions for constraint checking
    """

    def __init__(
        self,
        # ====== task specific ======
        prompt_template: str,
        task_meta: dict,
        swept_obj_express_types: Literal["image", "name"] = "image",
        bounds_express_types: Literal["image", "name"] = "image",
        constraint_express_types: Literal["image", "name"] = "image",
        prepend_color_to_name: bool = True,
        oracle_max_steps: int | None = None,
        oracle_step_to_env_step_ratio: int = 3,
        possible_dragged_obj_texture: str
        | list[str]
        | TextureEntry
        | list[TextureEntry]
        | None = None,
        possible_base_obj_texture: str
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
        # probability of sampling number of swept objects given different difficulties
        self._sample_prob_all = {
            "easy": {1: 0.5, 2: 0.25, 3: 0.25},
            "medium": {1: 0.25, 2: 0.5, 3: 0.25},
            "hard": {1: 0.25, 2: 0.25, 3: 0.5},
        }
        task_meta["sample_prob"] = (self._sample_prob_all["easy"],)

        placeholder_expression = {
            "swept_obj": {
                "types": [swept_obj_express_types],
                "prepend_color": prepend_color_to_name,
            },
            "bounds": {
                "types": [bounds_express_types],
                "prepend_color": prepend_color_to_name,
            },
            "constraint": {
                "types": [constraint_express_types],
                "prepend_color": prepend_color_to_name,
            },
        }

        self.possible_dragged_obj = ObjPedia.SMALL_BLOCK

        self.possible_base_obj = ObjPedia.THREE_SIDED_RECTANGLE

        self.possible_constraint_obj = ObjPedia.LINE

        texture_list = [
            possible_dragged_obj_texture,
            possible_base_obj_texture,
        ]
        texture_default_values = [
            TexturePedia.all_entries()[: int(len(TexturePedia.all_entries()) / 2)],
            TexturePedia.all_entries()[int(len(TexturePedia.all_entries()) / 2) :],
        ]
        for idx, texture in enumerate(texture_list):
            if texture is None:
                texture_list[idx] = texture_default_values[idx]
            elif isinstance(texture, str):
                texture_list[idx] = [TexturePedia.lookup_color_by_name(texture)]
            elif isinstance(texture, TextureEntry):
                texture_list[idx] = [texture]
            elif isinstance(texture, list):
                if isinstance(texture[0], str):
                    texture_list[idx] = [
                        TexturePedia.lookup_color_by_name(obj) for obj in texture
                    ]
                elif isinstance(texture[0], TextureEntry):
                    texture_list[idx] = texture
                else:
                    raise ValueError("texture must be a list of str or TextureEntry")
            else:
                raise ValueError("texture must be a str or list of str or TextureEntry")
        self.possible_dragged_obj_texture = texture_list[0]
        self.possible_base_obj_texture = texture_list[1]

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
        self.pos_eps = 0.05

        # override necessary variable for Spatula and Push primitive
        self.ee = Spatula
        self.primitive = Push()

        # check constraint in simulation steps
        self.constraint_checking = {
            "enabled": True,
            "checking_sim_interval": 50,
            "satisfied": True,
            "obj_pts": None,
            "constraint_zone": None,
            "constraint_zone_with_margin": None,
        }

        # possible position for generating objects to be swept
        self.possible_swept_obj_born_pos = [
            (self.bounds[0, 0] + 0.32, self.bounds[1, 0] + 0.28),
            (self.bounds[0, 0] + 0.33, self.bounds[1, 0] + 0.53),
        ]

        # dict of distractors' obj_id: obj_pts
        self.distractors_pts: dict[int:int] = {}

    def check_percentage_in_zone(
        self,
        obj_id: int,
        pts: np.ndarray,
        zone: tuple[tuple, np.ndarray],
        zone_with_margin: list[float] | None = None,
    ) -> float:
        obj_pose = p.getBasePositionAndOrientation(
            obj_id, physicsClientId=self.client_id
        )
        # Fast checking
        if zone_with_margin is not None:
            top, bottom, left, right = zone_with_margin
            obj_pos = obj_pose[0]
            if (
                (obj_pos[0] < top)
                or (obj_pos[0] > bottom)
                or (obj_pos[1] < left)
                or (obj_pos[1] > right)
            ):
                return 0.0

        # Count valid points in zone.
        zone_pose, zone_size = zone
        world_to_zone = utils.invert(zone_pose)
        obj_to_zone = utils.multiply(world_to_zone, obj_pose)
        pts = np.float32(utils.apply(obj_to_zone, pts))
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
        zone_pts = np.sum(np.float32(valid_pts))
        total_pts = pts.shape[1]
        percentage_in_zone = zone_pts / total_pts
        return percentage_in_zone

    def is_touching_zone(
        self,
        obj_id: int,
        pts: np.ndarray,
        zone: tuple[tuple, np.ndarray],
        zone_with_margin: list[float] | None = None,
    ) -> bool:
        percentage_in_zone = self.check_percentage_in_zone(
            obj_id, pts, zone, zone_with_margin
        )
        return percentage_in_zone > 0

    def is_all_in_zone(
        self,
        obj_id: int,
        pts: np.ndarray,
        zone: tuple[tuple, np.ndarray],
        zone_with_margin: list[float] | None = None,
    ) -> bool:
        percentage_in_zone = self.check_percentage_in_zone(
            obj_id, pts, zone, zone_with_margin
        )
        return np.abs(1 - percentage_in_zone) < 0.001

    def check_constraint(self):
        if not self.constraint_checking["satisfied"]:
            return
        obj_pts = self.constraint_checking["obj_pts"]
        constraint_zone = self.constraint_checking["constraint_zone"]
        zone_with_margin = self.constraint_checking["constraint_zone_with_margin"]
        for obj_id in obj_pts.keys():
            if self.is_touching_zone(
                obj_id, obj_pts[obj_id], constraint_zone, zone_with_margin
            ):
                self.constraint_checking["satisfied"] = False
                return

    def check_distractors(self, *args) -> bool:
        """check whether distractor (or any other object) is at the goal"""
        zone_pts = 0
        zones = self._all_goals[0][6][1]
        for zone_pose, zone_size in zones:
            for distractor_id, distractor_pts in self.distractors_pts.items():
                distractor_pose = p.getBasePositionAndOrientation(
                    distractor_id, physicsClientId=self.client_id
                )
                world_to_zone = utils.invert(zone_pose)
                obj_to_zone = utils.multiply(world_to_zone, distractor_pose)
                pts = np.float32(utils.apply(obj_to_zone, distractor_pts))
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
                    zone_pts += np.sum(np.float32(valid_pts))
                if zone_pts > 0:
                    return False
        return True

    def set_difficulty(self, difficulty: str):
        super().set_difficulty(difficulty)
        if difficulty == "easy":
            self.task_meta["sample_prob"] = self._sample_prob_all["easy"]
            self.task_meta["constraint_length"] = 0.75
        elif difficulty == "medium":
            self.task_meta["sample_prob"] = self._sample_prob_all["medium"]
            self.task_meta["constraint_length"] = 1.35
        else:
            self.task_meta["sample_prob"] = self._sample_prob_all["hard"]
            self.task_meta["constraint_length"] = 1.85
        self.constraint_checking["enabled"] = True

    def check_success(self, *args, **kwargs) -> ResultTuple:
        # check failure by checking constraint
        if (
            self.constraint_checking["enabled"]
            and not self.constraint_checking["satisfied"]
        ):
            return ResultTuple(success=False, failure=True, distance=None)
        # check if distractors in the zone
        if not self.check_distractors():
            return ResultTuple(success=False, failure=True, distance=None)

        # check success by checking the number of goals to be achieved
        # must achieve the exact number of goals
        if self._n_achieved == self._target_num:
            return ResultTuple(success=True, failure=False, distance=None)
        elif self._n_achieved > self._target_num:
            return ResultTuple(success=False, failure=True, distance=None)
        else:
            # iterative over all objects to be manipulated to check failure
            failures = []
            for goal in self.goals:
                obj_ids, _, bases, _, _, _, _, _ = goal
                for obj_id_packed in obj_ids:
                    obj_id, (symmetry, _) = obj_id_packed
                    obj_pose = getBasePositionAndOrientation(
                        obj_id, physicsClientId=self.client_id
                    )
                    failures.append(
                        not self._valid_workspace.contains(
                            np.array(obj_pose[0][:2], dtype=self._valid_workspace.dtype)
                        )
                    )
            return ResultTuple(
                success=False,
                failure=any(failures),
                distance=None,
            )

    def update_goals(self, skip_oracle=True):
        n_goals_before_update = len(self.goals)

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
            # Evaluate by measuring object intersection with zone.
            zone_pts, total_pts = 0, 0
            obj_pts, zones = params
            for zone_idx, (zone_pose, zone_size) in enumerate(zones):

                # Count valid points in zone.
                for obj_idx, obj_id in enumerate(obj_pts):
                    pts = obj_pts[obj_id]
                    obj_pose = getBasePositionAndOrientation(
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
                indices_to_pop.append(goal_index)

        # remove completed goals
        for index in sorted(indices_to_pop, reverse=True):
            del self.goals[index]

        n_goals_after_update = len(self.goals)
        self._n_achieved += n_goals_before_update - n_goals_after_update
