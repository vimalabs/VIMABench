from __future__ import annotations

import itertools
from functools import partial
from typing import NamedTuple, Literal

import numpy as np
import pybullet as p

from ..base import BaseTask
from ...components.encyclopedia import ObjPedia, TexturePedia
from ...components.encyclopedia.definitions import ObjEntry, TextureEntry
from ...components.placeholders import PlaceholderObj
from ...utils.misc_utils import eulerXYZ_to_quatXYZW


class ResultTuple(NamedTuple):
    success: bool
    failure: bool
    distance: float | None


class ManipulateOldNeighbor(BaseTask):
    """
    Tasks requiring memory: pick an object into object, and then put its old neighbor into that object.
    """

    task_name = "manipulate_old_neighbor"

    def __init__(
        self,
        # ====== task specific ======
        num_dragged_obj: int = 2,
        num_base_obj: int = 1,
        num_null_distractors: int = 1,
        num_array_rows: int = 3,
        num_array_columns: int = 2,
        dragged_obj_express_types: Literal["image", "name"] = "image",
        base_obj_express_types: Literal["image", "name"] = "image",
        prepend_color_to_name: bool = True,
        oracle_max_steps: int = 4,
        oracle_step_to_env_step_ratio: int = 4,
        possible_dragged_obj: str | list[str] | ObjEntry | list[ObjEntry] | None = None,
        possible_base_obj: str | list[str] | ObjEntry | list[ObjEntry] | None = None,
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
        assert (
            num_dragged_obj == 2
        ), "Only one dragged object and one of its neighbor to be dragged are allowed"
        assert num_null_distractors <= (
            num_array_rows * num_array_columns - num_dragged_obj
        ), "too many null distractors to sample!"

        task_meta = {
            "num_dragged_obj": num_dragged_obj,
            "num_base_obj": num_base_obj,
            # distractors consist of visible object and free distracting positions in the array.
            # to force the agent to memorize the initial position of the dragged object.
            "num_null_distractors": num_null_distractors,
            # workspace configuration (object_array: 2x2 2x3 2x4)
            "num_array_rows": num_array_rows,
            # num_array_columns could take the value of 2,3,4,...
            "num_array_columns": num_array_columns,
        }

        placeholder_expression = {
            "dragged_obj": {
                "types": [dragged_obj_express_types],
                "prepend_color": [prepend_color_to_name],
            },
            "base_obj": {
                "types": [base_obj_express_types],
                "prepend_color": [prepend_color_to_name],
            },
        }

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

        if possible_base_obj is None:
            self.possible_base_obj = [
                ObjPedia.PAN,
                ObjPedia.BOWL,
                ObjPedia.FRAME,
                ObjPedia.SQUARE,
                ObjPedia.CONTAINER,
                ObjPedia.PALLET,
            ]
        elif isinstance(possible_base_obj, str):
            self.possible_base_obj = [ObjPedia.lookup_object_by_name(possible_base_obj)]
        elif isinstance(possible_base_obj, ObjEntry):
            self.possible_base_obj = [possible_base_obj]
        elif isinstance(possible_base_obj, list):
            if isinstance(possible_base_obj[0], str):
                self.possible_base_obj = [
                    ObjPedia.lookup_object_by_name(obj) for obj in possible_base_obj
                ]
            elif isinstance(possible_base_obj[0], ObjEntry):
                self.possible_base_obj = possible_base_obj
            else:
                raise ValueError("possible_base_obj must be a list of str or ObjEntry")
        else:
            raise ValueError(
                "possible_base_obj must be a str or list of str or ObjEntry"
            )

        if possible_dragged_obj_texture is None:
            self.possible_dragged_obj_texture = TexturePedia.all_entries()[
                : int(len(TexturePedia.all_entries()) / 2)
            ]
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

        if possible_base_obj_texture is None:
            self.possible_base_obj_texture = TexturePedia.all_entries()[
                int(len(TexturePedia.all_entries()) / 2) :
            ]
        elif isinstance(possible_base_obj_texture, str):
            self.possible_base_obj_texture = [
                TexturePedia.lookup_color_by_name(possible_base_obj_texture)
            ]
        elif isinstance(possible_base_obj_texture, TextureEntry):
            self.possible_base_obj_texture = [possible_base_obj_texture]
        elif isinstance(possible_base_obj_texture, list):
            if isinstance(possible_base_obj_texture[0], str):
                self.possible_base_obj_texture = [
                    TexturePedia.lookup_color_by_name(obj)
                    for obj in possible_base_obj_texture
                ]
            elif isinstance(possible_base_obj_texture[0], TextureEntry):
                self.possible_base_obj_texture = possible_base_obj_texture
            else:
                raise ValueError(
                    "possible_base_obj_texture must be a list of str or TextureEntry"
                )
        else:
            raise ValueError(
                "possible_base_obj_texture must be a str or list of str or TextureEntry"
            )

        super().__init__(
            prompt_template=(
                "First put {dragged_obj} into {base_obj} "
                "then put the object that was previously at its {direction} into the same {base_obj}."
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

        self._neighbor_obj = None

    def _reset_prompt(self):

        # possible position for generating objects & squares
        # (position of center)
        # 1. calculate bounds
        num_row = self.task_meta["num_array_rows"]
        num_col = self.task_meta["num_array_columns"]
        x_bound, y_bound, _ = self.bounds
        pos_y_interval = (y_bound[1] - y_bound[0]) / (
            1 + num_col + 1  # additional one for base object
        )
        pos_x_interval = (x_bound[1] - x_bound[0]) / (1 + num_row)
        # possible born positions 2xN array
        self.possible_born_pos_center = [
            [
                (
                    self.bounds[0, 0] + (1 + x) * pos_x_interval,
                    self.bounds[1, 0] + (1 + y) * pos_y_interval,
                )
                for y in range(num_col)
            ]
            for x in range(num_row)
        ]
        self.base_pos = (
            self.bounds[0, 0] + (self.bounds[0, 1] - self.bounds[0, 0]) / 2,
            self.bounds[1, 1] - pos_y_interval,
        )
        # sample the index of dragged object and the indices of null distractors where no object will be put here
        candidate_indices = [
            r
            for r in itertools.product(
                list(range(0, num_row)),
                list(range(0, num_col)),
            )
        ]

        self.sampled_dragged_obj_idx = candidate_indices[
            self.rng.choice(range(len(candidate_indices)))
        ]
        # sample neighboring object to be dragged
        neighbor_sample_space = {
            "north": (-1, 0),
            "south": (1, 0),
            "west": (0, -1),
            "east": (0, 1),
        }

        def is_valid_idx(idx_to_check: tuple[int, int]) -> bool:
            if (
                idx_to_check[0] < 0
                or idx_to_check[1] < 0
                or idx_to_check[0] >= self.task_meta["num_array_rows"]
                or idx_to_check[1] >= self.task_meta["num_array_columns"]
            ):
                return False
            return True

        while True:
            sampled_neighbor_dir = self.rng.choice(list(neighbor_sample_space.keys()))
            self.sampled_neighbor_idx = (
                self.sampled_dragged_obj_idx[0]
                + neighbor_sample_space[sampled_neighbor_dir][0],
                self.sampled_dragged_obj_idx[1]
                + neighbor_sample_space[sampled_neighbor_dir][1],
            )
            if is_valid_idx(self.sampled_neighbor_idx):
                break

        def is_valid_null_distractor(
            idx_to_check: tuple[int, int], existing_null_distractors: list
        ):
            if (
                idx_to_check == self.sampled_dragged_obj_idx
                or idx_to_check == self.sampled_neighbor_idx
            ):
                # duplicate indexes
                return False
            elif idx_to_check in existing_null_distractors:
                return False
            else:
                return True

        self.sampled_null_distractor_indices = []
        for i in range(self.task_meta["num_null_distractors"]):
            while True:
                sampled_null_distractor_idx = candidate_indices[
                    self.rng.choice(range(len(candidate_indices)))
                ]
                if is_valid_null_distractor(
                    sampled_null_distractor_idx, self.sampled_null_distractor_indices
                ):
                    self.sampled_null_distractor_indices.append(
                        sampled_null_distractor_idx
                    )
                    break

        partial_str = partial(
            "First put {dragged_obj} into {base_obj} "
            "then put the object that was previously at its {direction} into the same {base_obj}.".format,
            direction=sampled_neighbor_dir,
        )
        self.prompt_template = partial_str(
            dragged_obj="{dragged_obj}", base_obj="{base_obj}"
        )

    def reset(self, env):
        super().reset(env)
        self._reset_prompt()
        zero_rot = eulerXYZ_to_quatXYZW((0, 0, 0))

        possible_dragged_combo = [
            r
            for r in itertools.product(
                self.possible_dragged_obj, self.possible_dragged_obj_texture
            )
        ]

        # 1. add array of dragged objects & its neighbor & distractors
        sampled_combos = [
            (e[0].value, e[1].value)
            for e in self.rng.choice(
                possible_dragged_combo,
                replace=False,
                size=self.task_meta["num_array_rows"]
                * self.task_meta["num_array_columns"],
            )
        ]
        sampled_dragged_objects = [e[0] for e in sampled_combos]
        sampled_dragged_obj_textures = [e[1] for e in sampled_combos]

        # add them to env
        dragged_obj = []
        dragged_obj_entries = []
        neighbor_obj = []
        self.distractors = []
        for row_idx, row in enumerate(self.possible_born_pos_center):
            for col_idx, pos_center in enumerate(row):
                one_dim_idx = row_idx * self.task_meta["num_array_columns"] + col_idx
                sampled_obj = sampled_dragged_objects[one_dim_idx]
                sampled_obj_texture = sampled_dragged_obj_textures[one_dim_idx]
                sampled_obj_size = self.rng.uniform(
                    low=sampled_obj.size_range.low,
                    high=sampled_obj.size_range.high,
                )
                obj_pose = (
                    (
                        pos_center[0],
                        pos_center[1],
                        sampled_obj_size[2] / 2,
                    ),
                    zero_rot,
                )
                if (row_idx, col_idx) not in self.sampled_null_distractor_indices:
                    obj_id, urdf, pose = self.add_object_to_env(
                        env,
                        sampled_obj,
                        sampled_obj_texture,
                        sampled_obj_size,
                        pose=obj_pose,
                        category="rigid",
                    )

                    if (
                        row_idx == self.sampled_dragged_obj_idx[0]
                        and col_idx == self.sampled_dragged_obj_idx[1]
                    ):
                        dragged_obj.append((obj_id, (0, None)))
                        dragged_obj_entries.append(
                            (sampled_obj, obj_id, urdf, sampled_obj_texture)
                        )
                    elif (
                        row_idx == self.sampled_neighbor_idx[0]
                        and col_idx == self.sampled_neighbor_idx[1]
                    ):
                        neighbor_obj.append((obj_id, (0, None)))
                    else:
                        self.distractors.append((obj_id, (0, None)))
                else:
                    pass

        # add base object
        sampled_base_obj = self.rng.choice(self.possible_base_obj).value
        sampled_base_texture = self.rng.choice(self.possible_base_obj_texture).value
        sampled_base_size = self.rng.uniform(
            low=sampled_base_obj.size_range.low,
            high=sampled_base_obj.size_range.high,
        )
        base_pose = (
            self.base_pos[0],
            self.base_pos[1],
            sampled_base_size[2] / 2,
        ), zero_rot
        base_obj_id, base_urdf, base_pose = self.add_object_to_env(
            env,
            sampled_base_obj,
            sampled_base_texture,
            sampled_base_size,
            pose=base_pose,
            category="fixed",
        )
        # --- set goals ---
        # goal of dragged object
        for obj in dragged_obj + neighbor_obj:
            self.goals.append(
                (
                    [obj],
                    np.ones((1, 1)),
                    [base_pose],
                    False,
                    True,
                    "pose",
                    None,
                    1,
                )
            )
        self._all_goals = self.goals.copy()
        assert len(neighbor_obj) == 1
        self._neighbor_obj = neighbor_obj[0]

        # add placeholder objects
        dragged_obj_entry = dragged_obj_entries[0]
        self.placeholders["dragged_obj"] = PlaceholderObj(
            name=dragged_obj_entry[0].name,
            obj_id=dragged_obj_entry[1],
            urdf=dragged_obj_entry[2],
            novel_name=dragged_obj_entry[0].novel_name,
            alias=dragged_obj_entry[0].alias,
            color=dragged_obj_entry[3],
            image_size=self._placeholder_img_size,
            seed=self.seed,
        )
        self.placeholders["base_obj"] = PlaceholderObj(
            name=sampled_base_obj.name,
            obj_id=base_obj_id,
            urdf=base_urdf,
            novel_name=sampled_base_obj.novel_name,
            alias=sampled_base_obj.alias,
            color=sampled_base_texture,
            image_size=self._placeholder_img_size,
            seed=self.seed,
        )

    def set_difficulty(self, difficulty: str):
        super().set_difficulty(difficulty)
        if difficulty == "easy":
            # 3x2 array, 1 null distractor, 3 "real" distractor
            pass
        elif difficulty == "medium":
            # 3x3 array, 3 null distractor, 4 "real" distractor
            self.task_meta["num_null_distractors"] = 3
            self.task_meta["num_array_rows"] = 3
            self.task_meta["num_array_columns"] = 3
        else:
            # 3x4 array, 4 null distractor, 6 "real" distractor
            self.task_meta["num_null_distractors"] = 4
            self.task_meta["num_array_rows"] = 3
            self.task_meta["num_array_columns"] = 4

    def check_success(self, *args, **kwargs) -> ResultTuple:
        # check if the agent manipulates the neighbor object first
        # if that happened, it should be a failure
        if len(self.goals) == 1:
            left_obj = self.goals[0][0][0]
            if left_obj != self._neighbor_obj:
                return ResultTuple(success=False, failure=True, distance=None)

        # check distractors
        for distractor in self.distractors:
            if not self.check_distractors(distractor):
                return ResultTuple(success=False, failure=True, distance=self.pos_eps)
        # check success by checking the number of goals to be achieved
        all_achieved = len(self.goals) == 0
        if all_achieved:
            # when all goals are achieved, distance is simply the threshold
            return ResultTuple(success=True, failure=False, distance=self.pos_eps)
        else:
            # iterative over all objects to be manipulated to check failure
            failures = []
            distances = []
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
            return ResultTuple(
                success=False,
                failure=any(failures),
                distance=sum(distances) / len(distances),
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

    def is_match(self, pose0, pose1, symmetry):
        return super().is_match(pose0, pose1, symmetry, position_only=True)
