from __future__ import annotations

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


class PickInOrderThenRestore(BaseTask):

    task_name = "pick_in_order_then_restore"

    def __init__(
        self,
        # ====== task specific ======
        num_dragged_obj: int = 1,
        num_distractor_dragged_obj: int = 0,
        num_target_base_obj: int = 2,
        num_array_rows: int = 2,
        num_array_columns: int = 3,
        same_obj: bool = False,
        same_bases: bool = True,
        dragged_obj_express_types: Literal["image", "name"] = "image",
        base_obj_express_types: Literal["image", "name"] = "image",
        prepend_color_to_name: bool = True,
        oracle_max_steps: int = 5,
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
            num_array_rows == 2
        ), f"The number of rows {num_array_rows} in array is not supported"

        task_meta = {
            "num_dragged_obj": num_dragged_obj,
            "num_distractor_dragged_obj": num_distractor_dragged_obj,
            "num_target_base_obj": num_target_base_obj,  # target square
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
        }
        for i in range(num_target_base_obj):
            placeholder_expression[f"base_obj_{i+1}"] = {
                "types": [base_obj_express_types],
                "prepend_color": [prepend_color_to_name],
            }

        if possible_dragged_obj is None:
            self.possible_dragged_obj = [
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
                "Put {dragged_obj} into {base_obj}. "
                "Finally restore it into its original container."
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
        self.pos_eps = 0.05
        # whether all dragged and distracted objs have the same shape
        self._same_obj = same_obj
        # whether all bases have the same shape or not
        self._same_bases = same_bases
        # in case no difficulty level is set
        self._update_task_meta()

    def _update_task_meta(self):
        # Infer the number of distractors based on the task meta
        self.task_meta["num_distractor_base_obj"] = (
            self.task_meta["num_array_rows"] * self.task_meta["num_array_columns"]
            - self.task_meta["num_target_base_obj"]
            - self.task_meta["num_dragged_obj"]
        )

        # possible position for generating objects & squares
        # (position of center)
        # 1. calculate bounds
        _, y_bound, _ = self.bounds
        pos_y_interval = (y_bound[1] - y_bound[0]) / (
            1 + (self.task_meta["num_array_columns"])
        )
        # possible born positions 2xN array
        self.possible_born_pos_center = [
            (
                self.bounds[0, 0] + (self.bounds[0, 1] - self.bounds[0, 0]) / 4,
                self.bounds[1, 0] + (1 + y) * pos_y_interval,
            )
            for y in range(self.task_meta["num_array_columns"])
        ] + [
            (
                self.bounds[0, 0] + (self.bounds[0, 1] - self.bounds[0, 0]) * 3 / 4,
                self.bounds[1, 0] + (1 + y) * pos_y_interval,
            )
            for y in range(self.task_meta["num_array_columns"])
        ]

    def _reset_prompt(self):
        # for more than one target bases
        num_base_obj = self.task_meta["num_target_base_obj"]
        bases_prompt = [" {" + f"base_obj_{i+1}" + "}" for i in range(num_base_obj)]
        if num_base_obj > 1:
            for i in range(1, num_base_obj * 2 - 1, 2):
                bases_prompt.insert(i, " then")
        self.prompt_template = "".join(
            [
                "Put {dragged_obj} into",
                *bases_prompt,
                ". Finally restore it into its original container.",
            ]
        )

    def reset(self, env):
        super().reset(env)
        self._reset_prompt()
        # sample everything first
        (
            n_row,
            n_col,
            num_target_base_obj,
            num_distractor_dragged_obj,
            num_dragged_obj,
        ) = (
            self.task_meta["num_array_rows"],
            self.task_meta["num_array_columns"],
            self.task_meta["num_target_base_obj"],
            self.task_meta["num_distractor_dragged_obj"],
            self.task_meta["num_dragged_obj"],
        )

        sampled_pos_centers = self.rng.choice(
            self.possible_born_pos_center,
            size=n_row * n_col,
            replace=False,
        )
        dragged_obj_base_pos_center = sampled_pos_centers[0]
        target_base_pos = sampled_pos_centers[1 : num_target_base_obj + 1]
        distractor_nonempty_base_pos_centers = sampled_pos_centers[
            num_target_base_obj
            + 1 : num_target_base_obj
            + 1
            + num_distractor_dragged_obj
        ]
        distractor_empty_base_pos_centers = sampled_pos_centers[
            1 + num_target_base_obj + num_distractor_dragged_obj :
        ]

        # sample dragged object and distractor objects
        possible_dragged_objs = (
            [self.rng.choice(self.possible_dragged_obj)]
            if self._same_obj
            else self.possible_dragged_obj
        )
        sampled_objs = [
            obj.value
            for obj in self.rng.choice(
                possible_dragged_objs,
                size=num_dragged_obj + num_distractor_dragged_obj,
            )
        ]
        sampled_dragged_obj = sampled_objs[0]
        sampled_distractors = sampled_objs[1:]
        sampled_obj_textures = [
            e.value
            for e in self.rng.choice(
                self.possible_dragged_obj_texture,
                size=num_dragged_obj + num_distractor_dragged_obj,
                replace=False,
            )
        ]
        sampled_dragged_obj_texture = sampled_obj_textures[0]
        sampled_distractor_obj_textures = sampled_obj_textures[1:]

        # sample all bases
        possible_base_obj = (
            [self.rng.choice(self.possible_base_obj)]
            if self._same_bases
            else self.possible_base_obj
        )
        sampled_bases = [
            obj.value
            for obj in self.rng.choice(
                possible_base_obj,
                size=n_row * n_col,
            )
        ]
        sampled_obj_base = sampled_bases[0]
        sampled_target_bases = sampled_bases[1 : num_target_base_obj + 1]
        sampled_distractor_bases = sampled_bases[
            num_target_base_obj
            + 1 : num_target_base_obj
            + num_distractor_dragged_obj
            + 1
        ]
        sampled_empty_bases = sampled_bases[
            num_target_base_obj + num_distractor_dragged_obj + 1 :
        ]

        sampled_base_textures = [
            e.value
            for e in self.rng.choice(
                self.possible_base_obj_texture,
                size=n_row * n_col,
                replace=False,
            )
        ]
        # sample dragged obj's base and target base(s) textures
        sampled_base_obj_texture = sampled_base_textures[0]
        sampled_base_target_textures = sampled_base_textures[
            1 : num_target_base_obj + 1
        ]

        sampled_distractor_base_textures = sampled_base_textures[
            num_target_base_obj
            + 1 : num_target_base_obj
            + 1
            + num_distractor_dragged_obj
        ]
        sampled_empty_base_textures = sampled_base_textures[
            num_target_base_obj + 1 + num_distractor_dragged_obj :
        ]

        dragged_obj_sampled_size = self.rng.uniform(
            low=sampled_dragged_obj.size_range.low,
            high=sampled_dragged_obj.size_range.high,
        )
        base_sampled_size = self.rng.uniform(
            low=sampled_obj_base.size_range.low,
            high=sampled_obj_base.size_range.high,
        )
        rot = eulerXYZ_to_quatXYZW((0, 0, 0))

        # add dragged object and its base
        dragged_obj_base_pose = (
            (
                dragged_obj_base_pos_center[0],
                dragged_obj_base_pos_center[1],
                base_sampled_size[2] / 2,
            ),
            rot,
        )
        _scale = (
            2
            if sampled_obj_base in [ObjPedia.BOWL.value, ObjPedia.PALLET.value]
            else 0.5
        )
        dragged_obj_pose = (
            (
                dragged_obj_base_pos_center[0],
                dragged_obj_base_pos_center[1],
                dragged_obj_sampled_size[2] * _scale,
            ),
            rot,
        )

        dragged_obj_id, dragged_obj_urdf_full_path, _ = self.add_object_to_env(
            env=env,
            obj_entry=sampled_dragged_obj,
            color=sampled_dragged_obj_texture,
            pose=dragged_obj_pose,
            size=dragged_obj_sampled_size,
            retain_temp=True,
        )

        dragged_obj_base_id, _, _ = self.add_object_to_env(
            env=env,
            obj_entry=sampled_obj_base,
            color=sampled_base_obj_texture,
            pose=dragged_obj_base_pose,
            size=base_sampled_size,
            category="fixed",
            retain_temp=True,
        )

        dragged_obj = (dragged_obj_id, (0, None))
        self.placeholders["dragged_obj"] = PlaceholderObj(
            name=sampled_dragged_obj.name,
            obj_id=dragged_obj_id,
            urdf=dragged_obj_urdf_full_path,
            novel_name=sampled_dragged_obj.novel_name,
            alias=sampled_dragged_obj.alias,
            color=sampled_dragged_obj_texture,
            image_size=self._placeholder_img_size,
            seed=self.seed,
        )

        # 2. add target object base(s)
        for i in range(num_target_base_obj):
            target_base_pose = (
                (
                    target_base_pos[i][0],
                    target_base_pos[i][1],
                    base_sampled_size[2] / 2,
                ),
                rot,
            )
            target_object_pose = (
                (
                    target_base_pos[i][0],
                    target_base_pos[i][1],
                    dragged_obj_sampled_size[2] / 2,
                ),
                rot,
            )
            target_obj_base_id, target_obj_base_urdf, _ = self.add_object_to_env(
                env=env,
                obj_entry=sampled_target_bases[i],
                color=sampled_base_target_textures[i],
                pose=target_base_pose,
                size=base_sampled_size,
                category="fixed",
            )
            self.placeholders[f"base_obj_{i+1}"] = PlaceholderObj(
                name=sampled_target_bases[i].name,
                obj_id=target_obj_base_id,
                urdf=target_obj_base_urdf,
                novel_name=sampled_target_bases[i].novel_name,
                alias=sampled_target_bases[i].alias,
                color=sampled_base_target_textures[i],
                image_size=self._placeholder_img_size,
                seed=self.seed,
            )
            self.goals.append(
                (
                    [dragged_obj],
                    np.ones((1, 1)),
                    [target_object_pose],
                    False,
                    True,
                    "pose",
                    None,
                    1 / (num_target_base_obj + 1),
                )
            )
            # add the initial obj base in the end
            if i == num_target_base_obj - 1:
                self.goals.append(
                    (
                        (
                            [dragged_obj],
                            np.ones((1, 1)),
                            [dragged_obj_base_pose],
                            False,
                            True,
                            "pose",
                            None,
                            1 / (num_target_base_obj + 1),
                        )
                    )
                )
        self._all_goals = self.goals.copy()

        # 3.1 add distractor objects & object's base
        self.distractors = []
        for (
            distractor_pose_center,
            sampled_distractor,
            sampled_distractor_obj_texture,
            sampled_distractor_base,
            sampled_distractor_base_texture,
        ) in zip(
            distractor_nonempty_base_pos_centers,
            sampled_distractors,
            sampled_distractor_obj_textures,
            sampled_distractor_bases,
            sampled_distractor_base_textures,
        ):
            distractor_base_pose = (
                (
                    distractor_pose_center[0],
                    distractor_pose_center[1],
                    base_sampled_size[2] / 2,
                ),
                rot,
            )
            self.add_object_to_env(
                env=env,
                obj_entry=sampled_distractor_base,
                color=sampled_distractor_base_texture,
                pose=distractor_base_pose,
                size=base_sampled_size,
                category="fixed",
            )

            distractor_pose = (
                (
                    distractor_pose_center[0],
                    distractor_pose_center[1],
                    dragged_obj_sampled_size[2] / 2,
                ),
                rot,
            )
            obj_id, _, _ = self.add_object_to_env(
                env=env,
                obj_entry=sampled_distractor,
                color=sampled_distractor_obj_texture,
                pose=distractor_pose,
                size=dragged_obj_sampled_size,
            )
            self.distractors.append((obj_id, (0, None)))

        # 3.2 add empty distractor base objects
        for sampled_empty_base, base_texture, base_pose_center in zip(
            sampled_empty_bases,
            sampled_empty_base_textures,
            distractor_empty_base_pos_centers,
        ):
            empty_base_pose = (
                (
                    base_pose_center[0],
                    base_pose_center[1],
                    base_sampled_size[2] / 2,
                ),
                rot,
            )
            self.add_object_to_env(
                env=env,
                obj_entry=sampled_empty_base,
                color=base_texture,
                pose=empty_base_pose,
                size=base_sampled_size,
                category="fixed",
            )

    def set_difficulty(self, difficulty: str):
        super().set_difficulty(difficulty)
        if difficulty == "easy":
            # 2x2 array
            pass
        elif difficulty == "medium":
            # 2x3
            self.task_meta["num_array_columns"] = 3
        else:
            # 2x3, add one more intermediate target base
            self.task_meta["num_array_columns"] = 3
            self.task_meta["num_target_base_obj"] = 3
            self.placeholder_expression["base_obj_3"] = {
                "types": self.placeholder_expression["base_obj_1"]["types"],
                "prepend_color": self.placeholder_expression["base_obj_1"][
                    "prepend_color"
                ],
            }
        self._update_task_meta()
        self.oracle_max_steps = self.task_meta["num_target_base_obj"] + 2

    def check_success(self, *args, **kwargs) -> ResultTuple:
        # check if the agent deviates from the required order
        original_container_goal = self._all_goals[-1]
        (
            objs,
            matches,
            targs,
            _,
            _,
            metric,
            params,
            max_progress,
        ) = original_container_goal
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
            at_original_container = True
        else:
            at_original_container = False

        if at_original_container:
            future_goals = self.goals[1:-1]
        else:
            future_goals = self.goals[1:]
        for future_goal in future_goals:
            objs, matches, targs, _, _, metric, params, max_progress = future_goal
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
                # the agent completes the future goal instead of the nearest goal
                # deviates from the order, failed
                return ResultTuple(success=False, failure=True, distance=None)

        # check success by checking the number of goals to be achieved
        for distractor in self.distractors:
            if not self.check_distractors(distractor):
                return ResultTuple(success=False, failure=True, distance=self.pos_eps)
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

    def is_match(self, pose0, pose1, symmetry):
        return super().is_match(pose0, pose1, symmetry, position_only=True)
