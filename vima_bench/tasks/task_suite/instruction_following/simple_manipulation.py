from __future__ import annotations

import itertools
from typing import NamedTuple, Literal

import numpy as np
import pybullet as p

from ..base import BaseTask
from ...components.encyclopedia import ObjPedia, TexturePedia
from ...components.encyclopedia.definitions import ObjEntry, TextureEntry
from ...components.placeholders import PlaceholderObj
from ...utils.pybullet_utils import if_in_hollow_object


class ResultTuple(NamedTuple):
    success: bool
    failure: bool
    distance: float


class SimpleManipulation(BaseTask):
    """
    Instruction following task: Put the {dragged_obj} into the {base_obj}.
    """

    task_name = "visual_manipulation"

    def __init__(
        self,
        # ====== task specific ======
        num_dragged_obj: int = 1,
        num_base_obj: int = 1,
        num_distractors_obj: int = 1,
        dragged_obj_express_types: Literal["image", "name"] = "image",
        base_obj_express_types: Literal["image", "name"] = "image",
        prepend_color_to_name: bool = True,
        oracle_max_steps: int = 3,  # two mistakes are allowed
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
        use_neutral_color: bool = False,
        exclude_distractor_by_geometry: bool = False,
        # ====== general ======
        obs_img_views: str | list[str] | None = None,
        obs_img_size: tuple[int, int] = (128, 256),
        placeholder_img_size: tuple[int, int] = (128, 256),
        seed: int | None = None,
        debug: bool = False,
    ):
        task_meta = {
            "num_dragged_obj": num_dragged_obj,
            "num_base_obj": num_base_obj,
            "num_distractors_obj": num_distractors_obj,
        }
        placeholder_expression = {
            "base_obj": {
                "types": [base_obj_express_types],
                "prepend_color": [prepend_color_to_name],
            },
        }
        for i in range(1, num_dragged_obj + 1):
            placeholder_expression[f"dragged_obj_{i}"] = {
                "types": [dragged_obj_express_types],
                "prepend_color": [prepend_color_to_name],
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
            prompt_template="Put the {dragged_obj} into the {base_obj}.",
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
        self._use_neutral_color = use_neutral_color
        self._exclude_distractor_by_geometry = exclude_distractor_by_geometry

    def reset(self, env):
        self._reset_prompt()
        super().reset(env)

        sampled_colors = [
            self.rng.choice(self.possible_base_obj_texture).value,
            [
                t.value
                for t in self.rng.choice(
                    self.possible_dragged_obj_texture,
                    size=self.task_meta["num_dragged_obj"],
                )
            ],
        ]

        # add base objects
        base_poses = []
        not_reach_max_times = False
        for i in range(self.REJECT_SAMPLING_MAX_TIMES):
            sampled_base_obj = self.rng.choice(self.possible_base_obj).value
            base_size = self.rng.uniform(
                low=sampled_base_obj.size_range.low,
                high=sampled_base_obj.size_range.high,
            )
            obj_id, urdf_full_path, pose = self.add_object_to_env(
                env=env,
                obj_entry=sampled_base_obj,
                color=sampled_colors[0],
                size=base_size,
                category="fixed",
                retain_temp=True,
            )
            if obj_id is not None:
                base_poses.append(pose)
                # add placeholder objects
                self.placeholders["base_obj"] = PlaceholderObj(
                    name=sampled_base_obj.name,
                    obj_id=obj_id,
                    urdf=urdf_full_path,
                    novel_name=sampled_base_obj.novel_name,
                    alias=sampled_base_obj.alias,
                    color=sampled_colors[0],
                    image_size=self._placeholder_img_size,
                    seed=self.seed,
                    use_neutral_color=self._use_neutral_color,
                )
                not_reach_max_times = True
                break
            else:
                print(
                    f"Warning: {i + 1} repeated sampling when try to spawn base object"
                )
        if not not_reach_max_times:
            raise ValueError("Error in sampling base object")

        # add dragged objects
        dragged_poses = []
        dragged = []
        not_reach_max_times = False
        n_added_dragged_obj = 0
        for i in range(self.REJECT_SAMPLING_MAX_TIMES):
            sampled_dragged_obj = self.rng.choice(self.possible_dragged_obj).value
            dragged_size = self.rng.uniform(
                low=sampled_dragged_obj.size_range.low,
                high=sampled_dragged_obj.size_range.high,
            )
            dragged_pose = self.get_random_pose(env, dragged_size)
            # noinspection PyUnboundLocalVariable
            if dragged_size[0] is None or dragged_pose[1] is None:
                print(
                    f"Warning: {i + 1} repeated sampling when try to spawn dragged object"
                )
                continue
            elif sampled_base_obj in [
                ObjPedia.FRAME.value,
                ObjPedia.SQUARE.value,
            ] and if_in_hollow_object(
                [dragged_pose], dragged_size, base_poses, base_size
            ):
                print(
                    f"Warning: {i + 1} repeated sampling when try to spawn dragged object"
                )
                continue
            obj_id, urdf_full_path, _ = self.add_object_to_env(
                env=env,
                obj_entry=sampled_dragged_obj,
                color=sampled_colors[1][n_added_dragged_obj],
                size=dragged_size,
                pose=dragged_pose,
                retain_temp=True,
            )
            if obj_id is not None:
                dragged_poses.append(dragged_pose)
                dragged.append((obj_id, (0, None)))
                # add placeholder objects
                self.placeholders[
                    f"dragged_obj_{n_added_dragged_obj + 1}"
                ] = PlaceholderObj(
                    name=sampled_dragged_obj.name,
                    obj_id=obj_id,
                    urdf=urdf_full_path,
                    novel_name=sampled_dragged_obj.novel_name,
                    alias=sampled_dragged_obj.alias,
                    color=sampled_colors[1][n_added_dragged_obj],
                    image_size=self._placeholder_img_size,
                    seed=self.seed,
                    use_neutral_color=self._use_neutral_color,
                )
                n_added_dragged_obj += 1
            else:
                print(
                    f"Warning: {i + 1} repeated sampling when try to spawn dragged object"
                )
                continue
            if n_added_dragged_obj == self.task_meta["num_dragged_obj"]:
                not_reach_max_times = True
                break
        if not not_reach_max_times:
            raise ValueError("Error in sampling dragged object")

        # set goal
        for dragged_obj in dragged:
            self.goals.append(
                (
                    [dragged_obj],
                    np.ones((1, 1)),
                    base_poses,
                    False,
                    True,
                    "pose",
                    None,
                    1 / self.task_meta["num_dragged_obj"],
                )
            )
        self._all_goals = self.goals.copy()

        # add distractor(s): we use the same object type for both dragged_obj and distractor_obj
        if self.task_meta["num_distractors_obj"] == 0:
            return

        possible_distractor_obj = self.possible_dragged_obj + self.possible_base_obj
        possible_distractor_texture = (
            self.possible_dragged_obj_texture + self.possible_base_obj_texture
        )
        _distracting_combos = [
            r
            for r in itertools.product(
                possible_distractor_obj, possible_distractor_texture
            )
        ]
        distracting_combos = []
        # remove used combo in dragg and base obj
        for combo in _distracting_combos:
            if (
                combo[0].value == sampled_base_obj
                and combo[1].value == sampled_colors[0]
            ) or (
                self._exclude_distractor_by_geometry
                and (
                    combo[0].value == sampled_base_obj
                    or combo[0].value == sampled_dragged_obj
                )
            ):
                pass
            elif (
                combo[0].value == sampled_dragged_obj
                and combo[1].value in sampled_colors[1]
            ) or (
                self._exclude_distractor_by_geometry
                and (
                    combo[0].value == sampled_dragged_obj
                    or combo[0].value == sampled_base_obj
                )
            ):
                pass
            else:
                distracting_combos.append(combo)
        self.distractors = []
        not_reach_max_times = False
        for i in range(
            self.task_meta["num_distractors_obj"] + self.REJECT_SAMPLING_MAX_TIMES
        ):
            distractor_obj, distractor_color = self.rng.choice(distracting_combos)
            distractor_obj_entry, distractor_color_entry = (
                distractor_obj.value,
                distractor_color.value,
            )
            distractor_size = self.rng.uniform(
                low=distractor_obj_entry.size_range.low,
                high=distractor_obj_entry.size_range.high,
            )

            distractor_pose = self.get_random_pose(env, distractor_size)
            if distractor_pose[0] is None or distractor_pose[1] is None:
                print(
                    f"Warning: {i + 1} repeated sampling when try to spawn distractors"
                )
                continue
            if distractor_obj in self.possible_base_obj:
                # prevent sampling in hollow square and frame cases
                # noinspection PyUnboundLocalVariable
                if distractor_obj in [
                    ObjPedia.FRAME,
                    ObjPedia.SQUARE,
                ] and if_in_hollow_object(
                    dragged_poses, dragged_size, [distractor_pose], distractor_size
                ):
                    print(
                        f"Warning: {i + 1} repeated sampling when try to spawn distractors"
                    )
                    continue
                obj_id, urdf_full_path, _ = self.add_object_to_env(
                    env=env,
                    obj_entry=distractor_obj_entry,
                    color=distractor_color_entry,
                    size=distractor_size,
                    pose=distractor_pose,
                    category="fixed",
                    retain_temp=True,
                )
            else:
                # prevent sampling in hollow square and frame cases
                if sampled_base_obj in [
                    ObjPedia.FRAME.value,
                    ObjPedia.SQUARE.value,
                ] and if_in_hollow_object(
                    [distractor_pose], distractor_size, base_poses, base_size
                ):
                    print(
                        f"Warning: {i + 1} repeated sampling when try to spawn distractors"
                    )
                    continue
                obj_id, urdf_full_path, distractor_pose = self.add_object_to_env(
                    env=env,
                    obj_entry=distractor_obj_entry,
                    color=distractor_color_entry,
                    size=distractor_size,
                    pose=distractor_pose,
                    retain_temp=True,
                )
            if obj_id is None:
                print(
                    f"Warning: {i + 1} repeated sampling when try to spawn distractors"
                )
                continue
            self.distractors.append((obj_id, (0, None)))
            if len(self.distractors) == self.task_meta["num_distractors_obj"]:
                not_reach_max_times = True
                break
        if not not_reach_max_times:
            raise ValueError("Error in sampling distractors")
        self._reset_prompt()

    def _reset_prompt(self):
        num_dragged_obj = self.task_meta["num_dragged_obj"]
        dragged_objs_prompt = [
            " {" + f"dragged_obj_{i}" + "}" for i in range(1, num_dragged_obj + 1)
        ]
        if num_dragged_obj > 1:
            for i in range(1, num_dragged_obj * 2 - 1, 2):
                dragged_objs_prompt.insert(i, " and")
        prompt = "".join(["Put the", *dragged_objs_prompt, " into the {base_obj}."])
        # updates
        self.prompt_template = prompt

    def check_success(self, *args, **kwargs) -> ResultTuple:
        # check success by checking the number of goals to be achieved
        all_achieved = len(self.goals) == 0
        if self.distractors is not None:
            for distractor in self.distractors:
                if not self.check_distractors(distractor):
                    return ResultTuple(
                        success=False, failure=True, distance=self.pos_eps
                    )
        if all_achieved:
            # when the goal is achieved, the distance is simply the threshold
            return ResultTuple(success=True, failure=False, distance=self.pos_eps)
        else:
            # check failure
            obj, _, base, _, _, _, _, _ = self.goals[0]
            obj_id, _ = obj[0]
            obj_pose = p.getBasePositionAndOrientation(
                obj_id, physicsClientId=self.client_id
            )
            is_failure = not self._valid_workspace.contains(
                np.array(obj_pose[0][:2], dtype=self._valid_workspace.dtype)
            )
            result = ResultTuple(
                success=False,
                failure=is_failure,
                distance=np.linalg.norm(
                    np.float32(obj_pose[0][:2]) - np.float32(base[0][0][:2])
                ),
            )
            return result

    def set_difficulty(self, difficulty: str):
        super().set_difficulty(difficulty)
        if difficulty == "easy":
            # "easy" is defined as same as trained
            pass
        elif difficulty == "medium":
            # exact one variation
            # randomly sample that should the variation happen in dragged obj or base obj
            if self.rng.random() > 0.5:
                self.possible_dragged_obj_texture = self.possible_base_obj_texture
            else:
                self.possible_base_obj_texture = self.possible_dragged_obj_texture
        elif difficulty == "hard":
            # more than one variation
            # in this case, both dragged and base obj will have novel combos
            self.possible_dragged_obj_texture, self.possible_base_obj_texture = (
                self.possible_base_obj_texture,
                self.possible_dragged_obj_texture,
            )

    def is_match(self, pose0, pose1, symmetry):
        return super().is_match(pose0, pose1, symmetry, position_only=True)
