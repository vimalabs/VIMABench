from __future__ import annotations

from typing import NamedTuple, Literal

import numpy as np
import pybullet as p

from ..base import BaseTask
from ...components.encyclopedia import ObjPedia, TexturePedia
from ...components.encyclopedia.definitions import ObjEntry, TextureEntry
from ...components.placeholders import PlaceholderObj, PlaceholderText
from ...utils.pybullet_utils import (
    get_urdf_path,
    if_in_hollow_object,
)


class ResultTuple(NamedTuple):
    success: bool
    failure: bool
    distance: float


class NovelAdj(BaseTask):

    task_name = "novel_adj"

    def __init__(
        self,
        # ====== task specific ======
        n_supports: int = 3,
        novel_adjectives: list[str] | str | None = None,
        adjective_types: list[str] | str | None = None,
        num_geometric_distractors: int = 0,
        dragged_obj_express_types: Literal["image", "name"] = "image",
        base_obj_express_types: Literal["image", "name"] = "image",
        demo_blicker_obj_express_types: Literal["image", "name"] = "image",
        demo_less_blicker_obj_express_types: Literal["image", "name"] = "image",
        prepend_color_to_name: bool = True,
        oracle_max_steps: int = 3,
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
        assert n_supports >= 1
        self._n_supports = n_supports
        task_meta = {
            "n_supports": n_supports,
            "difficulty": "easy",
            "num_geometric_distractors": num_geometric_distractors,
        }
        placeholder_expression = {
            "dragged_obj": {
                "types": [dragged_obj_express_types],
                "prepend_color": prepend_color_to_name,
            },
            "base_obj": {
                "types": [base_obj_express_types],
                "prepend_color": prepend_color_to_name,
            },
            "adv": {"types": ["text"]},
            "novel_adj": {"types": ["text"]},
        }
        for i in range(n_supports):
            placeholder_expression.update(
                {
                    f"demo_blicker_obj_{i+1}": {
                        "types": [demo_blicker_obj_express_types],
                        "prepend_color": prepend_color_to_name,
                    },
                    f"demo_less_blicker_obj_{i+1}": {
                        "types": [demo_less_blicker_obj_express_types],
                        "prepend_color": prepend_color_to_name,
                    },
                }
            )

        if novel_adjectives is None:
            self.novel_adjectives = ["daxer", "blicker", "modier", "kobar"]
        elif isinstance(novel_adjectives, (list, tuple)):
            for comp in novel_adjectives:
                assert isinstance(
                    comp, str
                ), f"{comp} has to be string in novel adjectives"
            self.novel_adjectives = novel_adjectives
        elif isinstance(novel_adjectives, str):
            self.novel_adjectives = [novel_adjectives]
        else:
            raise ValueError("novel_adjectives must be a str or list of str.")

        # define novel adjective
        if adjective_types is None:
            self.adjective_types = [
                # disable now for limited workspace, otherwise decrease objects sizes
                # "smaller/larger",
                "shorter/taller",
                "lighter color/darker color",
            ]
        elif isinstance(adjective_types, (list, tuple)):
            for comp in adjective_types:
                assert isinstance(
                    comp, str
                ), f"{comp} has to be string in adjective_types"
            self.adjective_types = adjective_types
        elif isinstance(adjective_types, str):
            self.adjective_types = [adjective_types]
        else:
            raise ValueError("adjective_types must be a str or list of str.")

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
            self.possible_dragged_obj_texture = TexturePedia.all_light_dark_entries()
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

        prompt_template = ""
        for i in range(n_supports):
            _blicker_str = "{" + f"demo_blicker_obj_{i+1}" + "}"
            _less_blicker_str = "{" + f"demo_less_blicker_obj_{i+1}" + "}"
            _novel_adj_str = "{novel_adj}"
            prompt_template += (
                f"{_blicker_str} is {_novel_adj_str} than {_less_blicker_str}. "
            )
        prompt_template += "Put the {adv}{novel_adj} {dragged_obj} into the {base_obj}."

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
        self.pos_eps = 0.1

    def _reset_prompt(self) -> bool:
        """reset prompt and return whether flip adjective"""
        adjective_name = self.rng.choice(self.novel_adjectives)
        flip_adjective = self.rng.choice([True, False])
        adverb = "less " if flip_adjective else ""
        self.placeholders["adv"] = PlaceholderText(adverb)
        self.placeholders["novel_adj"] = PlaceholderText(adjective_name)
        return flip_adjective

    def reset(self, env):
        super().reset(env)
        flip_adjective = self._reset_prompt()

        adjective_types = self.adjective_types.copy()
        self.rng.shuffle(adjective_types)
        adjective_definition = self.rng.choice(adjective_types[0].split("/"))

        # determine scalars
        ratio_small = self.rng.uniform(0.55, 0.6)
        ratio_large = ratio_small + 0.4
        ratio_short = self.rng.uniform(0.3, 0.4)
        ratio_tall = ratio_short + 2.0
        if adjective_definition in ["smaller", "larger"]:
            scalar_blicker, scalar_less_blicker = [ratio_small, ratio_small, 1.0], [
                ratio_large,
                ratio_large,
                1.0,
            ]
        elif adjective_definition in ["shorter", "taller"]:
            scalar_blicker, scalar_less_blicker = [1, 1, ratio_short], [
                1,
                1,
                ratio_tall,
            ]
        else:
            scalar_blicker, scalar_less_blicker = 1.0, 1.0
        if adjective_definition in ["larger", "taller"]:
            scalar_blicker, scalar_less_blicker = scalar_less_blicker, scalar_blicker
        dragged_obj_scalar, confusion_obj_scalar = scalar_blicker, scalar_less_blicker
        if flip_adjective:
            dragged_obj_scalar, confusion_obj_scalar = (
                confusion_obj_scalar,
                dragged_obj_scalar,
            )

        # determine colors
        sampled_base_obj_color = self.rng.choice(self.possible_base_obj_texture).value
        sampled_dragged_obj_color = self.rng.choice(
            self.possible_dragged_obj_texture
        ).value
        if adjective_definition == "lighter color":
            dragged_color = TexturePedia.find_lighter_variant(sampled_dragged_obj_color)
            confusion_color = TexturePedia.find_darker_variant(
                sampled_dragged_obj_color
            )
        elif adjective_definition == "darker color":
            dragged_color = TexturePedia.find_darker_variant(sampled_dragged_obj_color)
            confusion_color = TexturePedia.find_lighter_variant(
                sampled_dragged_obj_color
            )
        else:
            dragged_color = sampled_dragged_obj_color
            confusion_color = sampled_dragged_obj_color
        if flip_adjective:
            dragged_color, confusion_color = confusion_color, dragged_color

        # process demo objects
        for i in range(self._n_supports):
            sampled_demo_obj_color = self.rng.choice(
                self.possible_dragged_obj_texture
            ).value

            if adjective_definition == "lighter color":
                demo_blicker_color = TexturePedia.find_lighter_variant(
                    sampled_demo_obj_color
                )
                demo_less_blicker_color = TexturePedia.find_darker_variant(
                    sampled_demo_obj_color
                )
            elif adjective_definition == "darker color":
                demo_blicker_color = TexturePedia.find_darker_variant(
                    sampled_demo_obj_color
                )
                demo_less_blicker_color = TexturePedia.find_lighter_variant(
                    sampled_demo_obj_color
                )
            else:
                demo_blicker_color = sampled_demo_obj_color
                demo_less_blicker_color = sampled_demo_obj_color

            sampled_demo_obj = self.rng.choice(self.possible_dragged_obj).value
            demo_obj_size = self.rng.uniform(
                low=sampled_demo_obj.size_range.low,
                high=sampled_demo_obj.size_range.high,
            )
            demo_blicker_urdf = get_urdf_path(
                env=env,
                obj_entry=sampled_demo_obj,
                size=demo_obj_size,
                scaling=scalar_blicker,
            )
            self.placeholders[f"demo_blicker_obj_{i + 1}"] = PlaceholderObj(
                name=sampled_demo_obj.name,
                obj_id=-1,
                urdf=demo_blicker_urdf,
                novel_name=sampled_demo_obj.novel_name,
                alias=sampled_demo_obj.alias,
                color=demo_blicker_color,
                image_size=self._placeholder_img_size,
                seed=self.seed,
            )
            demo_less_blicker_urdf = get_urdf_path(
                env=env,
                obj_entry=sampled_demo_obj,
                size=demo_obj_size,
                scaling=scalar_less_blicker,
            )
            self.placeholders[f"demo_less_blicker_obj_{i + 1}"] = PlaceholderObj(
                name=sampled_demo_obj.name,
                obj_id=-1,
                urdf=demo_less_blicker_urdf,
                novel_name=sampled_demo_obj.novel_name,
                alias=sampled_demo_obj.alias,
                color=demo_less_blicker_color,
                image_size=self._placeholder_img_size,
                seed=self.seed,
            )

        # add base object
        sampled_base_obj = self.rng.choice(self.possible_base_obj).value
        base_size = self.rng.uniform(
            low=sampled_base_obj.size_range.low,
            high=sampled_base_obj.size_range.high,
        )
        base_obj_id, base_obj_urdf, base_obj_pose = self.add_object_to_env(
            env=env,
            obj_entry=sampled_base_obj,
            color=sampled_base_obj_color,
            size=base_size,
            retain_temp=True,
            category="fixed",
        )
        base_poses = [base_obj_pose]
        self.placeholders["base_obj"] = PlaceholderObj(
            name=sampled_base_obj.name,
            obj_id=base_obj_id,
            urdf=base_obj_urdf,
            novel_name=sampled_base_obj.novel_name,
            alias=sampled_base_obj.alias,
            color=sampled_base_obj_color,
            image_size=self._placeholder_img_size,
            seed=self.seed,
            use_neutral_color=True,
        )

        # add dragged object
        # sample obj type, color and size
        sampled_dragged_obj, distractor_obj = self.rng.choice(
            self.possible_dragged_obj, size=2, replace=False
        )
        sampled_dragged_obj = sampled_dragged_obj.value
        distractor_obj = distractor_obj.value

        not_reach_max_times = False
        for i in range(self.REJECT_SAMPLING_MAX_TIMES):
            dragged_obj_size = self.rng.uniform(
                low=sampled_dragged_obj.size_range.low,
                high=sampled_dragged_obj.size_range.high,
            )
            dragged_obj_scaled_size = self._scale_size(
                dragged_obj_size, dragged_obj_scalar
            )
            dragged_pose = self.get_random_pose(env, dragged_obj_scaled_size)
            if (
                dragged_pose[0] is None
                or dragged_pose[1] is None
                or (
                    sampled_base_obj
                    in [
                        ObjPedia.FRAME.value,
                        ObjPedia.SQUARE.value,
                    ]
                    and if_in_hollow_object(
                        dragged_pose, dragged_obj_scaled_size, base_poses, base_size
                    )
                )
            ):
                print(
                    f"Warning: {i + 1} repeated sampling when try to spawn dragged object"
                )
                continue
            dragged_obj_id, dragged_obj_urdf, _ = self.add_object_to_env(
                env=env,
                obj_entry=sampled_dragged_obj,
                color=dragged_color,
                size=dragged_obj_size,
                scalar=dragged_obj_scalar,
                pose=dragged_pose,
                retain_temp=True,
            )
            dragged = [(dragged_obj_id, (0, None))]
            # generates a normal-size gray obj for prompt to avoid spoiling which object to drag
            placeholder_obj_urdf = get_urdf_path(
                env=env,
                obj_entry=sampled_dragged_obj,
                size=dragged_obj_size,
                scaling=1.0,
            )
            self.placeholders["dragged_obj"] = PlaceholderObj(
                name=sampled_dragged_obj.name,
                obj_id=-1,
                urdf=placeholder_obj_urdf,
                novel_name=sampled_dragged_obj.novel_name,
                alias=sampled_dragged_obj.alias,
                image_size=self._placeholder_img_size,
                seed=self.seed,
                use_neutral_color=True,
            )

            # set goal
            self.goals.append(
                (
                    dragged,
                    np.ones((len(dragged), len(base_poses))),
                    base_poses,
                    False,
                    True,
                    "pose",
                    None,
                    1,
                )
            )
            self._all_goals = self.goals.copy()
            not_reach_max_times = True
            break
        if not not_reach_max_times:
            raise ValueError("Error in sampling dragged object")

        self.distractors = []
        # add first confusion object
        not_reach_max_times = False
        for i in range(self.REJECT_SAMPLING_MAX_TIMES):
            # add first confusion object
            confusion1_scaled_size = self._scale_size(
                dragged_obj_size, confusion_obj_scalar
            )
            pose = self.get_random_pose(env, confusion1_scaled_size)
            if (not all([pose[0], pose[1]])) or (
                sampled_base_obj
                in [
                    ObjPedia.FRAME.value,
                    ObjPedia.SQUARE.value,
                ]
                and if_in_hollow_object(
                    [pose],
                    [confusion1_scaled_size],
                    base_poses,
                    base_size,
                )
            ):
                print(f"Warning: {i + 1} repeated sampling confusion objects pair")
                continue
            confusion_obj_id, _, _ = self.add_object_to_env(
                env=env,
                obj_entry=sampled_dragged_obj,
                color=confusion_color,
                size=dragged_obj_size,
                scalar=confusion_obj_scalar,
                pose=pose,
                retain_temp=True,
            )
            if confusion_obj_id is None:
                print(f"Warning: {i + 1} repeated sampling the first confusion object")
                continue
            else:
                self.distractors.append((confusion_obj_id, (0, None)))
                not_reach_max_times = True
                break
        if not not_reach_max_times:
            raise ValueError("Error in sampling the first confusion object")

        # add extra distractor if difficulty is medium or hard
        if self.task_meta["difficulty"] in ["medium", "hard"]:
            # in medium level, the distractor has no common attribute as the target object
            # e.g., if the target object has darker color, then the distractor has lighter color
            # if the target object is taller, then the distractor is shorter
            if self.task_meta["difficulty"] == "medium":
                if adjective_definition in ["lighter color", "darker color"]:
                    distractor_color = confusion_color
                elif adjective_definition in ["shorter", "taller"]:
                    distractor_color = dragged_color
                else:
                    raise ValueError
                distractor_size = dragged_obj_size
                distractor_size_scalar = confusion_obj_scalar
            else:
                # in hard level, the distractor has one same attribute as the target object
                # e.g., if the target object has darker color, then the distractor has darker color
                # if the target object is taller, then the distractor is taller
                if adjective_definition in ["darker color", "lighter color"]:
                    distractor_color = dragged_color
                    distractor_size = dragged_obj_size
                    distractor_size_scalar = confusion_obj_scalar
                elif adjective_definition in ["taller", "shorter"]:
                    distractor_color = confusion_color
                    distractor_size = dragged_obj_size
                    distractor_size_scalar = dragged_obj_scalar
                else:
                    raise ValueError

            not_reach_max_times = False
            for i in range(self.REJECT_SAMPLING_MAX_TIMES):
                # add first confusion object
                distractor_scaled_size = self._scale_size(
                    distractor_size, distractor_size_scalar
                )
                pose = self.get_random_pose(env, distractor_scaled_size)
                if (not all([pose[0], pose[1]])) or (
                    sampled_base_obj
                    in [
                        ObjPedia.FRAME.value,
                        ObjPedia.SQUARE.value,
                    ]
                    and if_in_hollow_object(
                        [pose],
                        [distractor_scaled_size],
                        base_poses,
                        base_size,
                    )
                ):
                    print(f"Warning: {i + 1} repeated sampling confusion objects pair")
                    continue
                distracto_obj_id, _, _ = self.add_object_to_env(
                    env=env,
                    obj_entry=distractor_obj,
                    color=distractor_color,
                    size=distractor_size,
                    scalar=distractor_size_scalar,
                    pose=pose,
                    retain_temp=True,
                )
                if distracto_obj_id is None:
                    print(f"Warning: {i + 1} repeated sampling the extra distractor")
                    continue
                else:
                    self.distractors.append((distracto_obj_id, (0, None)))
                    not_reach_max_times = True
                    break
            if not not_reach_max_times:
                raise ValueError("Error in sampling the extra distractor")

        if self.task_meta["num_geometric_distractors"] == 0:
            return
        possible_geometric_distractors = [
            obj
            for obj in self.possible_dragged_obj + self.possible_base_obj
            if obj.value != sampled_base_obj and obj.value != sampled_dragged_obj
        ]
        not_reach_max_times = False
        geometric_distractors = []
        for i in range(
            self.task_meta["num_geometric_distractors"] + self.REJECT_SAMPLING_MAX_TIMES
        ):
            distractor_obj = self.rng.choice(possible_geometric_distractors)
            distractor_obj_entry = distractor_obj.value
            distractor_color_entry = self.rng.choice(
                self.possible_dragged_obj_texture + self.possible_base_obj_texture
            ).value

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
            obj_id, urdf_full_path, _ = self.add_object_to_env(
                env=env,
                obj_entry=distractor_obj_entry,
                color=distractor_color_entry,
                size=distractor_size,
                pose=distractor_pose,
                category="fixed"
                if distractor_obj in self.possible_base_obj
                else "rigid",
                retain_temp=True,
            )
            if obj_id is None:
                print(
                    f"Warning: {i + 1} repeated sampling when try to spawn distractors"
                )
                continue
            geometric_distractors.append((obj_id, (0, None)))
            if (
                len(geometric_distractors)
                == self.task_meta["num_geometric_distractors"]
            ):
                not_reach_max_times = True
                break
        if not not_reach_max_times:
            raise ValueError("Error in sampling distractors")
        self.distractors.extend(geometric_distractors)

    def check_success(self, *args, **kwargs) -> ResultTuple:
        # distractors checking
        for distractor in self.distractors:
            if not self.check_distractors(distractor):
                return ResultTuple(success=False, failure=True, distance=self.rot_eps)

        # check success by checking the number of goals to be achieved
        all_achieved = len(self.goals) == 0
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
        self.task_meta["difficulty"] = difficulty

    def is_match(self, pose0, pose1, symmetry):
        return super().is_match(pose0, pose1, symmetry, position_only=True)
