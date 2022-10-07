from __future__ import annotations

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


class SameColor(BaseTask):

    task_name = "same_texture"

    def __init__(
        self,
        # ====== task specific ======
        num_dragged_obj: int = 2,
        num_base_obj: int = 1,
        num_distractors_obj: int = 1,
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

        if possible_dragged_obj is None:
            self.possible_dragged_obj = [
                ObjPedia.SHORTER_BLOCK,
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
                ObjPedia.CONTAINER,
                ObjPedia.FRAME,
                ObjPedia.PALLET,
                ObjPedia.SQUARE,
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
            prompt_template="Put all objects with the same texture as {base_obj} into it.",
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
        self.pos_eps = 0.125
        self.sampled_base = None
        self.sampled_base_obj_texture = None

    def _update_sampling(self):
        self.sampled_dragged_obj_textures = [self.sampled_base_obj_texture]
        self.sampled_dragged_objs = self.possible_dragged_obj.copy()
        self.sampled_distractor_objs = self.possible_dragged_obj.copy()

    def reset(self, env):
        super().reset(env)
        self.sampled_base_obj_texture = self.rng.choice(self.possible_base_obj_texture)
        base_texture_entry = self.sampled_base_obj_texture.value
        # just in case, though it is hardly ever impossible to sample the first obj
        for i in range(self.REJECT_SAMPLING_MAX_TIMES):
            sampled_base = self.rng.choice(self.possible_base_obj).value
            if sampled_base.name == "bowl":
                self.pos_eps = self.pos_eps_bowl_multiplier * self.pos_eps_original
            base_sampled_size = self.rng.uniform(
                low=sampled_base.size_range.low,
                high=sampled_base.size_range.high,
            )
            base_id, base_urdf_full_path, base_pose = self.add_object_to_env(
                env=env,
                obj_entry=sampled_base,
                color=base_texture_entry,
                size=base_sampled_size,
                retain_temp=True,
                category="fixed",
            )
            if all([base_id, base_urdf_full_path, base_pose]):
                self.placeholders["base_obj"] = PlaceholderObj(
                    name=sampled_base.name,
                    obj_id=base_id,
                    urdf=base_urdf_full_path,
                    novel_name=sampled_base.novel_name,
                    alias=sampled_base.alias,
                    color=base_texture_entry,
                    image_size=self._placeholder_img_size,
                    seed=self.seed,
                )
                self.sampled_base = sampled_base
                break

        # varied by sampling same profiles/same colors
        self._update_sampling()

        # add dragged objects & distractors
        num_added_objects = 0
        not_reach_max_times = False
        sampled_dragged_obj_textures = []
        self.distractors = []
        for i in range(self.REJECT_SAMPLING_MAX_TIMES * 2):
            if num_added_objects < self.task_meta["num_dragged_obj"]:
                sampled_obj = self.rng.choice(self.sampled_dragged_objs).value
            else:
                sampled_obj = self.rng.choice(self.sampled_distractor_objs).value
            obj_sampled_size = self.rng.uniform(
                low=sampled_obj.size_range.low,
                high=sampled_obj.size_range.high,
            )
            obj_pose = self.get_random_pose(env, obj_sampled_size)
            # noinspection PyUnboundLocalVariable
            if (
                obj_pose[0] is None
                or obj_pose[1] is None
                or if_in_hollow_object(
                    obj_pose, obj_sampled_size, base_pose, base_sampled_size
                )
            ):
                print(
                    f"Warning: {i-num_added_objects+1} repeated sampling "
                    f"when try to spawn object pose"
                )
                continue
            # add dragged obj
            if num_added_objects < self.task_meta["num_dragged_obj"]:
                dragged_obj_texture_entry = self.rng.choice(
                    self.sampled_dragged_obj_textures
                ).value
                dragged_obj_id, _, _ = self.add_object_to_env(
                    env=env,
                    obj_entry=sampled_obj,
                    color=dragged_obj_texture_entry,
                    pose=obj_pose,
                    size=obj_sampled_size,
                    retain_temp=False,
                )
                if dragged_obj_id is not None:
                    self.goals.append(
                        (
                            [(dragged_obj_id, (0, None))],
                            np.ones((1, 1)),
                            [base_pose],
                            False,
                            True,
                            "pose",
                            None,
                            1 / self.task_meta["num_dragged_obj"],
                        )
                    )
                    sampled_dragged_obj_textures.append(dragged_obj_texture_entry)
                    num_added_objects += 1
                else:
                    print(
                        f"Warning: {i + 1} repeated sampling "
                        f"when try to spawn dragged object"
                    )
                    continue
            else:
                # distractor colors are different from dragged objs
                distractor_texture_entry = self.rng.choice(
                    [
                        t
                        for t in self.possible_dragged_obj_texture
                        if t.value not in sampled_dragged_obj_textures
                    ]
                ).value
                distractor_obj_id, _, _ = self.add_object_to_env(
                    env=env,
                    obj_entry=sampled_obj,
                    color=distractor_texture_entry,
                    pose=obj_pose,
                    size=obj_sampled_size,
                    retain_temp=False,
                )
                if distractor_obj_id is not None:
                    num_added_objects += 1
                    self.distractors.append((distractor_obj_id, (0, None)))
                else:
                    print(
                        f"Warning: {i-num_added_objects+1} repeated sampling "
                        f"when try to spawn distractor object"
                    )
                    continue
            if (
                num_added_objects
                == self.task_meta["num_dragged_obj"]
                + self.task_meta["num_distractors_obj"]
            ):
                not_reach_max_times = True
                break
        if not not_reach_max_times:
            raise ValueError("Error in sampling objects")
        self._all_goals = self.goals.copy()

    def set_difficulty(self, difficulty: str):
        # task difficulty depends on the number of objects
        super().set_difficulty(difficulty)
        if difficulty == "easy":
            pass
        elif difficulty == "medium":
            self.task_meta["num_dragged_obj"] = 2
            self.task_meta["num_distractors_obj"] = 2
        else:
            self.task_meta["num_dragged_obj"] = 3
            self.task_meta["num_distractors_obj"] = 3
        self.oracle_max_steps = self.task_meta["num_dragged_obj"] + 2

    def check_success(self, *args, **kwargs) -> ResultTuple:
        # distractors checking
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
