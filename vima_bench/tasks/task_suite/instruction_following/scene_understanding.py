from __future__ import annotations

import itertools
from typing import NamedTuple, Literal

import numpy as np
import pybullet as p

from ..base import BaseTask
from ...components.encyclopedia import ObjPedia, TexturePedia
from ...components.encyclopedia.definitions import ObjEntry, TextureEntry
from ...components.placeholders import PlaceholderScene, PlaceholderTexture
from ...components.placeholders.placeholder_scene import SceneRenderEnv
from ...utils.pybullet_utils import (
    add_object_id_reverse_mapping_info,
    add_any_object,
    if_in_hollow_object,
    p_change_texture,
)


class ResultTuple(NamedTuple):
    success: bool
    failure: bool
    distance: float


class SceneUnderstanding(BaseTask):
    """
    Instruction following task:
    Put the {dragged_texture} object in {image_scene} into any {base_texture} object
    """

    task_name = "scene_understanding"

    possible_distract_obj_texture = TexturePedia.all_entries()

    def __init__(
        self,
        # ====== task specific ======
        num_dragged_obj: int = 1,
        num_base_obj: int = 1,
        num_distractor_in_scene: int = 1,
        num_distractor_in_workspace: int = 1,
        dragged_obj_express_types: Literal["image", "name"] = "name",
        base_obj_express_types: Literal["image", "name"] = "name",
        scene_express_types: Literal["image", "name"] = "image",
        prepend_color_to_name: bool = True,
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
        placeholder_scene_views: str | list[str] | None = None,
        # ====== general ======
        obs_img_views: str | list[str] | None = None,
        obs_img_size: tuple[int, int] = (128, 256),
        placeholder_img_size: tuple[int, int] = (128, 256),
        seed: int | None = None,
        debug: bool = False,
    ):
        assert num_dragged_obj <= num_base_obj, (
            f"The number of dragged objects {num_dragged_obj} "
            f"should be no more than the number of base objects{num_base_obj}"
        )
        assert num_distractor_in_scene >= 1 and num_distractor_in_workspace >= 1, (
            f"The number of distracting objects in the scene {num_distractor_in_scene} and "
            f"n the workspace {num_distractor_in_workspace} should be at least 1"
        )
        for num in [
            num_dragged_obj,
            num_base_obj,
            num_distractor_in_scene,
            num_distractor_in_workspace,
        ]:
            assert num <= 3, (
                f"The current size of the workspace only supports the max number of each "
                f"kind of object as 3, but got {num}"
            )

        task_meta = {
            "num_dragged_obj": num_dragged_obj,
            "num_base_obj": num_base_obj,
            "num_distractor_in_scene": num_distractor_in_scene,
            "num_distractor_in_workspace": num_distractor_in_workspace,
        }
        placeholder_expression = {
            "dragged_texture": {
                "types": [dragged_obj_express_types],
            },
            "base_texture": {
                "types": [base_obj_express_types],
            },
            "scene": {
                "types": [scene_express_types],
            },
        }
        for expression, types in placeholder_expression.items():
            if types == "name":
                placeholder_expression[expression][
                    "prepend_color"
                ] = prepend_color_to_name

        oracle_max_steps = num_dragged_obj + 2

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
                ObjPedia.BOWL,
                ObjPedia.FRAME,
                ObjPedia.SQUARE,
                ObjPedia.CONTAINER,
                ObjPedia.PALLET,
                ObjPedia.PAN,
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

        assert (
            len(self.possible_dragged_obj_texture) >= 1
            and len(self.possible_base_obj_texture) >= 1
        )
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

        super().__init__(
            prompt_template="Put the {dragged_texture} object in {scene} into the {base_texture} object.",
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
        self._placeholder_scene_views = placeholder_scene_views

    def reset(self, env):
        super().reset(env)

        # sample textures for base objects and dragged objects
        # these two textures must be different! we do this by reject sampling.
        base_objs_texture, dragged_objs_texture = None, None
        while base_objs_texture == dragged_objs_texture:
            base_objs_texture, dragged_objs_texture = (
                self.rng.choice(self.possible_base_obj_texture).value,
                self.rng.choice(self.possible_dragged_obj_texture).value,
            )

        # The distractor could be either dragged obj or base obj, distracted the agent by
        # the same colors but different shapes for base distractor(s) and/or same colors but
        # different shapes for obj distractor(s)
        possible_distractor_obj = self.possible_dragged_obj + self.possible_base_obj
        possible_distractor_texture = (
            self.possible_dragged_obj_texture + self.possible_base_obj_texture
        )
        # make different textures from dragged and base objs
        remaining_textures = [
            texture
            for texture in possible_distractor_texture
            if texture.value != dragged_objs_texture
        ]

        # add base obj texture and dragged obj texture to self.placeholders for the prompt
        self.placeholders["base_texture"] = PlaceholderTexture(
            name=base_objs_texture.name,
            color_value=base_objs_texture.color_value,
            texture_asset=base_objs_texture.texture_asset,
            alias=base_objs_texture.alias,
            novel_name=base_objs_texture.novel_name,
            seed=self.seed,
        )
        self.placeholders["dragged_texture"] = PlaceholderTexture(
            name=dragged_objs_texture.name,
            color_value=dragged_objs_texture.color_value,
            texture_asset=dragged_objs_texture.texture_asset,
            alias=dragged_objs_texture.alias,
            novel_name=dragged_objs_texture.novel_name,
            seed=self.seed,
        )

        not_reach_max_times = False
        hollow_base_poses = []
        hollow_base_sizes = []
        for i in range(self.REJECT_SAMPLING_MAX_TIMES):
            # sample some base object entries
            sampled_base_obj_entries = [
                e.value
                for e in self.rng.choice(
                    self.possible_base_obj,
                    size=self.task_meta["num_base_obj"],
                    replace=False,
                )
            ]  # add base objects
            base_poses = []
            for base_obj_entry in sampled_base_obj_entries:
                size = self.rng.uniform(
                    low=base_obj_entry.size_range.low,
                    high=base_obj_entry.size_range.high,
                )
                pose = self.get_random_pose(env, size)
                if pose[0] is None or pose[1] is None:
                    print(
                        f"Warning: {i + 1} repeated sampling when try to spawn base object"
                    )
                    break
                obj_id, _ = add_any_object(
                    env=env,
                    obj_entry=base_obj_entry,
                    pose=pose,
                    size=size,
                    category="fixed",
                )
                # change color according to base texture
                p_change_texture(obj_id, base_objs_texture, self.client_id)
                # add item into the object id mapping dict
                add_object_id_reverse_mapping_info(
                    mapping_dict=env.obj_id_reverse_mapping,
                    obj_id=obj_id,
                    object_entry=base_obj_entry,
                    texture_entry=base_objs_texture,
                )
                base_poses.append(pose)
                if base_obj_entry in [
                    ObjPedia.FRAME.value,
                    ObjPedia.SQUARE.value,
                ]:
                    hollow_base_poses.append(pose)
                    hollow_base_sizes.append(size)
            if len(base_poses) == self.task_meta["num_base_obj"]:
                not_reach_max_times = True
                break
        if not not_reach_max_times:
            raise ValueError("Error in sampling base object")

        not_reach_max_times = False
        for i in range(self.REJECT_SAMPLING_MAX_TIMES):
            # sample some dragged object entries
            sampled_dragged_obj_entries = [
                e.value
                for e in self.rng.choice(
                    self.possible_dragged_obj,
                    size=self.task_meta["num_dragged_obj"],
                    replace=False,  # make the objs different
                )
            ]
            # add dragged objects
            dragged_poses = []
            dragged = []
            dragged_sampled_sizes = []
            for dragged_obj_entry in sampled_dragged_obj_entries:
                size = self.rng.uniform(
                    low=dragged_obj_entry.size_range.low,
                    high=dragged_obj_entry.size_range.high,
                )
                dragged_sampled_sizes.append(size)
                pose = self.get_random_pose(env, size)
                if pose[0] is None or pose[1] is None:
                    print(
                        f"Warning: {i + 1} repeated sampling when try to spawn dragged object"
                    )
                    break
                elif len(hollow_base_poses) > 0 and if_in_hollow_object(
                    [pose], size, hollow_base_poses, hollow_base_sizes
                ):
                    print(
                        f"Warning: {i + 1} repeated sampling when try to spawn dragged object"
                    )
                    break
                obj_id, _ = add_any_object(
                    env=env, obj_entry=dragged_obj_entry, pose=pose, size=size
                )
                # change color according to dragged texture
                p_change_texture(
                    obj_id=obj_id,
                    texture_entry=dragged_objs_texture,
                    client_id=self.client_id,
                )
                # add item into the object id mapping dict
                add_object_id_reverse_mapping_info(
                    mapping_dict=env.obj_id_reverse_mapping,
                    obj_id=obj_id,
                    object_entry=dragged_obj_entry,
                    texture_entry=dragged_objs_texture,
                )
                dragged_poses.append(pose)
                dragged.append((obj_id, (0, None)))
            if len(dragged_poses) == self.task_meta["num_dragged_obj"]:
                not_reach_max_times = True
                break
        if not not_reach_max_times:
            raise ValueError("Error in sampling base object")

        # add placeholder scene to self.placeholders
        def create_scene_fn(scene_render_env: SceneRenderEnv, sim_id: int):
            # all dragged objects should appear in the scene
            for dragged_obj_entry, sampled_size in zip(
                sampled_dragged_obj_entries, dragged_sampled_sizes
            ):
                pose = scene_render_env.renderer_get_random_pose(
                    sampled_size, rng=self.rng
                )
                obj_id, _ = add_any_object(
                    env=scene_render_env,
                    obj_entry=dragged_obj_entry,
                    pose=pose,
                    size=sampled_size,
                )
                # change color according to dragged texture
                p_change_texture(obj_id, dragged_objs_texture, sim_id)
                add_object_id_reverse_mapping_info(
                    scene_render_env.obj_id_reverse_mapping,
                    obj_id=obj_id,
                    object_entry=dragged_obj_entry,
                    texture_entry=dragged_objs_texture,
                )

            # add distracting objects
            # the set of possible distracting objects is the union of possible dragged and possible base objects
            sampled_distractor_entries = [
                e.value
                for e in self.rng.choice(
                    possible_distractor_obj,
                    size=self.task_meta["num_distractor_in_scene"],
                )
            ]
            sampled_distractor_texture = [
                e.value
                for e in self.rng.choice(
                    remaining_textures, size=self.task_meta["num_distractor_in_scene"]
                )
            ]
            for entry, texture in zip(
                sampled_distractor_entries, sampled_distractor_texture
            ):
                size = self.rng.uniform(
                    low=entry.size_range.low, high=entry.size_range.high
                )
                pose = scene_render_env.renderer_get_random_pose(size, rng=self.rng)
                if not pose:
                    continue
                obj_id, _ = add_any_object(
                    env=scene_render_env, obj_entry=entry, pose=pose, size=size
                )
                if not obj_id:
                    continue
                # change color according to texture
                p_change_texture(obj_id, texture, sim_id)
                add_object_id_reverse_mapping_info(
                    scene_render_env.obj_id_reverse_mapping,
                    obj_id=obj_id,
                    object_entry=entry,
                    texture_entry=texture,
                )

        self.placeholders["scene"] = PlaceholderScene(
            create_scene_fn=create_scene_fn,
            assets_root=self.assets_root,
            views=self._placeholder_scene_views,
            image_size=self._placeholder_img_size,
            seed=self.seed,
        )

        # set goal
        # because the task says "put object into ANY base objects",
        # we sample all possible combinations of obj-base-goals.
        # Note that all dragged objects have already set to be unique

        # noinspection PyUnboundLocalVariable
        obj_goal_combination = [comb for comb in itertools.product(dragged, base_poses)]
        # make the oracle demonstrations varied, especially for multiple dragged objects cases
        self.rng.shuffle(obj_goal_combination)
        for each_obj, each_goal in obj_goal_combination:
            self.goals.append(
                (
                    [each_obj],
                    np.ones((1, 1)),
                    [each_goal],
                    False,
                    True,
                    "pose",
                    None,
                    1,
                )
            )
        self._all_goals = self.goals.copy()

        # add distractors in the workspace
        # they should have the SAME color as the manipulated objects but DIFFERENT shape
        # or DIFFERENT color as the base objects but SAME shape
        # Here we allow the same distractor entries from the scene
        # noinspection PyUnboundLocalVariable
        possible_dragged_distractors = [
            obj
            for obj in self.possible_dragged_obj
            # different from dragged obj shape
            if obj.value not in sampled_dragged_obj_entries
        ]
        # noinspection PyUnboundLocalVariable
        possible_base_distractors = [
            obj
            for obj in self.possible_base_obj
            # same as base obj shape
            if obj.value in sampled_base_obj_entries
        ]

        self.distractors = []
        not_reach_max_times = False
        for i in range(
            self.task_meta["num_distractor_in_workspace"]
            + 2 * self.REJECT_SAMPLING_MAX_TIMES
        ):
            # Due to the limited size of the workspace, though here is a 0.5 probability for sampling,
            # the generated data would be more likely with dragged distractors
            sampled_distractor_entries = [
                (
                    self.rng.choice(
                        possible_dragged_distractors
                        if self.rng.random() < 0.5
                        else possible_base_distractors,
                    ).value
                )
                for _ in range(self.task_meta["num_distractor_in_workspace"])
            ]

            n_added_distractors = 0
            for entry in sampled_distractor_entries:
                size = self.rng.uniform(
                    low=entry.size_range.low, high=entry.size_range.high
                )
                pose = self.get_random_pose(env, size)
                if pose[0] is None or pose[1] is None:
                    print(
                        f"Warning: {i + 1} repeated sampling when try to spawn workspace distractor"
                    )
                    break
                if entry in sampled_base_obj_entries:
                    # noinspection PyUnboundLocalVariable
                    if entry in [
                        ObjPedia.FRAME.value,
                        ObjPedia.SQUARE.value,
                    ] and if_in_hollow_object(
                        dragged_poses, dragged_sampled_sizes, pose, size
                    ):
                        print(
                            f"Warning: {i + 1} repeated sampling when try to spawn workspace distractor"
                        )
                        break
                    distractor_category = "fixed"
                    distractor_texture = self.rng.choice(
                        [
                            texture
                            for texture in possible_distractor_texture
                            # different from base obj color
                            if texture.value != base_objs_texture
                        ]
                    ).value
                else:
                    if len(hollow_base_poses) > 0 and if_in_hollow_object(
                        pose, size, hollow_base_poses, hollow_base_sizes
                    ):
                        print(
                            f"Warning: {i + 1} repeated sampling when try to spawn workspace distractor"
                        )
                        break
                    distractor_category = "rigid"
                    # same as dragged obj color
                    distractor_texture = dragged_objs_texture

                obj_id, _ = add_any_object(
                    env=env,
                    obj_entry=entry,
                    pose=pose,
                    size=size,
                    retain_temp=True,
                    category=distractor_category,
                )
                if not obj_id:
                    print(
                        f"Warning: {i + 1} repeated sampling when try to spawn workspace distractor"
                    )
                    break
                # change color according to texture
                p_change_texture(obj_id, distractor_texture, self.client_id)
                # add item into the object id mapping dict
                add_object_id_reverse_mapping_info(
                    mapping_dict=env.obj_id_reverse_mapping,
                    obj_id=obj_id,
                    object_entry=entry,
                    texture_entry=distractor_texture,
                )
                n_added_distractors += 1
                self.distractors.append((obj_id, (0, None)))
            if n_added_distractors == self.task_meta["num_distractor_in_workspace"]:
                not_reach_max_times = True
                break
        if not not_reach_max_times:
            raise ValueError("Error in sampling distractors")

    def check_success(self, *args, **kwargs) -> ResultTuple:
        # check for distractors:
        for distractor in self.distractors:
            if not self.check_distractors(distractor):
                return ResultTuple(success=False, failure=True, distance=self.pos_eps)
        # for each dragged obj, there should be num_of_base_obj possible goals
        # to succeed, should exact - 1 for each, otherwise, it may not achieve the goal,
        # or one obj has dragged into more than one base objects.
        # Note that all dragged objects have already set to be unique
        current_obj_goal_map = {goal[0][0][0]: 0 for goal in self.goals}
        for goal in self.goals:
            current_obj_goal_map[goal[0][0][0]] += 1

        for obj_id in current_obj_goal_map.keys():
            if current_obj_goal_map[obj_id] == self.task_meta["num_base_obj"] - 1:
                need_to_remove = [
                    goal for goal in self.goals if goal[0][0][0] == obj_id
                ]
                for goal in need_to_remove:
                    self.goals.remove(goal)
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

    def set_difficulty(self, difficulty: str):
        super().set_difficulty(difficulty)
        if difficulty == "easy":
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
