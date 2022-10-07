from __future__ import annotations

from typing import NamedTuple, Literal

import numpy as np
import pybullet as p

from ..base import BaseTask
from ...components.encyclopedia import ObjPedia, TexturePedia
from ...components.encyclopedia.definitions import ObjEntry, TextureEntry
from ...components.placeholders.placeholder_scene import (
    SceneRenderEnv,
    PlaceholderScene,
)
from ...utils.pybullet_utils import (
    add_any_object,
    add_object_id_reverse_mapping_info,
    p_change_texture,
)


class ResultTuple(NamedTuple):
    success: bool
    failure: bool
    distance: float


class RearrangeIntoSceneBase(BaseTask):
    """Rearrangement task base class"""

    def __init__(
        self,
        # ====== task specific ======
        prompt_template: str,
        num_dragged_obj: int = 2,
        num_distractors_obj: int = 2,
        distractor_conflict_rate: float | int = 0.3,
        scene_express_types: Literal["image", "name"] = "image",
        oracle_max_steps: int = 10,
        oracle_step_to_env_step_ratio: int = 4,
        possible_dragged_obj: str | list[str] | ObjEntry | list[ObjEntry] | None = None,
        possible_dragged_obj_texture: str
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
        task_meta = {
            "num_dragged_obj": num_dragged_obj,
            "num_distractor_in_workspace": num_distractors_obj,
            # probability that the distractor is sampled to one of the target pose
            "distractor_conflict_rate": distractor_conflict_rate,
        }
        placeholder_expression = {
            "scene": {
                "types": [scene_express_types],
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

        self.possible_distractor_obj = self.possible_dragged_obj.copy()

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

        self.possible_distractor_obj_textures = self.possible_dragged_obj_texture.copy()

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

        # saving for restore
        self.conflict_distractors = []
        self.conflict_distractor_original_poses = []
        self.dragged = []
        self.dragged_poses = []

    def reset(self, env):
        super().reset(env)

        num_dragged_obj = self.rng.choice([1, 2, 3], p=[0.4, 0.4, 0.2])
        num_distractors = 1 if num_dragged_obj == 1 else 0
        self.task_meta["num_dragged_obj"] = num_dragged_obj
        self.task_meta["num_distractor_in_workspace"] = num_distractors
        self.task_meta["distractor_conflict_rate"] = 0.5

        sampled_dragged_obj_textures = [
            e.value
            for e in self.rng.choice(
                self.possible_dragged_obj_texture,
                size=self.task_meta["num_dragged_obj"],
            )
        ]

        # 1. add dragged objects to their "target" poses (for now, only 1)
        dragged_poses = []  # store poses
        self.dragged = []  # store tuples of (obj_id, (symmetry, ...))
        dragged_object_entries = []  # store object entries
        dragged_sampled_sizes = []  # store sampled object sizes
        num_dragged_obj_added = 0
        not_reach_max_times = False
        for i in range(
            self.task_meta["num_dragged_obj"] + self.REJECT_SAMPLING_MAX_TIMES
        ):
            sampled_dragged_obj_entry = self.rng.choice(
                self.possible_dragged_obj,
            ).value
            size = self.rng.uniform(
                low=sampled_dragged_obj_entry.size_range.low,
                high=sampled_dragged_obj_entry.size_range.high,
            )

            obj_id, _, pose = self.add_object_to_env(
                env=env,
                obj_entry=sampled_dragged_obj_entry,
                color=sampled_dragged_obj_textures[num_dragged_obj_added],
                size=size,
            )
            if not obj_id:
                print(
                    f"Warning: {i + 1} repeated sampling when try to spawn dragged object"
                )
                continue
            num_dragged_obj_added += 1
            dragged_poses.append(pose)
            self.dragged.append((obj_id, (0, None)))
            dragged_object_entries.append(sampled_dragged_obj_entry)
            dragged_sampled_sizes.append(size)

            if num_dragged_obj_added == self.task_meta["num_dragged_obj"]:
                not_reach_max_times = True
                break
        if not not_reach_max_times:
            raise ValueError("Error in sampling dragged object")

        target_poses = dragged_poses
        dragged_poses = []

        # 2. generate dragged object actual poses
        # Note: the reversed op really matters (associates with the order of goals)
        for k, ((obj_id, (_, _)), obj_size) in enumerate(
            zip(reversed(self.dragged), reversed(dragged_sampled_sizes))
        ):
            not_reach_max_times = False
            for i in range(self.REJECT_SAMPLING_MAX_TIMES):
                pose = self.get_random_pose(env, obj_size)
                bad_sample = self.is_distractor_needed_to_move_away(
                    pose, target_poses, obj_size, dragged_sampled_sizes
                )
                if all([pose[0], pose[1], not bad_sample]):
                    p.resetBasePositionAndOrientation(
                        obj_id, pose[0], pose[1], env.client_id
                    )
                    dragged_poses.append(pose)
                    not_reach_max_times = True
                    break
                else:
                    print(
                        f"Warning: {i + 1} repeated sampling random target of objects"
                    )
            if not not_reach_max_times:
                raise ValueError("Error in generating random target poses")
        dragged_poses.reverse()
        self.dragged_poses = dragged_poses

        # set goals
        for dragged_obj, target_pose in zip(self.dragged, target_poses):
            self.goals.append(
                (
                    [dragged_obj],
                    np.ones((1, 1)),
                    [target_pose],
                    False,
                    True,
                    "pose",
                    None,
                    1 / self.task_meta["num_dragged_obj"],
                )
            )

        # 3. add distractor in workspace
        self.distractors = []
        # for restore
        self.conflict_distractors = []
        self.conflict_distractor_original_poses = []
        (
            sampled_distractor_entries,
            sampled_distractor_textures,
        ) = self._sample_distractors(dragged_obj_entries=dragged_object_entries)
        possible_target_poses = target_poses.copy()
        for i in range(len(sampled_distractor_entries)):
            distractor_i = sampled_distractor_entries[i]
            distractor_i_texture = sampled_distractor_textures[i]

            not_reach_max_times = False
            if (
                self.rng.random() < self.task_meta["distractor_conflict_rate"]
                and len(possible_target_poses) > 0
            ):
                # conflict with the random target pose
                for k in range(self.REJECT_SAMPLING_MAX_TIMES * 2):
                    size = self.rng.uniform(
                        low=distractor_i.size_range.low,
                        high=distractor_i.size_range.high,
                    )
                    # sample the distractor target pose
                    conflict_idx = self.rng.integers(len(possible_target_poses))
                    conflict_pose = target_poses[conflict_idx]
                    distractor_targ_pose = self.get_random_pose(env, size)
                    if not all(
                        [
                            distractor_targ_pose[0],
                            distractor_targ_pose[1],
                            not self.is_distractor_needed_to_move_away(
                                distractor_targ_pose,
                                target_poses,
                                size,
                                dragged_sampled_sizes,
                            ),
                        ]
                    ):
                        print(f"Warning: {k + 1} repeated sampling distractor target")
                        continue

                    # add distractor
                    obj_id, _, _ = self.add_object_to_env(
                        env=env,
                        obj_entry=distractor_i,
                        color=distractor_i_texture,
                        pose=conflict_pose,
                        size=size,
                    )
                    if obj_id is not None:
                        possible_target_poses.pop(conflict_idx)
                        distractor_obj = (obj_id, (0, None))
                        self.distractors.append(distractor_obj)
                        self.conflict_distractors.append(distractor_obj)
                        self.conflict_distractor_original_poses.append(conflict_pose)
                        not_reach_max_times = True
                        # update with distractor target pose (oracle only)
                        self.goals.insert(
                            0,
                            (
                                [distractor_obj],
                                np.ones((1, 1)),
                                [distractor_targ_pose],
                                False,
                                True,
                                "pose",
                                {"oracle_only": True},
                                0.01,
                            ),
                        )
                        break
                    else:
                        print(
                            f"Warning: {k + 1} repeated sampling distractor with conflict"
                        )
                        continue
            else:
                # clean distractor with no conflicts
                for k in range(self.REJECT_SAMPLING_MAX_TIMES * 2):
                    size = self.rng.uniform(
                        low=distractor_i.size_range.low,
                        high=distractor_i.size_range.high,
                    )
                    pose = self.get_random_pose(env, size)
                    if self.is_distractor_needed_to_move_away(
                        pose, target_poses, size, dragged_sampled_sizes
                    ):
                        print(f"Warning: {k + 1} repeated sampling clean distractor")
                        continue
                    obj_id, _, _ = self.add_object_to_env(
                        env=env,
                        obj_entry=distractor_i,
                        color=distractor_i_texture,
                        size=size,
                        pose=pose,
                    )
                    if obj_id is None:
                        print(f"Warning: {k + 1} repeated sampling clean distractor")
                        continue
                    else:
                        self.distractors.append((obj_id, (0, None)))
                        not_reach_max_times = True
                        break
            if not not_reach_max_times:
                raise ValueError("Error in sampling distractor")

        # create scene placeholder in the prompt
        def create_scene_fn(scene_render_env: SceneRenderEnv, sim_id: int):
            # all dragged objects should appear in the scene
            for k, (dragged_obj_entry, sampled_size, target_pose) in enumerate(
                zip(dragged_object_entries, dragged_sampled_sizes, target_poses)
            ):
                obj_id, _ = add_any_object(
                    env=scene_render_env,
                    obj_entry=dragged_obj_entry,
                    pose=target_pose,
                    size=sampled_size,
                )
                # change color according to dragged texture
                p_change_texture(obj_id, sampled_dragged_obj_textures[k], sim_id)
                add_object_id_reverse_mapping_info(
                    scene_render_env.obj_id_reverse_mapping,
                    obj_id=obj_id,
                    object_entry=dragged_obj_entry,
                    texture_entry=sampled_dragged_obj_textures[k],
                )

        self.placeholders["scene"] = PlaceholderScene(
            create_scene_fn=create_scene_fn,
            assets_root=self.assets_root,
            views=self._placeholder_scene_views,
            image_size=self._placeholder_img_size,
            seed=self.seed,
        )
        self._all_goals = self.goals.copy()

    @staticmethod
    def is_distractor_needed_to_move_away(
        distractor_pose: tuple,
        object_target_poses: list[tuple],
        distractor_size: tuple,
        dragged_sampled_sizes: list[tuple],
    ) -> bool:
        """
        Identify distractors in the target positions of the dragged object.
        Args:
            distractor_pose: the pose of a distractor to be judged.
            object_target_poses: target poses of objects
            distractor_size: sampled distractor size
            dragged_sampled_sizes: for calculating the distance threshold

        Returns: boolean

        """
        # check the validity of pose
        if distractor_pose[0] is None or distractor_pose[1] is None:
            return True

        for obj_size, target_pose in zip(dragged_sampled_sizes, object_target_poses):
            # Get translational error.
            diff_pos = np.float32(distractor_pose[0][:2]) - np.float32(
                target_pose[0][:2]
            )
            dist_pos = np.linalg.norm(diff_pos)

            if dist_pos <= max(
                (obj_size[0] + distractor_size[0]) / 2 + 0.01,
                (obj_size[1] + distractor_size[1]) / 2 + 0.01,
            ):
                return True
        return False

    def _sample_distractors(
        self,
        dragged_obj_entries: list[ObjEntry] = None,
    ) -> tuple[list[ObjEntry], list[TextureEntry]]:
        possible_distractor_entries_in_workspace = [
            e
            for e in self.possible_distractor_obj
            if e.value not in dragged_obj_entries
        ]

        sampled_distractor_entries = [
            e.value
            for e in self.rng.choice(
                possible_distractor_entries_in_workspace,
                size=self.task_meta["num_distractor_in_workspace"],
                replace=False,
            )
        ]
        sampled_distractor_textures = [
            texture.value
            for texture in self.rng.choice(
                TexturePedia.all_entries(),
                size=self.task_meta["num_distractor_in_workspace"],
            )
        ]
        return sampled_distractor_entries, sampled_distractor_textures

    def check_success(self, *args, **kwargs) -> ResultTuple:
        # check success by checking the number of goals to be achieved
        all_achieved = len(self.goals) == 0
        if all_achieved:
            # distractors checking after achieved all goals.
            for distractor in self.distractors:
                if not self.check_distractors(distractor):
                    return ResultTuple(
                        success=False, failure=True, distance=self.pos_eps
                    )
            # when all goals are achieved, distance is simply the threshold
            return ResultTuple(success=True, failure=False, distance=self.pos_eps)
        else:
            # iterative over all objects to be manipulated to check failure
            failures = []
            distances = []
            num_oracle_only = 0
            for goal in self.goals:
                obj, _, bases, _, _, _, params, _ = goal

                # if a goal contains simply a distractor (for the oracle agent)
                if isinstance(params, dict) and params["oracle_only"]:
                    # in case if only distractor goals left
                    num_oracle_only += 1
                    continue

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

            if num_oracle_only == len(self.goals):
                # only oracle_only goals left
                for distractor in self.distractors:
                    if not self.check_distractors(distractor):
                        return ResultTuple(
                            success=False, failure=True, distance=self.pos_eps
                        )
                return ResultTuple(success=True, failure=False, distance=self.pos_eps)

            return ResultTuple(
                success=False,
                failure=any(failures),
                distance=sum(distances) / len(distances),
            )

    def set_difficulty(self, difficulty: str):
        super().set_difficulty(difficulty)
        if difficulty == "easy":
            self.task_meta["distractor_conflict_rate"] = 0.3
        elif difficulty == "medium":
            self.task_meta["distractor_conflict_rate"] = 0.5
        else:
            self.task_meta["distractor_conflict_rate"] = 0.7
