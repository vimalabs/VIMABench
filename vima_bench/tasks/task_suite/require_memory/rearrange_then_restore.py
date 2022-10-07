from __future__ import annotations

from typing import Literal

import numpy as np
import pybullet as p

from ..rearrangement.rearrange_base import RearrangeIntoSceneBase
from ...components.encyclopedia.definitions import ObjEntry, TextureEntry


class RearrangeThenRestore(RearrangeIntoSceneBase):
    """Rearrangement task:
    The agent is required to rearrange the objects in the workspace
    based on given placement configurations, and then restore the
    workspace to its original setup
    """

    task_name = "rearrange_then_restore"

    def __init__(
        self,
        # ====== task specific ======
        num_dragged_obj: int = 3,
        num_distractors_obj: int = 0,
        distractor_conflict_rate: float | int = 0,
        scene_express_types: Literal["image", "name"] = "image",
        oracle_max_steps: int = 8,
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
        super().__init__(
            prompt_template="Rearrange objects to this setup {scene} and then restore.",
            num_dragged_obj=num_dragged_obj,
            num_distractors_obj=num_distractors_obj,
            distractor_conflict_rate=distractor_conflict_rate,
            scene_express_types=scene_express_types,
            oracle_max_steps=oracle_max_steps,
            oracle_step_to_env_step_ratio=oracle_step_to_env_step_ratio,
            possible_dragged_obj=possible_dragged_obj,
            possible_dragged_obj_texture=possible_dragged_obj_texture,
            placeholder_scene_views=placeholder_scene_views,
            obs_img_views=obs_img_views,
            obs_img_size=obs_img_size,
            placeholder_img_size=placeholder_img_size,
            seed=seed,
            debug=debug,
        )

        self._n_deduped_goals = None
        self._deduped_achieved = 0

    def reset(self, env):
        super().reset(env)

        self._n_deduped_goals = len(self.dragged)
        self._deduped_achieved = 0

        # restore the dragged object
        permutation_indices = self.rng.permutation(len(self.dragged))
        for idx in permutation_indices:
            dragged_obj, dragged_pose = self.dragged[idx], self.dragged_poses[idx]
            self.goals.append(
                (
                    [dragged_obj],
                    np.ones((1, 1)),
                    [dragged_pose],
                    False,
                    True,
                    "pose",
                    None,
                    1 / self.task_meta["num_dragged_obj"],
                )
            )

        # restore the distractors that are moved away during this process
        for distractor_obj, distractor_pose in zip(
            self.conflict_distractors, self.conflict_distractor_original_poses
        ):
            self.goals.append(
                (
                    [distractor_obj],
                    np.ones((1, 1)),
                    [distractor_pose],
                    False,
                    True,
                    "pose",
                    # we want all the objects in the workspace to be unchanged
                    {"oracle_only": True},
                    1 / len(self.conflict_distractor_original_poses),
                )
            )
        self._all_goals = self.goals.copy()

    def update_goals(self, skip_oracle=True):
        """
        Need to check all goals every time step
        """
        # check len of goals, if zero, all goals have been achieved
        if len(self.goals) == 0:
            return

        # each step of checking needs to enumerate all goals
        indices_to_pop = []
        if self._deduped_achieved >= self._n_deduped_goals:
            enumeration = enumerate(self.goals)
        else:
            if len(self.conflict_distractors) == 0:
                enumeration = enumerate(self.goals[: -self._n_deduped_goals])
            else:
                enumeration = enumerate(
                    self.goals[
                        : -(self._n_deduped_goals + len(self.conflict_distractors))
                    ]
                )
        for index, goal in enumeration:
            if skip_oracle:
                _, _, _, _, _, _, params, _ = goal
                if isinstance(params, dict) and params["oracle_only"]:
                    continue
                else:
                    goal_to_check = goal
                    goal_index = index
            else:
                goal_to_check = goal
                goal_index = index

            # Unpack goal step.
            objs, matches, targs, _, _, metric, params, max_progress = goal_to_check

            # Evaluate by matching object poses.
            progress_per_goal = 0
            if metric == "pose":
                for i in range(len(objs)):
                    object_id, (symmetry, _) = objs[i]
                    pose = p.getBasePositionAndOrientation(
                        object_id, physicsClientId=self.client_id
                    )
                    targets_i = np.argwhere(matches[i, :]).reshape(-1)
                    for j in targets_i:
                        target_pose = targs[j]
                        if self.is_match(
                            pose, target_pose, symmetry, position_only=True
                        ):
                            progress_per_goal += max_progress / len(objs)
                            break
            if np.abs(max_progress - progress_per_goal) < 0.001:
                indices_to_pop.append(goal_index)
                self._deduped_achieved += 1
        # remove completed goals
        for index in sorted(indices_to_pop, reverse=True):
            del self.goals[index]
