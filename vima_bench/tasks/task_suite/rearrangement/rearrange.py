from __future__ import annotations

from typing import Literal

import numpy as np
import pybullet as p

from .rearrange_base import RearrangeIntoSceneBase
from ...components.encyclopedia.definitions import ObjEntry, TextureEntry


class Rearrange(RearrangeIntoSceneBase):
    """Rearrangement task:
    The agent is required to rearrange the objects in the workspace
    based on given placement configurations.
    """

    task_name = "rearrange"

    def __init__(
        self,
        # ====== task specific ======
        num_dragged_obj: int = 3,
        num_distractors_obj: int = 0,
        distractor_conflict_rate: float | int = 0,
        scene_express_types: Literal["image", "name"] = "image",
        oracle_max_steps: int = 5,
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
            prompt_template="Rearrange to this {scene}.",
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

    def reset(self, env):
        super().reset(env)

    def update_goals(self, skip_oracle=True):
        """
        Need to check all goals every time step
        """
        # check len of goals, if zero, all goals have been achieved
        if len(self.goals) == 0:
            return

        # each step of checking needs to enumerate all goals
        indices_to_pop = []

        for index, goal in enumerate(self.goals):
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
        # remove completed goals
        for index in sorted(indices_to_pop, reverse=True):
            del self.goals[index]
