from __future__ import annotations

import math
from functools import partial
from typing import Literal

import numpy as np

from ..instruction_following.rotate_base import RotateTheObjBase
from ...components.encyclopedia.definitions import ObjEntry, TextureEntry
from ...components.placeholders import PlaceholderScene, PlaceholderText
from ...components.placeholders.placeholder_scene import SceneRenderEnv
from ...utils.misc_utils import eulerXYZ_to_quatXYZW


class Twist(RotateTheObjBase):
    """
    Novel concept (verbs) grounding: Object Twisting Tasks
        The agent needs to learn from the prompt how is "twist" defined,
        and what is the exact angle to twist. Like the ring balancing tasks,
        it needs to ground verbs that are semantically similar but different in exact definitions.
    """

    task_name = "twist"

    def __init__(
        self,
        # ====== task specific ======
        num_total_objs: int = 3,
        n_supports: int = 3,
        before_twist_obj_express_types: Literal["image", "name"] = "image",
        after_twist_obj_express_types: Literal["image", "name"] = "image",
        oracle_max_steps: int = 5,
        oracle_step_to_env_step_ratio: int = 4,
        possible_angles_of_rotation: list[float | int] | float | int | None = None,
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
        assert n_supports >= 1
        self._n_supports = n_supports
        task_meta = {
            "num_total_objs": num_total_objs,
            "num_dragged_obj": None,
            "num_distractors": None,
            "n_supports": n_supports,
        }
        placeholder_expression = {
            "twist_obj_texture": {
                "types": ["text"],
            }
        }
        for i in range(n_supports):
            placeholder_expression.update(
                {
                    f"before_twist_{i+1}": {
                        "types": [before_twist_obj_express_types],
                    },
                    f"after_twist_{i+1}": {
                        "types": [after_twist_obj_express_types],
                    },
                }
            )

        if possible_angles_of_rotation is None:
            # [start = 1/6 * pi, end = 11/6 * pi], step_size = 1/6 * pi
            self.possible_angles_of_rotation = [
                1 / 6 * math.pi * i for i in range(1, 12)
            ]
        elif isinstance(possible_angles_of_rotation, (float, int)):
            self.possible_angles_of_rotation = [possible_angles_of_rotation]
        elif isinstance(possible_angles_of_rotation, list) and all(
            isinstance(angle, (float, int)) for angle in possible_angles_of_rotation
        ):
            self.possible_angles_of_rotation = [
                float(angle) for angle in possible_angles_of_rotation
            ]
        else:
            raise ValueError(
                "possible_angles_of_rotation must be None or float or a list of floats"
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
        self._placeholder_scene_views = placeholder_scene_views

        prompt_template = (
            """"Twist" is defined as rotating object a specific angle. For examples:"""
        )
        for i in range(n_supports):
            _start_str = "{" + f"before_twist_{i+1}" + "}"
            _end_str = "{" + f"after_twist_{i+1}" + "}"
            prompt_template += f" From {_start_str} to {_end_str}."
        prompt_template += " Now twist all {twist_obj_texture} objects."

        super().__init__(
            prompt_template=prompt_template,
            task_meta=task_meta,
            placeholder_expression=placeholder_expression,
            oracle_max_steps=oracle_max_steps,
            oracle_step_to_env_step_ratio=oracle_step_to_env_step_ratio,
            possible_angles_of_rotation=possible_angles_of_rotation,
            possible_dragged_obj=possible_dragged_obj,
            possible_dragged_obj_texture=possible_dragged_obj_texture,
            obs_img_views=obs_img_views,
            obs_img_size=obs_img_size,
            placeholder_img_size=placeholder_img_size,
            seed=seed,
            debug=debug,
            is_subclassed_by_twist_task=True,
        )
        self.pos_eps = 0.05

    def _reset_prompt(self, *args, **kwargs):
        self.task_meta["num_dragged_obj"] = self.rng.integers(
            1, self.task_meta["num_total_objs"]
        )
        self.task_meta["num_distractors"] = (
            self.task_meta["num_total_objs"] - self.task_meta["num_dragged_obj"]
        )
        super()._reset_prompt()
        self.placeholders["twist_obj_texture"] = PlaceholderText(
            self.sampled_dragged_obj_texture.name
        )

    def reset(self, env, *args, **kwargs):
        dragged_entries_archive = super().reset(
            env,
            # in this task, dragged obj shapes will be different,
            # but colors will be the same
            same_dragged_obj=False,
            same_dragged_color=True,
        )

        # add placeholder objects
        for i in range(self._n_supports):
            example_obj_entry_idx = self.rng.choice(range(len(dragged_entries_archive)))
            example_obj, example_obj_urdf = (
                dragged_entries_archive[example_obj_entry_idx][0],
                dragged_entries_archive[example_obj_entry_idx][2],
            )
            sampled_example_obj_size = self.rng.uniform(
                low=example_obj.size_range.low,
                high=example_obj.size_range.high,
            )

            sampled_example_obj_texture = self.rng.choice(
                self.possible_dragged_obj_texture
            ).value
            while sampled_example_obj_texture == self.sampled_dragged_obj_texture:
                sampled_example_obj_texture = self.rng.choice(
                    self.possible_dragged_obj_texture
                ).value

            theta = self.rng.random() * 2 * np.pi
            before_twist_rot = eulerXYZ_to_quatXYZW((0, 0, theta))
            after_twist_rot = eulerXYZ_to_quatXYZW(
                (
                    0,
                    0,
                    theta - self.sampled_angle_of_rotation,
                )
            )

            # --- calculate center pose (used in the scene prompt) ----
            # calculate bounds
            x_bound, y_bound, _ = self.bounds
            # calculate center
            x_center = x_bound[0] + (x_bound[1] - x_bound[0]) / 2
            y_center = y_bound[0] + (y_bound[1] - y_bound[0]) / 2
            center = np.array([x_center, y_center])
            # calculate length along x and y
            lengths = np.array([x_bound[1] - x_bound[0], y_bound[1] - y_bound[0]])
            restriction_factor = np.array([0.25, 0.75])
            lengths *= restriction_factor
            # calculate sample region
            low = center - lengths / 2
            high = center + lengths / 2
            sampled_obj_pos = self.rng.uniform(low=low, high=high)
            sampled_obj_pos = (sampled_obj_pos[0], sampled_obj_pos[1], 0)

            def create_scene_fn(
                scene_render_env: SceneRenderEnv,
                sim_id: int,
                scene_name: str,
                obj,
                texture,
                size,
                position,
                init_rot,
                post_rot,
            ):
                # [INFO] To present the definition of "twist" in this episode,
                # we use an example object to demonstrate the twist verb
                # the start scene
                if scene_name.startswith(
                    "before_twist"
                ):  # generate with placement configurations
                    # Add the example obj.
                    self.add_object_to_env(
                        scene_render_env,
                        obj,
                        texture,
                        size,
                        pose=(position, init_rot),
                        category="rigid",
                    )

                elif scene_name.startswith("after_twist"):
                    self.add_object_to_env(
                        scene_render_env,
                        obj,
                        texture,
                        size,
                        pose=(position, post_rot),
                        category="rigid",
                    )

                else:
                    raise NotImplementedError()

            for scene_name in self.placeholder_expression.keys():
                if scene_name not in [f"before_twist_{i + 1}", f"after_twist_{i + 1}"]:
                    continue
                self.placeholders[scene_name] = PlaceholderScene(
                    create_scene_fn=partial(
                        create_scene_fn,
                        scene_name=scene_name,
                        obj=example_obj,
                        texture=sampled_example_obj_texture,
                        size=sampled_example_obj_size,
                        position=sampled_obj_pos,
                        init_rot=before_twist_rot,
                        post_rot=after_twist_rot,
                    ),
                    assets_root=self.assets_root,
                    views=self._placeholder_scene_views,
                    image_size=self._placeholder_img_size,
                    seed=self.seed,
                )

    def set_difficulty(self, difficulty: str):
        # task difficulty depends on the number of objects
        super().set_difficulty(difficulty)
        if difficulty == "easy":
            # three objects in total - easy level
            self.task_meta["num_total_objs"] = 3
        elif difficulty == "medium":
            # four objects in total - medium level
            self.task_meta["num_total_objs"] = 4
        else:
            # five objects in total - hard level
            self.task_meta["num_total_objs"] = 5
        self.oracle_max_steps = self.task_meta["num_total_objs"] + 2
