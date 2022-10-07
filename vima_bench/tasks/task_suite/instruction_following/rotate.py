from __future__ import annotations

import math
from functools import partial
from typing import Literal

from .rotate_base import RotateTheObjBase
from ...components.encyclopedia.definitions import ObjEntry, TextureEntry


class Rotate(RotateTheObjBase):
    """
    Task on rotating objects: rotation around z-axis
        Previously we care most about achieving target (x-y) positions for our tasks.
        In this task, rotation is introduced by let the robot arm to rotate objects by
        certain degrees (either clockwise or counterclockwise). An angle of rotational symmetry
        (the smallest angle for which the figure can be rotated to coincide with itself) is
        added to the ObjectPedia and objects with no rotational symmetry are used in this task.
        A sample prompt is 'Rotate the red heart clockwise 120 degrees.'
        where the heart is the object to be dragged and 120 denotes the angle (in degrees) to be rotated.
        'clockwise' here is to indicate the direction of rotation. We set the angle to [1/6 * pi, 11/6* pi]
        with a step size of 1/6 *pi. The epsilon of rotation of the environment is set to 15 degrees.
        And we modify the oracle to support rotational demonstration.
    """

    task_name = "rotate"

    def __init__(
        self,
        # ====== task specific ======
        num_dragged_obj: int = 1,
        same_dragged_obj: bool = False,  # whether dragged obj different when obj > 1
        num_distractors_obj: int = 1,
        dragged_obj_express_types: Literal["image", "name"] = "image",
        prepend_color_to_name: bool = True,
        oracle_max_steps: int = 3,  # two mistakes are allowed
        oracle_step_to_env_step_ratio: int = 4,
        possible_angles_of_rotation: list[float | int] | float | int | None = None,
        possible_dragged_obj: str | list[str] | ObjEntry | list[ObjEntry] | None = None,
        possible_dragged_obj_texture: str
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
            "num_distractors": num_distractors_obj,
        }
        placeholder_expression = (
            {
                f"dragged_obj_{i}": {
                    "types": [dragged_obj_express_types],
                    "prepend_color": prepend_color_to_name,
                }
                for i in range(1, num_dragged_obj + 1)
            }
            if not same_dragged_obj and num_dragged_obj > 1
            else {
                f"dragged_obj": {
                    "types": [dragged_obj_express_types],
                    "prepend_color": prepend_color_to_name,
                }
            }
        )

        if possible_angles_of_rotation is None:
            # [start = 1/6 * pi, end = 5/6 * pi], step_size = 1/6 * pi
            # to try the best for keeping all rotations clockwise
            self.possible_angles_of_rotation = [
                1 / 6 * math.pi * i for i in range(1, 6)
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

        super().__init__(
            prompt_template="Rotate the {dragged_obj} {angle_in_degree} degrees.",
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
        )
        assert all(
            self.rot_eps < abs(e) for e in self.possible_angles_of_rotation
        ), f"Angle(s) of rotation {self.possible_angles_of_rotation} is too small."

        self.pos_eps = 0.1
        self.same_dragged_obj = same_dragged_obj

    def _reset_prompt(self, same_dragged_obj: bool = False):
        super()._reset_prompt()
        sampled_angle_in_degrees = round(math.degrees(self.sampled_angle_of_rotation))

        num_dragged_obj = self.task_meta["num_dragged_obj"]
        if not same_dragged_obj and num_dragged_obj > 1:
            dragged_objs_prompt = [
                " {" + f"dragged_obj_{i}" + "}" for i in range(1, num_dragged_obj + 1)
            ]
            if num_dragged_obj > 1:
                for i in range(1, num_dragged_obj * 2 - 1, 2):
                    dragged_objs_prompt.insert(i, " and")
            self.prompt_template = "".join(
                [
                    "Rotate the",
                    *dragged_objs_prompt,
                    f" {sampled_angle_in_degrees} degrees.",
                ]
            )
        else:
            prompt = "Rotate the {dragged_obj} {angle_in_degree} degrees."
            partial_str = partial(
                prompt.format,
                angle_in_degree=sampled_angle_in_degrees,
            )
            self.prompt_template = partial_str(dragged_obj="{dragged_obj}")

    def reset(self, env, *args, **kwargs):
        super().reset(
            env,
            # in this task, the color-shape will be same or different simultaneously
            same_dragged_obj=self.same_dragged_obj,
            same_dragged_color=self.same_dragged_obj,
        )

    def set_difficulty(self, difficulty: str):
        # task difficulty depends on the number of distractors
        super().set_difficulty(difficulty)
        if difficulty == "easy":
            # three objects in total - easy level
            self.task_meta["num_distractors"] = 1
        elif difficulty == "medium":
            # five objects  in total - medium level
            self.task_meta["num_distractors"] = 3
        elif difficulty == "hard":
            # seven objects in total - hard level
            self.task_meta["num_distractors"] = 5
