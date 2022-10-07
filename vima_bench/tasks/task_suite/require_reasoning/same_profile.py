from __future__ import annotations

from typing import Literal

from .same_color import (
    SameColor,
)
from ...components.encyclopedia import ProfilePedia
from ...components.encyclopedia.definitions import ObjEntry, TextureEntry


class SameProfile(SameColor):

    task_name = "same_shape"

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
        super().__init__(
            num_dragged_obj=num_dragged_obj,
            num_base_obj=num_base_obj,
            num_distractors_obj=num_distractors_obj,
            base_obj_express_types=base_obj_express_types,
            prepend_color_to_name=prepend_color_to_name,
            oracle_max_steps=oracle_max_steps,
            oracle_step_to_env_step_ratio=oracle_step_to_env_step_ratio,
            possible_dragged_obj=possible_dragged_obj,
            possible_base_obj=possible_base_obj,
            possible_dragged_obj_texture=possible_dragged_obj_texture,
            possible_base_obj_texture=possible_base_obj_texture,
            obs_img_views=obs_img_views,
            obs_img_size=obs_img_size,
            placeholder_img_size=placeholder_img_size,
            seed=seed,
            debug=debug,
        )
        square_profiles = 0
        else_profiles = 0
        for obj in self.possible_dragged_obj:
            if obj.value.profile == ProfilePedia.SQUARE_LIKE:
                square_profiles += 1
            else:
                else_profiles += 1
        assert (
            square_profiles and else_profiles
        ), "possible dragged object should include both square and other profiles"
        self.prompt_template = (
            "Put all objects with the same profile as {base_obj} into it."
        )

    def _update_sampling(self):
        self.sampled_dragged_obj_textures = self.possible_dragged_obj_texture
        self.sampled_dragged_objs = [
            obj
            for obj in self.possible_dragged_obj
            if obj.value.profile == self.sampled_base.profile
        ]
        self.sampled_distractor_objs = [
            obj
            for obj in self.possible_dragged_obj
            if obj.value.profile != self.sampled_base.profile
        ]
