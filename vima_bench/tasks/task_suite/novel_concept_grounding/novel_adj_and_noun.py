from __future__ import annotations

from copy import deepcopy
from typing import Literal

from .novel_adj import NovelAdj
from ...components.encyclopedia.definitions import ObjEntry, TextureEntry


class NovelAdjAndNoun(NovelAdj):
    """
    Novel Concept Grounding task: Drag the blicker dax into a blicket.
    An extension to the novel adjective task (drag the blicker obj into the obj)
    by also introducing novel object names.
    """

    task_name = "novel_adj_and_noun"

    def __init__(
        self,
        # ====== task specific ======
        n_supports: int = 3,
        novel_names: list[str] | str | None = None,
        novel_adjectives: list[str] | str | None = None,
        adjective_types: list[str] | str | None = None,
        num_geometric_distractors: int = 2,
        dragged_obj_express_types: Literal["image", "name"] = "image",
        base_obj_express_types: Literal["image", "name"] = "image",
        demo_blicker_obj_express_types: Literal["image", "name"] = "image",
        demo_less_blicker_obj_express_types: Literal["image", "name"] = "image",
        prepend_color_to_name: bool = True,
        oracle_max_steps: int = 3,  # two mistakes allowed
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
        if novel_names is None:
            self.novel_names = ["dax", "blicket", "wug", "zup"]
        elif isinstance(novel_names, (list, tuple)):
            for comp in novel_names:
                assert isinstance(comp, str), f"{comp} has to be string in novel names"
            self.novel_names = novel_names
        elif isinstance(novel_names, str):
            self.novel_names = [novel_names]
        else:
            raise ValueError("novel_names must be a str or list of str.")

        super().__init__(
            n_supports=n_supports,
            novel_adjectives=novel_adjectives,
            adjective_types=adjective_types,
            num_geometric_distractors=num_geometric_distractors,
            dragged_obj_express_types=dragged_obj_express_types,
            base_obj_express_types=base_obj_express_types,
            demo_blicker_obj_express_types=demo_blicker_obj_express_types,
            demo_less_blicker_obj_express_types=demo_less_blicker_obj_express_types,
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
        self._novel_name_definitions = [
            "This is a {novel_name_dragged_obj} {dragged_obj}.",
            "This is a {novel_name_base_obj} {base_obj}.",
        ]
        self._original_prompt_template = deepcopy(self.prompt_template)

    def _reset_prompt(self) -> bool:
        """reset prompt and return whether flip adjective"""
        flip_adjective = super()._reset_prompt()
        novel_name_dragged_obj, novel_name_base_obj = self.rng.choice(
            self.novel_names, size=2, replace=False
        )

        desc_dragged_obj = self._novel_name_definitions[0].format(
            novel_name_dragged_obj=novel_name_dragged_obj, dragged_obj="{dragged_obj}"
        )
        desc_base_obj = self._novel_name_definitions[1].format(
            novel_name_base_obj=novel_name_base_obj, base_obj="{base_obj}"
        )
        obj_desc = [desc_dragged_obj, desc_base_obj]
        self.rng.shuffle(obj_desc)

        prompt_template = deepcopy(self._original_prompt_template)

        prompt_template = prompt_template.replace(
            "{dragged_obj}", novel_name_dragged_obj
        )
        prompt_template = prompt_template.replace("{base_obj}", novel_name_base_obj)
        self.prompt_template = " ".join(obj_desc + [prompt_template])

        return flip_adjective
