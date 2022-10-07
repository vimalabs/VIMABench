from __future__ import annotations

from functools import partial
from typing import Literal

from ..instruction_following.simple_manipulation import SimpleManipulation
from ...components.encyclopedia.definitions import ObjEntry, TextureEntry


class NovelNoun(SimpleManipulation):

    task_name = "novel_noun"

    def __init__(
        self,
        # ====== task specific ======
        novel_names: list[str] | str | None = None,
        num_dragged_obj: int = 1,
        num_base_obj: int = 1,
        num_distractors_obj: int = 1,
        dragged_obj_express_types: Literal["image", "name"] = "image",
        base_obj_express_types: Literal["image", "name"] = "image",
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

        self.obj_desc = [
            "This is a {novel_name_dragged_obj} {dragged_obj}.",
            "This is a {novel_name_base_obj} {base_obj}.",
        ]
        super().__init__(
            num_dragged_obj=num_dragged_obj,
            num_base_obj=num_base_obj,
            num_distractors_obj=num_distractors_obj,
            dragged_obj_express_types=dragged_obj_express_types,
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
            use_neutral_color=True,
            exclude_distractor_by_geometry=True,
        )
        self.pos_eps = 0.1

    def _reset_prompt(self):
        # compose certain_objs string in the prompt
        num_dragged_obj = self.task_meta["num_dragged_obj"]
        num_base_obj = self.task_meta["num_base_obj"]
        assert num_dragged_obj + num_base_obj <= len(self.novel_names)
        assert num_base_obj == 1
        novel_name_objs = self.rng.choice(
            self.novel_names, size=num_dragged_obj + num_base_obj, replace=False
        )

        novel_name_dragged_objs, novel_name_base_objs = (
            novel_name_objs[:num_dragged_obj],
            novel_name_objs[num_dragged_obj:],
        )
        certain_objs = [f"a {obj}" for obj in novel_name_dragged_objs]
        if num_dragged_obj > 1:
            for i in range(1, num_dragged_obj * 2 - 1, 2):
                certain_objs.insert(i, " and ")
        desc = self.obj_desc.copy()
        desc_dragged_obj_pars = [
            partial(
                desc[0].format,
                novel_name_dragged_obj=novel_name_dragged_objs[i],
            )
            for i in range(num_dragged_obj)
        ]
        dragged_obj_descs = [
            desc_dragged_obj_par(dragged_obj="{dragged_obj" + f"_{i + 1}" + "}")
            for i, desc_dragged_obj_par in enumerate(desc_dragged_obj_pars)
        ]

        # only 1 base
        desc_base_obj_par = [
            partial(
                desc[1].format,
                novel_name_base_obj=novel_name_base_objs[i],
            )
            for i in range(num_base_obj)
        ]
        base_obj_desc = [desc_base_obj_par[0](base_obj="{base_obj}")]

        obj_desc = [*dragged_obj_descs, *base_obj_desc]
        self.rng.shuffle(obj_desc)

        self.prompt_template = (
            "Put {certain_objs} into a {novel_name_base_obj}.".format(
                certain_objs="".join(certain_objs),
                novel_name_base_obj="".join(novel_name_base_objs),
            )
        )

        # combine obj_desc_prompt and prompt_template to get the final prompt
        self.prompt_template = " ".join([*obj_desc, self.prompt_template])

    def reset(self, env):
        super().reset(env)
        self._reset_prompt()
