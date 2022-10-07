from typing import Optional, Tuple, List

import numpy as np

from .base import Placeholder


class PlaceholderTexture(Placeholder):
    allowed_expressions = {"name", "novel_name", "alias"}

    """
    The placeholder texture in the prompt.
    """

    def __init__(
        self,
        name: str,
        color_value: Optional[Tuple[float, float, float]] = None,
        texture_asset: Optional[str] = None,
        alias: Optional[List[str]] = None,
        novel_name: Optional[List[str]] = None,
        seed: Optional[int] = None,
    ):
        self.name = name
        self.color_value = color_value
        self.texture_asset = texture_asset
        self.alias = alias
        self.novel_name = novel_name

        self._rng = np.random.default_rng(seed=seed)

    def get_expression(self, types: List[str], *args, **kwargs):
        assert set(types).issubset(
            self.allowed_expressions
        ), "Unsupported type of expression provided"

        expressions = {}
        for each_type in types:
            if each_type == "name":
                expressions["name"] = self.name
            elif each_type == "novel_name":
                expressions["novel_name"] = self._rng.choice(self.novel_name)
            elif each_type == "alias":
                expressions["alias"] = self._rng.choice(self.alias)
        return expressions
