from __future__ import annotations

import colorsys
import os
from enum import Enum
from typing import Tuple, List, Literal

import importlib_resources
import numpy as np

from .definitions import TextureEntry


def _texture_fpath(fname):
    with importlib_resources.files("vima_bench.tasks.assets.textures") as p:
        return os.path.join(str(p), fname)


def convert_to_darker_color(color: TextureEntry, offset=0.3) -> TextureEntry:
    """
    Darken a color by converting it to HSV format and lowering its V value
    """
    color_value = color.color_value
    h, s, v = colorsys.rgb_to_hsv(*color_value)
    v = max(0.0, v - offset)
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    darker_color = TextureEntry(name="dark " + color.name, color_value=(r, g, b))
    return darker_color


def _color_combo_textures_entry(
    color1: str,
    color2: str,
    dark: bool = False,
    texture_category: Literal["stripe", "polka_dot"] = "stripe",
) -> TextureEntry:
    assert texture_category in ["stripe", "polka_dot"]
    asset_name = "dark_" if dark else ""
    asset_name += f"{color1}_{color2}_{texture_category}.jpg"
    prefix = "dark " if dark else ""
    texture_cat_in_name = texture_category.replace("_", " ")
    return TextureEntry(
        name=f"{prefix}{color1} and {color2} {texture_cat_in_name}",
        texture_asset=_texture_fpath(f"{texture_category}s/{asset_name}"),
    )


def _single_color_textures_entry(
    color: str,
    dark: bool = False,
    texture_category: Literal["swirl", "paisley"] = "swirl",
):
    assert texture_category in ["swirl", "paisley"]
    asset_name = "dark_" if dark else ""
    asset_name += f"{color}_{texture_category}.jpg"
    prefix = "dark " if dark else ""
    texture_cat_in_name = texture_category.replace("_", " ")
    return TextureEntry(
        name=f"{prefix}{color} {texture_cat_in_name}",
        texture_asset=_texture_fpath(f"{texture_category}s/{asset_name}"),
    )


class TexturePedia(Enum):
    """
    An encyclopedia of textures in VIMA world.

    Texture(color) could be further added from https://www.rapidtables.com/web/color/RGB_Color.html
    """

    BRICK = TextureEntry(
        name="brick",
        texture_asset=_texture_fpath("brick.jpg"),
    )

    TILES = TextureEntry(
        name="tiles",
        texture_asset=_texture_fpath("tiles.jpg"),
    )

    WOODEN = TextureEntry(
        name="wooden",
        texture_asset=_texture_fpath("wood_light.png"),
    )

    GRANITE = TextureEntry(
        name="granite",
        texture_asset=_texture_fpath("granite.png"),
    )

    PLASTIC = TextureEntry(
        name="plastic",
        texture_asset=_texture_fpath("plastic.jpg"),
    )

    POLKA_DOT = TextureEntry(
        name="polka dot",
        texture_asset=_texture_fpath("polka_dot.jpg"),
    )

    CHECKERBOARD = TextureEntry(
        name="checkerboard",
        texture_asset=_texture_fpath("checkerboard.png"),
    )

    TIGER = TextureEntry(
        name="tiger",
        texture_asset=_texture_fpath("tiger.jpg"),
    )

    MAGMA = TextureEntry(
        name="magma",
        texture_asset=_texture_fpath("magma.jpg"),
    )

    RAINBOW = TextureEntry(
        name="rainbow",
        texture_asset=_texture_fpath("rainbow.jpg"),
    )

    # Variations of stripe textures which consist of two colors and its dark variant
    # the color name is with the format: "(dark) {color1} and {color2} stripe"
    RED_AND_YELLOW_STRIPE = _color_combo_textures_entry(
        "red", "yellow", texture_category="stripe"
    )
    DARK_RED_AND_YELLOW_STRIPE = _color_combo_textures_entry(
        "red", "yellow", dark=True, texture_category="stripe"
    )

    RED_AND_GREEN_STRIPE = _color_combo_textures_entry(
        "red", "green", texture_category="stripe"
    )
    DARK_RED_AND_GREEN_STRIPE = _color_combo_textures_entry(
        "red", "green", dark=True, texture_category="stripe"
    )

    RED_AND_BLUE_STRIPE = _color_combo_textures_entry(
        "red", "blue", texture_category="stripe"
    )
    DARK_RED_AND_BLUE_STRIPE = _color_combo_textures_entry(
        "red", "blue", dark=True, texture_category="stripe"
    )

    RED_AND_PURPLE_STRIPE = _color_combo_textures_entry(
        "red", "purple", texture_category="stripe"
    )
    DARK_RED_AND_PURPLE_STRIPE = _color_combo_textures_entry(
        "red", "purple", dark=True, texture_category="stripe"
    )

    YELLOW_AND_GREEN_STRIPE = _color_combo_textures_entry(
        "yellow", "green", texture_category="stripe"
    )
    DARK_YELLOW_AND_GREEN_STRIPE = _color_combo_textures_entry(
        "yellow", "green", dark=True, texture_category="stripe"
    )

    YELLOW_AND_BLUE_STRIPE = _color_combo_textures_entry(
        "yellow", "blue", texture_category="stripe"
    )
    DARK_YELLOW_AND_BLUE_STRIPE = _color_combo_textures_entry(
        "yellow", "blue", dark=True, texture_category="stripe"
    )

    YELLOW_AND_PURPLE_STRIPE = _color_combo_textures_entry(
        "yellow", "purple", texture_category="stripe"
    )
    DARK_YELLOW_AND_PURPLE_STRIPE = _color_combo_textures_entry(
        "yellow", "purple", dark=True, texture_category="stripe"
    )

    GREEN_AND_BLUE_STRIPE = _color_combo_textures_entry(
        "green", "blue", texture_category="stripe"
    )
    DARK_GREEN_AND_BLUE_STRIPE = _color_combo_textures_entry(
        "green", "blue", dark=True, texture_category="stripe"
    )

    GREEN_AND_PURPLE_STRIPE = _color_combo_textures_entry(
        "green", "purple", texture_category="stripe"
    )
    DARK_GREEN_AND_PURPLE_STRIPE = _color_combo_textures_entry(
        "green", "purple", dark=True, texture_category="stripe"
    )

    BLUE_AND_PURPLE_STRIPE = _color_combo_textures_entry(
        "blue", "purple", texture_category="stripe"
    )
    DARK_BLUE_AND_PURPLE_STRIPE = _color_combo_textures_entry(
        "blue", "purple", dark=True, texture_category="stripe"
    )

    # Variations of polka dot textures which consist of two colors and its dark variant
    # the color name is with the format: "(dark) {color1} and {color2} polka dot"
    RED_AND_YELLOW_POLKA_DOT = _color_combo_textures_entry(
        "red", "yellow", texture_category="polka_dot"
    )
    DARK_RED_AND_YELLOW_POLKA_DOT = _color_combo_textures_entry(
        "red", "yellow", dark=True, texture_category="polka_dot"
    )

    RED_AND_GREEN_POLKA_DOT = _color_combo_textures_entry(
        "red", "green", texture_category="polka_dot"
    )
    DARK_RED_AND_GREEN_POLKA_DOT = _color_combo_textures_entry(
        "red", "green", dark=True, texture_category="polka_dot"
    )

    RED_AND_BLUE_POLKA_DOT = _color_combo_textures_entry(
        "red", "blue", texture_category="polka_dot"
    )
    DARK_RED_AND_BLUE_POLKA_DOT = _color_combo_textures_entry(
        "red", "blue", dark=True, texture_category="polka_dot"
    )

    RED_AND_PURPLE_POLKA_DOT = _color_combo_textures_entry(
        "red", "purple", texture_category="polka_dot"
    )
    DARK_RED_AND_PURPLE_POLKA_DOT = _color_combo_textures_entry(
        "red", "purple", dark=True, texture_category="polka_dot"
    )

    YELLOW_AND_GREEN_POLKA_DOT = _color_combo_textures_entry(
        "yellow", "green", texture_category="polka_dot"
    )
    DARK_YELLOW_AND_GREEN_POLKA_DOT = _color_combo_textures_entry(
        "yellow", "green", dark=True, texture_category="polka_dot"
    )

    YELLOW_AND_BLUE_POLKA_DOT = _color_combo_textures_entry(
        "yellow", "blue", texture_category="polka_dot"
    )
    DARK_YELLOW_AND_BLUE_POLKA_DOT = _color_combo_textures_entry(
        "yellow", "blue", dark=True, texture_category="polka_dot"
    )

    YELLOW_AND_PURPLE_POLKA_DOT = _color_combo_textures_entry(
        "yellow", "purple", texture_category="polka_dot"
    )
    DARK_YELLOW_AND_PURPLE_POLKA_DOT = _color_combo_textures_entry(
        "yellow", "purple", dark=True, texture_category="polka_dot"
    )

    GREEN_AND_BLUE_POLKA_DOT = _color_combo_textures_entry(
        "green", "blue", texture_category="polka_dot"
    )
    DARK_GREEN_AND_BLUE_POLKA_DOT = _color_combo_textures_entry(
        "green", "blue", dark=True, texture_category="polka_dot"
    )

    GREEN_AND_PURPLE_POLKA_DOT = _color_combo_textures_entry(
        "green", "purple", texture_category="polka_dot"
    )
    DARK_GREEN_AND_PURPLE_POLKA_DOT = _color_combo_textures_entry(
        "green", "purple", dark=True, texture_category="polka_dot"
    )

    BLUE_AND_PURPLE_POLKA_DOT = _color_combo_textures_entry(
        "blue", "purple", texture_category="polka_dot"
    )
    DARK_BLUE_AND_PURPLE_POLKA_DOT = _color_combo_textures_entry(
        "blue", "purple", dark=True, texture_category="polka_dot"
    )

    # Swirl with single color and its dark variant.
    # name format: (dark) {color} swirl
    RED_SWIRL = _single_color_textures_entry("red", texture_category="swirl")
    DARK_RED_SWIRL = _single_color_textures_entry(
        "red", dark=True, texture_category="swirl"
    )
    YELLOW_SWIRL = _single_color_textures_entry("yellow", texture_category="swirl")
    DARK_YELLOW_SWIRL = _single_color_textures_entry(
        "yellow", dark=True, texture_category="swirl"
    )
    GREEN_SWIRL = _single_color_textures_entry("green", texture_category="swirl")
    DARK_GREEN_SWIRL = _single_color_textures_entry(
        "green", dark=True, texture_category="swirl"
    )
    BLUE_SWIRL = _single_color_textures_entry("blue", texture_category="swirl")
    DARK_BLUE_SWIRL = _single_color_textures_entry(
        "blue", dark=True, texture_category="swirl"
    )
    PURPLE_SWIRL = _single_color_textures_entry("purple", texture_category="swirl")
    DARK_PURPLE_SWIRL = _single_color_textures_entry(
        "purple", dark=True, texture_category="swirl"
    )

    # paisley pattern textures
    # name format: {color} paisley
    RED_PAISLEY = _single_color_textures_entry("red", texture_category="paisley")
    YELLOW_PAISLEY = _single_color_textures_entry("yellow", texture_category="paisley")
    GREEN_PAISLEY = _single_color_textures_entry("green", texture_category="paisley")
    BLUE_PAISLEY = _single_color_textures_entry("blue", texture_category="paisley")
    PURPLE_PAISLEY = _single_color_textures_entry("purple", texture_category="paisley")

    # Normal RGB textures
    BLUE = TextureEntry(
        name="blue",
        color_value=(078.0 / 255.0, 121.0 / 255.0, 167.0 / 255.0),
    )
    RED = TextureEntry(
        name="red",
        color_value=(255.0 / 255.0, 087.0 / 255.0, 089.0 / 255.0),
    )
    GREEN = TextureEntry(
        name="green",
        color_value=(089.0 / 255.0, 169.0 / 255.0, 079.0 / 255.0),
    )
    ORANGE = TextureEntry(
        name="orange",
        color_value=(242.0 / 255.0, 142.0 / 255.0, 043.0 / 255.0),
    )
    YELLOW = TextureEntry(
        name="yellow",
        color_value=(237.0 / 255.0, 201.0 / 255.0, 072.0 / 255.0),
    )
    PURPLE = TextureEntry(
        name="purple",
        color_value=(176.0 / 255.0, 122.0 / 255.0, 161.0 / 255.0),
    )
    PINK = TextureEntry(
        name="pink",
        color_value=(255.0 / 255.0, 157.0 / 255.0, 167.0 / 255.0),
    )
    CYAN = TextureEntry(
        name="cyan", color_value=(118.0 / 255.0, 183.0 / 255.0, 178.0 / 255.0)
    )
    OLIVE = TextureEntry(
        name="olive", color_value=(128.0 / 255.0, 128.0 / 255.0, 0.0 / 255.0)
    )

    # Darker versions of some colors. Should only be used in novel concept grounding
    DARK_BLUE = convert_to_darker_color(BLUE)
    DARK_RED = convert_to_darker_color(RED)
    DARK_GREEN = convert_to_darker_color(GREEN)
    DARK_ORANGE = convert_to_darker_color(ORANGE)
    DARK_YELLOW = convert_to_darker_color(YELLOW)
    DARK_PURPLE = convert_to_darker_color(PURPLE)
    DARK_PINK = convert_to_darker_color(PINK)
    DARK_CYAN = convert_to_darker_color(CYAN)

    @classmethod
    def convert_to_darker_color(cls, color: TextureEntry) -> TextureEntry:
        """
        Darken a color by converting it to HSV format and lowering its V value
        """
        color_value = color.color_value
        h, s, v = colorsys.rgb_to_hsv(*color_value)
        v = max(0.0, v - 0.3)
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        darker_color = TextureEntry(name="dark " + color.name, color_value=(r, g, b))
        return darker_color

    @classmethod
    def find_lighter_variant(cls, color: TextureEntry):
        """
        Return the lighter version of a color if one exists
        """
        if "dark" not in color.name:
            return color
        return cls[(color.name[5:]).upper().replace(" ", "_")].value

    @classmethod
    def find_darker_variant(cls, color: TextureEntry):
        """
        Return the darker version of a color if one exists
        """
        if "dark" in color.name:
            return color
        return cls[("dark_" + color.name).upper().replace(" ", "_")].value

    @classmethod
    def find_opposite_variant(cls, color: TextureEntry):
        """
        Return the lighter/darker version of a dark/light color
        """
        if "dark" in color.name:
            return cls.find_lighter_variant(color)
        else:
            return cls.find_darker_variant(color)

    @classmethod
    def lookup_closest_color(cls, rgb: Tuple[float, float, float]) -> TextureEntry:
        """
        Given a RGB color value, find the closest color defined in the encyclopedia.
        Currently, for simplicity, we use Euclidean distance to measure color distance.
        But the Euclidean distance does not match the human perception very well as discussed here
        https://stackoverflow.com/questions/9018016/how-to-compare-two-colors-for-similarity-difference.
        Therefore, future work will include using a better color distance metrics, perhaps the deltaE mentioned in the
        above link.
        TODO: use better color distance metrics
        """
        assert all(
            [element <= 1.0 for element in rgb]
        ), "Please provide a normalized RGB color value"

        rgb = np.array(rgb)[np.newaxis, ...]
        all_defined_color_values = np.array([e.value.color_value for e in cls])
        return [e.value for e in cls][
            np.argmax(
                np.linalg.norm(rgb - all_defined_color_values, ord=2, axis=-1), axis=0
            )
        ]

    @classmethod
    def lookup_color_by_name(cls, name: str) -> TextureEntry:
        """
        Given a color name, return corresponding color entry
        """
        for e in cls:
            if name == e.value.name:
                return e
        raise ValueError(f"Cannot find provided color {name}")

    @classmethod
    def all_entries(cls) -> List[TextureEntry]:
        return [e for e in cls if "dark" not in e.value.name]

    @classmethod
    def all_light_dark_entries(cls) -> List[TextureEntry]:
        """
        Textures used in novel concept grounding tasks
        """
        return [
            # RGB colors
            cls.BLUE,
            cls.DARK_BLUE,
            cls.RED,
            cls.DARK_RED,
            cls.GREEN,
            cls.DARK_GREEN,
            cls.ORANGE,
            cls.DARK_ORANGE,
            cls.YELLOW,
            cls.DARK_YELLOW,
            cls.PURPLE,
            cls.DARK_PURPLE,
            cls.PINK,
            cls.DARK_PINK,
            cls.CYAN,
            cls.DARK_CYAN,
            # stripe textures
            cls.RED_AND_YELLOW_STRIPE,
            cls.DARK_RED_AND_YELLOW_STRIPE,
            cls.RED_AND_GREEN_STRIPE,
            cls.DARK_RED_AND_GREEN_STRIPE,
            cls.RED_AND_BLUE_STRIPE,
            cls.DARK_RED_AND_BLUE_STRIPE,
            cls.RED_AND_PURPLE_STRIPE,
            cls.DARK_RED_AND_PURPLE_STRIPE,
            cls.YELLOW_AND_GREEN_STRIPE,
            cls.DARK_YELLOW_AND_GREEN_STRIPE,
            cls.YELLOW_AND_BLUE_STRIPE,
            cls.DARK_YELLOW_AND_BLUE_STRIPE,
            cls.YELLOW_AND_PURPLE_STRIPE,
            cls.DARK_YELLOW_AND_PURPLE_STRIPE,
            cls.GREEN_AND_BLUE_STRIPE,
            cls.DARK_GREEN_AND_BLUE_STRIPE,
            cls.GREEN_AND_PURPLE_STRIPE,
            cls.DARK_GREEN_AND_PURPLE_STRIPE,
            cls.BLUE_AND_PURPLE_STRIPE,
            cls.DARK_BLUE_AND_PURPLE_STRIPE,
            # polka dots textures
            cls.RED_AND_YELLOW_POLKA_DOT,
            cls.DARK_RED_AND_YELLOW_POLKA_DOT,
            cls.RED_AND_GREEN_POLKA_DOT,
            cls.DARK_RED_AND_GREEN_POLKA_DOT,
            cls.RED_AND_BLUE_POLKA_DOT,
            cls.DARK_RED_AND_BLUE_POLKA_DOT,
            cls.RED_AND_PURPLE_POLKA_DOT,
            cls.DARK_RED_AND_PURPLE_POLKA_DOT,
            cls.YELLOW_AND_GREEN_POLKA_DOT,
            cls.DARK_YELLOW_AND_GREEN_POLKA_DOT,
            cls.YELLOW_AND_BLUE_POLKA_DOT,
            cls.DARK_YELLOW_AND_BLUE_POLKA_DOT,
            cls.YELLOW_AND_PURPLE_POLKA_DOT,
            cls.DARK_YELLOW_AND_PURPLE_POLKA_DOT,
            cls.GREEN_AND_BLUE_POLKA_DOT,
            cls.DARK_GREEN_AND_BLUE_POLKA_DOT,
            cls.GREEN_AND_PURPLE_POLKA_DOT,
            cls.DARK_GREEN_AND_PURPLE_POLKA_DOT,
            cls.BLUE_AND_PURPLE_POLKA_DOT,
            cls.DARK_BLUE_AND_PURPLE_POLKA_DOT,
            # swirl textures
            cls.RED_SWIRL,
            cls.DARK_RED_SWIRL,
            cls.YELLOW_SWIRL,
            cls.DARK_YELLOW_SWIRL,
            cls.GREEN_SWIRL,
            cls.DARK_GREEN_SWIRL,
            cls.BLUE_SWIRL,
            cls.DARK_BLUE_SWIRL,
            cls.PURPLE_SWIRL,
            cls.DARK_PURPLE_SWIRL,
        ]
