"""
Define data structures used in all encyclopedias
"""
from typing import NamedTuple, Tuple, Optional, List, Callable

from .profiles import ProfilePedia

__all__ = ["SizeRange", "ObjEntry", "TextureEntry"]


# size boundary is a tuple of three float numbers, represents x, y, z, respectively
_SizeBoundary = Tuple[float, float, float]

# color value is a tuple of R, G, B
_ColorValue = Tuple[float, float, float]


# we can sample the size of a certain object from the range
class SizeRange(NamedTuple):
    """
    low: the minimum possible size in x, y, z
    high: the maximum possible size in x, y, z
    """

    low: _SizeBoundary
    high: _SizeBoundary


# defines the data structure of each entry in the object encyclopedia
class ObjEntry(NamedTuple):
    """
    name: name of the object, this name will be used in the prompts
    assets: path of asset files
    size_range: the size range of this object.
        Note that for objects that are not from template, their sizes are fixed.
        In this case, the upper should equal to the lower.
    from_template: whether this object can be directly added or from a template file. If True, the asset file is
        actually a template urdf file
    replace_fn: use when from template. Take `size` as input and return a replace dict to fill in the template.
    pose_transform_fn: transform sampled pose.
    alias: alias names of the object
    novel_name: novel name of the object, which will be used in novel concepts grounding
    template_file: some objects are made from template. In this case, fill the `template_file` field with corresponding template file
    symmetry: (in radians) the angle of rotational symmetry -- the smallest angle for which the figure can be rotated to coincide with itself.
            symmetry == 0 means to not use it in tasks where symmetry matters
            symmetry == 2 * pi means one object has no rotational symmetry.
            symmetry == None means the symmetry property of an object is not determined
    profile: the profile type of each object, -1 for undermined, 0 for square, 1 for circle, ... you could add it here
    """

    name: str
    assets: str
    size_range: SizeRange
    from_template: bool = False
    replace_fn: Optional[Callable] = None
    pose_transform_fn: Optional[Callable] = None
    alias: Optional[List[str]] = None
    novel_name: Optional[List[str]] = None
    template_file: Optional[str] = None
    symmetry: Optional[float] = None
    profile: Optional[ProfilePedia] = ProfilePedia.UNDETERMINED


# defines the data structure of each entry in the texture encyclopedia
class TextureEntry(NamedTuple):
    """
    name: can be the name of the color, e.g., red, or name of the texture, e.g., wooden
    color_value: color value in (R, G, B), None for using img textures
    texture_asset: asset file for the texture
    alias: alias names of the color
    novel_name: novel name of the color, which will be used in novel concepts grounding
    """

    name: str
    color_value: Optional[_ColorValue] = None
    texture_asset: Optional[str] = None
    alias: Optional[List[str]] = None
    novel_name: Optional[List[str]] = None


# define the data structure of each entry in the adjective encyclopedia
class AdjectiveEntry(NamedTuple):
    """
    name:
    """

    name: str
    alias: Optional[List[str]] = None
    novel_name: Optional[List[str]] = None
