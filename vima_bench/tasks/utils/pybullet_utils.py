"""PyBullet utilities for loading assets."""
from __future__ import annotations

import os
import random
import string
import tempfile
from typing import Dict, Optional, Any, Union, List

import numpy as np
import pybullet as p

from ..components.encyclopedia.definitions import ObjEntry, TextureEntry
from ..components.encyclopedia.replace_fns import default_replace_fn

INVISIBLE_ALPHA = 0
VISIBLE_ALPHA = 1


def get_urdf_path(
    env,
    obj_entry: ObjEntry,
    size,
    replace: Optional[Dict[str, Any]] = None,
    scaling: Union[float, List[float]] = 1.0,
):
    if obj_entry.from_template:
        # handle obj from a template
        template = obj_entry.assets
        if replace is None:
            if obj_entry.replace_fn is not None:
                replace = obj_entry.replace_fn(
                    size=size, scaling=scaling, assets_root=env.assets_root
                )
            else:
                replace = default_replace_fn(size=size, scaling=scaling)
        urdf_full_path = fill_template(env.assets_root, template, replace)
    else:
        # handle obj that can be directly added from urdf file
        assert obj_entry.size_range.low == obj_entry.size_range.high, (
            f"Try to add an object not from a template. "
            f"In this case its size is fixed and expect `size_range.low` = `size_range.high`, "
            f"but got {obj_entry.size_range.low} and {obj_entry.size_range.high}"
        )
        urdf = obj_entry.assets
        urdf_full_path = os.path.join(env.assets_root, urdf)
    return urdf_full_path


def add_any_object(
    env,
    obj_entry: ObjEntry,
    pose,
    size,
    replace: Optional[Dict[str, Any]] = None,
    retain_temp: bool = False,
    category: str = "rigid",
    scaling: Union[float, List[float]] = 1.0,
):
    """
    Add objects either from urdf template or directly from urdf files.
    :param env: An env instance.
    :param obj_entry: ObjEntry.
    :param pose: Pose of the object.
    :param size: Size of the object.
    :param replace: Key strings to be replaced.
        If None, will first try to use obj_entry.replace_fn and then try replace size only.
    :param retain_temp: Keep the generated temporal file or not if the object is added from a template.
    :param category: Category of the object, can be one of "fixed", "rigid", "deformable"
    """
    fixed_base = 1 if category == "fixed" else 0

    urdf_full_path = get_urdf_path(env, obj_entry, size, replace, scaling)
    if (
        obj_entry.pose_transform_fn is not None
        and pose[0] is not None
        and pose[1] is not None
    ):
        pose = obj_entry.pose_transform_fn(original_pose=pose)
    obj_id = load_urdf(
        p,
        urdf_full_path,
        pose[0],
        pose[1],
        useFixedBase=fixed_base,
        physicsClientId=env.client_id,
    )
    if obj_id is not None:
        env.obj_ids[category].append(obj_id)
    if obj_entry.from_template and not retain_temp:
        os.remove(urdf_full_path)
    return obj_id, urdf_full_path


def add_object_id_reverse_mapping_info(
    mapping_dict, obj_id, object_entry, texture_entry
):
    """
    add item that maps object id to object information,
        including obj_name, object's entry in ObjPedia, and object's texture entry in TexturePedia
    Args:
        mapping_dict: dict to add the item
        obj_id: obj id in bullet env
        object_entry: corresponding object entry to add
        texture_entry: corresponding object texture entry to add

        Employ a feature in Enum Class, see https://docs.python.org/3/library/enum.html#programmatic-access-to-enumeration-members-and-their-attributes
    Returns: None

    """
    mapping_dict[obj_id] = {
        **{f"obj_{k}": v for k, v in object_entry._asdict().items()},
        **{f"texture_{k}": v for k, v in texture_entry._asdict().items()},
    }


def fill_template(assets_root, template, replace):
    """Read a file and replace key strings."""
    full_template_path = os.path.join(assets_root, template)
    with open(full_template_path, "r") as file:
        fdata = file.read()
    for field in replace:
        for i in range(len(replace[field])):
            fdata = fdata.replace(f"{field}{i}", str(replace[field][i]))
    alphabet = string.ascii_lowercase + string.digits
    rname = "".join(random.choices(alphabet, k=16))
    tmpdir = tempfile.gettempdir()
    template_filename = os.path.split(template)[-1]
    fname = os.path.join(tmpdir, f"{template_filename}.{rname}")
    with open(fname, "w") as file:
        file.write(fdata)
    return fname


# BEGIN GOOGLE-EXTERNAL
def load_urdf(pybullet_client, file_path, *args, **kwargs):
    """Loads the given URDF filepath."""
    assert (
        "physicsClientId" in kwargs
    ), "You MUST explicitly provide pybullet client ID for safety reason!"
    # Handles most general file open case.
    try:
        return pybullet_client.loadURDF(file_path, *args, **kwargs)
    except pybullet_client.error:
        pass


# END GOOGLE-EXTERNAL


def set_visibility_bullet(pybullet_client_id, object_id, alpha_value):
    """
    Set the visibility of an object in PyBullet
    Args:
        pybullet_client_id: pybullet client id
        object_id: object unique id in the bullet env
        alpha_value: alpha value in RGBA

    Returns:

    """
    visual_shape_list = p.getVisualShapeData(
        object_id, physicsClientId=pybullet_client_id
    )
    for visual_shape in visual_shape_list:
        object_id, link_idx, _, _, _, _, _, rgba_color = visual_shape
        p.changeVisualShape(
            object_id, link_idx, rgbaColor=list(rgba_color)[:3] + [alpha_value]
        )


def is_one_object_in_a_hollow_2d_object(
    test_object_pose, test_object_size, hollow_object_pose, hollow_object_size
):
    """
    Test whether an object is in some hollow object (2D) e.g. a hollow square
    Args:
        test_object_pose:
        test_object_size:
        hollow_object_pose:
        hollow_object_size:

    Returns: bool

    """
    # obj_max_size = np.sqrt(test_object_size[0] ** 2 + test_object_size[1] ** 2)

    # Get translational error.
    diff_pos = np.float32(test_object_pose[0][:2]) - np.float32(
        hollow_object_pose[0][:2]
    )
    dist_pos = np.linalg.norm(diff_pos)
    # Rough judgment with margin compensation
    if dist_pos <= max(
        (test_object_size[0] + hollow_object_size[0]) / 2 + 0.01,
        (test_object_size[1] + hollow_object_size[1]) / 2 + 0.01,
    ):
        return True
    else:
        return False


def if_in_hollow_object(
    test_object_poses: list | tuple,
    test_object_size: list | np.ndarray,
    hollow_object_poses: list | tuple,
    hollow_object_size: list | np.ndarray,
) -> bool:
    """
    Helper for testing list of test objects and hollow objects with
    `is_one_object_in_a_hollow_2d_object`

    Args:
        test_object_poses: list of tuples / a tuple of poses
        test_object_size: test object xyz size(s) corresponding with poses
        hollow_object_poses: list of tuples / a tuple of poses
        hollow_object_size: hollow object xyz size(s) corresponding with poses

    Returns: bool of whether any of test object in any of hollow object

    """
    if not isinstance(test_object_poses, list):
        assert isinstance(test_object_poses, tuple)
        test_object_poses = [test_object_poses]
    if not isinstance(hollow_object_poses, list):
        assert isinstance(hollow_object_poses, tuple)
        hollow_object_poses = [hollow_object_poses]
    if not isinstance(test_object_size, list):
        assert isinstance(test_object_size, (np.ndarray, tuple))
        test_object_size = [test_object_size for _ in range(len(test_object_poses))]
    if not isinstance(hollow_object_size, list):
        assert isinstance(hollow_object_size, (np.ndarray, tuple))
        hollow_object_size = [
            hollow_object_size for _ in range(len(hollow_object_poses))
        ]
    assert len(test_object_poses) == len(test_object_poses)
    assert len(hollow_object_poses) == len(hollow_object_size)

    for test_idx, test_object_pose in enumerate(test_object_poses):
        for hollow_idx, hollow_object_pose in enumerate(hollow_object_poses):
            if is_one_object_in_a_hollow_2d_object(
                test_object_pose,
                test_object_size[test_idx],
                hollow_object_pose,
                hollow_object_size[hollow_idx],
            ):
                return True
    return False


def p_change_texture(obj_id: int, texture_entry: TextureEntry, client_id: int):
    assert not (
        texture_entry.color_value is None and texture_entry.texture_asset is None
    )
    if texture_entry.color_value is not None:
        p.changeVisualShape(
            obj_id,
            -1,
            rgbaColor=list(texture_entry.color_value) + [1],
            physicsClientId=client_id,
        )
    else:
        p.changeVisualShape(
            obj_id,
            -1,
            textureUniqueId=p.loadTexture(texture_entry.texture_asset, client_id),
            physicsClientId=client_id,
        )
