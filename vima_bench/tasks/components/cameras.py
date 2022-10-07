"""Camera configs."""
from __future__ import annotations

import numpy as np
import pybullet as p

__all__ = [
    "get_agent_cam_config",
    "Oracle",
]


def get_agent_cam_config(image_size: tuple[int, int] = (128, 256)):
    """
    Using a near perfect camera put at infinity for solving
    Args:
        image_size:

    Returns:

    """
    # For more info, we could refer to https://github.com/cliport/cliport/blob/152610b6913a84b5d67a356e6b80c591d80d7d7a/cliport/tasks/cameras.py#L26
    if isinstance(image_size, (tuple, list)):
        assert len(image_size) == 2, "image_size should be a tuple of length 2"
        height, width = image_size
    else:
        raise NotImplementedError("Unsupported type for image size")

    assert height == 128 and width == 256, (
        f"Please add camera configs of image size (128,256), because the camera are tuned for this size; "
        f"current size ({height},{width})"
    )
    return {
        "front": NearPerfectCamera128x256.CONFIG[0],
        "top": NearPerfectCamera128x256.CONFIG[1],
    }


class NearPerfectCamera128x256(object):
    """Top-down noiseless image used only by the oracle demonstrator."""

    # Near-orthographic projection.
    image_size = (128, 256)
    front_intrinsics = (64e4 // 1.80, 0, 320.0, 0, 63e4, 240.0, 0, 0, 1)
    top_intrinsics = (64e4 // 2.55, 0, 320.0, 0, 63e4, 240.0, 0, 0, 1)

    # Set default camera poses.
    front_offset = 0.09  # offset from the center of workspace to make the workspace close to the bottom in the view
    front_position = (1000 + 0.5 - front_offset, 0, 1000)
    front_rotation = (np.pi / 4, np.pi, -np.pi / 2)
    front_rotation = p.getQuaternionFromEuler(front_rotation)
    top_position = (0.5, 0, 1000.0)
    top_rotation = (0, np.pi, -np.pi / 2)
    top_rotation = p.getQuaternionFromEuler(top_rotation)

    # Camera config.
    CONFIG = [
        {
            "image_size": image_size,
            "intrinsics": front_intrinsics,
            "position": front_position,
            "rotation": front_rotation,
            # "zrange": (999.7, 1001.0),
            "zrange": (900, 10001.0),
            "noise": False,
        },
        {
            "image_size": image_size,
            "intrinsics": top_intrinsics,
            "position": top_position,
            "rotation": top_rotation,
            # "zrange": (999.7, 1001.0),
            "zrange": (900, 1001.0),
            "noise": False,
        },
    ]


class Oracle(object):
    """Top-down noiseless image used only by the oracle demonstrator."""

    # Near-orthographic projection.
    image_size = (480, 640)
    intrinsics = (63e4, 0, 320.0, 0, 63e4, 240.0, 0, 0, 1)
    position = (0.5, 0, 1000.0)
    rotation = p.getQuaternionFromEuler((0, np.pi, -np.pi / 2))

    # Camera config.
    CONFIG = [
        {
            "image_size": image_size,
            "intrinsics": intrinsics,
            "position": position,
            "rotation": rotation,
            # "zrange": (999.7, 1001.0),
            "zrange": (999.7, 1001.0),
            "noise": False,
        }
    ]
