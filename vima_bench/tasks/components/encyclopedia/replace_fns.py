import os
from functools import partial

import numpy as np

__all__ = ["container_replace_fn", "kit_obj_fn", "google_scanned_obj_fn"]

Z_SHRINK_FACTOR = 1.1
XY_SHRINK_FACTOR = 1


def default_replace_fn(*args, **kwargs):
    size = kwargs["size"]
    scaling = kwargs["scaling"]
    if isinstance(scaling, float):
        size = [s * scaling for s in size]
    else:
        size = [s1 * s2 for s1, s2 in zip(scaling, size)]
    return {"DIM": size}


def container_replace_fn(*args, **kwargs):
    size = kwargs["size"]
    size = [
        value / XY_SHRINK_FACTOR if i < 2 else value / Z_SHRINK_FACTOR
        for i, value in enumerate(size)
    ]
    scaling = kwargs["scaling"]
    if isinstance(scaling, float):
        size = [s * scaling for s in size]
    else:
        size = [s1 * s2 for s1, s2 in zip(scaling, size)]

    return {"DIM": size, "HALF": np.float32(size) / 2}


def _kit_obj_common(*args, **kwargs):
    fname = kwargs["fname"]
    assets_root = kwargs["assets_root"]
    fname = os.path.join(assets_root, "kitting", fname)
    scale = get_scale_from_map(fname[:-4].split("/")[-1], _kit_obj_scale_map)
    scaling = kwargs["scaling"]
    if isinstance(scaling, float):
        scale = [s * scaling for s in scale]
    else:
        scale = [s1 * s2 for s1, s2 in zip(scaling, scale)]
    return {
        "FNAME": (fname,),
        "SCALE": [
            scale[0] / XY_SHRINK_FACTOR,
            scale[1] / XY_SHRINK_FACTOR,
            scale[2] / Z_SHRINK_FACTOR,
        ],
    }


def kit_obj_fn(fname):
    return partial(_kit_obj_common, fname=fname)


def _google_scanned_obj_common(*args, **kwargs):
    fname = kwargs["fname"]
    assets_root = kwargs["assets_root"]
    fname = os.path.join(assets_root, "google", "meshes_fixed", fname)
    scale = get_scale_from_map(fname[:-4].split("/")[-1], _google_scanned_obj_scale_map)
    scaling = kwargs["scaling"]
    if isinstance(scaling, float):
        scale = [s * scaling for s in scale]
    else:
        scale = [s1 * s2 for s1, s2 in zip(scaling, scale)]
    return {
        "FNAME": (fname,),
        "SCALE": [
            scale[0] / XY_SHRINK_FACTOR,
            scale[1] / XY_SHRINK_FACTOR,
            scale[2] / Z_SHRINK_FACTOR,
        ],
        "COLOR": (0.2, 0.2, 0.2),
    }


def google_scanned_obj_fn(fname):
    return partial(_google_scanned_obj_common, fname=fname)


def get_scale_from_map(key, map):
    scale = map.get(key)
    if isinstance(scale, float):
        scale = [scale, scale, scale]
    return scale


_kit_obj_scale_map = {
    "capital_letter_a": [0.003 * 1.8, 0.003 * 1.8, 0.001 * 1.8],
    "capital_letter_e": [0.003 * 1.8, 0.003 * 1.8, 0.001 * 1.8],
    "capital_letter_g": [0.003 * 1.8, 0.003 * 1.8, 0.001 * 1.8],
    "capital_letter_m": [0.003 * 1.7, 0.003 * 1.7, 0.001 * 1.7],
    "capital_letter_r": [0.003 * 1.8, 0.003 * 1.8, 0.001 * 1.8],
    "capital_letter_t": [0.003 * 1.8, 0.003 * 1.8, 0.001 * 1.8],
    "capital_letter_v": [0.003 * 1.8, 0.003 * 1.8, 0.001 * 1.8],
    "cross": [0.003 * 1.6, 0.003 * 1.6, 0.001 * 1.6],
    "diamond": [0.003 * 1.6, 0.003 * 1.6, 0.001 * 1.6],
    "triangle": [0.003 * 1.6, 0.003 * 1.6, 0.001 * 1.6],
    "flower": [0.003 * 1.8, 0.003 * 1.8, 0.001 * 1.8],
    "heart": [0.003 * 1.8, 0.003 * 1.8, 0.001 * 1.8],
    "hexagon": [0.003 * 1.8, 0.003 * 1.8, 0.001 * 1.8],
    "pentagon": [0.003 * 1.8, 0.003 * 1.8, 0.001 * 1.8],
    "right_angle": [0.003 * 1.6, 0.003 * 1.6, 0.001 * 1.6],
    "ring": [0.003 * 1.6, 0.003 * 1.6, 0.001 * 1.6],
    "round": [0.003 * 1.6, 0.003 * 1.6, 0.001 * 1.6],
    "square": [0.003 * 1.6, 0.003 * 1.6, 0.001 * 1.6],
    "star": [0.003 * 1.8, 0.003 * 1.8, 0.001 * 1.8],
}


_google_scanned_obj_scale_map = {
    "frypan": 0.275 * 3,
}
