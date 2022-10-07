from __future__ import annotations

import math
import os
import tempfile

import numpy as np
import pybullet as p

from .base import Placeholder
from ..cameras import get_agent_cam_config
from ..encyclopedia import TexturePedia, TextureEntry
from ...utils import misc_utils as utils
from ...utils.pybullet_utils import p_change_texture


class PlaceholderObj(Placeholder):
    allowed_expressions = {"image", "name", "novel_name", "alias"}

    """
    The placeholder object in the prompt.
    It should be a real instance of a certain object that appears in the prompt, instead of an encyclopedia entry.
    After loading an object with `pybullet.load_urdf`,
    objects that will appear in the prompt should be wrapped with this class.
    """

    def __init__(
        self,
        name: str,
        obj_id: int,
        urdf: str,
        novel_name: list[str] | None = None,
        obj_position: tuple[float, float, float] = None,
        obj_orientation: tuple[float, float, float, float] = None,
        alias: list[str] | None = None,
        color: str | tuple[float, float, float] | TextureEntry | None = None,
        use_neutral_color: bool = False,
        global_scaling: float = 1.0,
        image_size: tuple[int, int] = (128, 256),
        seed: int | None = None,
        retain_temp: bool = False,
    ):
        self.name = name
        self.obj_id = obj_id
        self.urdf = urdf
        self.novel_name = novel_name
        self.obj_position = obj_position
        self.obj_orientation = obj_orientation
        if self.obj_position is None:
            # use default position
            self.obj_position = (0.5, 0, 0)

        if self.obj_orientation is None:
            # use default orientation
            self.obj_orientation = utils.eulerXYZ_to_quatXYZW((0, 0, math.pi / 2))

        self.alias = alias
        self.retain_temp = retain_temp
        if use_neutral_color:
            color = TextureEntry(
                name="gray", color_value=(186.0 / 255.0, 176.0 / 255.0, 172.0 / 255.0)
            )
        else:
            # if `color` is not provided, will try to infer from the simulation
            if color is None:
                color_rgb = p.getVisualShapeData(obj_id)[0][7][:3]
                color = TexturePedia.lookup_closest_color(color_rgb)
            # if 'color' is provided in RGB format, will find the closest color in the encyclopedia
            elif isinstance(color, tuple) and len(color) == 3:
                color_rgb = color
                color = TexturePedia.lookup_closest_color(color_rgb)
            elif isinstance(color, str):
                color = TexturePedia.lookup_color_by_name(color)
            elif isinstance(color, TextureEntry):
                color = color
            else:
                raise ValueError(f"Unsupported type of color provided {type(color)}")
        self.color = color
        self.global_scaling = global_scaling
        self.image_views = get_agent_cam_config(image_size)

        self._rng = np.random.default_rng(seed=seed)

    def _render_camera(self, config, image_size=None, client_id=None):
        """Render RGB-D image with specified camera configuration."""
        if not image_size:
            image_size = config["image_size"]

        # OpenGL camera settings.
        lookdir = np.float32([0, 0, 1]).reshape(3, 1)
        updir = np.float32([0, -1, 0]).reshape(3, 1)

        rotation = p.getMatrixFromQuaternion(
            config["rotation"], physicsClientId=client_id
        )

        rotm = np.float32(rotation).reshape(3, 3)
        lookdir = (rotm @ lookdir).reshape(-1)
        updir = (rotm @ updir).reshape(-1)
        lookat = config["position"] + lookdir
        focal_len = config["intrinsics"][0]
        znear, zfar = config["zrange"]
        viewm = p.computeViewMatrix(
            config["position"], lookat, updir, physicsClientId=client_id
        )
        fovh = (image_size[0] / 2) / focal_len
        fovh = 180 * np.arctan(fovh) * 2 / np.pi

        # Notes: 1) FOV is vertical FOV 2) aspect must be float
        aspect_ratio = image_size[1] / image_size[0]
        projm = p.computeProjectionMatrixFOV(
            fovh, aspect_ratio, znear, zfar, physicsClientId=client_id
        )

        # Render with OpenGL camera settings.
        _, _, color, depth, segm = p.getCameraImage(
            width=image_size[1],
            height=image_size[0],
            viewMatrix=viewm,
            projectionMatrix=projm,
            flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
            renderer=p.ER_BULLET_HARDWARE_OPENGL,
            physicsClientId=client_id,
        )

        # Get color image.
        color_image_size = (image_size[0], image_size[1], 4)
        color = np.array(color, dtype=np.uint8).reshape(color_image_size)
        color = color[:, :, :3]  # remove alpha channel
        if config["noise"]:
            color = np.int32(color)
            color += np.int32(self._rng.normal(0, 3, image_size))
            color = np.uint8(np.clip(color, 0, 255))
        # transpose from HWC to CHW
        color = np.transpose(color, (2, 0, 1))

        # Get segmentation image.
        segm = np.uint8(segm).reshape((image_size[0], image_size[1]))
        return color, segm

    def get_expression(self, types: list[str], prepend_color: bool = False):
        assert set(types).issubset(
            self.allowed_expressions
        ), "Unsupported type of expression provided"

        expressions = {}
        for each_type in types:
            if each_type == "image":
                image_expressions = {
                    "rgb": {},
                    "segm": {},
                    "placeholder_type": "object",
                }
                # start a new pybullet simulation only for showing object images
                obj_showing_sim_id = p.connect(p.DIRECT)

                # load the object
                _obj_id = p.loadURDF(
                    self.urdf,
                    basePosition=self.obj_position,
                    baseOrientation=self.obj_orientation,
                    globalScaling=self.global_scaling,
                    physicsClientId=obj_showing_sim_id,
                )
                # change object's color
                p_change_texture(_obj_id, self.color, obj_showing_sim_id)
                # aspect_ratio = self.image_size[1] / self.image_size[0]
                aspect_ratio = 1  # 1:1 camera with white color padding along y axis
                # now take pictures for this object
                for view_name, view_config in self.image_views.items():
                    rgb, seg = self._render_camera(
                        view_config, client_id=obj_showing_sim_id
                    )

                    image_expressions["rgb"][view_name] = rgb
                    image_expressions["segm"][view_name] = seg
                    # segmentation must include several pixels equal to the object id
                    assert (
                        np.sum(seg == _obj_id) >= 0
                    ), "INTERNAL: segmentation must include several pixels equal to object id"
                image_expressions["segm"]["obj_info"] = {
                    "obj_id": _obj_id,
                    "obj_name": self.name,
                    "obj_color": self.color.name,
                }

                # disconnect the simulation
                p.disconnect(physicsClientId=obj_showing_sim_id)
                expressions["image"] = image_expressions
            elif each_type == "name":
                expressions["name"] = (
                    f"{self.color.name} {self.name}" if prepend_color else self.name
                )
            elif each_type == "novel_name":
                novel_name = self._rng.choice(self.novel_name)
                expressions["novel_name"] = (
                    f"{self.color.name} {novel_name}" if prepend_color else novel_name
                )
            elif each_type == "alias":
                alias = self._rng.choice(self.alias)
                expressions["alias"] = (
                    f"{self.color.name} {alias}" if prepend_color else alias
                )
        if not self.retain_temp:
            # this method is only called once at `env.reset()`, so we delete temporal urdf files here
            tmpdir = tempfile.gettempdir()
            if os.path.normpath(os.path.dirname(self.urdf)) == os.path.normpath(tmpdir):
                os.remove(self.urdf)
        return expressions
