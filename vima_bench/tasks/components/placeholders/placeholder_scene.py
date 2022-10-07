import os
import tempfile
from typing import Callable, Optional, Union, List

import cv2
import numpy as np
import pybullet as p

from .base import Placeholder
from ..cameras import get_agent_cam_config, Oracle
from ...utils import pybullet_utils, misc_utils as utils

PLANE_URDF_PATH = "plane/plane.urdf"
UR5_WORKSPACE_URDF_PATH = "ur5/workspace.urdf"


class PlaceholderScene(Placeholder):
    allowed_expressions = {"image"}
    all_views = {"top", "front"}

    """
    The placeholder scene in the prompt.
    """

    def __init__(
        self,
        create_scene_fn: Callable,
        assets_root: str,
        views: Optional[Union[str, List[str]]] = None,
        image_size: tuple[int, int] = (128, 256),
        seed: Optional[int] = None,
    ):
        """
        :param create_scene_fn: A callable function used to create the scene. It takes one argument `sim_id`,
        which is the physics client ID of pybullet simulation that used for scene rendering.
        :param views: Specify the camera views.
        :param image_size: Specify the image size.
        """
        self._create_scene_fn = create_scene_fn
        self._assets_root = assets_root

        views = views or ["top", "front"]
        if isinstance(views, str):
            views = [views]
        assert isinstance(views, list)

        assert set(views).issubset(self.all_views)

        # get view camera config
        all_cam_config = get_agent_cam_config(image_size)
        self._cam_config = {view: all_cam_config[view] for view in views}

        self._rng = np.random.default_rng(seed=seed)

    def get_expression(self, types: List[str], *args, **kwargs):
        assert set(types).issubset(
            self.allowed_expressions
        ), "Unsupported type of expression provided"

        expressions = {}
        for each_type in types:
            if each_type == "image":
                image_expressions = {
                    "rgb": {},
                    "segm": {},
                    "placeholder_type": "scene",
                }
                # create scene
                scene_renderer = SceneRenderEnv(self._assets_root)
                self._create_scene_fn(scene_renderer, scene_renderer.client_id)
                # now take pictures for this scene
                for name, config in self._cam_config.items():
                    rgb, depth, segm = _render_camera(config, scene_renderer.client_id)
                    image_expressions["rgb"][name] = rgb
                    image_expressions["segm"][name] = segm
                # check obj ids should match
                assert len(scene_renderer.obj_id_reverse_mapping) > 0
                ids_from_reverse_map = list(
                    scene_renderer.obj_id_reverse_mapping.keys()
                )
                ids_from_obj_ids = [
                    x for v in scene_renderer.obj_ids.values() for x in v
                ]
                assert set(ids_from_reverse_map) == set(ids_from_obj_ids)
                image_expressions["segm"]["obj_info"] = [
                    {
                        "obj_id": k,
                        "obj_name": v["obj_name"],
                        "obj_color": v["texture_name"],
                    }
                    for k, v in scene_renderer.obj_id_reverse_mapping.items()
                ]
                expressions["image"] = image_expressions
                # disconnect simulation
                p.disconnect(physicsClientId=scene_renderer.client_id)
                del scene_renderer
        return expressions


class SceneRenderEnv(object):
    def __init__(self, assets_root, hz=240):
        self.pix_size = 0.003125
        self.bounds = np.array([[0.25, 0.75], [-0.5, 0.5], [0, 0.3]])
        self.obj_ids = {"fixed": [], "rigid": [], "deformable": []}
        self.assets_root = assets_root
        self.oracle_cams = Oracle.CONFIG

        client_id = p.connect(p.DIRECT)
        self.client_id = client_id
        file_io = p.loadPlugin("fileIOPlugin", physicsClientId=client_id)
        if file_io < 0:
            raise RuntimeError("pybullet: cannot load FileIO!")
        if file_io >= 0:
            p.executePluginCommand(
                file_io,
                textArgument=assets_root,
                intArgs=[p.AddFileIOAction],
                physicsClientId=client_id,
            )
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=self.client_id)
        p.setPhysicsEngineParameter(enableFileCaching=0, physicsClientId=self.client_id)
        p.setAdditionalSearchPath(assets_root, physicsClientId=self.client_id)
        p.setAdditionalSearchPath(tempfile.gettempdir(), physicsClientId=self.client_id)
        p.setTimeStep(1.0 / hz, physicsClientId=self.client_id)

        self._setup_workspace()

    def _setup_workspace(self):
        self.obj_ids = {"fixed": [], "rigid": [], "deformable": []}
        self.obj_id_reverse_mapping = {}
        p.resetSimulation(p.RESET_USE_DEFORMABLE_WORLD, physicsClientId=self.client_id)

        # Temporarily disable rendering to load scene faster.
        p.configureDebugVisualizer(
            p.COV_ENABLE_RENDERING, 0, physicsClientId=self.client_id
        )

        pybullet_utils.load_urdf(
            p,
            os.path.join(self.assets_root, PLANE_URDF_PATH),
            [0, 0, -0.001],
            physicsClientId=self.client_id,
        )
        pybullet_utils.load_urdf(
            p,
            os.path.join(self.assets_root, UR5_WORKSPACE_URDF_PATH),
            [0.5, 0, 0],
            physicsClientId=self.client_id,
        )
        # Re-enable rendering.
        p.configureDebugVisualizer(
            p.COV_ENABLE_RENDERING, 1, physicsClientId=self.client_id
        )

    def renderer_get_random_pose(self, obj_size, rng):
        # Get erosion size of object in pixels.
        max_size = np.sqrt(obj_size[0] ** 2 + obj_size[1] ** 2)
        erode_size = int(np.round(max_size / self.pix_size))

        _, hmap, obj_mask = self._get_true_image()

        # Randomly sample an object pose within free-space pixels.
        free = np.ones(obj_mask.shape, dtype=np.uint8)
        for obj_ids in self.obj_ids.values():
            for obj_id in obj_ids:
                free[obj_mask == obj_id] = 0
        free[0, :], free[:, 0], free[-1, :], free[:, -1] = 0, 0, 0, 0
        free = cv2.erode(free, np.ones((erode_size, erode_size), np.uint8))
        if np.sum(free) == 0:
            return None, None
        pix = utils.sample_distribution(prob=np.float32(free), rng=rng)
        pos = utils.pix_to_xyz(pix, hmap, self.bounds, self.pix_size)
        pos = (pos[0], pos[1], obj_size[2] / 2)
        theta = rng.random() * 2 * np.pi
        rot = utils.eulerXYZ_to_quatXYZW((0, 0, theta))
        return pos, rot

    def _get_true_image(self):
        # Capture near-orthographic RGB-D images and segmentation masks.
        # color: (C, H, W), depth: (1, H, W) within [0, 1]
        color, depth, segm = _render_camera(self.oracle_cams[0], self.client_id)
        # process images to be compatible with oracle input
        # rgb image, from CHW to HWC
        color = np.transpose(color, (1, 2, 0))
        # depth image, from (1, H, W) to (H, W) within [0, 20]
        depth = 20.0 * np.squeeze(depth, axis=0)

        # Combine color with masks for faster processing.
        color = np.concatenate((color, segm[Ellipsis, None]), axis=2)

        # Reconstruct real orthographic projection from point clouds.
        hmaps, cmaps = utils.reconstruct_heightmaps(
            [color], [depth], self.oracle_cams, self.bounds, self.pix_size
        )

        # Split color back into color and masks.
        cmap = np.uint8(cmaps)[0, Ellipsis, :3]
        hmap = np.float32(hmaps)[0, Ellipsis]
        mask = np.int32(cmaps)[0, Ellipsis, 3:].squeeze()
        return cmap, hmap, mask


def _render_camera(config, client_id, image_size=None):
    if not image_size:
        image_size = config["image_size"]

    # OpenGL camera settings.
    lookdir = np.float32([0, 0, 1]).reshape(3, 1)
    updir = np.float32([0, -1, 0]).reshape(3, 1)
    rotation = p.getMatrixFromQuaternion(config["rotation"], physicsClientId=client_id)
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
    # transpose from HWC to CHW
    color = np.transpose(color, (2, 0, 1))

    # Get depth image.
    depth_image_size = (image_size[0], image_size[1])
    zbuffer = np.array(depth).reshape(depth_image_size)
    depth = zfar + znear - (2.0 * zbuffer - 1.0) * (zfar - znear)
    depth = (2.0 * znear * zfar) / depth
    # normalize depth to be within range [0, 1]
    depth /= 20.0
    # add 'C' dimension
    depth = depth[np.newaxis, ...]

    # Get segmentation image.
    segm = np.uint8(segm).reshape(depth_image_size)

    return color, depth, segm
