from __future__ import annotations

from typing import Literal

import numpy as np
from pybullet import getBasePositionAndOrientation

from .base import SweepObjectsToZoneBase, det_to_integer
from ..base import OracleAgent
from ...components.encyclopedia.definitions import TextureEntry
from ...components.placeholders import PlaceholderObj, PlaceholderText
from ...utils.misc_utils import (
    eulerXYZ_to_quatXYZW,
    quatXYZW_to_eulerXYZ,
    invert,
    multiply,
)
from ...utils.pybullet_utils import (
    add_any_object,
    add_object_id_reverse_mapping_info,
    p_change_texture,
)


class WithoutExceeding(SweepObjectsToZoneBase):

    task_name = "sweep_without_exceeding"

    def __init__(
        self,
        # ====== task specific ======
        max_swept_obj: int = 3,
        constraint_range: list[float, float] | tuple[float, float] | None = None,
        swept_obj_express_types: Literal["image", "name"] = "image",
        bounds_express_types: Literal["image", "name"] = "image",
        constraint_express_types: Literal["image", "name"] = "image",
        prepend_color_to_name: bool = True,
        oracle_max_steps: int = 8,
        oracle_step_to_env_step_ratio: int = 3,
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
        if constraint_range is None:
            constraint_range = [0.25, 0.6]
        elif isinstance(constraint_range, (tuple, list)):
            assert len(constraint_range) == 2
        else:
            raise ValueError("constraint_range must be a tuple or list of 2 floats")

        task_meta = {
            "max_swept_obj": max_swept_obj,
            "constraint_range": constraint_range,
        }

        assert oracle_max_steps <= 20

        super().__init__(
            prompt_template="Sweep {det} {swept_obj} into {bounds} without exceeding {constraint}.",
            task_meta=task_meta,
            swept_obj_express_types=swept_obj_express_types,
            bounds_express_types=bounds_express_types,
            constraint_express_types=constraint_express_types,
            prepend_color_to_name=prepend_color_to_name,
            oracle_max_steps=oracle_max_steps,
            oracle_step_to_env_step_ratio=oracle_step_to_env_step_ratio,
            possible_dragged_obj_texture=possible_dragged_obj_texture,
            possible_base_obj_texture=possible_base_obj_texture,
            obs_img_views=obs_img_views,
            obs_img_size=obs_img_size,
            placeholder_img_size=placeholder_img_size,
            seed=seed,
            debug=debug,
        )

        self.placeholder_expression["det"] = dict(types=["text"])
        self._target_num = None
        self._n_achieved = None

    def reset(self, env):
        super().reset(env)

        probs = [
            self.task_meta["sample_prob"][i]
            for i in sorted(self.task_meta["sample_prob"])
        ]
        sampled_num_swept_objs = sorted(self.task_meta["sample_prob"])[
            self.rng.choice(len(probs), p=probs)
        ]

        sampled_swept_obj = self.possible_dragged_obj.value
        sampled_swept_obj_texture = self.rng.choice(
            self.possible_dragged_obj_texture
        ).value
        sampled_base_obj = self.possible_base_obj.value
        sampled_base_obj_texture = self.rng.choice(self.possible_base_obj_texture).value
        sampled_constraint_obj = self.possible_constraint_obj.value

        sampled_constraint_obj_texture = self.rng.choice(
            [
                e
                for e in self.possible_base_obj_texture
                if e.value != sampled_base_obj_texture
            ]
        ).value

        sampled_swept_obj_born_pos, sampled_distractor_obj_born_pos = [
            e
            for e in self.rng.choice(
                self.possible_swept_obj_born_pos, size=2, replace=False
            )
        ]

        sampled_distractor_obj_texture = self.rng.choice(
            [
                e
                for e in self.possible_dragged_obj_texture
                if e.value != sampled_swept_obj_texture
            ]
        ).value
        # add 3-sided square
        base_obj_sampled_size = self.rng.uniform(
            low=sampled_base_obj.size_range.low,
            high=sampled_base_obj.size_range.high,
        )
        base_obj_sampled_size *= 1.35

        # fixed base pose
        base_pos_x = self.bounds[0, 0] + 0.16
        base_pos_y = 0
        base_rot = eulerXYZ_to_quatXYZW((0, 0, -np.pi / 2))
        base_pos = (base_pos_x, base_pos_y, base_obj_sampled_size[2] / 2)
        base_pose = (base_pos, base_rot)

        obj_id, base_urdf_full_path = add_any_object(
            env=env,
            obj_entry=sampled_base_obj,
            pose=base_pose,
            size=base_obj_sampled_size,
            category="fixed",
            retain_temp=True,
        )

        p_change_texture(obj_id, sampled_base_obj_texture, self.client_id)
        add_object_id_reverse_mapping_info(
            mapping_dict=env.obj_id_reverse_mapping,
            obj_id=obj_id,
            object_entry=sampled_base_obj,
            texture_entry=sampled_base_obj_texture,
        )
        self.placeholders["bounds"] = PlaceholderObj(
            name=sampled_base_obj.name,
            obj_id=obj_id,
            urdf=base_urdf_full_path,
            novel_name=sampled_base_obj.novel_name,
            alias=sampled_base_obj.alias,
            color=sampled_base_obj_texture,
            image_size=self._placeholder_img_size,
            seed=self.seed,
        )
        # add constraint object
        constraint_obj_sampled_size = self.rng.uniform(
            low=sampled_constraint_obj.size_range.low,
            high=sampled_constraint_obj.size_range.high,
        )
        constraint_obj_sampled_size[0] *= 1.35
        # random constraint pose in the x direction
        constraint_pos_x = self.rng.uniform(
            low=base_pos_x
            + (self.task_meta["constraint_range"][0] - 0.5) * base_obj_sampled_size[0],
            high=base_pos_x
            + (self.task_meta["constraint_range"][1] - 0.5) * base_obj_sampled_size[0],
        )
        constraint_pos_y = base_pos_y
        constraint_rot = eulerXYZ_to_quatXYZW((0, 0, -np.pi / 2))

        constraint_pose = (
            (constraint_pos_x, constraint_pos_y, constraint_obj_sampled_size[2] / 2),
            constraint_rot,
        )
        obj_id, constraint_urdf_full_path = add_any_object(
            env=env,
            obj_entry=sampled_constraint_obj,
            pose=constraint_pose,
            size=constraint_obj_sampled_size,
            category="fixed",
            retain_temp=True,
        )
        p_change_texture(obj_id, sampled_constraint_obj_texture, self.client_id)
        # add item into the object id mapping dict
        add_object_id_reverse_mapping_info(
            mapping_dict=env.obj_id_reverse_mapping,
            obj_id=obj_id,
            object_entry=sampled_constraint_obj,
            texture_entry=sampled_constraint_obj_texture,
        )

        self.placeholders["constraint"] = PlaceholderObj(
            name=sampled_constraint_obj.name,
            obj_id=obj_id,
            urdf=constraint_urdf_full_path,
            novel_name=sampled_constraint_obj.novel_name,
            alias=sampled_constraint_obj.alias,
            color=sampled_constraint_obj_texture,
            image_size=self._placeholder_img_size,
            seed=self.seed,
        )

        # Add pile of small blocks.
        obj_pts = {}
        obj_ids = []
        swept_obj_size = self.rng.uniform(
            low=sampled_swept_obj.size_range.low,
            high=sampled_swept_obj.size_range.high,
        )

        # some samplings may be out of valid workspace
        not_reach_max_times = False
        for i in range(self.REJECT_SAMPLING_MAX_TIMES):
            poses = []
            for _ in range(sampled_num_swept_objs):
                rx = sampled_swept_obj_born_pos[0] + self.rng.random() * 0.18
                ry = sampled_swept_obj_born_pos[1] + self.rng.random() * 0.18
                xyz = (rx, ry, swept_obj_size[2] / 2)
                theta = self.rng.random() * 2 * np.pi
                xyzw = eulerXYZ_to_quatXYZW((0, 0, theta))
                pose = (xyz, xyzw)
                if not self._valid_workspace.contains(
                    np.array(pose[0][:2], dtype=self._valid_workspace.dtype)
                ):
                    print(
                        f"Warning: {i + 1} repeated sampling when try to spawn swept objects"
                    )
                    break
                poses.append(pose)
            if len(poses) != sampled_num_swept_objs:
                continue
            for pose in poses:
                obj_id, urdf_full_path = add_any_object(
                    env=env,
                    obj_entry=sampled_swept_obj,
                    pose=pose,
                    size=swept_obj_size,
                    retain_temp=True,
                )
                p_change_texture(obj_id, sampled_swept_obj_texture, self.client_id)
                # add item into the object id mapping dict
                add_object_id_reverse_mapping_info(
                    mapping_dict=env.obj_id_reverse_mapping,
                    obj_id=obj_id,
                    object_entry=sampled_swept_obj,
                    texture_entry=sampled_swept_obj_texture,
                )
                obj_pts[obj_id] = self.get_box_object_points(obj_id)
                obj_ids.append((obj_id, (0, None)))
            if len(obj_ids) == sampled_num_swept_objs:
                not_reach_max_times = True
                break
        if not not_reach_max_times:
            raise ValueError("Error in sampling swept objects")

        self.placeholders["swept_obj"] = PlaceholderObj(
            name=sampled_swept_obj.name,
            obj_id=obj_id,
            urdf=urdf_full_path,
            novel_name=sampled_swept_obj.novel_name,
            alias=sampled_swept_obj.alias,
            color=sampled_swept_obj_texture,
            image_size=self._placeholder_img_size,
            seed=self.seed,
            global_scaling=7,
        )

        # set goals
        for obj_id, (obj_pts_k, obj_pts_v) in zip(obj_ids, obj_pts.items()):
            obj_ids_add_to_goal = [obj_id]
            obj_pts_add_to_goal = {obj_pts_k: obj_pts_v}

            # Get zone dimension
            top_x = constraint_pos_x + constraint_obj_sampled_size[1] / 2
            bottom_x = base_pos_x + base_obj_sampled_size[0] / 2
            left_y = base_pos_y - base_obj_sampled_size[1] / 2
            right_y = base_pos_y + base_obj_sampled_size[1] / 2

            matches = np.ones((len(obj_ids_add_to_goal), 1))
            pos_x = (bottom_x + top_x) / 2.0 + 0.015
            target = [((pos_x, base_pos[1], base_pos[2]), base_rot)]

            zone_pose = ((pos_x, base_pos[1], base_pos[2]), base_rot)
            # !!! the size has to be (y, x, z)
            zone_size = (right_y - left_y, bottom_x - top_x, base_obj_sampled_size[2])

            # Goal: all small blocks must be in zone.
            # (objs, matches, targs, replace, rotations, metric, params, max_reward)
            self.goals.append(
                (
                    obj_ids_add_to_goal,
                    matches,
                    target,
                    True,
                    False,
                    "zone",
                    (obj_pts_add_to_goal, [(zone_pose, zone_size)]),
                    1,
                )
            )
        self._all_goals = self.goals.copy()

        # add distractors at sampled_distractor_obj_born_pos
        self.distractors_pts = {}
        not_reach_max_times = False
        for i in range(self.REJECT_SAMPLING_MAX_TIMES):
            poses = []
            for _ in range(sampled_num_swept_objs):
                rx = sampled_distractor_obj_born_pos[0] + self.rng.random() * 0.09
                ry = sampled_distractor_obj_born_pos[1] + self.rng.random() * 0.09
                xyz = (rx, ry, swept_obj_size[2] / 2)
                theta = self.rng.random() * 2 * np.pi
                xyzw = eulerXYZ_to_quatXYZW((0, 0, theta))
                pose = (xyz, xyzw)
                if not self._valid_workspace.contains(
                    np.array(pose[0][:2], dtype=self._valid_workspace.dtype)
                ):
                    print(
                        f"Warning: {i + 1} repeated sampling when try to spawn distractors"
                    )
                    break
                poses.append(pose)
            if len(poses) != sampled_num_swept_objs:
                continue
            for pose in poses:
                obj_id, urdf_full_path = add_any_object(
                    env=env,
                    obj_entry=sampled_swept_obj,
                    pose=pose,
                    size=swept_obj_size,
                    retain_temp=True,
                )
                p_change_texture(obj_id, sampled_distractor_obj_texture, self.client_id)
                # add item into the object id mapping dict
                add_object_id_reverse_mapping_info(
                    mapping_dict=env.obj_id_reverse_mapping,
                    obj_id=obj_id,
                    object_entry=sampled_swept_obj,
                    texture_entry=sampled_distractor_obj_texture,
                )
                self.distractors_pts[obj_id] = self.get_box_object_points(obj_id)
            if len(self.distractors_pts.keys()) == sampled_num_swept_objs:
                not_reach_max_times = True
                break
        if not not_reach_max_times:
            raise ValueError("Error in sampling distractor objects")

        # store constraint
        margin = swept_obj_size[0]
        constraint_top_x = constraint_pos_x - constraint_obj_sampled_size[1] / 2
        constraint_bottom_x = top_x
        constraint_left_y = constraint_pos_y - constraint_obj_sampled_size[0] / 2
        constraint_right_y = constraint_pos_y + constraint_obj_sampled_size[0] / 2
        self.constraint_checking["constraint_zone_with_margin"] = [
            constraint_top_x - margin,
            constraint_bottom_x + margin,
            constraint_left_y - margin,
            constraint_right_y + margin,
        ]
        self.constraint_checking["obj_pts"] = obj_pts
        self.constraint_checking["constraint_zone"] = (
            constraint_pose,
            constraint_obj_sampled_size,
        )
        self.constraint_checking["satisfied"] = True

        # sample determiner
        dets = [k for k, v in det_to_integer.items() if v <= sampled_num_swept_objs] + [
            "all"
        ]
        sampled_det = self.rng.choice(dets)
        self.placeholders["det"] = PlaceholderText(sampled_det)
        # determine target number
        if sampled_det in det_to_integer:
            self._target_num = det_to_integer[sampled_det]
        elif sampled_det == "all":
            self._target_num = sampled_num_swept_objs
        else:
            raise ValueError

        self._n_achieved = 0

    def oracle(self, env):
        def act(obs):
            """Calculate action."""

            # Oracle uses perfect RGB-D orthographic images and segmentation masks.
            _, hmap, obj_mask = self.get_true_image(env)

            # Unpack next goal step.
            objs, matches, targs, replace, rotations, _, params, _ = self.goals[0]
            obj_pts, zones = params

            # Match objects to targets without replacement.
            if not replace:

                # Modify a copy of the match matrix.
                matches = matches.copy()

                # Ignore already matched objects.
                for i in range(len(objs)):
                    object_id, (symmetry, _) = objs[i]
                    targets_i = np.argwhere(matches[i, :]).reshape(-1)
                    for j in targets_i:
                        if self.is_all_in_zone(object_id, obj_pts[object_id], zones[0]):
                            matches[i, :] = 0
                            matches[:, j] = 0

            # Get objects to be picked (prioritize farthest from nearest neighbor).
            nn_dists = []
            nn_targets = []
            obj_xyz = []
            for i in range(len(objs)):
                object_id, (symmetry, _) = objs[i]
                xyz, _ = getBasePositionAndOrientation(
                    object_id, physicsClientId=self.client_id
                )
                obj_xyz.append(xyz)
                targets_i = np.argwhere(matches[i, :]).reshape(-1)

                if len(targets_i) > 0:  # pylint: disable=g-explicit-length-test
                    targets_xyz = np.float32([targs[j][0] for j in targets_i])
                    dists = np.linalg.norm(
                        targets_xyz - np.float32(xyz).reshape(1, 3), axis=1
                    )
                    nn = np.argmin(dists)
                    nn_dists.append(dists[nn])
                    nn_targets.append(targets_i[nn])

                # Handle ignored objects.
                else:
                    nn_dists.append(0)
                    nn_targets.append(-1)
            order = np.argsort(nn_dists)[::-1]

            # Filter out matched objects.
            order = [
                i
                for i in order
                if not self.is_all_in_zone(objs[i][0], obj_pts[objs[i][0]], zones[0])
            ]
            pick_mask = None
            for pick_i in order:
                pick_mask = np.uint8(obj_mask == objs[pick_i][0])
                if np.sum(pick_mask) > 0:
                    break

            # Trigger task reset if no object is visible.
            if pick_mask is None or np.sum(pick_mask) == 0:
                self.goals = []
                self.lang_goals = []
                print("Object for pick is not visible. Skipping demonstration.")
                return

            # Get pushing start pose.
            # noinspection PyUnboundLocalVariable
            push_pos = obj_xyz[pick_i]
            push_start_pose = (np.asarray(push_pos), np.asarray((0, 0, 0, 1)))

            # Get pushing end pose.
            targ_pose = targs[
                nn_targets[pick_i]
            ]  # pylint: disable=undefined-loop-variable
            obj_pose = getBasePositionAndOrientation(
                objs[pick_i][0], physicsClientId=self.client_id
            )  # pylint: disable=undefined-loop-variable
            if not self.sixdof:
                obj_euler = quatXYZW_to_eulerXYZ(obj_pose[1])
                obj_quat = eulerXYZ_to_quatXYZW((0, 0, obj_euler[2]))
                obj_pose = (obj_pose[0], obj_quat)
            world_to_pick = invert(push_start_pose)
            obj_to_pick = multiply(world_to_pick, obj_pose)
            pick_to_obj = invert(obj_to_pick)
            push_end_pose = multiply(targ_pose, pick_to_obj)

            # adjust push start and end positions
            pos0 = np.float32((push_start_pose[0][0], push_start_pose[0][1], 0))
            pos1 = np.float32((push_end_pose[0][0], push_end_pose[0][1], 0))
            push_start_pose = (pos0, np.asarray((0, 0, 0, 1)))
            push_end_pose = (pos1, np.asarray((0, 0, 0, 1)))

            return {
                "pose0_position": push_start_pose[0].astype(np.float32)[:-1],
                "pose0_rotation": push_start_pose[1].astype(np.float32),
                "pose1_position": push_end_pose[0].astype(np.float32)[:-1],
                "pose1_rotation": push_end_pose[1].astype(np.float32),
            }

        return OracleAgent(act)
