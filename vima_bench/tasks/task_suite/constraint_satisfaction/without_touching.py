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


class WithoutTouching(SweepObjectsToZoneBase):

    task_name = "sweep_without_touching"

    def __init__(
        self,
        # ====== task specific ======
        max_swept_obj: int = 3,
        constraint_length: int | float = 1,
        swept_obj_express_types: Literal["image", "name"] = "image",
        bounds_express_types: Literal["image", "name"] = "image",
        constraint_express_types: Literal["image", "name"] = "image",
        prepend_color_to_name: bool = True,
        oracle_max_steps: int = 14,
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
        task_meta = {
            "max_swept_obj": max_swept_obj,
            "constraint_length": constraint_length,
        }

        assert oracle_max_steps <= 20

        super().__init__(
            prompt_template="Sweep {det} {swept_obj} into {bounds} without touching {constraint}.",
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
        # define phases used for oracle
        self.phases = {}

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
        base_obj_sampled_size[0] *= 0.86
        base_obj_sampled_size[1] *= 1.5

        # fixed base pose
        base_pos_x = self.bounds[0, 0] + 0.14
        base_pos_y = 0
        base_rot = eulerXYZ_to_quatXYZW((0, 0, -np.pi / 2))
        base_pos = (base_pos_x, base_pos_y, base_obj_sampled_size[2] / 2)
        base_pose = (base_pos, base_rot)

        base_obj_id, base_urdf_full_path = add_any_object(
            env=env,
            obj_entry=sampled_base_obj,
            pose=base_pose,
            size=base_obj_sampled_size,
            category="fixed",
            retain_temp=True,
        )
        p_change_texture(base_obj_id, sampled_base_obj_texture, self.client_id)
        add_object_id_reverse_mapping_info(
            mapping_dict=env.obj_id_reverse_mapping,
            obj_id=base_obj_id,
            object_entry=sampled_base_obj,
            texture_entry=sampled_base_obj_texture,
        )
        self.placeholders["bounds"] = PlaceholderObj(
            name=sampled_base_obj.name,
            obj_id=base_obj_id,
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
        constraint_obj_sampled_size[0] *= self.task_meta["constraint_length"]
        # fixed constraint pose
        constraint_pos_x = (
            base_pos_x
            + 0.5 * base_obj_sampled_size[0]
            + 0.5 * constraint_obj_sampled_size[1]
        )
        constraint_pos_y = base_pos_y
        constraint_rot = eulerXYZ_to_quatXYZW((0, 0, -np.pi / 2))

        constraint_pose = (
            (constraint_pos_x, constraint_pos_y, constraint_obj_sampled_size[2] / 2),
            constraint_rot,
        )
        constraint_obj_id, constraint_urdf_full_path = add_any_object(
            env=env,
            obj_entry=sampled_constraint_obj,
            pose=constraint_pose,
            size=constraint_obj_sampled_size,
            category="fixed",
            retain_temp=True,
        )
        p_change_texture(
            constraint_obj_id, sampled_constraint_obj_texture, self.client_id
        )
        add_object_id_reverse_mapping_info(
            mapping_dict=env.obj_id_reverse_mapping,
            obj_id=constraint_obj_id,
            object_entry=sampled_constraint_obj,
            texture_entry=sampled_constraint_obj_texture,
        )
        self.placeholders["constraint"] = PlaceholderObj(
            name=sampled_constraint_obj.name,
            obj_id=constraint_obj_id,
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
                obj_pts[obj_id] = self.get_box_object_points(obj_id)
                obj_ids.append((obj_id, (0, None)))
                add_object_id_reverse_mapping_info(
                    mapping_dict=env.obj_id_reverse_mapping,
                    obj_id=obj_id,
                    object_entry=sampled_swept_obj,
                    texture_entry=sampled_swept_obj_texture,
                )
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

        # Goal: all blocks must be in zone.
        zone_size = (
            base_obj_sampled_size[1],
            base_obj_sampled_size[0],
            base_obj_sampled_size[2],
        )
        for obj_id, (obj_pts_k, obj_pts_v) in zip(obj_ids, obj_pts.items()):
            obj_ids_add_to_goal = [obj_id]
            obj_pts_add_to_goal = {obj_pts_k: obj_pts_v}

            # (objs, matches, targs, replace, rotations, metric, params, max_reward)
            self.goals.append(
                (
                    obj_ids_add_to_goal,
                    np.ones((len(obj_ids), 1)),
                    [base_pose],
                    True,
                    False,
                    "zone",
                    (obj_pts_add_to_goal, [(base_pose, zone_size)]),
                    1,
                )
            )
        self._all_goals = self.goals.copy()

        # add distractors at sampled_distractor_obj_born_pos
        self.distractors_pts = {}
        not_reach_max_times = False
        for i in range(self.REJECT_SAMPLING_MAX_TIMES):
            poses = []
            # add distractors at sampled_distractor_obj_born_pos
            for _ in range(sampled_num_swept_objs):
                rx = sampled_distractor_obj_born_pos[0] + self.rng.random() * 0.18
                ry = sampled_distractor_obj_born_pos[1] + self.rng.random() * 0.18
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
        constraint_bottom_x = constraint_pos_x + constraint_obj_sampled_size[1] / 2
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

        # define phases used for oracle
        self.phases = {
            "constraint_pos": (constraint_pos_x, constraint_pos_y),
            "left_divider_y": constraint_left_y - 0.05,
            "right_divider_y": constraint_right_y + 0.05,
            "phase_14_divider_x": constraint_pos_x,
            "phase_23_divider_x": constraint_top_x - 0.02,
        }

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
            objs, matches, targs, _, rotations, _, params, _ = self.goals[0]
            obj_pts, zones = params

            # Find current phases of objects to determine order and their end positions
            phases = {1: [], 2: [], 3: []}
            phase12_left_y = self.phases["left_divider_y"] - 0.05
            phase12_right_y = self.phases["right_divider_y"] + 0.05
            phase1_x = (self.phases["phase_14_divider_x"] + self.bounds[0][1]) / 2
            phase2_x = targs[0][0][0] - 0.02

            obj_xyz = []
            obj_isleft = []
            obj_target = []
            obj_tgt_dist = []
            for i in range(len(objs)):
                object_id, (symmetry, _) = objs[i]
                xyz, _ = getBasePositionAndOrientation(
                    object_id, physicsClientId=self.client_id
                )
                obj_xyz.append(xyz)

                # Figure out if object is going left or right
                isleft = True if xyz[1] < self.phases["constraint_pos"][1] else False
                obj_isleft.append(isleft)
                # Obtain the phase that the object is in
                if (
                    self.phases["left_divider_y"]
                    < xyz[1]
                    < self.phases["right_divider_y"]
                ):
                    if xyz[0] > self.phases["phase_14_divider_x"]:  # phase 1
                        phase = 1
                    else:  # phase 3
                        phase = 3
                else:
                    if xyz[0] > self.phases["phase_23_divider_x"]:  # phase 2
                        phase = 2
                    else:  # phase 3
                        phase = 3
                phases[phase].append(i)

                if phase == 1:
                    if isleft:
                        dist = xyz[1] - self.phases["left_divider_y"]
                        target_xyz = (phase1_x, phase12_left_y, 0)
                    else:
                        dist = self.phases["right_divider_y"] - xyz[1]
                        target_xyz = (phase1_x, phase12_right_y, 0)
                elif phase == 2:
                    dist = xyz[0] - self.phases["phase_23_divider_x"]
                    if isleft:
                        target_xyz = (phase2_x, phase12_left_y, 0)
                    else:
                        target_xyz = (phase2_x, phase12_right_y, 0)
                elif phase == 3:
                    target_xyz = np.float32(targs[0][0])
                    dist = np.linalg.norm(target_xyz - np.float32(xyz))
                else:
                    raise ValueError("Invalid phase type")
                obj_target.append(target_xyz)
                obj_tgt_dist.append(dist)

            # Compute order for each of the phases (prioritize farthest from nearest)
            order = []
            for ph in [3, 2, 1]:
                phase_obj = phases[ph]
                phase_dists = list(map(lambda x: obj_tgt_dist[x], phase_obj))
                # sort objects in the same phase based on their distances to the next goal. The farther is prioritized
                sorted_phase_obj = [
                    x
                    for _, x in sorted(
                        zip(phase_dists, phase_obj), key=lambda pair: -pair[0]
                    )
                ]
                order.extend(sorted_phase_obj)

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
            targ_pose = (obj_target[pick_i], np.asarray((0, 0, 0, 1)))
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
