from __future__ import annotations

import copy
import queue
from functools import partial
from typing import NamedTuple, Literal

import numpy as np
import pybullet as p

from ..base import BaseTask
from ...components.encyclopedia import ObjPedia, TexturePedia
from ...components.encyclopedia.definitions import ObjEntry, TextureEntry
from ...components.placeholders import PlaceholderScene
from ...components.placeholders.placeholder_scene import SceneRenderEnv
from ...utils.misc_utils import eulerXYZ_to_quatXYZW
from ...utils.pybullet_utils import (
    add_any_object,
    add_object_id_reverse_mapping_info,
    p_change_texture,
)


class ResultTuple(NamedTuple):
    success: bool
    failure: bool
    distance: float


class FollowOrder(BaseTask):
    """One shot imitation task: Stack objects in this order {video frames}
    In this task, the agent is required to imitate the moves in the given one shot video trajectory.
    Here we define video trajectory by variable length of frames of image.
    """

    task_name = "follow_order"

    def __init__(
        self,
        # ====== task specific ======
        num_dragged_obj: int = 3,
        num_distractor_in_workspace: int = 1,
        num_frames: int = 3,
        scene_express_types: Literal["image", "name"] = "image",
        placeholder_scene_views: str | list[str] | None = None,
        oracle_max_steps: int = 4,
        oracle_step_to_env_step_ratio: int = 4,
        possible_dragged_obj: str | list[str] | ObjEntry | list[ObjEntry] | None = None,
        possible_dragged_obj_texture: str
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
            "num_dragged_obj": num_dragged_obj,
            "num_distractor_in_workspace": num_distractor_in_workspace,
            "num_frames": num_frames,
        }
        placeholder_expression = {
            f"frame_{i}": {
                "types": [scene_express_types],
            }
            for i in range(num_frames)
        }
        self._scene_express_types = scene_express_types

        assert (
            placeholder_scene_views is None
            or str
            or (
                isinstance(placeholder_scene_views, list)
                and all(isinstance(views, str) for views in placeholder_scene_views)
            )
        ), (
            f"`placeholder_scene_views` must be a str, list of str, or None, "
            f"but get {type(placeholder_scene_views)}"
        )
        self._placeholder_scene_views = placeholder_scene_views

        if possible_dragged_obj is None:
            self.possible_dragged_obj = [
                ObjPedia.BLOCK,
                ObjPedia.L_BLOCK,
                ObjPedia.CAPITAL_LETTER_A,
                ObjPedia.CAPITAL_LETTER_E,
                ObjPedia.CAPITAL_LETTER_G,
                ObjPedia.CAPITAL_LETTER_M,
                ObjPedia.CAPITAL_LETTER_R,
                ObjPedia.CAPITAL_LETTER_T,
                ObjPedia.CAPITAL_LETTER_V,
                ObjPedia.CROSS,
                ObjPedia.DIAMOND,
                ObjPedia.TRIANGLE,
                ObjPedia.FLOWER,
                ObjPedia.HEART,
                ObjPedia.HEXAGON,
                ObjPedia.PENTAGON,
                ObjPedia.RING,
                ObjPedia.ROUND,
                ObjPedia.STAR,
            ]
        elif isinstance(possible_dragged_obj, str):
            self.possible_dragged_obj = [
                ObjPedia.lookup_object_by_name(possible_dragged_obj)
            ]
        elif isinstance(possible_dragged_obj, ObjEntry):
            self.possible_dragged_obj = [possible_dragged_obj]
        elif isinstance(possible_dragged_obj, list):
            if isinstance(possible_dragged_obj[0], str):
                self.possible_dragged_obj = [
                    ObjPedia.lookup_object_by_name(obj) for obj in possible_dragged_obj
                ]
            elif isinstance(possible_dragged_obj[0], ObjEntry):
                self.possible_dragged_obj = possible_dragged_obj
            else:
                raise ValueError(
                    "possible_dragged_obj must be a list of str or ObjEntry"
                )
        else:
            raise ValueError(
                "possible_dragged_obj must be a str or list of str or ObjEntry"
            )

        if possible_dragged_obj_texture is None:
            self.possible_dragged_obj_texture = TexturePedia.all_entries()
        elif isinstance(possible_dragged_obj_texture, str):
            self.possible_dragged_obj_texture = [
                TexturePedia.lookup_color_by_name(possible_dragged_obj_texture)
            ]
        elif isinstance(possible_dragged_obj_texture, TextureEntry):
            self.possible_dragged_obj_texture = [possible_dragged_obj_texture]
        elif isinstance(possible_dragged_obj_texture, list):
            if isinstance(possible_dragged_obj_texture[0], str):
                self.possible_dragged_obj_texture = [
                    TexturePedia.lookup_color_by_name(obj)
                    for obj in possible_dragged_obj_texture
                ]
            elif isinstance(possible_dragged_obj_texture[0], TextureEntry):
                self.possible_dragged_obj_texture = possible_dragged_obj_texture
            else:
                raise ValueError(
                    "possible_dragged_obj_texture must be a list of str or TextureEntry"
                )
        else:
            raise ValueError(
                "possible_dragged_obj_texture must be a str or list of str or TextureEntry"
            )

        super().__init__(
            prompt_template=(
                "Stack objects in this order "
                + " ".join(["{{frame_{i}}}".format(i=i) for i in range(num_frames)])
                + "."
            ),
            task_meta=task_meta,
            placeholder_expression=placeholder_expression,
            oracle_max_steps=oracle_max_steps,
            oracle_step_to_env_step_ratio=oracle_step_to_env_step_ratio,
            obs_img_views=obs_img_views,
            obs_img_size=obs_img_size,
            placeholder_img_size=placeholder_img_size,
            seed=seed,
            debug=debug,
        )
        self.pos_eps = 0.1
        self.num_blocks_in_each_pose = None
        self.base_poses = None
        self._n_goals_before, self._n_goals_after = None, None

    def all_stacking_sequence(self) -> list:
        """
        Get all valid stacking sequence

        Returns: list of all possible moves of length ``num_op''

        """
        all_sequences = []
        # no op (i==j) is not allowed
        possible_moves = tuple(
            (i, j)
            for i in range(self.task_meta["num_dragged_obj"])
            for j in range(self.task_meta["num_dragged_obj"])
            if i != j
        )
        q = queue.Queue()
        initial_state = tuple(1 for _ in range(self.task_meta["num_dragged_obj"]))
        q.put(
            (initial_state, 0, [])
        )  # a queue stores (num_of_blocks_at_each_position_tuple, depth,op_sequence)
        # breadth first search
        while not q.empty():
            front_obj = q.get()
            objs_at_base_position, depth, op_sequence = front_obj
            if depth == self.num_operations:
                all_sequences.append(op_sequence)
                continue

            for move in possible_moves:
                if objs_at_base_position[move[0]] == 0:
                    continue
                else:
                    new_objs_at_base_position = list(objs_at_base_position)
                    new_objs_at_base_position[move[0]] -= 1
                    new_objs_at_base_position[move[1]] += 1
                    new_op_sequence = copy.deepcopy(op_sequence)
                    new_op_sequence.append(move)
                    q.put(
                        (tuple(new_objs_at_base_position), depth + 1, new_op_sequence)
                    )

        return all_sequences

    def _reset_prompt(self):
        self.num_operations = self.task_meta["num_frames"] - 1
        self.possible_moving_sequence = self.all_stacking_sequence()
        self.rng.shuffle(self.possible_moving_sequence)
        self.prompt_template = (
            "Stack objects in this order "
            + " ".join(
                [
                    "{{frame_{i}}}".format(i=i)
                    for i in range(self.task_meta["num_frames"])
                ]
            )
            + "."
        )
        self.placeholder_expression = {
            f"frame_{i}": {
                "types": [self._scene_express_types],
            }
            for i in range(self.task_meta["num_frames"])
        }

    def reset(self, env):
        super().reset(env)
        self._reset_prompt()

        num_dragged_obj = self.task_meta["num_dragged_obj"]
        self.num_blocks_in_each_pose = [0] * num_dragged_obj

        sampled_block_object = self.rng.choice(self.possible_dragged_obj).value
        sampled_block_texture = [
            e.value
            for e in self.rng.choice(
                self.possible_dragged_obj_texture,
                size=num_dragged_obj,
                replace=False,
            )
        ]
        sampled_move_seq = self.rng.choice(self.possible_moving_sequence)

        # calculate bounds
        x_bound, y_bound, _ = self.bounds
        pos_x_center = (x_bound[1] + x_bound[0]) / 2
        pos_y_interval = (y_bound[1] - y_bound[0]) / (1 + num_dragged_obj)
        pos_y_start = y_bound[0]
        # calculate rot of objects
        rot = eulerXYZ_to_quatXYZW((0, 0, 0))

        num_base_position = num_dragged_obj
        # we record a block and its pose based on the base position it is at.
        blocks = [list() for _ in range(num_base_position)]
        # block_poses for each block. the index will not be changed after creation
        block_poses = [list() for _ in range(num_base_position)]
        block_textures = [list() for _ in range(num_base_position)]

        # In this task, blocks are put in a line; no sampling for poses are needs
        # add objects to workspace
        block_sampled_size = self.rng.uniform(
            low=sampled_block_object.size_range.low,
            high=sampled_block_object.size_range.high,
        )
        # we put blocks in a line, hence it is necessary to check whether the workspace is large enough
        # to contain all the objects
        if pos_y_interval <= block_sampled_size[0]:
            raise ValueError(
                "ERROR: WORKSPACE CANNOT ACCOMEDATE ALL THE OBJECTS. Please decrease the number of objects."
            )
        # list of poses at base location; the objects could only be placed at base poses
        self.base_poses = []
        for i in range(num_dragged_obj):
            # add blocks into workspace,
            # each base position one block
            pos = (
                pos_x_center,
                pos_y_start + pos_y_interval * (i + 1),
                block_sampled_size[2] / 2,
            )
            pose = (pos, rot)
            self.base_poses.append(pose)

            obj_id, _, _ = self.add_object_to_env(
                env=env,
                color=sampled_block_texture[i],
                obj_entry=sampled_block_object,
                pose=pose,
                size=block_sampled_size,
            )
            self.num_blocks_in_each_pose[i] += 1
            # each base position one block
            blocks[i].append((obj_id, (0, None)))
            block_poses[i].append(pose)
            block_textures[i].append(sampled_block_texture[i])
        if len(self.base_poses) != num_base_position:
            raise ValueError("Failed to sample objects")

        # generate video prompt
        blocks_in_scene = copy.deepcopy(blocks)
        block_poses_in_scene = copy.deepcopy(block_poses)
        block_textures_in_scene = copy.deepcopy(block_textures)

        # generate frames and set goals
        # (objs, matches, targets, replace, rotations, metric, params, max_reward)
        frames_of_block_poses_in_scene = [
            (
                copy.deepcopy(block_poses_in_scene),
                copy.deepcopy(block_textures_in_scene),
            )
        ]
        for i, (from_idx, to_idx) in enumerate(sampled_move_seq):
            (
                blocks_in_scene,
                block_poses_in_scene,
                block_textures_in_scene,
            ) = self.move_blocks_from_to(
                blocks_in_scene,
                block_poses_in_scene,
                block_textures_in_scene,
                from_idx,
                to_idx,
            )
            dragged_obj = blocks_in_scene[to_idx][-1]
            target_pose = block_poses_in_scene[to_idx][-1]
            self.goals.append(
                (
                    [dragged_obj],
                    np.ones((1, 1)),
                    [target_pose],
                    False,
                    True,
                    "pose",
                    None,
                    1 / self.num_operations,
                )
            )
            frames_of_block_poses_in_scene.append(
                (block_poses_in_scene, block_textures_in_scene)
            )

        # add distractors
        num_distractor = self.task_meta["num_distractor_in_workspace"]
        if num_distractor > 0:
            not_reach_max_times = False
            num_added = 0
            possible_dragged_distractors = [
                obj
                for obj in self.possible_dragged_obj
                # different from dragged obj shape
                if obj.value != sampled_block_object
            ]
            for i in range(self.REJECT_SAMPLING_MAX_TIMES + num_distractor):
                entry = self.rng.choice(possible_dragged_distractors).value
                size = self.rng.uniform(
                    low=entry.size_range.low, high=entry.size_range.high
                )
                texture = self.rng.choice(
                    [t for t in self.possible_dragged_obj_texture]
                ).value
                obj_id, _, _ = self.add_object_to_env(
                    env=env, obj_entry=entry, color=texture, size=size
                )
                if obj_id is None:
                    print(
                        f"Warning: {i + 1} repeated sampling when try to spawn workspace distractor"
                    )
                else:
                    num_added += 1
                if num_added == num_distractor:
                    not_reach_max_times = True
                    break
            if not not_reach_max_times:
                raise ValueError("Error in sampling distractors")

        # create scene placeholder in the prompt
        def create_scene_fn(
            scene_render_env: SceneRenderEnv, sim_id: int, frame_idx: int
        ):
            (
                block_poses_frame_i,
                block_textures_frame_i,
            ) = frames_of_block_poses_in_scene[frame_idx]

            for block_poses_base_pos_i, block_textures_base_pos_i in zip(
                block_poses_frame_i, block_textures_frame_i
            ):
                for block_pose, block_texture in zip(
                    block_poses_base_pos_i, block_textures_base_pos_i
                ):
                    # add blocks into workspace
                    obj_id, _ = add_any_object(
                        env=scene_render_env,
                        obj_entry=sampled_block_object,
                        pose=block_pose,
                        size=block_sampled_size,
                    )
                    # change color according to dragged texture
                    p_change_texture(obj_id, block_texture, sim_id)
                    add_object_id_reverse_mapping_info(
                        scene_render_env.obj_id_reverse_mapping,
                        obj_id=obj_id,
                        object_entry=sampled_block_object,
                        texture_entry=block_texture,
                    )

        for i in range(self.task_meta["num_frames"]):
            self.placeholders[f"frame_{i}"] = PlaceholderScene(
                create_scene_fn=partial(create_scene_fn, frame_idx=i),
                assets_root=self.assets_root,
                views=self._placeholder_scene_views,
                image_size=self._placeholder_img_size,
                seed=self.seed,
            )

    def move_blocks_from_to(
        self,
        blocks_in_scene: list[list[tuple]],
        block_poses_in_scene: list[list[tuple]],
        block_textures_in_scene: list[list[TextureEntry]],
        from_idx: int,
        to_idx: int,
    ) -> tuple[list[list[tuple]], list[list[tuple]], list[list[TextureEntry]]]:
        """
        Calculate poses after a move operation
        block base idx starts from 0

        Args:
            block_textures_in_scene:
            block_poses_in_scene:
            blocks_in_scene:
            from_idx: index of object that move from
            to_idx: index of object that move to

        Returns: a tuple of (new_blocks_in_scene, new_block_poses_in_scene,new_block_textures_in_scene)

        """
        assert (
            len(blocks_in_scene[from_idx]) > 0
        ), "Error in moving block at position i to position j, position i is empty"

        new_blocks_in_scene = copy.deepcopy(blocks_in_scene)
        new_block_poses_in_scene = copy.deepcopy(block_poses_in_scene)
        new_block_textures_in_scene = copy.deepcopy(block_textures_in_scene)
        # correct the z in pose
        half_height = self.base_poses[to_idx][0][2]
        new_pose_height = (
            len(new_block_poses_in_scene[to_idx]) * 2 * half_height + half_height
        )
        pos, rot = self.base_poses[to_idx]
        new_pose = ((pos[0], pos[1], new_pose_height), rot)
        new_block_poses_in_scene[to_idx].append(new_pose)

        new_blocks_in_scene[to_idx].append(new_blocks_in_scene[from_idx].pop())
        new_block_textures_in_scene[to_idx].append(
            new_block_textures_in_scene[from_idx].pop()
        )

        self.num_blocks_in_each_pose[from_idx] -= 1
        self.num_blocks_in_each_pose[to_idx] += 1
        return (
            new_blocks_in_scene,
            new_block_poses_in_scene,
            new_block_textures_in_scene,
        )

    def check_success(self, *args, **kwargs) -> ResultTuple:
        # check success by checking the number of goals to be achieved
        all_achieved = len(self.goals) == 0
        if all_achieved:
            # when all goals are achieved, distance is simply the threshold
            return ResultTuple(success=True, failure=False, distance=self.pos_eps)
        else:
            # iterative over all objects to be manipulated to check failure
            failures = []
            distances = []
            # check not in_workspace failure
            for goal in self.goals:
                obj, _, bases, _, _, _, _, _ = goal
                obj_id, _ = obj[0]
                obj_pose = p.getBasePositionAndOrientation(
                    obj_id, physicsClientId=self.client_id
                )
                failures.append(
                    not self._valid_workspace.contains(
                        np.array(obj_pose[0][:2], dtype=self._valid_workspace.dtype)
                    )
                )
                distances.append(
                    np.linalg.norm(
                        np.float32(obj_pose[0][:2]) - np.float32(bases[0][0][:2])
                    )
                )
            is_failure = any(failures) or (
                self._n_goals_before - self._n_goals_after == 0
            )
            return ResultTuple(
                success=False,
                failure=is_failure,
                distance=sum(distances) / len(distances),
            )

    def set_difficulty(self, difficulty: str):
        super().set_difficulty(difficulty)
        if difficulty == "easy":
            # 3 obj, 3 frames, 1 distractor
            pass
        elif difficulty == "medium":
            self.task_meta["num_dragged_obj"] = 4
            self.task_meta["num_frames"] = 4
        else:
            self.task_meta["num_dragged_obj"] = 4
            self.task_meta["num_frames"] = 5
        self.oracle_max_steps = self.task_meta["num_frames"] + 1

    def update_goals(self, skip_oracle=True):
        self._n_goals_before = len(self.goals)
        rtn = super().update_goals(skip_oracle=skip_oracle)
        self._n_goals_after = len(self.goals)
        return rtn

    def is_match(self, pose0, pose1, symmetry):
        return super().is_match(pose0, pose1, symmetry, position_only=True)
