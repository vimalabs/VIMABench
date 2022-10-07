"""Motion primitives."""

import numpy as np

from ..utils import misc_utils as utils

__all__ = ["PickPlace", "Push"]


class MoveEndEffector(object):
    """
    Move end effector primitive.
    """

    def __init__(self, height=0.32, speed=0.01, debug: bool = False):
        self._height = height
        self._speed = speed
        # delta z when adjusting the height
        self._delta_z = 0.001
        # max number of height adjustments is determined by the max height to avoid infinite loop
        self._max_adjust = int(self._height * 2 / self._delta_z)

    @property
    def height(self):
        return self._height

    def __call__(self, movej, movep, ee, target_pose, release: bool):
        """
        Execute move end effector and (may) release primitive.
        This primitive includes several steps.
        1. Hover the robot arm to target 2D position (x, y).
        2. Adjust the height of the arm to the desired height.
           If contact is detected and the end effector doesn't hold anything,
           automatically grip that object and stop adjustment.
           If contact is detected and the end effector holds something, stop the adjustment.
        3. After height adjustment, release the object if `release == True`.

        :param movej: function to move robot joints.
        :param movep: function to move robot end effector pose.
        :param ee: robot end effector.
        :param target_pose: SE(3) target pose.
        :param release: Boolean value to indicate release or not.
                        If the end effector is not activated, this command will be ignored.
        :return timeout: robot movement timed out if True.
        """
        # the release command may only be executed if the end effector has gripped something
        has_gripped_something = ee.check_grasp()

        # 1. hover the robot arm to target 2D position (x, y)
        tar_2d_pose = ((0, 0, ee.cur_pose[0][2]), (0, 0, 0, 1))
        tar_2d_pose = utils.multiply(target_pose, tar_2d_pose)
        timeout = movep(tar_2d_pose, self._speed)

        # 2. adjust the height of the arm to the desired height.
        achieved_tar_pose = tar_2d_pose
        n_adjustments = 0
        while abs(ee.cur_pose[0][2] - target_pose[0][2] > self._delta_z):
            delta = (
                np.float32(
                    [
                        0,
                        0,
                        -self._delta_z
                        if ee.cur_pose[0][2] > target_pose[0][2]
                        else self._delta_z,
                    ]
                ),
                utils.eulerXYZ_to_quatXYZW((0, 0, 0)),
            )
            achieved_tar_pose = utils.multiply(achieved_tar_pose, delta)
            timeout |= movep(achieved_tar_pose, self._speed)
            if timeout:
                return True
            # detect contact
            if ee.detect_contact():
                if not has_gripped_something:
                    ee.activate()
                break
            # break if exceeds max number of adjustments to avoid infinite loop
            if n_adjustments > self._max_adjust:
                print(
                    f"Warning: height adjustments exceeded {self._max_adjust} times. Skipping."
                )
                break
            n_adjustments += 1

        # 3. after height adjustment, release the object if 'has_gripped_something == True and release == True'
        if has_gripped_something and release:
            ee.release()
        return timeout


class MoveEndEffectorForPush(object):
    """
    Move end effector primitive.
    """

    def __init__(
        self,
        rest_height=0.31,
        operation_height=0.005,
        speed=0.002,
        advance_distance=0.08,
        step_size=0.001,
        free_move_steps_threshold=100,
    ):
        self._rest_height = rest_height
        self._operation_height = operation_height
        self._speed = speed
        self._advance_distance = advance_distance
        self._step_size = step_size
        self._free_move_steps_threshold = free_move_steps_threshold

    @property
    def height(self):
        return self._rest_height

    def __call__(self, movej, movep, ee, target_pose, push: bool):
        """
        Execute Move End Effector for pushing
        This primitive involves multiple steps:
        If push is False, execute a simple action:
            1. move arm to target pos with a safe height
        If push is True, execute the following 3 sub actions:
            1. rotate end effector such that it's perpendicular to the pushing direction
            2. descend arm to the ground
            3. push to the target pos
        """
        cur_pos = ee.cur_pose[0]
        if not push:
            # Move ee to the position with a good height, don't change orientation
            pos1 = np.float32((target_pose[0][0], target_pose[0][1], self._rest_height))
            timeout = movep((pos1, np.asarray((0, 0, 0, 1))))
        else:
            # Push from current position to the target position
            over0 = np.float32((cur_pos[0], cur_pos[1], self._rest_height))
            pos0 = np.float32((cur_pos[0], cur_pos[1], self._operation_height))
            pos1 = np.float32(
                (target_pose[0][0], target_pose[0][1], self._operation_height)
            )
            vec = np.float32(pos1) - np.float32(pos0)
            actual_dist_to_push = np.linalg.norm(vec)
            vec = vec / actual_dist_to_push

            over0[:2] -= vec[:2] * self._advance_distance
            pos0[:2] -= vec[:2] * self._advance_distance

            theta = np.arctan2(vec[1], vec[0])
            rot = utils.eulerXYZ_to_quatXYZW((0, 0, theta))

            # step 1: rotate ee at a good height and leave an advance distance
            timeout = movep((over0, rot))
            # step 2: descend (and leave an advance distance to push)
            timeout |= movep((pos0, rot), speed=self._speed)
            # step 3: moving forward until touches the object (push forward by "advance_distance")
            n_step_taken_freely = 0
            while (
                (not ee.detect_contact())
                and (not timeout)
                and (n_step_taken_freely < self._free_move_steps_threshold)
            ):
                n_step_taken_freely += 1
                target = pos0 + vec * n_step_taken_freely * self._step_size
                timeout |= movep((target, rot), speed=self._speed)
            # step 4: push forward by "actual_dist_to_push"
            n_step_needed = np.int32(np.floor(actual_dist_to_push / self._step_size))
            for step_i in range(n_step_needed):
                target = (
                    pos0
                    + vec * n_step_taken_freely * self._step_size
                    + vec * step_i * self._step_size
                )
                timeout |= movep((target, rot), speed=self._speed)
        return timeout


class PickPlace:
    """Pick and place primitive."""

    def __init__(self, height=0.32, speed=0.01):
        self.height, self.speed = height, speed

    def __call__(self, movej, movep, ee, pose0, pose1):
        """Execute pick and place primitive.

        Args:
          movej: function to move robot joints.
          movep: function to move robot end effector pose.
          ee: robot end effector.
          pose0: SE(3) picking pose.
          pose1: SE(3) placing pose.

        Returns:
          timeout: robot movement timed out if True.
        """

        pick_pose = (np.array([pose0[0][0], pose0[0][1], 0]), pose0[1])
        place_pose = (np.array([pose1[0][0], pose1[0][1], 0]), pose1[1])

        # Execute picking primitive.
        prepick_to_pick = ((0, 0, 0.32), (0, 0, 0, 1))
        postpick_to_pick = ((0, 0, self.height), (0, 0, 0, 1))
        prepick_pose = utils.multiply(pick_pose, prepick_to_pick)
        postpick_pose = utils.multiply(pick_pose, postpick_to_pick)
        timeout = movep(prepick_pose)

        # Move towards pick pose until contact is detected.
        delta = (np.float32([0, 0, -0.001]), utils.eulerXYZ_to_quatXYZW((0, 0, 0)))
        targ_pose = prepick_pose
        while not ee.detect_contact():  # and target_pose[2] > 0:
            targ_pose = utils.multiply(targ_pose, delta)
            timeout |= movep(targ_pose)
            if timeout:
                return True, False

        # Activate end effector, move up, and check picking success.
        ee.activate()
        timeout |= movep(postpick_pose, self.speed)
        pick_success = ee.check_grasp()

        # Execute placing primitive if pick is successful.
        if pick_success:
            preplace_to_place = ((0, 0, self.height), (0, 0, 0, 1))
            postplace_to_place = ((0, 0, 0.32), (0, 0, 0, 1))
            preplace_pose = utils.multiply(place_pose, preplace_to_place)
            postplace_pose = utils.multiply(place_pose, postplace_to_place)
            targ_pose = preplace_pose
            while not ee.detect_contact():
                targ_pose = utils.multiply(targ_pose, delta)
                timeout |= movep(targ_pose, self.speed)
                if timeout:
                    return True, False
            ee.release()
            timeout |= movep(postplace_pose)

        # Move to prepick pose if pick is not successful.
        else:
            ee.release()
            timeout |= movep(prepick_pose)

        return timeout, pick_success


class Push:
    def __init__(
        self, rest_height=0.31, operation_height=0.005, speed=0.002, step_size=0.001
    ):
        self._rest_height = rest_height
        self._operation_height = operation_height
        self._speed = speed
        self._step_size = step_size

    @property
    def height(self):
        return self._rest_height

    def __call__(self, movej, movep, ee, pose0, pose1):
        """Execute pushing primitive.

        Args:
          movej: function to move robot joints.
          movep: function to move robot end effector pose.
          ee: robot end effector.
          pose0: SE(3) starting pose.
          pose1: SE(3) ending pose.

        Returns:
          timeout: robot movement timed out if True.
        """

        # Adjust push start and end positions.
        pos0 = np.float32((pose0[0][0], pose0[0][1], self._operation_height))
        pos1 = np.float32((pose1[0][0], pose1[0][1], self._operation_height))
        vec = np.float32(pos1) - np.float32(pos0)
        length = np.linalg.norm(vec)
        vec = vec / length
        pos0 -= vec * 0.02
        pos1 -= vec * 0.05

        # Align spatula against push direction.
        theta = np.arctan2(vec[1], vec[0])
        rot = utils.eulerXYZ_to_quatXYZW((0, 0, theta))

        over0 = (pos0[0], pos0[1], self._rest_height)
        over1 = (pos1[0], pos1[1], self._rest_height)

        # Execute push.
        timeout = movep((over0, rot))
        timeout |= movep((pos0, rot))
        n_push = np.int32(np.floor(np.linalg.norm(pos1 - pos0) / self._step_size))
        for _ in range(n_push):
            target = pos0 + vec * n_push * self._step_size
            timeout |= movep((target, rot), speed=self._speed)
        timeout |= movep((pos1, rot), speed=self._speed)
        timeout |= movep((over1, rot))
        return timeout
