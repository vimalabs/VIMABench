"""Motion primitives."""
import time

import numpy as np
import pybullet as p

from ..utils import misc_utils as utils

__all__ = ["PickPlace", "MicroPickPlace", "Push"]


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

    def __call__(self, movej, movep, ee, pose0, pose1, episode, task_name):
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
        timeout = movep(prepick_pose, self.speed, [episode,0,0,"%s_move_to_pick" % (task_name)])

        # Move towards pick pose until contact is detected.
        delta = (np.float32([0, 0, -0.001]), utils.eulerXYZ_to_quatXYZW((0, 0, 0)))
        targ_pose = prepick_pose

        # 视角矩阵示例
        # viewMat = [-0.5120397806167603, 0.7171027660369873, -0.47284144163131714, 0.0, -0.8589617609977722,
        #            -0.42747554183006287, 0.28186774253845215, 0.0, 0.0, 0.5504802465438843, 0.8348482847213745, 0.0,
        #            0.1925382763147354, -0.24935829639434814, -0.4401884973049164, 1.0]
        # projMatrix = [0.75, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0000200271606445, -1.0, 0.0, 0.0,
        #               -0.02000020071864128, 0.0]

        # 保存频率
        freq_save = 00
        # 所要保存的视角List
        viewList = []

        cam_pos1 = np.array([0.0, 0.5, 0.7])
        cam_orientation1 = np.array([0.7, 0.7, 0, 0])

        cam_pos2 = np.array([0.2, 0.3, 0.8])
        cam_orientation2 = np.array([0.7, 0.4, 0.1, 0])

        cam_pos3 = np.array([0.0, 0.5, 0.7])
        cam_orientation3 = np.array([0.3, 0.3, 0.7, 0])

        for [pos, cam_orientation] in [[cam_pos1, cam_orientation1], [cam_pos2, cam_orientation2], [cam_pos3, cam_orientation3]]:
            viewMat, projMatrix = self.get_view(pos, cam_orientation)
            viewList.append([viewMat, projMatrix])

        while not ee.detect_contact():  # and target_pose[2] > 0:
            # time.sleep(0.02)
            # print("move to pick")

            # freq_save += 1
            # if freq_save % 50 == 1:
            #     self.view_image_save(episode, viewList, freq_save, is_act=1, is_end=1, stage="%s_move_to_pick"%(task_name))

            targ_pose = utils.multiply(targ_pose, delta)
            timeout |= movep(targ_pose, self.speed, [episode,0,0,"%s_move_to_pick" % (task_name)])

            if timeout:
                return True, False

        # Activate end effector, move up, and check picking success.
        ee.activate()
        # timeout |= movep(postpick_pose, self.speed, [episode, 1, 0, "%s_picking_success" % (task_name)])
        timeout |= movep(postpick_pose, self.speed)
        pick_success = ee.check_grasp()

        # Execute placing primitive if pick is successful.
        if pick_success:
            preplace_to_place = ((0, 0, self.height), (0, 0, 0, 1))
            postplace_to_place = ((0, 0, 0.32), (0, 0, 0, 1))
            preplace_pose = utils.multiply(place_pose, preplace_to_place)
            postplace_pose = utils.multiply(place_pose, postplace_to_place)
            targ_pose = preplace_pose
            # print("while begin!")
            while not ee.detect_contact():
                # time.sleep(0.5)
                # print("pick_success")

                # print("place_pose: ", list(place_pose[0]),list(place_pose[1]))
                # print("targ_pose: ", targ_pose)

                # freq_save += 1
                # if freq_save % 50 == 1:
                #     self.view_image_save(episode, viewList, freq_save, is_act=1, is_end=0, stage="%s_picking_success"%(task_name))

                targ_pose = utils.multiply(targ_pose, delta)
                timeout |= movep(targ_pose, self.speed, [episode, 1, 0, "%s_picking_success" % (task_name)])

                if timeout:
                    return True, False

            # print("while end")

            # 因为要存reward=1的状态，最终状态一定要保存
            # self.view_image_save(episode, viewList, freq_save, is_act=1, is_end=1, stage="%s_goal_state"%(task_name))

            ee.release()
            # 只保存最后一次的goal state
            timeout |= movep(prepick_pose, self.speed, [episode, 1, 1, "%s_goal_state" % (task_name)])

        # Move to prepick pose if pick is not successful.
        else:
            ee.release()
            timeout |= movep(prepick_pose, self.speed, [episode, 1, 1, "%s_goal_state" % (task_name)])

        return timeout, pick_success

    def view_image_save(self, episode, viewList, freq_save, is_end, is_act, stage):
        '''
        Args:
            episode:
            viewList:
            freq_save:
            stage:  [move_to_pick, picking_success, goal_state]
        Returns:

        '''
        for ind, view in enumerate(viewList):
            images = p.getCameraImage(300,
                                      300,
                                      view[0],
                                      view[1],
                                      shadow=True,
                                      renderer=p.ER_BULLET_HARDWARE_OPENGL)
            # print(images[2])
            # import cv2
            from PIL import Image
            # import numpy as np
            info = p.getLinkState(3, 0)
            info = list(info)[:2]  # 只保留坐标与转角
            info.insert(0, is_act)  # 夹了东西
            info.insert(-1, is_end)  # 没有到终点
            # cv2.imwrite("traj_%d_view_%d_%s_%d.jpg" % (episode, ind, stage, freq_save),
            #             images[2])  # [2]是rpg图像
            im = Image.fromarray(images[2])
            # rgb_im = rgb_im.convert('RGB')
            if im.mode in ("RGBA", "P"): im = im.convert("RGB")
            im.save("traj_%d_view_%d_%s_%d.jpg" % (episode, ind, stage, freq_save))
            # scipy.misc.imsave("traj_%d_view_%d_%s_%d.jpg" % (episode, ind, stage, freq_save),
            #             images[2])
        np.save(file="traj_%d_%s_%d.npy" % (episode, stage, freq_save), arr=info)

    def get_view(self, cam_pos=0, cam_orientation=0):
        """
        Arguments
            cam_pos: camera position
            cam_orientation: camera orientation in quaternion
        """
        width = 300
        height = 300
        fov = 90
        aspect = width / height
        near = 0.001
        far = 5

        # cam_pos = np.array([0.0, 0.5, 0.7])
        # cam_orientation = np.array([0.7, 0.7, 0, 0])

        use_maximal_coordinates = False
        if use_maximal_coordinates:
            # cam_orientation has problem when enable bt_rigid_body,
            # looking at 0.0, 0.0, 0.0 instead
            # this does not affect performance
            cam_pos_offset = cam_pos + np.array([0.0, 0.0, 0.3])
            target_pos = np.array([0.0, 0.0, 0.0])
        else:
            # camera pos, look at, camera up direction
            rot_matrix = p.getMatrixFromQuaternion(cam_orientation)
            # offset to base pos
            cam_pos_offset = cam_pos + np.dot(
                np.array(rot_matrix).reshape(3, 3), np.array([0.1, 0.0, 0.3]))
            target_pos = cam_pos_offset + np.dot(
                np.array(rot_matrix).reshape(3, 3), np.array([-1.0, 0.0, 0.0]))
        # compute view matrix
        view_matrix = p.computeViewMatrix(cam_pos_offset, target_pos, [0, 0, 1])
        projection_matrix = p.computeProjectionMatrixFOV(fov, aspect, near, far)

        return view_matrix, projection_matrix


class MicroPickPlace:
    """Pick and place primitive with micro actions."""

    def __init__(self, height=0.32, speed=0.01):
        self.height, self.speed = height, speed

    def __call__(self, movej, movep, ee, micro_action, episode, task_name):
        """Execute pick and place primitive.

        Args:
          movej: function to move robot joints.
          movep: function to move robot end effector pose.
          ee: robot end effector.
          micro_action: micro_action.

        Returns:
          timeout: robot movement timed out if True.
        """
        pose0 = []
        pose1 = []

        pick_pose = (np.array([pose0[0][0], pose0[0][1], 0]), pose0[1])
        place_pose = (np.array([pose1[0][0], pose1[0][1], 0]), pose1[1])

        # Execute picking primitive.
        prepick_to_pick = ((0, 0, 0.32), (0, 0, 0, 1))
        postpick_to_pick = ((0, 0, self.height), (0, 0, 0, 1))
        prepick_pose = utils.multiply(pick_pose, prepick_to_pick)
        postpick_pose = utils.multiply(pick_pose, postpick_to_pick)
        timeout = movep(prepick_pose, self.speed, [episode, 0, 0, "%s_move_to_pick" % (task_name)])

        # Move towards pick pose until contact is detected.
        delta = (np.float32([0, 0, -0.001]), utils.eulerXYZ_to_quatXYZW((0, 0, 0)))
        targ_pose = prepick_pose

        while not ee.detect_contact():  # and target_pose[2] > 0:
            # time.sleep(0.02)
            # print("move to pick")

            # freq_save += 1
            # if freq_save % 50 == 1:
            #     self.view_image_save(episode, viewList, freq_save, is_act=1, is_end=1, stage="%s_move_to_pick"%(task_name))

            targ_pose = utils.multiply(targ_pose, delta)
            timeout |= movep(targ_pose, self.speed, [episode, 0, 0, "%s_move_to_pick" % (task_name)])

            if timeout:
                return True, False

        # Activate end effector, move up, and check picking success.
        ee.activate()
        # timeout |= movep(postpick_pose, self.speed, [episode, 1, 0, "%s_picking_success" % (task_name)])
        timeout |= movep(postpick_pose, self.speed)
        pick_success = ee.check_grasp()

        # Execute placing primitive if pick is successful.
        if pick_success:
            preplace_to_place = ((0, 0, self.height), (0, 0, 0, 1))
            postplace_to_place = ((0, 0, 0.32), (0, 0, 0, 1))
            preplace_pose = utils.multiply(place_pose, preplace_to_place)
            postplace_pose = utils.multiply(place_pose, postplace_to_place)
            targ_pose = preplace_pose
            print("while begin!")
            while not ee.detect_contact():
                # time.sleep(0.5)
                # print("pick_success")

                print("place_pose: ", list(place_pose[0]), list(place_pose[1]))
                print("targ_pose: ", targ_pose)

                # freq_save += 1
                # if freq_save % 50 == 1:
                #     self.view_image_save(episode, viewList, freq_save, is_act=1, is_end=0, stage="%s_picking_success"%(task_name))

                targ_pose = utils.multiply(targ_pose, delta)
                timeout |= movep(targ_pose, self.speed, [episode, 1, 0, "%s_picking_success" % (task_name)])

                if timeout:
                    return True, False

            print("while end")

            # 因为要存reward=1的状态，最终状态一定要保存
            # self.view_image_save(episode, viewList, freq_save, is_act=1, is_end=1, stage="%s_goal_state"%(task_name))

            ee.release()
            # 只保存最后一次的goal state
            timeout |= movep(prepick_pose, self.speed, [episode, 1, 1, "%s_goal_state" % (task_name)])

        # Move to prepick pose if pick is not successful.
        else:
            ee.release()
            timeout |= movep(prepick_pose, self.speed, [episode, 1, 1, "%s_goal_state" % (task_name)])

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
        print("pushing")
        timeout = movep((over0, rot))
        timeout |= movep((pos0, rot))
        n_push = np.int32(np.floor(np.linalg.norm(pos1 - pos0) / self._step_size))
        for _ in range(n_push):
            target = pos0 + vec * n_push * self._step_size
            timeout |= movep((target, rot), speed=self._speed)
        timeout |= movep((pos1, rot), speed=self._speed)
        timeout |= movep((over1, rot))
        return timeout
