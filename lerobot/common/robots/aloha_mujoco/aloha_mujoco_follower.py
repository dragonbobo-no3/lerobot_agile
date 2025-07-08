#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import math
import time
from functools import cached_property
from gc import enable
from typing import Any
import os
import mujoco.msh2obj_test
import mujoco.viewer
import numpy as np

from lerobot.common.cameras.utils import make_cameras_from_configs
from lerobot.common.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from lerobot.common.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.common.motors.dynamixel import (
    DynamixelMotorsBus,
    OperatingMode,
)

import mujoco
import cv2
import glfw

from ..robot import Robot
from ..utils import ensure_safe_goal_position
from .config_aloha_mujoco_follower import AlohaMujocoFollowerConfig

logger = logging.getLogger(__name__)
class AlohaMujocoFollower(Robot):
    config_class = AlohaMujocoFollowerConfig
    name = "aloha_mujoco_follower"

    def __init__(self, config: AlohaMujocoFollowerConfig):
        super().__init__(config)
        self.config = config

        #mujoco环境初始化
        #创建opengl上下文
        self.resolution = (640, 480)  # Resolution for the offscreen rendering
        glfw.init()
        glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
        window = glfw.create_window(*self.resolution, "Offscreen", None, None)
        glfw.make_context_current(window)

        # 寻找模型路径
        current_path = os.path.dirname(os.path.dirname(__file__))
        current_path = current_path + "/aloha_mujoco/meshes_mujoco"
        model_path = current_path + "/aloha_v1.xml"
        print("The directory is ",model_path)

        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.scene = mujoco.MjvScene(self.model, maxgeom=10000)
        self.context = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_150.value)

        #创建相机
        self.type = "mujoco" # Camera type for the cameras
        self.fps = 30  # Frames per second for the cameras
        self.cameras = {i:{"type": self.type, "width": self.resolution[0], "height": self.resolution[1], "fps": self.fps} for i in range(3)}
        self.offscreens = {f"offscreen_{i}": mujoco.MjvCamera() for i in range(3)}
        for i in range(3):
            self.offscreens[f"offscreen_{i}"].type = mujoco.mjtCamera.mjCAMERA_FIXED
            self.offscreens[f"offscreen_{i}"].fixedcamid = i

        #创建其他不做数据的固定相机
        # self.viewer = mujoco.MjvCamera()
        # self.viewer.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        # tracking_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "fl_link1")
        # self.viewer.trackbodyid = tracking_body_id  # 追踪的物体ID
        # self.viewer.distance = 2  # 相机与目标的距离
        # self.viewer.azimuth = 45    # 水平方位角（度）
        # self.viewer.elevation = -45 # 俯仰角（度）

        # 创建帧缓冲对象
        framebuffer = mujoco.MjrRect(0, 0, *self.resolution)
        mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, self.context)

        self.is_enabled_ = True
        self.is_robot_connected_ = True

        # 校准参数（和现实松灵臂）
        self.right_bias = np.array([9.83900375e-03, 3.55023312e-02, 3.11918405e+00, -1.70167349e-01,                                   
                                    -8.71561797e-03, -1.39412725e-02, 4.46428572e-05])
        self.left_bias = np.array([4.91205376e-03, 3.08380829e-02, 3.12512528e+00, -1.52382061e-01,
                                     -1.31789072e-02, -1.28610098e-02, 8.97072712e-05])
        self.right_scale = np.array([1.16113983, 0.9187596, 1.02111805, 1.02622979, 1.1946608,
                                      1.48471411, 0.03654068])
        self.left_scale = np.array([1.16057805, 0.91882059, 1.02681781, 1.02765416, 1.19477187, 1.4826585,
                                     0.0367132])
        
        # self.max = np.zeros(7)
        # self.min = np.ones(7)

    @property
    def _motors_ft(self) -> dict[str, type]:
        return {self.id + f".joint{i}.pos": float for i in range(7)}

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3) for cam in self.cameras
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._motors_ft

    @property
    def is_connected(self) -> bool:
        ## TODO: add piper is connected
        return True

    def connect(self, calibrate: bool = True) -> None:
        """
        We assume that at connection time, arm is in a rest position,
        and torque can be safely disabled to run calibration.
        """
        logger.info(f"{self} connected.")

    @property
    def is_calibrated(self) -> bool:
        ## TODO: add calibration process
        return True

    def calibrate(self) -> None:
        logger.info(f"\nNo calibration {self} can be run")

    def configure(self) -> None:
        ## TODO: we need to configure the torque
        logger.info(f"\nNo configuration {self} can be run")
        return

    def setup_motors(self) -> None:
        logger.info(f"\nWe don't have to setup motor for {self}")
        return

    def get_observation(self, need_to_show) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # Read arm position
        start = time.perf_counter()
        obs_dict = {
            *{f"fl.joint{i}.pos":
                self.data.joint(f"fl.joint{i + 1}").qpos for i in range(7)}  # 从 1 到 7
            *{f"fr.joint{i}.pos":
                self.data.joint(f"fr.joint{i + 1}").qpos for i in range(7,15)}  # 从 8 到 14
        }
        
        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read state: {dt_ms:.1f}ms")

        # Capture images from cameras
        for i in range(3):
            start = time.perf_counter()
            bgr = self.render_image(i, need_to_show)    
            obs_dict[f"camera{i}"] = bgr  
            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug(f"{self} read camera{i}: {dt_ms:.1f}ms")

        # time.sleep(0.01)   

        return obs_dict

    def send_action(self, action: dict[str, float], need_to_show) -> dict[str, float]:
        """Command arm to move to a target joint configuration.

        The relative action magnitude may be clipped depending on the configuration parameter
        `max_relative_target`. In this case, the action sent differs from original action.
        Thus, this function always returns the action actually sent.

        Args:
            action (dict[str, float]): The goal positions for the motors.

        Returns:
            dict[str, float]: The action sent to the motors, potentially clipped.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        if not self.is_enabled:
            self.enable()
        # print(action)
        goal_pos_left = {key.removesuffix(".pos").removeprefix(f"left."): val for key, val in action.items() if
                    (key.endswith(".pos") and key.startswith("left"))}
        goal_pos_right = {key.removesuffix(".pos").removeprefix(f"right."): val for key, val in action.items() if
                    (key.endswith(".pos") and key.startswith("right"))}

        # Send goal position to the arm
        factor = 1000 * 180 / math.pi

        for i in range(7):

            if i == 2: #仿真的第三轴反了
                goal_left = 3.14 - (float(goal_pos_left[f"joint{i}"].item()/factor) * self.left_scale[i] + self.left_bias[i])
                goal_right = 3.14 - (float(goal_pos_right[f"joint{i}"].item()/factor) * self.right_scale[i] + self.right_bias[i])
            else:
                goal_left = float(goal_pos_left[f"joint{i}"].item()/factor) * self.left_scale[i] + self.left_bias[i]
                goal_right = float(goal_pos_right[f"joint{i}"].item()/factor) * self.right_scale[i] + self.right_bias[i]

            self.data.actuator(f"fl_joint{i+1}").ctrl = goal_left
            self.data.actuator(f"fr_joint{i+1}").ctrl = goal_right

            if i == 6: #仿真的夹爪把两边分开定义了
                self.data.actuator(f"fl_joint{i+2}").ctrl = goal_left
                self.data.actuator(f"fr_joint{i+2}").ctrl = goal_right

            # if goal_left > self.max[i]:
            #     self.max[i] = goal_left
            # if goal_left < self.min[i]:
            #     self.min[i] = goal_left
            # if i == 2:
            #     print(f"fl_joint{i+1} ctrl: {goal_left}, fr_joint{i+1} ctrl: {goal_right}")

        # print(f"max: {self.max}, min: {self.min}")

        if need_to_show:
            for i in range(3):
                start = time.perf_counter()
                self.render_image(i, need_to_show)    
                dt_ms = (time.perf_counter() - start) * 1e3
                logger.debug(f"{self} read camera{i}: {dt_ms:.1f}ms")

        return True

    @property
    def is_enabled(self) -> bool:
        return self.is_enabled_

    def enable(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        self.is_enabled_ = True
        return

    def disconnect(self):
        # if not self.is_connected:
        #     raise DeviceNotConnectedError(f"{self} is not connected.")

        # self.bus.disconnect(self.config.disable_torque_on_disconnect)

        logger.info(f"{self} disconnected.")

    def step(self): # Step the simulation to update the state
        mujoco.mj_step(self.model, self.data)  

    def render_image(self, camera_id, need_to_show):
        viewport = mujoco.MjrRect(0, 0, *self.resolution)
        mujoco.mjv_updateScene(self.model, self.data, mujoco.MjvOption(), 
                        mujoco.MjvPerturb(), self.offscreens[f"offscreen_{camera_id}"],
                        mujoco.mjtCatBit.mjCAT_ALL, self.scene)
        mujoco.mjr_render(viewport, self.scene, self.context)   
        rgb = np.zeros((self.resolution[1], self.resolution[0], 3), dtype=np.uint8)
        mujoco.mjr_readPixels(rgb, None, viewport, self.context)
        bgr = cv2.cvtColor(np.flipud(rgb), cv2.COLOR_RGB2BGR)
        # cv2.imwrite(f"debug_output{camera_id}_{idx}.png", bgr)
        if (need_to_show):
            cv2.imshow(f'Camera{camera_id}', bgr)
            cv2.waitKey(1)  # Wait for a short time to allow the image show
        return bgr
    
    def render_viewer(self):
        "Render the viewer with the current model and data."
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self.viewer.cam.lookat[:] = [0.0, 0.0, 1.0]   # 目标点坐标（模型中心）
        self.viewer.cam.distance = 2                  # 到目标的距离
        self.viewer.cam.azimuth = 0                  # 水平旋转角度
        self.viewer.cam.elevation = -45               # 向下俯视

    def viewer_step(self):
        "make the viewer move forward one step"
        self.viewer.sync()

       

