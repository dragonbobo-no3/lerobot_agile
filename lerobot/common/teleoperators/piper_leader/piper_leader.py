from functools import cached_property
import logging
import time
import torch

from lerobot.common.cameras import make_cameras_from_configs
from lerobot.common.motors import Motor, MotorNormMode
from lerobot.common.motors.piper.piper import PiperMotorsBus
from lerobot.common.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from lerobot.common.robots import Robot

from .config_piper_leader import PiperConfig

logger = logging.getLogger(__name__)

class PiperRobot(Robot):
    config_class = PiperConfig
    name = "piper_leader"

    def __init__(self, config: PiperConfig):
        super().__init__(config)
        self.config = config
        self.bus = PiperMotorsBus(
            port=self.config.port,
            motors={
                "joint_1": Motor(1, "agilex_piper", MotorNormMode.RANGE_M100_100),
                "joint_2": Motor(2, "agilex_piper", MotorNormMode.RANGE_M100_100),
                "joint_3": Motor(3, "agilex_piper", MotorNormMode.RANGE_M100_100),
                "joint_4": Motor(4, "agilex_piper", MotorNormMode.RANGE_M100_100),
                "joint_5": Motor(5, "agilex_piper", MotorNormMode.RANGE_M100_100),
                "joint_6": Motor(6, "agilex_piper", MotorNormMode.RANGE_M100_100),
                "gripper": Motor(7, "agilex_piper", MotorNormMode.RANGE_0_100),
            },
            calibration=self.calibration,
        )
        self.cameras = make_cameras_from_configs(config.cameras)
        self.logs = {}
        self.is_connected = False

    @property
    def _motors_ft(self) -> dict[str, type]:
        return {f"{motor}.pos": float for motor in self.bus.motors}
    
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
    
    # @property
    # def is_connected(self) -> bool:
    #     return self.bus.pb.is_connected and all(cam.is_connected for cam in self.cameras.values())
    
    # def connect(self, calibrate: bool = True) -> None:
    #     """
    #     We assume that at connection time, arm is in a rest position,
    #     and torque can be safely disabled to run calibration.
    #     """
    #     if self.is_connected:
    #         raise DeviceAlreadyConnectedError(f"{self} already connected")

    #     self.bus.pb.connect()
    #     if not self.is_calibrated and calibrate:
    #         self.calibrate()

    #     for cam in self.cameras.values():
    #         cam.connect()

    #     self.configure()
    #     logger.info(f"{self} connected.")

    def connect(self) -> None:
        """Connect piper and cameras"""
        if self.is_connected:
            raise DeviceAlreadyConnectedError(
                "Piper is already connected. Do not run `robot.connect()` twice."
            )
        
        # connect piper
        self.bus.pb.connect(enable=True)
        print("piper conneted")

        # connect cameras
        for name in self.cameras:
            self.cameras[name].connect()
            self.is_connected = self.is_connected and self.cameras[name].is_connected
            print(f"camera {name} conneted")
        
        print("All connected")
        self.is_connected = True
        
        self.calibrate()

    def disconnect(self) -> None:
        """move to home position, disenable piper and cameras"""

        # disconnect piper
        self.bus.pb.safe_disconnect()
        print("piper disable after 5 seconds")
        time.sleep(5)
        self.bus.pb.connect(enable=False)

        # disconnect cameras
        if len(self.cameras) > 0:
            for cam in self.cameras.values():
                cam.disconnect()

        self.is_connected = False

    # def disconnect(self):
    #     if not self.is_connected:
    #         raise DeviceNotConnectedError(f"{self} is not connected.")

    #     self.bus.disconnect(self.config.disable_torque_on_disconnect)
    #     for cam in self.cameras.values():
    #         cam.disconnect()

    #     logger.info(f"{self} disconnected.")

    #not for now, but might be useful in the future
    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass
        # with self.bus.torque_disabled():
        #     self.bus.configure_motors()
        #     for motor in self.bus.motors:
        #         self.bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)
        #         self.bus.write("P_Coefficient", motor, 16)
        #         self.bus.write("I_Coefficient", motor, 0)
        #         self.bus.write("D_Coefficient", motor, 32)

    def get_observation(self) -> dict:
        """get current images and joint positions"""
        if not self.is_connected:
            raise DeviceNotConnectedError(
                "Piper is not connected. You need to run `robot.connect()`."
            )
        
        # # Read arm position
        # obs_dict = self.bus.sync_read("Present_Position")
        # obs_dict = {f"{motor}.pos": val for motor, val in obs_dict.items()}

        # # Capture images from cameras
        # for cam_key, cam in self.cameras.items():
        #     obs_dict[cam_key] = cam.async_read()

    # below is the old get_observation method, kept for reference
        # read current joint positions
        before_read_t = time.perf_counter()
        state = self.bus.pb.read()  # 6 joints + 1 gripper
        self.logs["read_pos_dt_s"] = time.perf_counter() - before_read_t

        state = torch.as_tensor(list(state.values()), dtype=torch.float32)

        # read images from cameras
        images = {}
        for name in self.cameras:
            before_camread_t = time.perf_counter()
            images[name] = self.cameras[name].async_read()
            images[name] = torch.from_numpy(images[name])
            self.logs[f"read_camera_{name}_dt_s"] = self.cameras[name].logs["delta_timestamp_s"]
            self.logs[f"async_read_camera_{name}_dt_s"] = time.perf_counter() - before_camread_t

        # Populate output dictionnaries and format to pytorch
        obs_dict = {}
        obs_dict["observation.state"] = state
        for name in self.cameras:
            obs_dict[f"observation.images.{name}"] = images[name]
        return obs_dict
    
    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        goal_pos = {key.removesuffix(".pos"): val for key, val in action.items()}

        # Send goal position to the arm
        self.bus.sync_write("Goal_Position", goal_pos)

    # below is the old send_action method, kept for reference    
        # """Write the predicted actions from policy to the motors"""
        # if not self.is_connected:
        #     raise DeviceNotConnectedError(
        #         "Piper is not connected. You need to run `robot.connect()`."
        #     )

        # # send to motors, torch to list
        # target_joints = action.tolist()
        # self.arm.write(target_joints)

        return action
    