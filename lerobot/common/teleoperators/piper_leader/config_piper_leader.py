from dataclasses import dataclass, field

from lerobot.common.cameras import CameraConfig
from lerobot.common.cameras.opencv import OpenCVCameraConfig
from lerobot.common.robots import RobotConfig


@RobotConfig.register_subclass("piper_leader")
@dataclass
class PiperConfig(RobotConfig):
    port: str