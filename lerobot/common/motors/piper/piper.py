import time
from typing import Dict

from lerobot.common.utils.encoding_utils import decode_sign_magnitude, encode_sign_magnitude

from ..motors_bus import Motor, MotorCalibration, MotorsBus, NameOrID, Value, get_address
import PortHandler



class PiperMotorsBus(MotorsBus):

    def __init__(
        self,
        port: str,
        motors: dict[str, Motor],
        calibration: dict[str, MotorCalibration] | None = None,
    ):
        super().__init__(port, motors, calibration)
        from piper_config import PiperBus
        self.pb = PiperBus(port, motors)
        # HACK: monkeypatch
        # self.port_handler.setPacketTimeout = patch_setPacketTimeout.__get__(
        #     self.port_handler, scs.PortHandler
        # )
        # self.packet_handler = scs.PacketHandler(protocol_version)
        # self.sync_reader = self.pb.read()
        # self.sync_writer = self.pb.write()
        # self._comm_success = scs.COMM_SUCCESS
        # self._no_error = 0x00