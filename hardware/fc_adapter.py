"""fc_adapter.py - Flight Controller Hardware Adapter Interface.

Implements an abstracted adapter for the RadioLink CrossFlight flight controller.

STRICT SAFETY CONSTRAINTS:
1. CrossFlight adapter is READ-ONLY initially. Command transmission is blocked
   until the exact firmware, serial port, baud rate, and failsafe modes are verified.
2. Hardware tests must be conducted ONLY on a secured bench with propellers removed.
3. Loss of companion heartbeat triggers the FC's internal failsafe.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
import time
from typing import Dict, Optional, Tuple
import numpy as np

from core.contracts import CoordinateFrame, VelocityAction


class FlightMode(Enum):
    """ArduPilot / CrossFlight operating modes."""
    STABILIZE = "STABILIZE"
    ALT_HOLD = "ALT_HOLD"
    LOITER = "LOITER"
    GUIDED = "GUIDED" # Mode required for offboard companion velocity targets
    RTL = "RTL"
    LAND = "LAND"
    UNKNOWN = "UNKNOWN"


@dataclass
class DroneTelemetry:
    """Telemetry packet received from the flight controller."""
    timestamp_s: float
    is_armed: bool
    mode: FlightMode
    battery_voltage_v: float
    battery_remaining_pct: float
    roll_rad: float
    pitch_rad: float
    yaw_rad: float
    roll_rate_radps: float
    pitch_rate_radps: float
    yaw_rate_radps: float
    altitude_relative_m: float
    is_failsafe_active: bool
    frame: CoordinateFrame = CoordinateFrame.NED


class IFlightControllerAdapter(ABC):
    """Abstract interface for UAV flight controller communication."""

    @abstractmethod
    def connect(self, port: str, baudrate: int) -> bool:
        """Establishes communication with the flight controller."""
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """Closes communication link."""
        pass

    @abstractmethod
    def is_connected(self) -> bool:
        """Returns True if heartbeat is active."""
        pass

    @abstractmethod
    def get_telemetry(self) -> Optional[DroneTelemetry]:
        """Fetches the latest state telemetry packet."""
        pass

    @abstractmethod
    def send_velocity_command(self, action: VelocityAction) -> bool:
        """Transmits high-level velocity setpoints to the autopilot."""
        pass


class RadioLinkCrossFlightAdapter(IFlightControllerAdapter):
    """Hardware driver for RadioLink CrossFlight flight controller.
    
    HARDWARE PROFILE & INTEGRATION SPECIFICATION:
    - Autopilot Model: RadioLink CrossFlight (STM32F405 processor)
    - Underlying Firmware: Proprietary fork of ArduCopter (ArduPilot-RL)
    - Communication Protocol: MAVLink v2
    - Target Serial Port: TELEM1 / TELEM2 (typically 57600 or 115200 baud) or USB CDC
    - Offboard Command Message: SET_POSITION_TARGET_LOCAL_NED (Message #84)
    - Coordinate Frame: Local NED (North-East-Down)
    - Required Flight Mode: GUIDED mode (Mode 4)
    - Failsafe Behavior: If companion computer heartbeat ceases for > 1.0s,
      the FC triggers its internal GCS/companion failsafe (FS_GCS_ENABLE).
    """

    def __init__(self, read_only: bool = True):
        self._is_connected = False
        self._read_only = read_only
        self._latest_telemetry: Optional[DroneTelemetry] = None
        self._port = ""
        self._baudrate = 0

    @property
    def is_read_only(self) -> bool:
        return self._read_only

    def connect(self, port: str = "/dev/ttyAMA0", baudrate: int = 115200) -> bool:
        """Opens serial link in read-only mode to verify telemetry reception first."""
        self._port = port
        self._baudrate = baudrate
        # In hardware phase, this initializes pymavlink.mavutil.mavlink_connection
        self._is_connected = True
        return True

    def disconnect(self) -> None:
        self._is_connected = False

    def is_connected(self) -> bool:
        return self._is_connected

    def get_telemetry(self) -> Optional[DroneTelemetry]:
        """Reads incoming MAVLink HEARTBEAT (#0), ATTITUDE (#30), and SYS_STATUS (#1)."""
        return self._latest_telemetry

    def send_velocity_command(self, action: VelocityAction) -> bool:
        """SAFETY INTERLOCK: Disallows command transmission while in read-only mode."""
        if self._read_only:
            raise PermissionError(
                "RadioLink CrossFlight adapter is in READ-ONLY mode. "
                "Command transmission is locked until hardware model, firmware version, "
                "and failsafe responses are verified on a secured test bench with propellers removed."
            )
        # When authorized, sends MAVLink SET_POSITION_TARGET_LOCAL_NED
        return False


class MockFlightControllerAdapter(IFlightControllerAdapter):
    """Loopback adapter for testing and simulation."""

    def __init__(self):
        self._connected = True
        self._telemetry = DroneTelemetry(
            timestamp_s=time.monotonic(),
            is_armed=False,
            mode=FlightMode.GUIDED,
            battery_voltage_v=12.4,
            battery_remaining_pct=95.0,
            roll_rad=0.0,
            pitch_rad=0.0,
            yaw_rad=0.0,
            roll_rate_radps=0.0,
            pitch_rate_radps=0.0,
            yaw_rate_radps=0.0,
            altitude_relative_m=1.0,
            is_failsafe_active=False
        )
        self.last_command: Optional[VelocityAction] = None

    def connect(self, port: str = "mock", baudrate: int = 0) -> bool:
        self._connected = True
        return True

    def disconnect(self) -> None:
        self._connected = False

    def is_connected(self) -> bool:
        return self._connected

    def get_telemetry(self) -> Optional[DroneTelemetry]:
        return self._telemetry

    def send_velocity_command(self, action: VelocityAction) -> bool:
        self.last_command = action
        return True
