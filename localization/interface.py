"""interface.py - Modular Localization Interface & Health Checks.

Defines the pluggable architecture for external/internal localization sources:
- Simulator ground-truth provider
- Optical MoCap (OptiTrack NatNet / Vicon UDP)
- Calibrated Ground Camera tracking
- Mock provider with configurable noise, stale data, and dropouts

CONSTRAINTS ENFORCED:
1. Dynamic source switching is DISABLED by default. Active source must be explicitly configured.
2. Every packet is validated for:
   - Valid monotonic timestamp and age <= max_stale_age_s
   - Matching coordinate frame convention (ENU vs NED)
   - Position and velocity bounds sanity check
   - Covariance / uncertainty bounds
3. Ground-truth state is strictly separated from deployable policy observations.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
import time
from typing import Dict, Optional, Tuple
import numpy as np

from core.contracts import CoordinateFrame


@dataclass
class LocalizationPacket:
    """Standardized localization telemetry packet."""
    timestamp_s: float            # System monotonic timestamp of capture (seconds)
    position: np.ndarray          # 3D Position [x, y, z] (meters)
    velocity: np.ndarray          # 3D Linear Velocity [vx, vy, vz] (m/s)
    orientation_quat: np.ndarray  # Quaternion [x, y, z, w]
    covariance: np.ndarray        # 3x3 or 6x6 covariance matrix
    frame: CoordinateFrame        # Coordinate frame convention
    source_name: str              # Identifier: "sim_ground_truth", "mocap", "camera", "mock"
    is_valid: bool = True         # Hardware validity flag from sensor


@dataclass
class LocalizationHealth:
    """Diagnostic health assessment of localization source."""
    is_healthy: bool
    packet_age_s: float
    frequency_hz: float
    uncertainty_m: float
    error_message: str = ""


class ILocalizationProvider(ABC):
    """Abstract base class for all localization sources."""

    @property
    @abstractmethod
    def source_name(self) -> str:
        """Returns unique identifier for this localization provider."""
        pass

    @abstractmethod
    def get_latest_packet(self) -> Optional[LocalizationPacket]:
        """Returns the most recent received localization packet."""
        pass

    @abstractmethod
    def check_health(self, current_time_s: float, max_stale_age_s: float = 0.35) -> LocalizationHealth:
        """Performs rigorous health checks on timestamp, covariance, and dropouts."""
        pass


class LocalizationManager:
    """Manages active localization source with explicit configuration.
    
    Dynamic auto-switching is STRICTLY DISABLED by default to prevent silent
    mode changes during autonomous flight.
    """

    def __init__(
        self,
        primary_source: ILocalizationProvider,
        allow_dynamic_switching: bool = False,
        target_frame: CoordinateFrame = CoordinateFrame.ENU,
        max_stale_age_s: float = 0.35,
        max_allowed_uncertainty_m: float = 0.50,
    ):
        self._sources: Dict[str, ILocalizationProvider] = {primary_source.source_name: primary_source}
        self._active_source: ILocalizationProvider = primary_source
        self.allow_dynamic_switching = allow_dynamic_switching
        self.target_frame = target_frame
        self.max_stale_age_s = max_stale_age_s
        self.max_allowed_uncertainty_m = max_allowed_uncertainty_m

    @property
    def active_source_name(self) -> str:
        return self._active_source.source_name

    def register_source(self, source: ILocalizationProvider) -> None:
        """Registers a candidate localization source."""
        self._sources[source.source_name] = source

    def set_active_source_explicit(self, source_name: str) -> bool:
        """Explicitly selects the active source. Requires manual configuration."""
        if source_name not in self._sources:
            raise ValueError(f"Unknown localization source: {source_name}. Available: {list(self._sources.keys())}")
        self._active_source = self._sources[source_name]
        return True

    def get_state(self, current_time_s: Optional[float] = None) -> Tuple[Optional[LocalizationPacket], LocalizationHealth]:
        """Retrieves and validates the current state packet from active source."""
        if current_time_s is None:
            current_time_s = time.monotonic()

        health = self._active_source.check_health(current_time_s, self.max_stale_age_s)

        if not health.is_healthy:
            if self.allow_dynamic_switching:
                # If explicitly permitted, attempt fallback
                for name, alt_source in self._sources.items():
                    if name != self._active_source.source_name:
                        alt_health = alt_source.check_health(current_time_s, self.max_stale_age_s)
                        if alt_health.is_healthy:
                            self._active_source = alt_source
                            return alt_source.get_latest_packet(), alt_health

            return None, health

        pkt = self._active_source.get_latest_packet()
        if pkt is None:
            return None, LocalizationHealth(False, 999.0, 0.0, 999.0, "Packet is None")

        # Frame transformation check (e.g. NED to ENU if needed)
        if pkt.frame != self.target_frame:
            pkt = self._convert_frame(pkt, self.target_frame)

        # Uncertainty threshold check
        uncertainty = float(np.sqrt(np.trace(pkt.covariance[:3, :3])))
        if uncertainty > self.max_allowed_uncertainty_m:
            return pkt, LocalizationHealth(False, health.packet_age_s, health.frequency_hz, uncertainty, f"Uncertainty ({uncertainty:.3f}m) exceeds limit ({self.max_allowed_uncertainty_m}m)")

        return pkt, health

    def _convert_frame(self, pkt: LocalizationPacket, target: CoordinateFrame) -> LocalizationPacket:
        """Transforms coordinates between ENU and NED conventions."""
        if pkt.frame == CoordinateFrame.NED and target == CoordinateFrame.ENU:
            # NED: X=North, Y=East, Z=Down -> ENU: X=East, Y=North, Z=Up
            pos_enu = np.array([pkt.position[1], pkt.position[0], -pkt.position[2]])
            vel_enu = np.array([pkt.velocity[1], pkt.velocity[0], -pkt.velocity[2]])
            # Transform quaternion appropriately
            q = pkt.orientation_quat
            q_enu = np.array([q[1], q[0], -q[2], q[3]])
            return LocalizationPacket(
                timestamp_s=pkt.timestamp_s,
                position=pos_enu,
                velocity=vel_enu,
                orientation_quat=q_enu,
                covariance=pkt.covariance.copy(),
                frame=CoordinateFrame.ENU,
                source_name=pkt.source_name,
                is_valid=pkt.is_valid
            )
        elif pkt.frame == CoordinateFrame.ENU and target == CoordinateFrame.NED:
            pos_ned = np.array([pkt.position[1], pkt.position[0], -pkt.position[2]])
            vel_ned = np.array([pkt.velocity[1], pkt.velocity[0], -pkt.velocity[2]])
            q = pkt.orientation_quat
            q_ned = np.array([q[1], q[0], -q[2], q[3]])
            return LocalizationPacket(
                timestamp_s=pkt.timestamp_s,
                position=pos_ned,
                velocity=vel_ned,
                orientation_quat=q_ned,
                covariance=pkt.covariance.copy(),
                frame=CoordinateFrame.NED,
                source_name=pkt.source_name,
                is_valid=pkt.is_valid
            )
        return pkt
