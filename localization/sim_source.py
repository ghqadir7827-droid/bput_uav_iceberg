"""sim_source.py - Simulator Ground-Truth Localization Provider.

Queries internal PyBullet physics simulation directly to supply privileged ground-truth
measurements for reward computation, expert demonstrations, and simulation evaluation.
"""

from typing import Optional
import numpy as np

from core.contracts import CoordinateFrame
from localization.interface import ILocalizationProvider, LocalizationHealth, LocalizationPacket


class SimGroundTruthSource(ILocalizationProvider):
    """Provides exact ground truth from the simulation environment."""

    def __init__(self, source_name: str = "sim_ground_truth"):
        self._source_name = source_name
        self._latest_packet: Optional[LocalizationPacket] = None
        self._last_update_time: float = 0.0

    @property
    def source_name(self) -> str:
        return self._source_name

    def update_from_sim(
        self,
        position: np.ndarray,
        velocity: np.ndarray,
        orientation_quat: np.ndarray,
        sim_time_s: float,
    ) -> None:
        """Called by simulation step to publish the latest true state."""
        # Simulated ground truth has zero measurement covariance (near-zero for numerical stability)
        cov = np.eye(3, dtype=np.float64) * 1e-6
        self._latest_packet = LocalizationPacket(
            timestamp_s=sim_time_s,
            position=np.asarray(position, dtype=np.float64).copy(),
            velocity=np.asarray(velocity, dtype=np.float64).copy(),
            orientation_quat=np.asarray(orientation_quat, dtype=np.float64).copy(),
            covariance=cov,
            frame=CoordinateFrame.ENU,
            source_name=self._source_name,
            is_valid=True
        )
        self._last_update_time = sim_time_s

    def get_latest_packet(self) -> Optional[LocalizationPacket]:
        return self._latest_packet

    def check_health(self, current_time_s: float, max_stale_age_s: float = 0.35) -> LocalizationHealth:
        if self._latest_packet is None:
            return LocalizationHealth(False, 999.0, 0.0, 0.0, "No simulation state received yet")

        age = current_time_s - self._latest_packet.timestamp_s
        if age > max_stale_age_s:
            return LocalizationHealth(False, age, 0.0, 0.0, f"Simulation step lagged (age={age:.3f}s > {max_stale_age_s}s)")

        return LocalizationHealth(True, age, 100.0, 0.001, "Healthy")
