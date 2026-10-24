"""mock_source.py - Configurable Mock Localization Source for Unit & Failsafe Testing.

Simulates sensor noise, latency buffers, packet dropouts, and stale data to verify
system resilience under realistic imperfect localization.
"""

from collections import deque
from typing import Deque, Optional
import numpy as np

from core.contracts import CoordinateFrame
from localization.interface import ILocalizationProvider, LocalizationHealth, LocalizationPacket


class MockLocalizationSource(ILocalizationProvider):
    """Configurable mock source for testing sensor noise, latency, and dropouts."""

    def __init__(
        self,
        source_name: str = "mock_tracker",
        noise_std_m: float = 0.03,
        latency_s: float = 0.05,
        dropout_probability: float = 0.0,
        frame: CoordinateFrame = CoordinateFrame.ENU,
        seed: int = 42,
    ):
        self._source_name = source_name
        self.noise_std_m = noise_std_m
        self.latency_s = latency_s
        self.dropout_probability = dropout_probability
        self.frame = frame
        self._rng = np.random.default_rng(seed)

        self._queue: Deque[LocalizationPacket] = deque()
        self._latest_delivered: Optional[LocalizationPacket] = None
        self.force_stale: bool = False
        self.force_invalid: bool = False

    @property
    def source_name(self) -> str:
        return self._source_name

    def push_true_state(
        self,
        true_position: np.ndarray,
        true_velocity: np.ndarray,
        true_quat: np.ndarray,
        timestamp_s: float,
    ) -> None:
        """Pushes a true state into the mock pipeline with noise and latency queueing."""
        # Check dropout
        if self._rng.random() < self.dropout_probability:
            return # Dropped packet

        # Add Gaussian noise
        pos_noisy = true_position + self._rng.normal(0, self.noise_std_m, size=3)
        vel_noisy = true_velocity + self._rng.normal(0, self.noise_std_m * 0.5, size=3)

        cov = np.eye(3, dtype=np.float64) * (self.noise_std_m**2)

        packet = LocalizationPacket(
            timestamp_s=timestamp_s,
            position=pos_noisy,
            velocity=vel_noisy,
            orientation_quat=true_quat.copy(),
            covariance=cov,
            frame=self.frame,
            source_name=self._source_name,
            is_valid=(not self.force_invalid)
        )
        self._queue.append(packet)

    def advance_time(self, current_time_s: float) -> None:
        """Flushes packets from the latency queue whose delay has elapsed."""
        if self.force_stale:
            return # Freeze delivery to simulate stale telemetry

        while self._queue and (current_time_s - self._queue[0].timestamp_s) >= self.latency_s:
            self._latest_delivered = self._queue.popleft()

    def get_latest_packet(self) -> Optional[LocalizationPacket]:
        return self._latest_delivered

    def check_health(self, current_time_s: float, max_stale_age_s: float = 0.35) -> LocalizationHealth:
        if self._latest_delivered is None:
            return LocalizationHealth(False, 999.0, 0.0, 999.0, "No mock packet delivered yet")

        if not self._latest_delivered.is_valid:
            return LocalizationHealth(False, 0.0, 0.0, 999.0, "Sensor flagged invalid data")

        age = current_time_s - self._latest_delivered.timestamp_s
        if age > max_stale_age_s:
            return LocalizationHealth(False, age, 0.0, self.noise_std_m, f"Stale telemetry (age={age:.3f}s > {max_stale_age_s}s)")

        uncertainty = float(np.sqrt(np.trace(self._latest_delivered.covariance[:3, :3])))
        return LocalizationHealth(True, age, 30.0, uncertainty, "Mock healthy")
