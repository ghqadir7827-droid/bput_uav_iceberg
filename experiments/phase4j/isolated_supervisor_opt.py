"""isolated_supervisor_opt.py - Phase 4J Isolated Experimental Supervisor.

Provides an isolated, parameter-validated wrapper around the safety layer to evaluate
caution-zone distance optimizations without modifying canonical controllers/safety_supervisor.py.

Key constraints:
- Only obstacle_caution_dist_m is swept.
- critical_obstacle_dist_m (0.30m) and emergency_reverse_dist_m (0.15m) are strictly immutable.
- Parameter validation guarantees d_caut > d_crit + 0.05m to preserve numerical stability.
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Union
import numpy as np

from controllers.safety_supervisor import SafetyState, SafetySupervisor, SafetySupervisorConfig
from core.contracts import ActionLimits, DeployableObservation, VelocityAction
from core.geometry import Corridor3D


# Canonical invariant thresholds
CANONICAL_CRITICAL_DIST_M: float = 0.30
CANONICAL_REVERSE_DIST_M: float = 0.15
CANONICAL_BOUNDARY_BUFFER_M: float = 0.25


class IsolatedOptimizedSupervisor:
    """Isolated experimental supervisor enforcing parameter invariants during sweeps."""

    def __init__(
        self,
        corridor: Corridor3D,
        caution_distance_m: float = 0.55,
        limits: ActionLimits = ActionLimits(),
    ):
        if caution_distance_m <= CANONICAL_CRITICAL_DIST_M:
            raise ValueError(
                f"Invalid caution_distance_m={caution_distance_m:.3f}m: must be strictly greater than "
                f"critical_obstacle_dist_m={CANONICAL_CRITICAL_DIST_M:.3f}m."
            )
        if caution_distance_m < (CANONICAL_CRITICAL_DIST_M + 0.05):
            raise ValueError(
                f"caution_distance_m={caution_distance_m:.3f}m provides insufficient margin above critical threshold ({CANONICAL_CRITICAL_DIST_M}m)."
            )

        self.caution_distance_m = float(caution_distance_m)
        self.corridor = corridor
        self.limits = limits

        # Instantiate canonical SafetySupervisor with custom validated config
        self.config = SafetySupervisorConfig(
            boundary_buffer_m=CANONICAL_BOUNDARY_BUFFER_M,
            critical_obstacle_dist_m=CANONICAL_CRITICAL_DIST_M,
            emergency_reverse_dist_m=CANONICAL_REVERSE_DIST_M,
            obstacle_caution_dist_m=self.caution_distance_m,
        )

        self._supervisor = SafetySupervisor(
            corridor=corridor,
            limits=limits,
            config=self.config,
        )

    @property
    def loss_timer_s(self) -> float:
        return self._supervisor.loss_timer_s

    @loss_timer_s.setter
    def loss_timer_s(self, value: float):
        self._supervisor.loss_timer_s = value

    @property
    def stalled_timer_s(self) -> float:
        return self._supervisor.stalled_timer_s

    @stalled_timer_s.setter
    def stalled_timer_s(self, value: float):
        self._supervisor.stalled_timer_s = value

    @property
    def last_safe_action(self) -> VelocityAction:
        return self._supervisor.last_safe_action

    @last_safe_action.setter
    def last_safe_action(self, value: VelocityAction):
        self._supervisor.last_safe_action = value

    def reset_state(self):
        """Full reset of internal supervisor timers and state."""
        self._supervisor.loss_timer_s = 0.0
        self._supervisor.stalled_timer_s = 0.0
        self._supervisor.last_safe_action = VelocityAction(0.0, 0.0, 0.0, 0.0)

    def filter_action(
        self,
        proposed_action: VelocityAction,
        current_position: np.ndarray,
        current_velocity: np.ndarray,
        min_front_range: float,
        dt: float,
        is_localization_healthy: bool = True,
        telemetry_age_s: float = 0.0,
        observation: Optional[Union[DeployableObservation, np.ndarray]] = None,
        range_rays: Optional[np.ndarray] = None,
    ) -> Tuple[VelocityAction, SafetyState, str]:
        return self._supervisor.filter_action(
            proposed_action=proposed_action,
            current_position=current_position,
            current_velocity=current_velocity,
            min_front_range=min_front_range,
            dt=dt,
            is_localization_healthy=is_localization_healthy,
            telemetry_age_s=telemetry_age_s,
            observation=observation,
            range_rays=range_rays,
        )
