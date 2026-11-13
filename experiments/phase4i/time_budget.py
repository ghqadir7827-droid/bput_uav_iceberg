"""time_budget.py - Isolated Time-Budget Diagnostic Instrumentation for Phase 4I.

Defines mutually exclusive movement mode classification, time-budget accounting,
and obstacle interaction tracking without altering canonical control behavior.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from controllers.safety_supervisor import SafetyState
from core.contracts import VelocityAction


class MovementMode(str, Enum):
    """Mutually exclusive categories of drone movement during an episode."""
    UNCONSTRAINED_FORWARD = "unconstrained_forward"
    SUPERVISOR_CONSTRAINED_FORWARD = "supervisor_constrained_forward"
    LATERAL_AVOIDANCE = "lateral_avoidance"
    BRAKING_OR_STOPPING = "braking_or_stopping"
    RETREAT_OR_RECOVERY = "retreat_or_recovery"
    RE_ACCELERATION = "re_acceleration"
    GOAL_APPROACH_BRAKING = "goal_approach_braking"
    OTHER_UNCLASSIFIED = "other_unclassified"


def classify_step_mode(
    pos: np.ndarray,
    vel: np.ndarray,
    raw_action: VelocityAction,
    safe_action: VelocityAction,
    safety_state: SafetyState,
    front_clearance_m: float,
    min_obs_clearance_m: float,
    goal_dist_m: float,
    reason: str = "",
) -> MovementMode:
    """Classifies a single simulation timestep into exactly one mutually exclusive MovementMode.

    Priority order ensures disjoint partitioning of the time budget.
    """
    vx = float(vel[0])
    vy = float(vel[1])
    cmd_vx = float(safe_action.vx)
    cmd_vy = float(safe_action.vy)

    # 1. Retreat or recovery (backward movement commanded or occurring)
    if cmd_vx < -0.05 or vx < -0.05 or "Reverse" in reason or "recovery" in reason.lower():
        return MovementMode.RETREAT_OR_RECOVERY

    # 2. Braking or stopping (drone halted or near standstill)
    if abs(vx) <= 0.10 and abs(vy) <= 0.10:
        return MovementMode.BRAKING_OR_STOPPING

    # 3. Goal approach and terminal deceleration (final stretch approaching destination)
    if pos[0] >= 8.0 and goal_dist_m <= 2.0 and front_clearance_m >= 1.50:
        return MovementMode.GOAL_APPROACH_BRAKING

    # 4. Lateral avoidance (active lateral maneuver to bypass obstacle)
    if abs(vy) >= 0.15 or abs(cmd_vy) >= 0.15:
        return MovementMode.LATERAL_AVOIDANCE

    # 5. Forward movement under active supervisor constraint (speed throttled by supervisor)
    if safety_state != SafetyState.NOMINAL and cmd_vx > 0.05:
        return MovementMode.SUPERVISOR_CONSTRAINED_FORWARD

    # 6. Re-acceleration (cleared obstacle, accelerating forward back to cruise speed)
    if front_clearance_m >= 0.80 and vx < 0.40 and cmd_vx >= 0.40 and pos[0] < 8.0:
        return MovementMode.RE_ACCELERATION

    # 7. Unconstrained forward movement (freely flying forward at nominal speed)
    if safety_state == SafetyState.NOMINAL and vx >= 0.35 and abs(vy) < 0.15:
        return MovementMode.UNCONSTRAINED_FORWARD

    # 8. Residual transition
    return MovementMode.OTHER_UNCLASSIFIED


@dataclass
class TimeBudgetRecord:
    """Detailed time-budget breakdown for a single episode."""
    total_steps: int = 0
    dt_s: float = 0.05
    step_counts: Dict[str, int] = field(default_factory=lambda: {m.value: 0 for m in MovementMode})
    time_spent_s: Dict[str, float] = field(default_factory=lambda: {m.value: 0.0 for m in MovementMode})
    time_percentages: Dict[str, float] = field(default_factory=lambda: {m.value: 0.0 for m in MovementMode})

    # Obstacle interaction encounters
    caution_entries: int = 0
    complete_stops: int = 0
    stalled_recoveries: int = 0
    obstacles_cleared: int = 0

    def record_step(self, mode: MovementMode):
        self.total_steps += 1
        self.step_counts[mode.value] += 1
        self.time_spent_s[mode.value] = round(self.step_counts[mode.value] * self.dt_s, 3)

    def finalize(self):
        """Calculates percentages and reconciles total time."""
        total_time = self.total_steps * self.dt_s
        for m in MovementMode:
            val = self.time_spent_s[m.value]
            self.time_percentages[m.value] = round((val / total_time * 100.0) if total_time > 0 else 0.0, 2)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_steps": self.total_steps,
            "total_time_s": round(self.total_steps * self.dt_s, 3),
            "step_counts": self.step_counts,
            "time_spent_s": self.time_spent_s,
            "time_percentages": self.time_percentages,
            "encounters": {
                "caution_entries": self.caution_entries,
                "complete_stops": self.complete_stops,
                "stalled_recoveries": self.stalled_recoveries,
                "obstacles_cleared": self.obstacles_cleared,
            }
        }
