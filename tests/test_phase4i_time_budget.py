"""test_phase4i_time_budget.py - Unit tests for Phase 4I Time-Budget Instrumentation."""

import numpy as np
import pytest

from controllers.safety_supervisor import SafetyState
from core.contracts import VelocityAction
from experiments.phase4i.time_budget import (
    classify_step_mode,
    MovementMode,
    TimeBudgetRecord,
)


def test_time_budget_mutually_exclusive_classification():
    """Verifies that diverse physical states map to valid, mutually exclusive modes."""
    # 1. Reverse retreat
    m1 = classify_step_mode(
        pos=np.array([3.0, 0.0, 1.0]),
        vel=np.array([-0.10, 0.0, 0.0]),
        raw_action=VelocityAction(0.5, 0.0, 0.0, 0.0),
        safe_action=VelocityAction(-0.10, 0.0, 0.0, 0.0),
        safety_state=SafetyState.OBSTACLE_BRAKE,
        front_clearance_m=0.12,
        min_obs_clearance_m=0.12,
        goal_dist_m=6.5,
        reason="Emergency obstacle proximity -> Reversing",
    )
    assert m1 == MovementMode.RETREAT_OR_RECOVERY

    # 2. Braking or stopped
    m2 = classify_step_mode(
        pos=np.array([2.8, 0.0, 1.0]),
        vel=np.array([0.02, 0.01, 0.0]),
        raw_action=VelocityAction(0.5, 0.0, 0.0, 0.0),
        safe_action=VelocityAction(0.0, 0.0, 0.0, 0.0),
        safety_state=SafetyState.OBSTACLE_BRAKE,
        front_clearance_m=0.22,
        min_obs_clearance_m=0.22,
        goal_dist_m=6.7,
        reason="Front obstacle detected -> Forward brake",
    )
    assert m2 == MovementMode.BRAKING_OR_STOPPING

    # 3. Goal approach braking
    m3 = classify_step_mode(
        pos=np.array([8.8, 0.0, 1.0]),
        vel=np.array([0.25, 0.0, 0.0]),
        raw_action=VelocityAction(0.25, 0.0, 0.0, 0.0),
        safe_action=VelocityAction(0.25, 0.0, 0.0, 0.0),
        safety_state=SafetyState.NOMINAL,
        front_clearance_m=3.0,
        min_obs_clearance_m=2.0,
        goal_dist_m=0.7,
        reason="",
    )
    assert m3 == MovementMode.GOAL_APPROACH_BRAKING

    # 4. Lateral avoidance
    m4 = classify_step_mode(
        pos=np.array([2.6, 0.2, 1.0]),
        vel=np.array([0.40, 0.25, 0.0]),
        raw_action=VelocityAction(0.4, 0.3, 0.0, 0.2),
        safe_action=VelocityAction(0.4, 0.25, 0.0, 0.2),
        safety_state=SafetyState.NOMINAL,
        front_clearance_m=0.9,
        min_obs_clearance_m=0.35,
        goal_dist_m=6.9,
        reason="",
    )
    assert m4 == MovementMode.LATERAL_AVOIDANCE

    # 5. Supervisor constrained forward
    m5 = classify_step_mode(
        pos=np.array([2.2, 0.0, 1.0]),
        vel=np.array([0.22, 0.02, 0.0]),
        raw_action=VelocityAction(0.6, 0.0, 0.0, 0.0),
        safe_action=VelocityAction(0.22, 0.0, 0.0, 0.0),
        safety_state=SafetyState.OBSTACLE_BRAKE,
        front_clearance_m=0.45,
        min_obs_clearance_m=0.45,
        goal_dist_m=7.3,
        reason="Obstacle caution zone -> Speed throttled",
    )
    assert m5 == MovementMode.SUPERVISOR_CONSTRAINED_FORWARD

    # 6. Re-acceleration
    m6 = classify_step_mode(
        pos=np.array([3.8, 0.0, 1.0]),
        vel=np.array([0.28, 0.02, 0.0]),
        raw_action=VelocityAction(0.6, 0.0, 0.0, 0.0),
        safe_action=VelocityAction(0.6, 0.0, 0.0, 0.0),
        safety_state=SafetyState.NOMINAL,
        front_clearance_m=1.8,
        min_obs_clearance_m=0.85,
        goal_dist_m=5.7,
        reason="",
    )
    assert m6 == MovementMode.RE_ACCELERATION

    # 7. Unconstrained forward
    m7 = classify_step_mode(
        pos=np.array([1.5, 0.0, 1.0]),
        vel=np.array([0.55, 0.01, 0.0]),
        raw_action=VelocityAction(0.6, 0.0, 0.0, 0.0),
        safe_action=VelocityAction(0.6, 0.0, 0.0, 0.0),
        safety_state=SafetyState.NOMINAL,
        front_clearance_m=2.5,
        min_obs_clearance_m=2.5,
        goal_dist_m=8.0,
        reason="",
    )
    assert m7 == MovementMode.UNCONSTRAINED_FORWARD


def test_time_budget_record_reconciliation():
    """Verifies that recorded steps sum exactly to total_steps and total_time."""
    tb = TimeBudgetRecord(dt_s=0.05)
    modes = [
        MovementMode.UNCONSTRAINED_FORWARD,
        MovementMode.SUPERVISOR_CONSTRAINED_FORWARD,
        MovementMode.LATERAL_AVOIDANCE,
        MovementMode.BRAKING_OR_STOPPING,
        MovementMode.RETREAT_OR_RECOVERY,
        MovementMode.RE_ACCELERATION,
        MovementMode.GOAL_APPROACH_BRAKING,
        MovementMode.OTHER_UNCLASSIFIED,
    ]

    for i in range(100):
        m = modes[i % len(modes)]
        tb.record_step(m)

    tb.finalize()
    d = tb.to_dict()

    assert tb.total_steps == 100
    assert abs(d["total_time_s"] - 5.0) < 1e-6
    assert sum(tb.step_counts.values()) == 100
    assert abs(sum(tb.time_percentages.values()) - 100.0) < 0.1
