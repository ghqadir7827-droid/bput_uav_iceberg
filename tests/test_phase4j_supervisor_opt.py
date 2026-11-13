"""test_phase4j_supervisor_opt.py - Unit tests for Phase 4J Isolated Supervisor Implementation."""

import numpy as np
import pytest

from controllers.safety_supervisor import SafetyState
from core.contracts import VelocityAction
from core.geometry import Corridor3D
from experiments.phase4j.isolated_supervisor_opt import (
    CANONICAL_BOUNDARY_BUFFER_M,
    CANONICAL_CRITICAL_DIST_M,
    CANONICAL_REVERSE_DIST_M,
    IsolatedOptimizedSupervisor,
)


def test_isolated_supervisor_valid_instantiation():
    """Verifies that valid caution distances initialize properly with invariant thresholds."""
    corridor = Corridor3D.get_10m_benchmark()
    for d_caut in [0.80, 0.70, 0.60, 0.55, 0.50, 0.45]:
        sup = IsolatedOptimizedSupervisor(corridor=corridor, caution_distance_m=d_caut)
        assert sup.caution_distance_m == d_caut
        assert sup.config.critical_obstacle_dist_m == CANONICAL_CRITICAL_DIST_M
        assert sup.config.emergency_reverse_dist_m == CANONICAL_REVERSE_DIST_M
        assert sup.config.boundary_buffer_m == CANONICAL_BOUNDARY_BUFFER_M


def test_isolated_supervisor_invalid_rejection():
    """Verifies that parameter values violating physical margin are rejected."""
    corridor = Corridor3D.get_10m_benchmark()

    # <= critical threshold
    with pytest.raises(ValueError):
        IsolatedOptimizedSupervisor(corridor=corridor, caution_distance_m=0.30)

    with pytest.raises(ValueError):
        IsolatedOptimizedSupervisor(corridor=corridor, caution_distance_m=0.20)

    # Margin < 0.05m
    with pytest.raises(ValueError):
        IsolatedOptimizedSupervisor(corridor=corridor, caution_distance_m=0.32)


def test_isolated_supervisor_throttling_profile():
    """Verifies that throttling behavior activates strictly at the configured caution distance."""
    corridor = Corridor3D.get_10m_benchmark()
    pos = np.array([2.0, 0.0, 1.0])
    vel = np.array([0.5, 0.0, 0.0])
    proposed = VelocityAction(vx=0.60, vy=0.0, vz=0.0, yaw_rate=0.0)

    # 1. Under d_caut = 0.55m, front range 0.70m should be UNCONSTRAINED
    sup_055 = IsolatedOptimizedSupervisor(corridor=corridor, caution_distance_m=0.55)
    safe_act, state, _ = sup_055.filter_action(
        proposed_action=proposed,
        current_position=pos,
        current_velocity=vel,
        min_front_range=0.70,
        dt=0.05,
    )
    assert state == SafetyState.NOMINAL
    assert abs(safe_act.vx - 0.60) < 1e-4

    # 2. Under d_caut = 0.80m, front range 0.70m should be THROTTLED
    sup_080 = IsolatedOptimizedSupervisor(corridor=corridor, caution_distance_m=0.80)
    safe_act_80, state_80, _ = sup_080.filter_action(
        proposed_action=proposed,
        current_position=pos,
        current_velocity=vel,
        min_front_range=0.70,
        dt=0.05,
    )
    assert state_80 == SafetyState.OBSTACLE_BRAKE
    assert safe_act_80.vx < 0.60

    # 3. Critical brake must trigger at 0.25m (< 0.30m) regardless of caution distance
    safe_crit, state_crit, _ = sup_055.filter_action(
        proposed_action=proposed,
        current_position=pos,
        current_velocity=vel,
        min_front_range=0.25,
        dt=0.05,
    )
    assert state_crit == SafetyState.OBSTACLE_BRAKE
    assert safe_crit.vx == 0.0
