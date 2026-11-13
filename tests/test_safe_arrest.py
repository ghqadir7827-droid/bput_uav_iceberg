"""test_safe_arrest.py - Unit tests for Safe-Arrest State Machine and Infeasibility Classification."""

import numpy as np
import pytest

from controllers.safe_arrest import SafeArrestConfig, SafeArrestState, SafeArrestStateMachine
from core.contracts import DeployableObservation


def _make_blocked_obs(c_front: float, c_left: float, c_right: float) -> np.ndarray:
    """Helper creating 16-ray observation vector with specified directional clearances."""
    rays = np.full(16, 1.0, dtype=np.float32)
    # Front rays (0, 1, 2)
    rays[0] = c_front / 4.0
    rays[1] = c_front / 4.0
    rays[2] = c_front / 4.0
    # Left rays (1, 3, 5, 12, 14)
    rays[3] = c_left / 4.0
    rays[5] = c_left / 4.0
    rays[12] = c_left / 4.0
    rays[14] = c_left / 4.0
    # Right rays (2, 4, 6, 13, 15)
    rays[4] = c_right / 4.0
    rays[6] = c_right / 4.0
    rays[13] = c_right / 4.0
    rays[15] = c_right / 4.0

    obs = np.zeros(27, dtype=np.float32)
    obs[:16] = rays
    return obs


def test_nominal_approach_remains_approaching():
    """Verifies that clear space keeps the state machine in APPROACHING state."""
    sm = SafeArrestStateMachine()
    clear_obs = np.ones(27, dtype=np.float32) # All rays at 4.0m

    pos = np.array([2.0, 0.0, 1.2])
    vel = np.array([0.4, 0.0, 0.0])

    state, terminate, diag = sm.update(
        current_position=pos,
        current_velocity=vel,
        observation=clear_obs,
        min_obstacle_clearance=2.5,
        min_boundary_clearance=0.8,
    )
    assert state == SafeArrestState.APPROACHING
    assert terminate is False
    assert diag.suppress_lateral_guidance is False


def test_impassable_barrier_triggers_verified_infeasible():
    """Verifies that an impassable barrier (gap < 0.65m on all sides) transitions to VERIFIED_INFEASIBLE."""
    sm = SafeArrestStateMachine()
    blocked_obs = _make_blocked_obs(c_front=0.50, c_left=0.55, c_right=0.55)

    pos = np.array([6.5, 0.0, 1.2])
    vel = np.array([0.02, 0.0, 0.0])

    # Step 1: APPROACHING -> BLOCKED
    state, _, _ = sm.update(pos, vel, blocked_obs, min_obstacle_clearance=0.30, min_boundary_clearance=0.55)
    assert state == SafeArrestState.BLOCKED

    # Step 2: BLOCKED -> ASSESSING_TRAVERSABILITY
    state, _, _ = sm.update(pos, vel, blocked_obs, min_obstacle_clearance=0.30, min_boundary_clearance=0.55)
    assert state == SafeArrestState.ASSESSING_TRAVERSABILITY

    # Accumulate 10 consecutive steps of impassable barrier evidence
    for _ in range(10):
        state, _, diag = sm.update(pos, vel, blocked_obs, min_obstacle_clearance=0.30, min_boundary_clearance=0.55)

    assert state == SafeArrestState.VERIFIED_INFEASIBLE

    # Next update enters SAFE_ARREST with lateral guidance suppressed
    state, _, diag = sm.update(pos, vel, blocked_obs, min_obstacle_clearance=0.30, min_boundary_clearance=0.55)
    assert state == SafeArrestState.SAFE_ARREST
    assert diag.suppress_lateral_guidance is True


def test_stable_dwell_verification_succeeds():
    """Verifies that 40 steps (2.0s) of stable hover completes safe arrest."""
    sm = SafeArrestStateMachine()
    blocked_obs = _make_blocked_obs(c_front=0.50, c_left=0.55, c_right=0.55)

    pos_anchor = np.array([6.5, 0.0, 1.2])
    vel_hover = np.array([0.01, -0.01, 0.0])

    # Drive to SAFE_ARREST
    for _ in range(15):
        sm.update(pos_anchor, vel_hover, blocked_obs, min_obstacle_clearance=0.30, min_boundary_clearance=0.55)
    assert sm.state == SafeArrestState.SAFE_ARREST

    # Dwell for 40 steps with small noise well within limits (speed <= 0.05, drift <= 0.10, c >= 0.05)
    terminated = False
    for i in range(40):
        drift_noise = np.array([0.01 * np.sin(i), 0.01 * np.cos(i), 0.0])
        current_pos = pos_anchor + drift_noise
        state, term, diag = sm.update(
            current_pos,
            vel_hover,
            blocked_obs,
            min_obstacle_clearance=0.28,
            min_boundary_clearance=0.55
        )
        if term:
            terminated = True
            break

    assert terminated is True
    assert diag.is_stable is True
    assert sm.dwell_counter >= 40


def test_arrest_fails_if_clearance_breached():
    """Verifies that if clearance drops <= 0.05m during dwell, state becomes UNSAFE_ARREST."""
    sm = SafeArrestStateMachine()
    blocked_obs = _make_blocked_obs(c_front=0.50, c_left=0.55, c_right=0.55)
    pos_anchor = np.array([6.5, 0.0, 1.2])
    vel_hover = np.array([0.01, 0.0, 0.0])

    # Drive to SAFE_ARREST
    for _ in range(15):
        sm.update(pos_anchor, vel_hover, blocked_obs, min_obstacle_clearance=0.30, min_boundary_clearance=0.55)

    # Clearance breached (< 0.05m)
    state, term, diag = sm.update(
        pos_anchor,
        vel_hover,
        blocked_obs,
        min_obstacle_clearance=0.03, # Breached!
        min_boundary_clearance=0.55
    )
    assert state == SafeArrestState.UNSAFE_ARREST
    assert term is True
    assert diag.is_stable is False
