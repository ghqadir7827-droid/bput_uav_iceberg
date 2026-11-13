"""test_phase4jf_lateral_guidance.py - Unit tests for Phase 4J-F Proactive Lateral Guidance Supervisor."""

import numpy as np
import pytest

from controllers.safety_supervisor import SafetyState
from core.contracts import ActionLimits, VelocityAction
from core.geometry import Corridor3D
from experiments.phase4j.phase4jf_lateral_guidance_20261029_182017.experimental_supervisor import (
    GuidanceState,
    IsolatedLateralGuidanceSupervisor,
)


@pytest.fixture
def corridor():
    return Corridor3D.get_10m_benchmark()


@pytest.fixture
def guidance_supervisor(corridor):
    return IsolatedLateralGuidanceSupervisor(
        corridor=corridor,
        mode="lateral_guidance",
        caution_distance_m=0.50,
    )


@pytest.fixture
def baseline_supervisor(corridor):
    return IsolatedLateralGuidanceSupervisor(
        corridor=corridor,
        mode="baseline",
        caution_distance_m=0.50,
    )


def test_observation_contract_compatibility(guidance_supervisor):
    """Verifies that the supervisor accepts 23-float observation vectors."""
    obs = np.ones(23, dtype=np.float32) * 0.5
    action = VelocityAction(vx=0.5, vy=0.0, vz=0.0, yaw_rate=0.0)
    pos = np.array([2.5, 0.0, 1.0])
    vel = np.array([0.5, 0.0, 0.0])

    safe_act, state, reason = guidance_supervisor.filter_action(
        proposed_action=action,
        current_position=pos,
        current_velocity=vel,
        min_front_range=2.0,
        dt=0.05,
        observation=obs,
    )
    assert isinstance(safe_act, VelocityAction)
    assert state == SafetyState.NOMINAL


def test_blocked_progress_detection(guidance_supervisor):
    """Verifies that stagnation over rolling window triggers blocked-progress detection."""
    pos = np.array([2.5, 0.0, 1.0])
    vel = np.array([0.02, 0.01, 0.0])  # Near standstill

    # Feed 29 steps: should not trigger yet
    for _ in range(29):
        is_blocked = guidance_supervisor.detect_blocked_progress(front_dist=0.35, current_pos=pos, current_vel=vel)
        assert not is_blocked

    # 30th step: should trigger
    is_blocked = guidance_supervisor.detect_blocked_progress(front_dist=0.35, current_pos=pos, current_vel=vel)
    assert is_blocked


def test_left_side_preference(guidance_supervisor):
    """Verifies that Left (+Y) is selected when left opening is wider."""
    obs = np.ones(23, dtype=np.float32)
    obs[1] = 0.8  # Front-Left = 3.2m
    obs[2] = 0.2  # Front-Right = 0.8m (obstacle on right)
    pos = np.array([2.5, 0.0, 1.0])  # Centered in corridor

    direction, reason = guidance_supervisor.evaluate_lateral_openings(obs, pos)
    assert direction == +1
    assert "Left opening superior" in reason


def test_right_side_preference(guidance_supervisor):
    """Verifies that Right (-Y) is selected when right opening is wider."""
    obs = np.ones(23, dtype=np.float32)
    obs[1] = 0.2  # Front-Left = 0.8m (obstacle on left)
    obs[2] = 0.8  # Front-Right = 3.2m
    pos = np.array([2.5, 0.0, 1.0])  # Centered in corridor

    direction, reason = guidance_supervisor.evaluate_lateral_openings(obs, pos)
    assert direction == -1
    assert "Right opening superior" in reason


def test_deterministic_tie_breaking(guidance_supervisor):
    """Verifies deterministic tie-breaking under symmetric obstacle encounters."""
    obs = np.ones(23, dtype=np.float32)
    obs[1] = 0.3  # Symmetric 1.2m
    obs[2] = 0.3  # Symmetric 1.2m

    # If centered at Y=0.0: tie-breaks left (+1)
    dir_center, _ = guidance_supervisor.evaluate_lateral_openings(obs, np.array([2.5, 0.0, 1.0]))
    assert dir_center == +1

    # If positioned at Y=+0.2: tie-breaks right (-1) toward center
    dir_right, _ = guidance_supervisor.evaluate_lateral_openings(obs, np.array([2.5, 0.2, 1.0]))
    assert dir_right == -1


def test_commitment_persistence(guidance_supervisor):
    """Verifies that guidance commitment persists over multiple steps."""
    obs = np.ones(23, dtype=np.float32)
    obs[1] = 0.8; obs[2] = 0.2
    pos = np.array([2.5, 0.0, 1.0])
    vel = np.array([0.01, 0.0, 0.0])
    action = VelocityAction(vx=0.5, vy=0.0, vz=0.0, yaw_rate=0.0)

    # Trigger deadlock
    for _ in range(30):
        guidance_supervisor.filter_action(action, pos, vel, min_front_range=0.35, dt=0.05, observation=obs)

    assert guidance_supervisor.commit_dir == +1
    assert guidance_supervisor.commit_remaining_steps > 0
    assert guidance_supervisor.guidance_state == GuidanceState.GUIDANCE_ACTIVE_LEFT


def test_abort_when_wall_limit_reached(guidance_supervisor):
    """Verifies that lateral guidance aborts before violating corridor wall buffer."""
    obs = np.ones(23, dtype=np.float32)
    obs[1] = 0.8; obs[2] = 0.2
    pos = np.array([2.5, 0.0, 1.0])
    vel = np.array([0.01, 0.0, 0.0])
    action = VelocityAction(vx=0.5, vy=0.0, vz=0.0, yaw_rate=0.0)

    # Trigger left commitment
    for _ in range(30):
        guidance_supervisor.filter_action(action, pos, vel, min_front_range=0.35, dt=0.05, observation=obs)

    assert guidance_supervisor.commit_dir == +1

    # Move drone near left wall (y = 0.60 -> wall clearance = 1.0 - 0.60 - 0.15 = 0.25m <= wall_abort 0.26m)
    pos_near_wall = np.array([2.5, 0.60, 1.0])
    safe_act, _, reason = guidance_supervisor.filter_action(action, pos_near_wall, vel, min_front_range=0.35, dt=0.05, observation=obs)

    # Commitment should be aborted
    assert guidance_supervisor.commit_dir == 0
    assert "Left Wall Limit" in reason


def test_emergency_braking_precedence(guidance_supervisor):
    """Verifies that critical obstacle emergency braking strictly overrides forward crawl guidance."""
    obs = np.ones(23, dtype=np.float32)
    obs[1] = 0.8; obs[2] = 0.2
    pos = np.array([2.5, 0.0, 1.0])
    vel = np.array([0.01, 0.0, 0.0])
    action = VelocityAction(vx=0.5, vy=0.0, vz=0.0, yaw_rate=0.0)

    # Trigger commitment
    for _ in range(30):
        guidance_supervisor.filter_action(action, pos, vel, min_front_range=0.35, dt=0.05, observation=obs)

    # If front obstacle drops below critical distance 0.30m (e.g. 0.25m)
    safe_act, state, reason = guidance_supervisor.filter_action(action, pos, vel, min_front_range=0.25, dt=0.05, observation=obs)

    # Forward velocity MUST be clamped to 0.0 despite guidance crawl
    assert safe_act.vx == 0.0
    assert state == SafetyState.OBSTACLE_BRAKE
    assert "Forward brake" in reason


def test_conservative_fallback_when_both_sides_blocked(guidance_supervisor):
    """Verifies fallback when both lateral paths are constrained."""
    obs = np.zeros(23, dtype=np.float32)  # All rays at zero distance
    pos = np.array([2.5, 0.0, 1.0])
    direction, reason = guidance_supervisor.evaluate_lateral_openings(obs, pos)
    assert direction == 0
    assert "Fallback" in reason


def test_reset_state_isolation(guidance_supervisor):
    """Verifies that reset_state completely purges internal history and commitments."""
    obs = np.ones(23, dtype=np.float32)
    pos = np.array([2.5, 0.0, 1.0])
    vel = np.array([0.01, 0.0, 0.0])
    action = VelocityAction(vx=0.5, vy=0.0, vz=0.0, yaw_rate=0.0)

    for _ in range(30):
        guidance_supervisor.filter_action(action, pos, vel, min_front_range=0.35, dt=0.05, observation=obs)

    assert len(guidance_supervisor.x_history) > 0
    assert guidance_supervisor.commit_dir != 0

    guidance_supervisor.reset_state()
    assert len(guidance_supervisor.x_history) == 0
    assert guidance_supervisor.commit_dir == 0
    assert guidance_supervisor.commit_remaining_steps == 0
    assert guidance_supervisor.guidance_state == GuidanceState.IDLE

