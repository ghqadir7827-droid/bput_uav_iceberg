"""test_multidirectional_safety.py - Focused tests for multi-directional safety arbitration."""

import numpy as np
import pytest

from controllers.safety_supervisor import SafetyState, SafetySupervisor, SafetySupervisorConfig
from core.contracts import ActionLimits, DeployableObservation, VelocityAction
from core.geometry import Corridor3D


def _make_observation(ray_distances_m: np.ndarray) -> np.ndarray:
    """Helper creating a 23-float observation vector from 12 ray distances in meters."""
    norm_rays = np.clip(np.asarray(ray_distances_m, dtype=np.float32) / 4.0, 0.0, 1.0)
    obs = np.zeros(23, dtype=np.float32)
    obs[:12] = norm_rays
    # Rel goal: along +X
    obs[12:15] = [1.0, 0.0, 0.0]
    # Body vel: [0.3, 0, 0]
    obs[15:18] = [0.3, 0.0, 0.0]
    # Attitude trig: roll=0, pitch=0
    obs[18:22] = [0.0, 1.0, 0.0, 1.0]
    obs[22] = 0.0
    return obs


def test_left_obstacle_threat_blocks_unsafe_leftward_motion():
    """Case 1: Obstacle threat on the left with an unsafe leftward action."""
    corridor = Corridor3D(width=2.0, height=2.0)
    sup = SafetySupervisor(corridor)

    # Ray 1 (front left 30 deg) and Ray 5 (direct left 90 deg) detect obstacle at 0.20m
    rays = np.full(12, 3.5, dtype=np.float32)
    rays[1] = 0.20
    rays[5] = 0.20
    obs = _make_observation(rays)

    pos = np.array([5.0, 0.0, 1.0]) # Centered in corridor (wall is 1.0m away)
    action = VelocityAction(vx=0.35, vy=0.30, vz=0.0, yaw_rate=0.0) # Unsafe leftward command

    safe_act, state, reason = sup.filter_action(
        proposed_action=action,
        current_position=pos,
        current_velocity=np.zeros(3),
        min_front_range=0.20,
        dt=0.05,
        observation=obs
    )

    # Unsafe leftward velocity must be clamped to 0 or repelled rightward
    assert safe_act.vy <= 0.0
    assert "left_obs" in reason or "Front obstacle" in reason


def test_right_obstacle_threat_blocks_unsafe_rightward_motion():
    """Case 2: Obstacle threat on the right with an unsafe rightward action."""
    corridor = Corridor3D(width=2.0, height=2.0)
    sup = SafetySupervisor(corridor)

    # Ray 2 (front right 30 deg) and Ray 6 (direct right 90 deg) detect obstacle at 0.20m
    rays = np.full(12, 3.5, dtype=np.float32)
    rays[2] = 0.20
    rays[6] = 0.20
    obs = _make_observation(rays)

    pos = np.array([5.0, 0.0, 1.0]) # Centered
    action = VelocityAction(vx=0.35, vy=-0.30, vz=0.0, yaw_rate=0.0) # Unsafe rightward command

    safe_act, state, reason = sup.filter_action(
        proposed_action=action,
        current_position=pos,
        current_velocity=np.zeros(3),
        min_front_range=0.20,
        dt=0.05,
        observation=obs
    )

    # Unsafe rightward velocity must be clamped
    assert safe_act.vy >= 0.0
    assert "right_obs" in reason or "Front obstacle" in reason


def test_front_obstacle_while_proposed_lateral_motion_unsafe():
    """Case 3: Front obstacle while the proposed lateral motion is unsafe."""
    corridor = Corridor3D(width=2.0, height=2.0)
    sup = SafetySupervisor(corridor)

    # Ray 0 (front) = 0.25m, Ray 2 (front right) = 0.22m
    rays = np.full(12, 3.5, dtype=np.float32)
    rays[0] = 0.25
    rays[2] = 0.22
    obs = _make_observation(rays)

    pos = np.array([5.0, 0.0, 1.0])
    action = VelocityAction(vx=0.45, vy=-0.35, vz=0.0, yaw_rate=0.0)

    safe_act, state, reason = sup.filter_action(
        proposed_action=action,
        current_position=pos,
        current_velocity=np.zeros(3),
        min_front_range=0.22,
        dt=0.05,
        observation=obs
    )

    # Forward motion must brake to 0.0
    assert safe_act.vx == 0.0
    # Lateral motion toward right obstacle must be clamped
    assert safe_act.vy >= 0.0
    assert state == SafetyState.OBSTACLE_BRAKE


def test_obstacle_one_side_and_corridor_boundary_other():
    """Case 4: Obstacle on one side and corridor boundary on the other."""
    corridor = Corridor3D(width=2.0, height=2.0, drone_radius=0.15)
    # y = +0.75 -> clearance to left wall = 1.0 - 0.75 - 0.15 = 0.10m (< buffer 0.25m)
    sup = SafetySupervisor(corridor)

    # Obstacle on the right at 0.25m
    rays = np.full(12, 3.5, dtype=np.float32)
    rays[6] = 0.25 # Right obstacle
    obs = _make_observation(rays)

    pos = np.array([5.0, 0.75, 1.0])
    action = VelocityAction(vx=0.30, vy=-0.20, vz=0.0, yaw_rate=0.0)

    safe_act, state, reason = sup.filter_action(
        proposed_action=action,
        current_position=pos,
        current_velocity=np.zeros(3),
        min_front_range=3.0,
        dt=0.05,
        observation=obs
    )

    # Left wall disallows positive vy; right obstacle disallows negative vy
    # Supervisor must NOT push the drone into the right obstacle!
    assert safe_act.vy >= 0.0 or abs(safe_act.vy) < 0.15


def test_simultaneous_obstacle_and_boundary_intervention():
    """Case 5: Simultaneous obstacle and boundary intervention."""
    corridor = Corridor3D(width=2.0, height=2.0, drone_radius=0.15)
    # y = -0.75 -> right wall clearance = 0.10m
    sup = SafetySupervisor(corridor)

    # Front obstacle detected at 0.22m
    rays = np.full(12, 3.5, dtype=np.float32)
    rays[0] = 0.22
    obs = _make_observation(rays)

    pos = np.array([5.0, -0.75, 1.0])
    action = VelocityAction(vx=0.40, vy=-0.25, vz=0.0, yaw_rate=0.0)

    safe_act, state, reason = sup.filter_action(
        proposed_action=action,
        current_position=pos,
        current_velocity=np.zeros(3),
        min_front_range=0.22,
        dt=0.05,
        observation=obs
    )

    assert safe_act.vx == 0.0
    assert safe_act.vy > 0.0 # Repelled inward away from right wall
    assert state == SafetyState.BOUNDARY_AND_OBSTACLE_BRAKE


def test_both_lateral_directions_constrained():
    """Case 6: Both lateral directions constrained."""
    corridor = Corridor3D(width=2.0, height=2.0)
    sup = SafetySupervisor(corridor)

    # Left obstacle at 0.22m, Right obstacle at 0.22m (narrow squeeze)
    rays = np.full(12, 3.5, dtype=np.float32)
    rays[5] = 0.22
    rays[6] = 0.22
    obs = _make_observation(rays)

    pos = np.array([5.0, 0.0, 1.0])
    action = VelocityAction(vx=0.30, vy=0.25, vz=0.0, yaw_rate=0.0)

    safe_act, state, reason = sup.filter_action(
        proposed_action=action,
        current_position=pos,
        current_velocity=np.zeros(3),
        min_front_range=2.0,
        dt=0.05,
        observation=obs
    )

    # When both lateral directions are blocked, lateral velocity must be zeroed
    assert safe_act.vy == 0.0


def test_angled_obstacle_approach_forward_clearance_insufficient():
    """Case 7: Angled obstacle approach where forward clearance alone is insufficient."""
    corridor = Corridor3D(width=2.0, height=2.0)
    sup = SafetySupervisor(corridor)

    # Front center ray 0 is clear (2.5m), but Ray 1 (front left 30 deg) detects obstacle at 0.25m
    rays = np.full(12, 3.5, dtype=np.float32)
    rays[0] = 2.50
    rays[1] = 0.25 # Angled front left obstacle!
    obs = _make_observation(rays)

    pos = np.array([5.0, 0.0, 1.0])
    action = VelocityAction(vx=0.40, vy=0.30, vz=0.0, yaw_rate=0.0) # Steering left toward obstacle

    safe_act, state, reason = sup.filter_action(
        proposed_action=action,
        current_position=pos,
        current_velocity=np.zeros(3),
        min_front_range=2.50, # Even if legacy caller reported only front ray
        dt=0.05,
        observation=obs
    )

    # Multi-directional supervisor must catch the angled threat and clamp leftward motion
    assert safe_act.vy <= 0.0


def test_vertical_threat_constrained():
    """Case 8: Vertical threat when sensor evidence indicates obstacle."""
    corridor = Corridor3D(width=2.0, height=2.0)
    sup = SafetySupervisor(corridor)

    # Ray 10 (zenith ceiling ray) detects obstacle at 0.20m
    rays = np.full(12, 3.5, dtype=np.float32)
    rays[10] = 0.20
    obs = _make_observation(rays)

    pos = np.array([5.0, 0.0, 1.0]) # Z=1.0 (corridor height 2.0, ceiling clearance 0.85m)
    action = VelocityAction(vx=0.30, vy=0.0, vz=0.25, yaw_rate=0.0) # Climbing into overhead obstacle

    safe_act, state, reason = sup.filter_action(
        proposed_action=action,
        current_position=pos,
        current_velocity=np.zeros(3),
        min_front_range=3.0,
        dt=0.05,
        observation=obs
    )

    # Upward velocity must be clamped
    assert safe_act.vz <= 0.0


def test_emergency_reverse_preservation():
    """Case 9: Emergency reverse preservation."""
    corridor = Corridor3D()
    sup = SafetySupervisor(corridor)

    rays = np.full(12, 3.5, dtype=np.float32)
    rays[0] = 0.10 # Closer than 0.15m emergency reverse threshold
    obs = _make_observation(rays)

    pos = np.array([5.0, 0.0, 1.0])
    action = VelocityAction(vx=0.40, vy=0.0, vz=0.0, yaw_rate=0.0)

    safe_act, state, reason = sup.filter_action(
        proposed_action=action,
        current_position=pos,
        current_velocity=np.zeros(3),
        min_front_range=0.10,
        dt=0.05,
        observation=obs
    )

    assert safe_act.vx == -0.10
    assert state == SafetyState.OBSTACLE_BRAKE


def test_no_obstacle_threat_action_unchanged():
    """Case 10: No obstacle threat: ordinary policy action remains unchanged."""
    corridor = Corridor3D(width=2.0, height=2.0)
    sup = SafetySupervisor(corridor)

    # All rays completely clear (3.5m)
    rays = np.full(12, 3.5, dtype=np.float32)
    obs = _make_observation(rays)

    pos = np.array([5.0, 0.0, 1.0]) # Centered
    action = VelocityAction(vx=0.45, vy=0.15, vz=-0.10, yaw_rate=0.20)

    safe_act, state, reason = sup.filter_action(
        proposed_action=action,
        current_position=pos,
        current_velocity=np.zeros(3),
        min_front_range=3.5,
        dt=0.05,
        observation=obs
    )

    assert np.isclose(safe_act.vx, 0.45)
    assert np.isclose(safe_act.vy, 0.15)
    assert np.isclose(safe_act.vz, -0.10)
    assert np.isclose(safe_act.yaw_rate, 0.20)
    assert state == SafetyState.NOMINAL


def test_valid_escape_direction_available_when_supported():
    """Case 11: A valid escape direction remains available when sensor evidence supports it."""
    corridor = Corridor3D(width=2.0, height=2.0)
    sup = SafetySupervisor(corridor)

    # Obstacle on the right (0.20m), Left side wide open (3.5m)
    rays = np.full(12, 3.5, dtype=np.float32)
    rays[2] = 0.20
    rays[6] = 0.20
    obs = _make_observation(rays)

    pos = np.array([5.0, 0.0, 1.0])
    action = VelocityAction(vx=0.30, vy=-0.20, vz=0.0, yaw_rate=0.0)

    safe_act, state, reason = sup.filter_action(
        proposed_action=action,
        current_position=pos,
        current_velocity=np.zeros(3),
        min_front_range=0.20,
        dt=0.05,
        observation=obs
    )

    # Rightward motion prevented; escape to left available
    assert safe_act.vy >= 0.0


def test_output_actions_strictly_within_action_limits():
    """Case 12: Output actions remain within the shared physical action limits."""
    corridor = Corridor3D(width=2.0, height=2.0)
    sup = SafetySupervisor(corridor)

    rays = np.full(12, 0.10, dtype=np.float32) # Extreme proximity everywhere
    obs = _make_observation(rays)

    wild_action = VelocityAction(vx=100.0, vy=-50.0, vz=25.0, yaw_rate=-30.0)
    pos = np.array([5.0, 0.85, 1.85])

    safe_act, _, _ = sup.filter_action(
        proposed_action=wild_action,
        current_position=pos,
        current_velocity=np.zeros(3),
        min_front_range=0.10,
        dt=0.05,
        observation=obs
    )

    limits = ActionLimits()
    assert limits.min_fwd_vel_mps <= safe_act.vx <= limits.max_fwd_vel_mps
    assert -limits.max_lat_vel_mps <= safe_act.vy <= limits.max_lat_vel_mps
    assert -limits.max_vert_vel_mps <= safe_act.vz <= limits.max_vert_vel_mps
    assert -limits.max_yaw_rate_radps <= safe_act.yaw_rate <= limits.max_yaw_rate_radps
