"""test_safety_supervisor.py - Unit Tests for Deterministic Safety Supervisor."""

import numpy as np
import pytest

from controllers.safety_supervisor import SafetyState, SafetySupervisor, SafetySupervisorConfig
from core.contracts import ActionLimits, VelocityAction
from core.geometry import Corridor3D


def test_nominal_action_passthrough():
    """When inside safe boundaries with clear front distance, actions should pass through."""
    corridor = Corridor3D(width=2.4, height=2.0, z_floor=0.0, z_ceiling=2.0)
    sup = SafetySupervisor(corridor)

    action = VelocityAction(vx=0.40, vy=0.10, vz=0.05, yaw_rate=0.10)
    pos = np.array([5.0, 0.0, 1.0]) # Centered
    vel = np.array([0.4, 0.1, 0.0])

    safe_act, state, _ = sup.filter_action(
        proposed_action=action,
        current_position=pos,
        current_velocity=vel,
        min_front_range=2.5,
        dt=0.05,
        is_localization_healthy=True,
        telemetry_age_s=0.02
    )

    assert state == SafetyState.NOMINAL
    assert np.isclose(safe_act.vx, 0.40)
    assert np.isclose(safe_act.vy, 0.10)
    assert np.isclose(safe_act.vz, 0.05)


def test_boundary_buffer_repulsion():
    """When near the left wall, positive vy must be clamped and inward repulsion applied."""
    corridor = Corridor3D(width=2.0, height=2.0, drone_radius=0.15)
    # y_max = 1.0. Place drone at y = 0.75 -> clearance = 1.0 - 0.75 - 0.15 = 0.10m (< buffer 0.25m)
    sup = SafetySupervisor(corridor, config=SafetySupervisorConfig(boundary_buffer_m=0.25, repulsion_gain=2.0))

    action = VelocityAction(vx=0.30, vy=0.20, vz=0.0, yaw_rate=0.0) # Proposing to fly further left
    pos = np.array([5.0, 0.75, 1.0])
    vel = np.array([0.3, 0.1, 0.0])

    safe_act, state, _ = sup.filter_action(
        proposed_action=action,
        current_position=pos,
        current_velocity=vel,
        min_front_range=3.0,
        dt=0.05
    )

    assert state == SafetyState.BOUNDARY_INTERVENTION
    # vy must be negative (inward toward center)
    assert safe_act.vy < 0.0


def test_obstacle_emergency_brake():
    """When front range drops below critical distance, forward speed must be clamped to 0."""
    corridor = Corridor3D()
    sup = SafetySupervisor(corridor, config=SafetySupervisorConfig(critical_obstacle_dist_m=0.30))

    action = VelocityAction(vx=0.50, vy=0.0, vz=0.0, yaw_rate=0.0)
    pos = np.array([5.0, 0.0, 1.0])
    vel = np.array([0.4, 0.0, 0.0])

    safe_act, state, _ = sup.filter_action(
        proposed_action=action,
        current_position=pos,
        current_velocity=vel,
        min_front_range=0.22, # Less than 0.30m
        dt=0.05
    )

    assert state == SafetyState.OBSTACLE_BRAKE
    assert safe_act.vx == 0.0


def test_loss_of_localization_failsafe():
    """When localization is unhealthy or stale, safety supervisor must execute controlled deceleration."""
    corridor = Corridor3D()
    cfg = SafetySupervisorConfig(max_stale_duration_s=0.30, deceleration_limit_mps2=1.0)
    sup = SafetySupervisor(corridor, config=cfg)

    action = VelocityAction(vx=0.50, vy=0.20, vz=0.0, yaw_rate=0.2)
    pos = np.array([5.0, 0.0, 1.0])
    vel = np.array([0.40, 0.20, 0.0]) # Currently flying forward at 0.40 m/s

    safe_act, state, _ = sup.filter_action(
        proposed_action=action,
        current_position=pos,
        current_velocity=vel,
        min_front_range=2.0,
        dt=0.10,
        is_localization_healthy=False, # Unhealthy!
        telemetry_age_s=0.50 # Stale!
    )

    assert state == SafetyState.LOCALIZATION_FAILSAFE
    # Velocity should decelerate: 0.40 - (1.0 * 0.1) = 0.30 m/s
    assert np.isclose(safe_act.vx, 0.30, atol=1e-3)
    assert safe_act.yaw_rate == 0.0


def test_obstacle_braking_permits_boundary_checks():
    """Obstacle braking must not bypass boundary enforcement; both must execute in the same cycle."""
    corridor = Corridor3D(width=2.0, height=2.0, drone_radius=0.15)
    # y_min = -1.0. Place drone at y = -0.75 -> clearance = -0.75 - (-1.0) - 0.15 = 0.10m (< buffer 0.25m)
    sup = SafetySupervisor(corridor, config=SafetySupervisorConfig(boundary_buffer_m=0.25, critical_obstacle_dist_m=0.30))

    # Propose forward progress and lateral movement further right into the right wall
    action = VelocityAction(vx=0.50, vy=-0.30, vz=0.0, yaw_rate=0.0)
    pos = np.array([5.0, -0.75, 1.0])
    vel = np.array([0.4, -0.1, 0.0])

    safe_act, state, reason = sup.filter_action(
        proposed_action=action,
        current_position=pos,
        current_velocity=vel,
        min_front_range=0.20, # Obstacle critical brake triggered!
        dt=0.05
    )

    # 1. Forward velocity must be clamped to 0
    assert safe_act.vx == 0.0
    # 2. Right wall boundary check MUST have executed: negative vy disallowed, inward repulsion applied
    assert safe_act.vy > 0.0
    # 3. State must report unified intervention
    assert state == SafetyState.BOUNDARY_AND_OBSTACLE_BRAKE
    # 4. Reason must mention both constraints
    assert "Front obstacle detected" in reason
    assert "right=" in reason


def test_obstacle_ahead_blocks_unsafe_lateral_movement_toward_boundary():
    """When an obstacle is ahead, lateral commands heading into a boundary must be blocked."""
    corridor = Corridor3D(width=2.0, height=2.0, drone_radius=0.15)
    # y_max = 1.0. Place drone at y = 0.75 -> clearance = 0.10m to left wall
    sup = SafetySupervisor(corridor, config=SafetySupervisorConfig(boundary_buffer_m=0.25, critical_obstacle_dist_m=0.30))

    # Propose forward progress and lateral movement into left wall
    action = VelocityAction(vx=0.45, vy=0.35, vz=0.0, yaw_rate=0.0)
    pos = np.array([5.0, 0.75, 1.0])
    vel = np.array([0.3, 0.1, 0.0])

    safe_act, state, reason = sup.filter_action(
        proposed_action=action,
        current_position=pos,
        current_velocity=vel,
        min_front_range=0.25,
        dt=0.05
    )

    assert safe_act.vx == 0.0
    # Left wall must disallow positive vy and repel right (negative)
    assert safe_act.vy < 0.0
    assert state == SafetyState.BOUNDARY_AND_OBSTACLE_BRAKE


def test_unsafe_vertical_movement_constrained():
    """Ceiling and floor boundaries must be enforced during obstacle encounters."""
    corridor = Corridor3D(width=2.0, height=2.0, z_floor=0.0, z_ceiling=2.0, drone_radius=0.15)
    sup = SafetySupervisor(corridor, config=SafetySupervisorConfig(boundary_buffer_m=0.25, critical_obstacle_dist_m=0.30))

    # Near ceiling: z = 1.75 -> d_ceil = 2.0 - 1.75 - 0.15 = 0.10m (< 0.25m)
    pos_ceil = np.array([5.0, 0.0, 1.75])
    act_ceil = VelocityAction(vx=0.40, vy=0.0, vz=0.25, yaw_rate=0.0)
    safe_ceil, state_ceil, _ = sup.filter_action(act_ceil, pos_ceil, np.zeros(3), min_front_range=0.25, dt=0.05)
    assert safe_ceil.vx == 0.0
    assert safe_ceil.vz < 0.0 # Repelled downward from ceiling
    assert state_ceil == SafetyState.BOUNDARY_AND_OBSTACLE_BRAKE

    # Near floor: z = 0.20 -> d_floor = 0.20 - 0.0 - 0.15 = 0.05m (< 0.25m)
    pos_floor = np.array([5.0, 0.0, 0.20])
    act_floor = VelocityAction(vx=0.40, vy=0.0, vz=-0.25, yaw_rate=0.0)
    safe_floor, state_floor, _ = sup.filter_action(act_floor, pos_floor, np.zeros(3), min_front_range=0.25, dt=0.05)
    assert safe_floor.vx == 0.0
    assert safe_floor.vz > 0.0 # Repelled upward from floor
    assert state_floor == SafetyState.BOUNDARY_AND_OBSTACLE_BRAKE


def test_safe_escape_movement_preserved():
    """An escape command away from the violating boundary must not be blocked."""
    corridor = Corridor3D(width=2.0, height=2.0, drone_radius=0.15)
    # Drone near right wall (y = -0.75, d_right = 0.10m < 0.25m)
    sup = SafetySupervisor(corridor, config=SafetySupervisorConfig(boundary_buffer_m=0.25, critical_obstacle_dist_m=0.30))

    # Policy recognizes right wall and commands safe escape to the LEFT (vy = +0.25 m/s)
    escape_action = VelocityAction(vx=0.40, vy=0.25, vz=0.0, yaw_rate=0.0)
    pos = np.array([5.0, -0.75, 1.0])

    safe_act, state, _ = sup.filter_action(
        proposed_action=escape_action,
        current_position=pos,
        current_velocity=np.zeros(3),
        min_front_range=0.25,
        dt=0.05
    )

    assert safe_act.vx == 0.0 # Forward stopped
    # Escape command preserved and reinforced by inward repulsion: vy >= 0.25
    assert safe_act.vy >= 0.25
    assert safe_act.vy <= sup.limits.max_lat_vel_mps
    assert state == SafetyState.BOUNDARY_AND_OBSTACLE_BRAKE


def test_boundary_intervention_throttles_unsafe_forward_transit():
    """When boundary buffer is penetrated and an obstacle is ahead, forward speed must be throttled."""
    corridor = Corridor3D(width=2.0, height=2.0, drone_radius=0.15)
    # y = -0.75 -> clearance = 0.10m (< 0.25m)
    sup = SafetySupervisor(corridor, config=SafetySupervisorConfig(boundary_buffer_m=0.25, obstacle_caution_dist_m=0.80))

    # Policy commands maximum forward velocity
    action = VelocityAction(vx=0.60, vy=0.0, vz=0.0, yaw_rate=0.0)
    pos = np.array([5.0, -0.75, 1.0])

    safe_act, state, reason = sup.filter_action(
        proposed_action=action,
        current_position=pos,
        current_velocity=np.zeros(3),
        min_front_range=0.50, # Obstacle in caution range (< 0.80m)
        dt=0.05
    )

    # Forward velocity must be throttled below full speed
    assert safe_act.vx < 0.20
    assert safe_act.vy > 0.0 # Repelled inward away from right wall
    assert state == SafetyState.BOUNDARY_AND_OBSTACLE_BRAKE


def test_clear_corridor_safe_progress_permitted():
    """When the corridor is clear ahead, boundary intervention permits safe forward progress."""
    corridor = Corridor3D(width=2.0, height=2.0, drone_radius=0.15)
    sup = SafetySupervisor(corridor, config=SafetySupervisorConfig(boundary_buffer_m=0.25))

    # Slightly near left wall (y = 0.70 -> d_left = 0.15m < 0.25m), but corridor clear (front = 3.5m)
    action = VelocityAction(vx=0.50, vy=0.0, vz=0.0, yaw_rate=0.0)
    pos = np.array([5.0, 0.70, 1.0])

    safe_act, state, _ = sup.filter_action(
        proposed_action=action,
        current_position=pos,
        current_velocity=np.zeros(3),
        min_front_range=3.50, # Clear
        dt=0.05
    )

    # Forward transit remains possible
    assert safe_act.vx > 0.0
    # Inward repulsion applied
    assert safe_act.vy < 0.0
    assert state == SafetyState.BOUNDARY_INTERVENTION


def test_action_outputs_strictly_finite_and_bounded():
    """All output actions must remain finite and strictly clamped within physical limits."""
    corridor = Corridor3D()
    sup = SafetySupervisor(corridor)

    wild_action = VelocityAction(vx=100.0, vy=-50.0, vz=25.0, yaw_rate=-30.0)
    pos = np.array([5.0, -0.90, 1.95]) # Near right wall and ceiling

    safe_act, _, _ = sup.filter_action(
        proposed_action=wild_action,
        current_position=pos,
        current_velocity=np.zeros(3),
        min_front_range=0.10,
        dt=0.05
    )

    assert np.isfinite(safe_act.vx)
    assert np.isfinite(safe_act.vy)
    assert np.isfinite(safe_act.vz)
    assert np.isfinite(safe_act.yaw_rate)

    assert -sup.limits.max_lat_vel_mps <= safe_act.vy <= sup.limits.max_lat_vel_mps
    assert -sup.limits.max_vert_vel_mps <= safe_act.vz <= sup.limits.max_vert_vel_mps
    assert -sup.limits.max_yaw_rate_radps <= safe_act.yaw_rate <= sup.limits.max_yaw_rate_radps


def test_emergency_reverse_behavior():
    """When an obstacle is closer than emergency_reverse_dist_m, slow reverse is commanded."""
    corridor = Corridor3D()
    sup = SafetySupervisor(corridor, config=SafetySupervisorConfig(emergency_reverse_dist_m=0.15))

    action = VelocityAction(vx=0.40, vy=0.0, vz=0.0, yaw_rate=0.0)
    pos = np.array([5.0, 0.0, 1.0])

    safe_act, state, reason = sup.filter_action(
        proposed_action=action,
        current_position=pos,
        current_velocity=np.zeros(3),
        min_front_range=0.10, # Very close!
        dt=0.05
    )

    assert state == SafetyState.OBSTACLE_BRAKE
    assert safe_act.vx == -0.10
    assert "Reversing" in reason

