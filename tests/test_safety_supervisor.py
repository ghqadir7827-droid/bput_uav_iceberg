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
