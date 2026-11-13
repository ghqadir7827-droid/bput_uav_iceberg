"""test_contracts.py - Verification of Observation & Action Interfaces."""

import numpy as np
import pytest

from core.contracts import (
    ActionLimits,
    DeployableObservation,
    PrivilegedState,
    SensorConfig,
    VelocityAction,
)


def test_deployable_observation_shape():
    """Verifies that the deployable student observation vector is exactly 23 floats."""
    cfg = SensorConfig()
    obs = DeployableObservation(
        distance_rays=np.ones(cfg.num_rays, dtype=np.float32) * 0.5,
        rel_goal_direction_body=np.array([1.0, 0.0, 0.0], dtype=np.float32),
        body_velocity=np.array([0.2, 0.0, 0.0], dtype=np.float32),
        attitude_trig=np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float32), # sin/cos roll & pitch
        yaw_rate=0.05
    )
    vec = obs.to_vector()
    assert vec.shape == (23,), f"Expected shape (23,), got {vec.shape}"
    assert vec.dtype == np.float32


def test_velocity_action_roundtrip():
    """Verifies conversion between physical velocity commands and normalized [-1, 1] actions."""
    limits = ActionLimits()
    act = VelocityAction(vx=0.30, vy=-0.20, vz=0.15, yaw_rate=0.25)
    norm = act.to_normalized(limits)

    assert np.all(norm >= -1.0) and np.all(norm <= 1.0)

    recovered = VelocityAction.from_normalized(norm, limits)
    assert np.isclose(act.vx, recovered.vx, atol=1e-5)
    assert np.isclose(act.vy, recovered.vy, atol=1e-5)
    assert np.isclose(act.vz, recovered.vz, atol=1e-5)
    assert np.isclose(act.yaw_rate, recovered.yaw_rate, atol=1e-5)


def test_action_clamping():
    """Verifies that out-of-bound actions are clamped properly."""
    limits = ActionLimits()
    extreme_norm = np.array([5.0, -10.0, 2.5, -4.0], dtype=np.float32)
    act = VelocityAction.from_normalized(extreme_norm, limits)

    assert np.isclose(act.vx, limits.max_fwd_vel_mps)
    assert np.isclose(act.vy, -limits.max_lat_vel_mps)
    assert np.isclose(act.vz, limits.max_vert_vel_mps)
    assert np.isclose(act.yaw_rate, -limits.max_yaw_rate_radps)

    # Extreme negative forward norm (-5.0) should clamp to min_fwd_vel_mps (-0.20)
    extreme_neg_norm = np.array([-5.0, 0.0, 0.0, 0.0], dtype=np.float32)
    act_neg = VelocityAction.from_normalized(extreme_neg_norm, limits)
    assert np.isclose(act_neg.vx, limits.min_fwd_vel_mps)


def test_normalization_roundtrips_comprehensive():
    """Verifies round-trip accuracy across all dimensions and key operating setpoints."""
    limits = ActionLimits()

    # Forward velocity test points including min, emergency reverse, stop, crawl, and max
    vx_points = [limits.min_fwd_vel_mps, -0.10, -0.05, 0.00, 0.15, 0.30, 0.45, limits.max_fwd_vel_mps]
    for vx in vx_points:
        act = VelocityAction(vx=vx, vy=0.0, vz=0.0, yaw_rate=0.0)
        norm = act.to_normalized(limits)
        assert -1.0 <= norm[0] <= 1.0, f"Normalized vx={norm[0]} out of [-1, 1]"
        rec = VelocityAction.from_normalized(norm, limits)
        assert np.isclose(rec.vx, vx, atol=1e-5), f"vx round-trip mismatch: {vx} vs {rec.vx}"

    # Lateral velocity test points
    vy_points = [-limits.max_lat_vel_mps, -0.20, 0.00, 0.20, limits.max_lat_vel_mps]
    for vy in vy_points:
        act = VelocityAction(vx=0.20, vy=vy, vz=0.0, yaw_rate=0.0)
        norm = act.to_normalized(limits)
        assert -1.0 <= norm[1] <= 1.0
        rec = VelocityAction.from_normalized(norm, limits)
        assert np.isclose(rec.vy, vy, atol=1e-5)

    # Vertical velocity test points
    vz_points = [-limits.max_vert_vel_mps, -0.10, 0.00, 0.15, limits.max_vert_vel_mps]
    for vz in vz_points:
        act = VelocityAction(vx=0.20, vy=0.0, vz=vz, yaw_rate=0.0)
        norm = act.to_normalized(limits)
        assert -1.0 <= norm[2] <= 1.0
        rec = VelocityAction.from_normalized(norm, limits)
        assert np.isclose(rec.vz, vz, atol=1e-5)

    # Yaw rate test points
    yr_points = [-limits.max_yaw_rate_radps, -0.25, 0.00, 0.25, limits.max_yaw_rate_radps]
    for yr in yr_points:
        act = VelocityAction(vx=0.20, vy=0.0, vz=0.0, yaw_rate=yr)
        norm = act.to_normalized(limits)
        assert -1.0 <= norm[3] <= 1.0
        rec = VelocityAction.from_normalized(norm, limits)
        assert np.isclose(rec.yaw_rate, yr, atol=1e-5)


def test_stop_versus_reverse_decoding():
    """Verifies that a stop command (0.0 m/s) and an emergency reverse command (-0.10 m/s) decode distinctly."""
    limits = ActionLimits()

    stop_act = VelocityAction(vx=0.00, vy=0.0, vz=0.0, yaw_rate=0.0)
    norm_stop = stop_act.to_normalized(limits)
    rec_stop = VelocityAction.from_normalized(norm_stop, limits)

    rev_act = VelocityAction(vx=-0.10, vy=0.0, vz=0.0, yaw_rate=0.0)
    norm_rev = rev_act.to_normalized(limits)
    rec_rev = VelocityAction.from_normalized(norm_rev, limits)

    # 1. Stop must decode exactly to 0.0 m/s
    assert np.isclose(rec_stop.vx, 0.00, atol=1e-5)
    assert np.isclose(norm_stop[0], -0.50, atol=1e-5)

    # 2. Reverse must decode exactly to -0.10 m/s
    assert np.isclose(rec_rev.vx, -0.10, atol=1e-5)
    assert np.isclose(norm_rev[0], -0.75, atol=1e-5)

    # 3. Stop is not decoded as reverse or max reverse
    assert not np.isclose(rec_stop.vx, -0.10)
    assert not np.isclose(rec_stop.vx, -0.20)
    assert not np.isclose(norm_stop[0], -1.00)

    # 4. Reverse is not decoded as stop
    assert not np.isclose(rec_rev.vx, 0.00)
    assert rec_rev.vx < -0.05


def test_component_action_bounds_consistency():
    """Verifies authoritative ActionLimits bounds across components."""
    from controllers.safety_supervisor import SafetySupervisor
    from sim.corridor_env import CorridorSimEnv

    limits = ActionLimits()
    assert limits.min_fwd_vel_mps == -0.20
    assert limits.max_fwd_vel_mps == 0.60
    assert limits.max_lat_vel_mps == 0.40
    assert limits.max_vert_vel_mps == 0.30
    assert limits.max_yaw_rate_radps == 0.50

    # SafetySupervisor uses ActionLimits
    from core.geometry import Corridor3D
    sup = SafetySupervisor(Corridor3D())
    assert sup.limits.min_fwd_vel_mps == limits.min_fwd_vel_mps
    assert sup.limits.max_fwd_vel_mps == limits.max_fwd_vel_mps
    assert sup.limits.max_lat_vel_mps == limits.max_lat_vel_mps

    # CorridorSimEnv defaults to ActionLimits
    env = CorridorSimEnv()
    assert env.action_limits.min_fwd_vel_mps == limits.min_fwd_vel_mps
    assert env.action_limits.max_fwd_vel_mps == limits.max_fwd_vel_mps
    env.close()


def test_simulator_reverse_deterministic_yaw():
    """Verifies that negative body-forward velocity produces backward physical motion in the simulator across yaw angles."""
    import pybullet as p
    from sim.corridor_env import CorridorSimEnv

    limits = ActionLimits()
    env = CorridorSimEnv(action_limits=limits)
    env.reset(seed=100)

    rev_act = VelocityAction(vx=-0.10, vy=0.0, vz=0.0, yaw_rate=0.0)
    norm_rev = rev_act.to_normalized(limits)

    # Position in clear obstacle-free zone: x=1.5, y=0.0, z=1.0 (obstacles are at x=3, 5, 7)
    clear_pos = [1.5, 0.0, 1.0]

    # Case 1: Heading along world +X (yaw = 0.0)
    quat0 = p.getQuaternionFromEuler([0, 0, 0.0])
    p.resetBasePositionAndOrientation(env.drone_id, clear_pos, quat0, physicsClientId=env.client_id)
    p.resetBaseVelocity(env.drone_id, [0, 0, 0], [0, 0, 0], physicsClientId=env.client_id)
    pos_start = env._get_drone_position()
    for _ in range(10):
        env.step(norm_rev)
    pos_end = env._get_drone_position()
    dx = pos_end[0] - pos_start[0]
    dy = pos_end[1] - pos_start[1]
    # Body reverse at yaw=0 moves in world -X direction
    assert dx < -0.04, f"Expected negative X displacement at yaw=0, got dx={dx}"
    assert abs(dy) < 0.005, f"Expected near-zero Y displacement, got dy={dy}"

    # Case 2: Heading along world +Y (yaw = pi/2 = 90 deg)
    quat90 = p.getQuaternionFromEuler([0, 0, np.pi / 2.0])
    p.resetBasePositionAndOrientation(env.drone_id, clear_pos, quat90, physicsClientId=env.client_id)
    p.resetBaseVelocity(env.drone_id, [0, 0, 0], [0, 0, 0], physicsClientId=env.client_id)
    pos_start = env._get_drone_position()
    for _ in range(10):
        env.step(norm_rev)
    pos_end = env._get_drone_position()
    dx = pos_end[0] - pos_start[0]
    dy = pos_end[1] - pos_start[1]
    # Body reverse at yaw=90 deg moves in world -Y direction
    assert dy < -0.04, f"Expected negative Y displacement at yaw=pi/2, got dy={dy}"
    assert abs(dx) < 0.005, f"Expected near-zero X displacement, got dx={dx}"

    # Case 3: Heading along world -X (yaw = pi = 180 deg)
    quat180 = p.getQuaternionFromEuler([0, 0, np.pi])
    p.resetBasePositionAndOrientation(env.drone_id, clear_pos, quat180, physicsClientId=env.client_id)
    p.resetBaseVelocity(env.drone_id, [0, 0, 0], [0, 0, 0], physicsClientId=env.client_id)
    pos_start = env._get_drone_position()
    for _ in range(10):
        env.step(norm_rev)
    pos_end = env._get_drone_position()
    dx = pos_end[0] - pos_start[0]
    dy = pos_end[1] - pos_start[1]
    # Body reverse at yaw=180 deg moves in world +X direction
    assert dx > 0.04, f"Expected positive X displacement at yaw=pi, got dx={dx}"
    assert abs(dy) < 0.005, f"Expected near-zero Y displacement, got dy={dy}"

    env.close()


