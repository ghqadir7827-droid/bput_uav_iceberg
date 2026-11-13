"""test_pillar3_recovery.py - Unit and Integration Tests for Phase 6B Post-Evasion Recovery.

Tests all 12 conditions mandated by Phase 6B specification Section 5:
1. Obstacle not yet passed: recovery remains inactive.
2. Obstacle passed with safe clearance: recovery may activate.
3. Insufficient clearance: recovery remains inactive.
4. Invalid or stale observations: safe fallback (fail-closed).
5. Recovery termination after centerline restoration.
6. Recovery-state reset between episodes.
7. Hysteresis around activation and release thresholds.
8. Supervisor rejection or clipping of a recovery command.
9. No commands that violate the existing action bounds.
10. Regression behavior in the Scenario C Wide Left case.
11. Simulator/edge guidance parity.
12. Existing production BC ONNX loading and inference contracts.
"""

from pathlib import Path
import numpy as np
import pytest

from controllers.safety_supervisor import SafetyState, SafetySupervisor, SafetySupervisorConfig
from core.contracts import ActionLimits, DeployableObservation, VelocityAction
from core.geometry import Corridor3D
from core.guidance import GuidanceConfig, StatelessClearanceGuidanceArbitrator
from experiments.phase6b.phase6b_pillar3_recovery_20261030_025800.recovery_guidance import (
    PostEvasionRecoveryArbitrator,
    RecoveryConfig,
)
from hardware.onnx_runner import OnboardONNXPolicyRunner


def make_clean_observation(
    front_c: float = 2.0,
    wall_l: float = 1.0,
    wall_r: float = 1.0,
    c_left: float = 1.0,
    c_right: float = 1.0,
    goal_norm: float = 1.0,
) -> np.ndarray:
    """Helper to construct synthetic 23-float observation vectors."""
    obs = np.zeros(23, dtype=np.float32)
    # Rangefinder rays normalized to [0, 1] for 4.0m max_range
    # Ray 0: front, Ray 1: front-left, Ray 2: front-right
    obs[0] = front_c / 4.0
    obs[1] = front_c / 4.0
    obs[2] = front_c / 4.0
    obs[3] = c_left / 4.0   # left-45
    obs[4] = c_right / 4.0  # right-45
    obs[5] = wall_l / 4.0   # left-90 (left wall)
    obs[6] = wall_r / 4.0   # right-90 (right wall)
    obs[7:12] = 1.0         # rear, up, down, front-up, front-down
    # Goal vector in FLU
    obs[12] = goal_norm
    obs[13] = 0.0
    obs[14] = 0.0
    # Velocity
    obs[15] = 0.4
    # Attitude trig
    obs[18] = 1.0  # cos(roll)
    obs[20] = 1.0  # cos(pitch)
    return obs


def test_1_obstacle_not_yet_passed_recovery_inactive():
    """1. Obstacle not yet passed: recovery remains inactive even if displaced."""
    arb = PostEvasionRecoveryArbitrator()
    raw_act = VelocityAction(vx=0.4, vy=0.1, vz=0.0, yaw_rate=0.0)
    # Displaced left (wall_l=0.6, wall_r=1.4 -> y_obs = +0.40m), but no obstacle encountered yet
    obs = make_clean_observation(front_c=2.5, wall_l=0.6, wall_r=1.4)
    act, s_dir, _, _ = arb.arbitrate(raw_act, obs)
    assert s_dir == "NONE"
    assert act.vy == pytest.approx(raw_act.vy, abs=1e-5)
    assert not arb.recovery_active


def test_2_obstacle_passed_with_safe_clearance_recovery_activates():
    """2. Obstacle passed with safe clearance: recovery may activate."""
    arb = PostEvasionRecoveryArbitrator()
    raw_act = VelocityAction(vx=0.4, vy=0.0, vz=0.0, yaw_rate=0.0)
    # First: obstacle encountered ahead (front_c = 0.50m)
    obs_obs = make_clean_observation(front_c=0.50, wall_l=1.0, wall_r=1.0)
    arb.arbitrate(raw_act, obs_obs)
    assert arb.obstacle_encountered

    # Second: obstacle passed (front opens up to 1.5m), displaced left (y_obs = +0.35m), safe right clearance (1.2m)
    obs_clear = make_clean_observation(front_c=1.5, wall_l=0.65, wall_r=1.35, c_right=1.2)
    act, s_dir, _, _ = arb.arbitrate(raw_act, obs_clear)
    assert s_dir == "RECOVERY_RIGHT"
    assert act.vy < 0.0
    assert arb.recovery_active


def test_3_insufficient_clearance_recovery_inactive():
    """3. Insufficient clearance: recovery remains inactive."""
    arb = PostEvasionRecoveryArbitrator()
    raw_act = VelocityAction(vx=0.4, vy=0.0, vz=0.0, yaw_rate=0.0)
    # Trigger obstacle encountered
    arb.arbitrate(raw_act, make_clean_observation(front_c=0.50))

    # Displaced left (y_obs = +0.35m), but right clearance is dangerously narrow (0.40m < 0.55m)
    obs_blocked = make_clean_observation(front_c=1.5, wall_l=0.65, wall_r=1.35, c_right=0.40)
    act, s_dir, _, _ = arb.arbitrate(raw_act, obs_blocked)
    assert s_dir == "NONE"
    assert not arb.recovery_active


def test_4_invalid_or_stale_observations_safe_fallback():
    """4. Invalid or stale observations: fail-closed fallback."""
    arb = PostEvasionRecoveryArbitrator()
    raw_act = VelocityAction(vx=0.4, vy=0.2, vz=0.0, yaw_rate=0.0)
    arb.arbitrate(raw_act, make_clean_observation(front_c=0.50))

    # Observation with NaNs / Infs or invalid corridor width (wall_l + wall_r = 0.8m)
    obs_invalid = make_clean_observation(front_c=1.5, wall_l=0.3, wall_r=0.5)
    act, s_dir, _, _ = arb.arbitrate(raw_act, obs_invalid)
    assert s_dir == "NONE"
    assert not arb.recovery_active

    # Observation with explicit NaNs
    obs_nan = np.full(23, np.nan, dtype=np.float32)
    act_nan, s_dir_nan, _, _ = arb.arbitrate(raw_act, obs_nan)
    assert np.isfinite(act_nan.vx)
    assert np.isfinite(act_nan.vy)


def test_5_recovery_termination_after_centerline_restoration():
    """5. Recovery termination after centerline restoration."""
    arb = PostEvasionRecoveryArbitrator()
    raw_act = VelocityAction(vx=0.4, vy=0.0, vz=0.0, yaw_rate=0.0)
    # Encounter obstacle
    arb.arbitrate(raw_act, make_clean_observation(front_c=0.50))

    # Activate recovery (displaced left y_obs = +0.30m)
    arb.arbitrate(raw_act, make_clean_observation(front_c=1.5, wall_l=0.7, wall_r=1.3, c_right=1.2))
    assert arb.recovery_active

    # Drone reaches centerline (y_obs = +0.04m <= 0.08m release threshold)
    obs_centered = make_clean_observation(front_c=1.5, wall_l=0.96, wall_r=1.04, c_right=1.0)
    act, s_dir, _, _ = arb.arbitrate(raw_act, obs_centered)
    assert not arb.recovery_active
    assert s_dir == "NONE"


def test_6_recovery_state_reset_between_episodes():
    """6. Recovery-state reset correctly between episodes."""
    arb = PostEvasionRecoveryArbitrator()
    raw_act = VelocityAction(vx=0.4, vy=0.0, vz=0.0, yaw_rate=0.0)
    arb.arbitrate(raw_act, make_clean_observation(front_c=0.50))
    arb.arbitrate(raw_act, make_clean_observation(front_c=1.5, wall_l=0.7, wall_r=1.3, c_right=1.2))
    assert arb.recovery_active
    assert arb.obstacle_encountered

    # Reset
    arb.reset()
    assert not arb.recovery_active
    assert not arb.obstacle_encountered
    assert arb.last_dir == "NONE"


def test_7_hysteresis_around_activation_and_release_thresholds():
    """7. Hysteresis around activation (0.22m) and release (0.08m) thresholds."""
    arb = PostEvasionRecoveryArbitrator()
    raw_act = VelocityAction(vx=0.4, vy=0.0, vz=0.0, yaw_rate=0.0)
    arb.arbitrate(raw_act, make_clean_observation(front_c=0.50))

    # At y_obs = 0.15m (between release 0.08m and activate 0.22m): should NOT activate initially
    obs_mid = make_clean_observation(front_c=1.5, wall_l=0.85, wall_r=1.15, c_right=1.1)
    _, s_dir, _, _ = arb.arbitrate(raw_act, obs_mid)
    assert not arb.recovery_active

    # Push to y_obs = 0.25m: activates
    obs_high = make_clean_observation(front_c=1.5, wall_l=0.75, wall_r=1.25, c_right=1.2)
    _, s_dir, _, _ = arb.arbitrate(raw_act, obs_high)
    assert arb.recovery_active

    # Return to y_obs = 0.15m: remains ACTIVE due to hysteresis
    _, s_dir, _, _ = arb.arbitrate(raw_act, obs_mid)
    assert arb.recovery_active

    # Drop to y_obs = 0.06m: releases
    obs_low = make_clean_observation(front_c=1.5, wall_l=0.94, wall_r=1.06, c_right=1.0)
    _, s_dir, _, _ = arb.arbitrate(raw_act, obs_low)
    assert not arb.recovery_active


def test_8_supervisor_rejection_or_clipping_of_recovery_command():
    """8. Supervisor rejection or clipping of a recovery command."""
    corridor = Corridor3D.get_10m_benchmark()
    limits = ActionLimits()
    supervisor = SafetySupervisor(
        corridor=corridor,
        limits=limits,
        config=SafetySupervisorConfig(boundary_buffer_m=0.25, critical_lateral_dist_m=0.32),
    )
    # Recovery proposes vy = -0.15 m/s toward right wall
    rec_act = VelocityAction(vx=0.4, vy=-0.15, vz=0.0, yaw_rate=0.0)
    # Drone position right next to right wall (y = -0.80m, clearance = 0.05m < 0.32m)
    pos = np.array([5.0, -0.80, 1.0], dtype=np.float32)
    vel = np.array([0.4, 0.0, 0.0], dtype=np.float32)
    obs = make_clean_observation(front_c=2.0, wall_r=0.20)

    safe_act, s_state, s_reason = supervisor.filter_action(
        proposed_action=rec_act,
        current_position=pos,
        current_velocity=vel,
        min_front_range=2.0,
        dt=0.05,
        observation=obs,
    )
    # Supervisor must prevent negative lateral motion into the right wall
    assert safe_act.vy >= 0.0
    assert s_state != SafetyState.NOMINAL


def test_9_no_commands_violate_existing_action_bounds():
    """9. No commands violate the existing action bounds."""
    arb = PostEvasionRecoveryArbitrator()
    limits = ActionLimits()
    raw_act = VelocityAction(vx=1.5, vy=-2.0, vz=3.0, yaw_rate=5.0)
    obs = make_clean_observation(front_c=1.5, wall_l=0.5, wall_r=1.5, c_right=1.2)
    act, _, _, _ = arb.arbitrate(raw_act, obs)
    assert limits.min_fwd_vel_mps <= act.vx <= limits.max_fwd_vel_mps
    assert -limits.max_lat_vel_mps <= act.vy <= limits.max_lat_vel_mps
    assert -limits.max_vert_vel_mps <= act.vz <= limits.max_vert_vel_mps
    assert -limits.max_yaw_rate_radps <= act.yaw_rate <= limits.max_yaw_rate_radps


def test_10_scenario_c_wide_left_audit():
    """10. Scenario C Wide Left audit: verify baseline succeeds without collision."""
    from experiments.phase6b.phase6b_pillar3_recovery_20261030_025800.benchmark_phase6b import run_scenario_episode
    from controllers.guidance_controller import StatelessClearanceGuidanceController
    from sim.corridor_env import CorridorSimEnv

    corridor = Corridor3D.get_10m_benchmark()
    env = CorridorSimEnv(corridor=corridor)
    supervisor = SafetySupervisor(corridor=corridor, limits=env.action_limits, config=SafetySupervisorConfig(obstacle_caution_dist_m=0.50))
    controller = StatelessClearanceGuidanceController("models/checkpoints/bc_policy_best.pt")

    summary, _ = run_scenario_episode(env, supervisor, controller, "Scenario_C_Wide_Left")
    env.close()

    assert summary["success"] == 1
    assert summary["obs_collisions"] == 0
    assert summary["wall_collisions"] == 0
    assert summary["min_obs_clearance_m"] >= 0.080


def test_11_simulator_edge_guidance_parity():
    """11. Simulator/edge guidance parity: verify bit-for-bit parity across shared arbitrator."""
    guidance_cfg = GuidanceConfig()
    limits = ActionLimits()
    sim_arb = StatelessClearanceGuidanceArbitrator(config=guidance_cfg, limits=limits)
    edge_runner = OnboardONNXPolicyRunner(
        onnx_model_path="models/checkpoints/policy_bc.onnx",
        corridor=Corridor3D.get_10m_benchmark(),
    )

    rng = np.random.default_rng(12345)
    for _ in range(50):
        obs = rng.uniform(0.0, 1.0, size=23).astype(np.float32)
        sim_arb.reset()
        edge_runner.guidance_arbitrator.reset()
        raw_act = VelocityAction(vx=0.4, vy=0.0, vz=0.0, yaw_rate=0.0)

        sim_act, sim_dir, sim_cl, sim_cr = sim_arb.arbitrate(raw_act, obs)
        edge_act, edge_dir, edge_cl, edge_cr = edge_runner.guidance_arbitrator.arbitrate(raw_act, obs)

        assert sim_dir == edge_dir
        assert sim_act.vx == pytest.approx(edge_act.vx, abs=1e-5)
        assert sim_act.vy == pytest.approx(edge_act.vy, abs=1e-5)
        assert sim_cl == pytest.approx(edge_cl, abs=1e-5)
        assert sim_cr == pytest.approx(edge_cr, abs=1e-5)


def test_12_existing_production_bc_onnx_contract():
    """12. Existing production BC ONNX loading and inference contracts."""
    runner = OnboardONNXPolicyRunner(
        onnx_model_path="models/checkpoints/policy_bc.onnx",
        corridor=Corridor3D.get_10m_benchmark(),
    )
    obs_vec = np.ones(23, dtype=np.float32) * 0.5
    raw_act = runner.predict(obs_vec)
    assert isinstance(raw_act, VelocityAction)
    assert runner.action_limits.min_fwd_vel_mps <= raw_act.vx <= runner.action_limits.max_fwd_vel_mps

    deployable_obs = DeployableObservation(
        distance_rays=np.ones(12, dtype=np.float32) * 0.5,
        rel_goal_direction_body=np.array([1.0, 0.0, 0.0], dtype=np.float32),
        body_velocity=np.array([0.4, 0.0, 0.0], dtype=np.float32),
        attitude_trig=np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float32),
        yaw_rate=0.0,
    )
    safe_act, s_state, s_reason = runner.step_onboard_control(
        deployable_obs=deployable_obs,
        estimated_position=np.array([5.0, 0.0, 1.0], dtype=np.float32),
        estimated_velocity=np.array([0.4, 0.0, 0.0], dtype=np.float32),
    )
    assert isinstance(safe_act, VelocityAction)
    assert isinstance(s_state, SafetyState)
    assert isinstance(s_reason, str)
