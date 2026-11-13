"""test_onnx_runner.py - Unit tests for Embedded ONNX Edge Inference Engine."""

from pathlib import Path
import time
import numpy as np
import pytest

from controllers.safety_supervisor import SafetyState
from core.contracts import ActionLimits, DeployableObservation, VelocityAction
from hardware.onnx_runner import OnboardONNXPolicyRunner


# ==============================================================================
# 1. Production Behavior Cloning (BC) ONNX Edge Tests (Default Deployment Path)
# ==============================================================================

def test_production_bc_default_initialization():
    """Verifies default runner initializes with policy_bc.onnx and canonical 0.50m caution distance."""
    runner = OnboardONNXPolicyRunner()
    assert runner.onnx_path == Path("models/checkpoints/policy_bc.onnx")
    assert runner.onnx_path.exists(), "Production artifact models/checkpoints/policy_bc.onnx must exist"
    assert runner.caution_distance_m == 0.50
    assert runner.supervisor_config.obstacle_caution_dist_m == 0.50
    assert runner.enable_guidance is True


def test_production_bc_inference_contract_and_bounds():
    """Verifies that production BC model input/output shapes and action limits conform to contract."""
    runner = OnboardONNXPolicyRunner(onnx_model_path="models/checkpoints/policy_bc.onnx")
    limits = ActionLimits()

    # Input shape contract
    inputs = runner.session.get_inputs()
    outputs = runner.session.get_outputs()
    assert len(inputs) == 1
    assert inputs[0].shape[1] == 23
    assert len(outputs) == 1
    assert outputs[0].shape[1] == 4

    # Run inference across random observations
    rng = np.random.default_rng(42)
    for _ in range(50):
        obs = rng.uniform(0.0, 1.0, size=(23,)).astype(np.float32)
        act = runner.predict(obs)

        assert isinstance(act, VelocityAction)
        assert np.isfinite(act.vx) and np.isfinite(act.vy) and np.isfinite(act.vz) and np.isfinite(act.yaw_rate)
        assert limits.min_fwd_vel_mps - 1e-6 <= act.vx <= limits.max_fwd_vel_mps + 1e-6
        assert -limits.max_lat_vel_mps - 1e-6 <= act.vy <= limits.max_lat_vel_mps + 1e-6
        assert -limits.max_vert_vel_mps - 1e-6 <= act.vz <= limits.max_vert_vel_mps + 1e-6
        assert -limits.max_yaw_rate_radps - 1e-6 <= act.yaw_rate <= limits.max_yaw_rate_radps + 1e-6


def test_production_bc_guidance_invocation_in_control_loop():
    """Verifies that clearance guidance is invoked by step_onboard_control on obstacle approach."""
    runner = OnboardONNXPolicyRunner()

    # Obstacle detected directly ahead at 0.50m (ray 0, 1, 2 = 0.50 / 4.0 = 0.125 < 0.85m threshold)
    # Goal far away (norm = 1.0)
    obs = DeployableObservation(
        distance_rays=np.array([0.125, 0.125, 0.125, 0.5, 0.5, 0.8, 0.8, 0.5, 0.5, 0.5, 0.5, 0.5], dtype=np.float32),
        rel_goal_direction_body=np.array([1.0, 0.0, 0.0], dtype=np.float32),
        body_velocity=np.array([0.4, 0.0, 0.0], dtype=np.float32),
        attitude_trig=np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float32),
        yaw_rate=0.0,
    )
    pos = np.array([1.0, 0.0, 1.0], dtype=np.float32)
    vel = np.array([0.4, 0.0, 0.0], dtype=np.float32)

    safe_act, s_state, s_reason = runner.step_onboard_control(
        deployable_obs=obs,
        estimated_position=pos,
        estimated_velocity=vel,
        dt=0.05,
    )

    assert isinstance(safe_act, VelocityAction)
    # Guidance must actively steer lateral velocity (|vy| >= 0.25 m/s before supervisor filter)
    assert runner.last_guidance_dir != "NONE"
    assert "TIE_BREAK" in runner.last_guidance_dir or runner.last_guidance_dir in ["LEFT", "RIGHT"]


def test_production_bc_terminal_guidance_release():
    """Verifies that guidance yields when goal is within 1.0m (goal_norm < 0.50)."""
    runner = OnboardONNXPolicyRunner()

    # Obstacle detected at 0.60m, but drone is in terminal approach (goal_norm = 0.30 < 0.50)
    obs = DeployableObservation(
        distance_rays=np.array([0.15, 0.15, 0.15, 0.5, 0.5, 0.8, 0.8, 0.5, 0.5, 0.5, 0.5, 0.5], dtype=np.float32),
        rel_goal_direction_body=np.array([0.30, 0.0, 0.0], dtype=np.float32),  # norm = 0.30 < 0.50
        body_velocity=np.array([0.3, 0.0, 0.0], dtype=np.float32),
        attitude_trig=np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float32),
        yaw_rate=0.0,
    )
    pos = np.array([9.2, 0.0, 1.0], dtype=np.float32)
    vel = np.array([0.3, 0.0, 0.0], dtype=np.float32)

    safe_act, s_state, s_reason = runner.step_onboard_control(
        deployable_obs=obs,
        estimated_position=pos,
        estimated_velocity=vel,
        dt=0.05,
    )

    # In terminal approach, guidance must release control (NONE)
    assert runner.last_guidance_dir == "NONE"


def test_production_bc_safety_supervisor_filter_active():
    """Verifies that the SafetySupervisor actively filters guided commands to prevent envelope breach."""
    runner = OnboardONNXPolicyRunner()

    # Place drone extremely close to corridor ceiling (z = 1.95m, ceiling is at 2.0m -> wall clearance = 0.05m < 0.25m buffer)
    obs = DeployableObservation(
        distance_rays=np.array([0.8] * 12, dtype=np.float32),
        rel_goal_direction_body=np.array([1.0, 0.0, 0.0], dtype=np.float32),
        body_velocity=np.array([0.4, 0.0, 0.2], dtype=np.float32),
        attitude_trig=np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float32),
        yaw_rate=0.0,
    )
    pos = np.array([2.0, 0.0, 1.95], dtype=np.float32)
    vel = np.array([0.4, 0.0, 0.2], dtype=np.float32)

    safe_act, s_state, s_reason = runner.step_onboard_control(
        deployable_obs=obs,
        estimated_position=pos,
        estimated_velocity=vel,
        dt=0.05,
    )

    # Supervisor must intervene to push down (vz < 0.0)
    assert s_state != SafetyState.NOMINAL
    assert safe_act.vz <= 0.0


def test_onnx_runner_nonexistent_model_raises_filenotfound():
    """Verifies that requesting a non-existent ONNX model raises FileNotFoundError and does NOT silently fall back."""
    with pytest.raises(FileNotFoundError, match="ONNX model not found"):
        _ = OnboardONNXPolicyRunner(onnx_model_path="models/checkpoints/non_existent_policy.onnx")


def test_production_bc_latency_budget():
    """Verifies that production BC ONNX inference easily satisfies the 3.0ms edge latency budget."""
    runner = OnboardONNXPolicyRunner()
    dummy_obs = np.ones(23, dtype=np.float32) * 0.5

    # Warmup
    for _ in range(20):
        _ = runner.predict(dummy_obs)

    # 100 trials
    t0 = time.perf_counter()
    for _ in range(100):
        _ = runner.predict(dummy_obs)
    avg_lat_ms = (time.perf_counter() - t0) * 10.0  # ms for 100 trials / 100 * 1000 = * 10

    assert avg_lat_ms < 0.5, f"Production BC ONNX runner too slow: {avg_lat_ms:.3f} ms"


# ==============================================================================
# 2. Legacy PPO ONNX Regression Tests
# ==============================================================================

def test_legacy_ppo_loading_and_prediction():
    """Verifies backward compatibility with historical PPO checkpoint."""
    runner = OnboardONNXPolicyRunner(onnx_model_path="models/checkpoints/policy_ppo.onnx")

    dummy_obs = np.ones(23, dtype=np.float32) * 0.5
    action = runner.predict(dummy_obs)

    assert isinstance(action, VelocityAction)
    assert -0.20 <= action.vx <= 0.80
    assert -0.60 <= action.vy <= 0.60
    assert -0.40 <= action.vz <= 0.40
    assert -0.60 <= action.yaw_rate <= 0.60


def test_legacy_ppo_full_control_loop_step():
    """Verifies that full control loop operates with legacy PPO model."""
    runner = OnboardONNXPolicyRunner(onnx_model_path="models/checkpoints/policy_ppo.onnx")

    obs = DeployableObservation(
        distance_rays=np.array([0.7] * 12, dtype=np.float32),
        rel_goal_direction_body=np.array([1.0, 0.0, 0.0], dtype=np.float32),
        body_velocity=np.array([0.5, 0.0, 0.0], dtype=np.float32),
        attitude_trig=np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float32),
        yaw_rate=0.0,
    )
    pos = np.array([1.0, 0.0, 1.0], dtype=np.float32)
    vel = np.array([0.5, 0.0, 0.0], dtype=np.float32)

    safe_action, s_state, s_reason = runner.step_onboard_control(
        deployable_obs=obs,
        estimated_position=pos,
        estimated_velocity=vel,
        dt=0.05,
    )

    assert isinstance(safe_action, VelocityAction)
    assert safe_action.vx > 0.0


def test_legacy_ppo_nan_inf_sanitization():
    """Verifies that corrupted observations are sanitized under PPO runner."""
    runner = OnboardONNXPolicyRunner(onnx_model_path="models/checkpoints/policy_ppo.onnx")
    corrupted_obs = np.array([np.nan, np.inf, -np.inf] + [0.5] * 20, dtype=np.float32)

    action = runner.predict(corrupted_obs)
    assert isinstance(action, VelocityAction)
    assert not np.isnan(action.vx)
    assert not np.isnan(action.vy)
    assert not np.isnan(action.vz)
    assert not np.isnan(action.yaw_rate)


def test_onnx_controller_registry_integration():
    """Verifies that registry exposes the ONNX edge controller properly."""
    from controllers.registry import ControllerRegistry
    from controllers.onnx_controller import ONNXEdgePolicyController

    registry = ControllerRegistry()
    ctrl = registry.get_controller("onnx_ppo_edge")
    assert ctrl is not None
    assert isinstance(ctrl, ONNXEdgePolicyController)
    assert ctrl.metadata.is_implemented is True
    assert ctrl.metadata.checkpoint_id == "onnx_5.3kb_0.016ms"

    obs = np.ones(23, dtype=np.float32) * 0.5
    action = ctrl.predict(obs)
    assert isinstance(action, VelocityAction)
    assert action.vx > 0.0
