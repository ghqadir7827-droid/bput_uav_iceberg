"""test_onnx_runner.py - Unit tests for Embedded ONNX Edge Inference Engine."""

import time
import numpy as np
import pytest

from core.contracts import DeployableObservation, VelocityAction
from hardware.onnx_runner import OnboardONNXPolicyRunner


def test_onnx_runner_loading_and_prediction():
    runner = OnboardONNXPolicyRunner(onnx_model_path="models/checkpoints/policy_ppo.onnx")

    dummy_obs = np.ones(23, dtype=np.float32) * 0.5
    action = runner.predict(dummy_obs)

    assert isinstance(action, VelocityAction)
    assert -0.20 <= action.vx <= 0.80
    assert -0.60 <= action.vy <= 0.60
    assert -0.40 <= action.vz <= 0.40
    assert -0.60 <= action.yaw_rate <= 0.60


def test_onnx_runner_full_control_loop_step():
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


def test_onnx_runner_latency_budget():
    runner = OnboardONNXPolicyRunner(onnx_model_path="models/checkpoints/policy_ppo.onnx")
    dummy_obs = np.ones(23, dtype=np.float32) * 0.5

    # Warmup
    for _ in range(20):
        _ = runner.predict(dummy_obs)

    # 100 trials
    t0 = time.perf_counter()
    for _ in range(100):
        _ = runner.predict(dummy_obs)
    avg_lat_ms = (time.perf_counter() - t0) * 10.0 # ms

    # Must be under 0.5 ms (well within 3.0 ms budget)
    assert avg_lat_ms < 0.5, f"ONNX runner too slow: {avg_lat_ms:.3f} ms"


def test_onnx_controller_registry_integration():
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

