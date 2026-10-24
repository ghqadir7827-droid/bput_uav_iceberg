"""test_ppo_policy.py - Unit tests for PPO Fine-Tuned Policy Controller."""

import time
import numpy as np
import pytest

from controllers.ppo_policy import PPOPolicyController
from controllers.registry import ControllerRegistry
from core.contracts import VelocityAction


def test_ppo_controller_registration_and_metadata():
    registry = ControllerRegistry()
    ctrl = registry.get_controller("ppo_fine_tuned")
    # If checkpoint exists, it should be registered
    if ctrl is not None:
        meta = ctrl.metadata
        assert meta.controller_id == "ppo_fine_tuned"
        assert meta.is_implemented is True
        assert "Milestone 4" in meta.milestone_tag


def test_ppo_controller_action_bounds():
    ctrl = PPOPolicyController()
    if not ctrl.metadata.is_implemented:
        pytest.skip("PPO checkpoint not yet trained")

    rng = np.random.default_rng(42)
    for _ in range(50):
        obs = rng.uniform(-1.0, 1.0, size=23).astype(np.float32)
        obs[0:12] = rng.uniform(0.05, 1.0, size=12)
        action = ctrl.predict(obs)

        assert isinstance(action, VelocityAction)
        assert -0.20 <= action.vx <= 0.80
        assert -0.60 <= action.vy <= 0.60
        assert -0.40 <= action.vz <= 0.40
        assert -0.60 <= action.yaw_rate <= 0.60


def test_ppo_controller_inference_latency():
    ctrl = PPOPolicyController()
    if not ctrl.metadata.is_implemented:
        pytest.skip("PPO checkpoint not yet trained")

    obs = np.ones(23, dtype=np.float32) * 0.5
    for _ in range(20):
        _ = ctrl.predict(obs)

    t0 = time.perf_counter()
    for _ in range(100):
        _ = ctrl.predict(obs)
    elapsed_ms = (time.perf_counter() - t0) * 10.0

    assert elapsed_ms < 1.0, f"Average inference time too high: {elapsed_ms:.3f} ms"
