"""test_bc_policy.py - Unit tests for Behavior Cloning Imitation Policy Controller."""

import time
import numpy as np
import pytest

from controllers.bc_policy import BehaviorCloningPolicyController
from controllers.registry import ControllerRegistry
from core.contracts import VelocityAction


def test_bc_controller_registration_and_metadata():
    registry = ControllerRegistry()
    ctrl = registry.get_controller("imitation_bc_policy")
    assert ctrl is not None
    meta = ctrl.metadata
    assert meta.controller_id == "imitation_bc_policy"
    assert meta.is_implemented is True
    assert "Milestone 3" in meta.milestone_tag
    assert "bc_epoch" in meta.checkpoint_id


def test_bc_controller_prediction_bounds():
    ctrl = BehaviorCloningPolicyController()
    assert ctrl.metadata.is_implemented is True

    # Test random observation vectors
    rng = np.random.default_rng(42)
    for _ in range(50):
        # 23-float observation
        obs = rng.uniform(-1.0, 1.0, size=23).astype(np.float32)
        # Rays [0:12] normalized [0, 1]
        obs[0:12] = rng.uniform(0.05, 1.0, size=12)
        action = ctrl.predict(obs)

        assert isinstance(action, VelocityAction)
        assert -0.20 <= action.vx <= 0.80
        assert -0.60 <= action.vy <= 0.60
        assert -0.40 <= action.vz <= 0.40
        assert -0.60 <= action.yaw_rate <= 0.60


def test_bc_controller_inference_latency():
    ctrl = BehaviorCloningPolicyController()
    obs = np.ones(23, dtype=np.float32) * 0.5

    # Warmup
    for _ in range(20):
        _ = ctrl.predict(obs)

    # 100 calls
    t0 = time.perf_counter()
    for _ in range(100):
        _ = ctrl.predict(obs)
    elapsed_ms = (time.perf_counter() - t0) * 10.0 # avg ms per call

    # Strict latency check: must be < 1.0 ms (target < 3.0 ms)
    assert elapsed_ms < 1.0, f"Average inference time too high: {elapsed_ms:.3f} ms"
