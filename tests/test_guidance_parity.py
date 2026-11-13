"""test_guidance_parity.py - Parity Tests Between Simulation & Edge Guidance Engines."""

import numpy as np
import pytest

from core.contracts import ActionLimits, DeployableObservation, VelocityAction
from core.guidance import GuidanceConfig, StatelessClearanceGuidanceArbitrator
from controllers.guidance_controller import StatelessClearanceGuidanceController
from hardware.onnx_runner import OnboardONNXPolicyRunner


def test_shared_arbitrator_parity_with_controller():
    """Verifies that StatelessClearanceGuidanceController produces identical decisions to core arbitrator."""
    arbitrator = StatelessClearanceGuidanceArbitrator()
    controller = StatelessClearanceGuidanceController()

    rng = np.random.default_rng(42)
    raw_action = VelocityAction(vx=0.40, vy=0.0, vz=0.0, yaw_rate=0.0)

    for i in range(200):
        # Generate random observation vectors covering all geometric regimes
        obs = rng.uniform(0.0, 1.0, size=(23,)).astype(np.float32)
        
        # Test obstacle ahead (c_front < 0.85m)
        if i % 3 == 0:
            obs[0:3] = rng.uniform(0.05, 0.20, size=(3,)) # < 0.85m physical (0.20 * 4 = 0.80m)
        
        # Test terminal zone (goal_norm < 0.50)
        if i % 5 == 0:
            obs[12:15] = rng.uniform(-0.1, 0.1, size=(3,)) # norm < 0.50
        else:
            obs[12:15] = np.array([1.0, 0.0, 0.0], dtype=np.float32)

        # Test tight wall proximity
        if i % 7 == 0:
            obs[5] = rng.uniform(0.02, 0.08) # left wall tight (< 0.35m physical)
        if i % 11 == 0:
            obs[6] = rng.uniform(0.02, 0.08) # right wall tight (< 0.35m physical)

        # Arbitrator prediction
        arb_act, arb_dir, arb_cl, arb_cr = arbitrator.arbitrate(raw_action, obs)

        # Controller prediction
        ctrl_act, ctrl_dir, ctrl_cl, ctrl_cr = controller.predict_with_diagnostics(obs)

        # In case the controller loaded a base model, the base vx/vy might differ,
        # but the guidance direction and arbitration branch must be strictly identical!
        assert arb_dir == ctrl_dir, f"Mismatch at trial {i}: arbitrator={arb_dir} vs controller={ctrl_dir}"
        assert abs(arb_cl - ctrl_cl) < 1e-5
        assert abs(arb_cr - ctrl_cr) < 1e-5


def test_onnx_runner_guidance_parity_with_arbitrator():
    """Verifies that OnboardONNXPolicyRunner.apply_clearance_guidance matches core arbitrator identically."""
    runner = OnboardONNXPolicyRunner()
    arbitrator = StatelessClearanceGuidanceArbitrator()

    rng = np.random.default_rng(101)
    raw_action = VelocityAction(vx=0.40, vy=0.05, vz=0.0, yaw_rate=0.0)

    for i in range(200):
        obs = rng.uniform(0.0, 1.0, size=(23,)).astype(np.float32)
        if i % 2 == 0:
            obs[0:3] = rng.uniform(0.05, 0.18, size=(3,))
        if i % 4 == 0:
            obs[12:15] = np.array([0.2, 0.1, 0.0], dtype=np.float32) # Terminal zone
        else:
            obs[12:15] = np.array([1.0, 0.0, 0.0], dtype=np.float32)

        run_act = runner.apply_clearance_guidance(raw_action, obs)
        arb_act, arb_dir, _, _ = arbitrator.arbitrate(raw_action, obs)

        assert abs(run_act.vx - arb_act.vx) < 1e-5
        assert abs(run_act.vy - arb_act.vy) < 1e-5
        assert abs(run_act.vz - arb_act.vz) < 1e-5
        assert abs(run_act.yaw_rate - arb_act.yaw_rate) < 1e-5
        assert runner.last_guidance_dir == arb_dir

