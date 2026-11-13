"""test_regression_harness_integrity.py - Regression Harness Integrity Tests.

Verifies:
1. Reset cleans all internal controller state (specifically arbitrator.last_dir).
2. Supervisor reset clears all failsafe timers, previous commands, and states.
3. Deterministic repeatability: repeated runs with the same seed yield identical trajectory and metrics.
4. Independent execution: running seed B after seed A yields the exact same trajectory as running seed B directly.
"""

import numpy as np
import pytest

from controllers.bc_policy import BehaviorCloningPolicyController
from controllers.safety_supervisor import SafetySupervisor
from core.contracts import VelocityAction
from core.guidance import GuidanceConfig, StatelessClearanceGuidanceArbitrator
from core.physical_dimensions import DifficultyLevel, ScaledEnvironmentFactory
from scripts.run_exceed_90_experiments import GenericGuidedController
from scripts.run_controlled_regression_campaign import run_episode, RouteFeasibilityChecker, PhysicalDroneModel
from sim.corridor_env import CorridorSimEnv


def test_controller_and_arbitrator_reset():
    g_cfg = GuidanceConfig.for_65cm_uav_v3()
    arb = StatelessClearanceGuidanceArbitrator(config=g_cfg)
    arb.last_dir = "LEFT"
    assert arb.last_dir == "LEFT"

    # Test reset
    arb.reset()
    assert arb.last_dir in (None, "NONE"), "Arbitrator.reset() must reset last_dir"

    ckpt_path = "models/checkpoints/bc_policy_scaled_v2.pt"
    bc = BehaviorCloningPolicyController(ckpt_path)
    ctrl = GenericGuidedController(bc, arb, "test_ctrl")
    arb.last_dir = "RIGHT"
    ctrl.reset()
    assert arb.last_dir in (None, "NONE"), "Controller.reset() must cascade reset to arbitrator"


def test_supervisor_reset():
    cfg = ScaledEnvironmentFactory.get_environment_config(DifficultyLevel.LEVEL_2_STANDARD)
    env = CorridorSimEnv(scenario=cfg, seed=20001)
    sup = SafetySupervisor(env.corridor)

    # Induce state changes
    sup.last_safe_action = VelocityAction(0.5, 0.2, 0.0, 0.1)
    sup.loss_timer_s = 1.0
    sup.stalled_timer_s = 0.3

    sup.reset()
    assert sup.last_safe_action.vx == 0.0
    assert sup.last_safe_action.vy == 0.0
    assert sup.loss_timer_s == 0.0
    assert sup.stalled_timer_s == 0.0
    env.close()


def test_seed_determinism_and_no_state_leakage():
    cfg = ScaledEnvironmentFactory.get_environment_config(DifficultyLevel.LEVEL_2_STANDARD)
    env = CorridorSimEnv(scenario=cfg, seed=20000)
    sup = SafetySupervisor(env.corridor)
    checker = RouteFeasibilityChecker(PhysicalDroneModel())

    ckpt_path = "models/checkpoints/bc_policy_scaled_v2.pt"
    bc = BehaviorCloningPolicyController(ckpt_path)
    g_cfg = GuidanceConfig.for_65cm_uav_v3()
    arb = StatelessClearanceGuidanceArbitrator(config=g_cfg)
    ctrl = GenericGuidedController(bc, arb, "scaled_guided_bc_v3")

    # Run seed 20002 first time
    rec1 = run_episode(
        env=env,
        controller=ctrl,
        supervisor=sup,
        checker=checker,
        tier_name="Level 2",
        level_num=2,
        seed=20002,
        controller_id="scaled_guided_bc_v3",
    )

    # Run intervening seed 20003
    _ = run_episode(
        env=env,
        controller=ctrl,
        supervisor=sup,
        checker=checker,
        tier_name="Level 2",
        level_num=2,
        seed=20003,
        controller_id="scaled_guided_bc_v3",
    )

    # Run seed 20002 second time after intervening seed
    rec2 = run_episode(
        env=env,
        controller=ctrl,
        supervisor=sup,
        checker=checker,
        tier_name="Level 2",
        level_num=2,
        seed=20002,
        controller_id="scaled_guided_bc_v3",
    )

    env.close()

    assert rec1.status == rec2.status, f"Status mismatch: {rec1.status} vs {rec2.status}"
    assert rec1.total_steps == rec2.total_steps, f"Steps mismatch: {rec1.total_steps} vs {rec2.total_steps}"
    assert np.isclose(rec1.flight_distance_m, rec2.flight_distance_m, atol=1e-4), "Flight distance mismatch"
    assert np.isclose(rec1.min_obstacle_clearance_m, rec2.min_obstacle_clearance_m, atol=1e-4), "Clearance mismatch"
    assert rec1.safety_interventions == rec2.safety_interventions, "Intervention count mismatch"
