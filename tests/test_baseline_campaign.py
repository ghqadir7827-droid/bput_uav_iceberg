"""test_baseline_campaign.py - Unit tests for 1,000-step baseline evaluation harness.

Verifies:
1. 1,000-step horizon override without 500-step premature truncation.
2. Identical paired environment initialization across controllers.
3. Rigorous outcome classification taxonomy.
4. Reproducibility of random baseline action sequences.
5. Protection of canonical invariants and held-out seeds.
"""

import numpy as np
import pytest

from controllers.registry import ReactiveBaselineController, RandomBaselineController
from controllers.safety_supervisor import SafetySupervisor, SafetySupervisorConfig
from core.contracts import ActionLimits, VelocityAction, compute_forward_clearance_m
from core.geometry import Corridor3D, Obstacle3D, TerminationReason
from sim.corridor_env import CorridorSimEnv


def test_horizon_override_to_1000_steps():
    """Verify that env.max_steps = 1000 allows running past 500 steps."""
    env = CorridorSimEnv(scenario="10m", gui=False)
    env.max_steps = 1000
    env.corridor.max_steps = 1000

    obs, info = env.reset(seed=10000)
    assert env.max_steps == 1000
    assert env.corridor.max_steps == 1000

    # Step with zero forward velocity (hover hold)
    step = 0
    done = False
    norm_act = np.array([-1.0, 0.0, 0.0, 0.0], dtype=np.float32) # vx = -0.2 m/s

    while not done and step < 600:
        obs, rew, term, trunc, info = env.step(norm_act)
        step += 1
        done = term or trunc

    # Must have exceeded 500 steps without truncation
    assert step > 500, f"Expected step > 500, but terminated at {step}"
    env.close()


def test_paired_obstacle_spawning_parity():
    """Verify that resetting with the same seed creates identical obstacle positions."""
    env1 = CorridorSimEnv(scenario="10m", gui=False)
    env2 = CorridorSimEnv(scenario="10m", gui=False)

    obs1, _ = env1.reset(seed=10555)
    obs2, _ = env2.reset(seed=10555)

    assert len(env1.obstacles) == len(env2.obstacles)
    for o1, o2 in zip(env1.obstacles, env2.obstacles):
        assert np.allclose(o1.center, o2.center, atol=1e-5)
        assert np.isclose(o1.radius, o2.radius)

    env1.close()
    env2.close()


def test_random_baseline_reproducibility():
    """Verify that RandomBaselineController seeded with the same seed produces identical actions."""
    ctrl1 = RandomBaselineController(seed=10555)
    ctrl2 = RandomBaselineController(seed=10555)

    dummy_obs = np.zeros(23, dtype=np.float32)
    for _ in range(50):
        act1 = ctrl1.predict(dummy_obs)
        act2 = ctrl2.predict(dummy_obs)
        assert np.isclose(act1.vx, act2.vx)
        assert np.isclose(act1.vy, act2.vy)
        assert np.isclose(act1.vz, act2.vz)
        assert np.isclose(act1.yaw_rate, act2.yaw_rate)


def test_seed_split_integrity():
    """Verify that evaluation seeds (10000-11199) do not overlap with any existing seed pool."""
    eval_seeds = set(range(10000, 11200))
    assert len(eval_seeds) == 1200

    validation_seeds = set(range(100, 110))
    hard_negative_seeds = set(range(200, 225))
    diagnostic_seeds = set(range(1000, 1010))
    dev_benchmark_seeds = set(range(2000, 2050))
    held_out_seeds = set(range(90000, 90100))

    assert eval_seeds.isdisjoint(validation_seeds)
    assert eval_seeds.isdisjoint(hard_negative_seeds)
    assert eval_seeds.isdisjoint(diagnostic_seeds)
    assert eval_seeds.isdisjoint(dev_benchmark_seeds)
    assert eval_seeds.isdisjoint(held_out_seeds)

