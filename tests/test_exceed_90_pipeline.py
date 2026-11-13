"""test_exceed_90_pipeline.py - Verification and regression tests for Exceed 90% Campaign."""

import hashlib
import json
from pathlib import Path
import time
import numpy as np
import pytest
import torch

from controllers.bc_policy import BehaviorCloningPolicyController
from core.contracts import ActionLimits, VelocityAction
from core.guidance import GuidanceConfig, StatelessClearanceGuidanceArbitrator


def test_guidance_v2_configuration():
    """Verify GuidanceConfig.for_65cm_uav_v2 aligns with 5-ray forward sector."""
    cfg = GuidanceConfig.for_65cm_uav_v2(drone_radius=0.325)
    assert cfg.forward_ray_indices == (0, 1, 2, 3, 4)
    assert np.isclose(cfg.activation_threshold_m, 1.275)
    assert np.isclose(cfg.min_lateral_override_mps, 0.35)
    assert np.isclose(cfg.wall_tight_threshold_m, 0.675)
    assert np.isclose(cfg.hysteresis_m, 0.06)
    assert np.isclose(cfg.deadband_m, 0.08)


def test_guidance_v2_blind_sector_elimination():
    """Verify that obstacles in the 30-60 degree azimuth window (rays 3 or 4) trigger evasion."""
    cfg = GuidanceConfig.for_65cm_uav_v2(drone_radius=0.325)
    arb = StatelessClearanceGuidanceArbitrator(config=cfg)

    # Construct observation (23 dims):
    # ray 0-2 clear (2.5m), ray 4 (right 60 deg) close (0.80m / 4.0 = 0.20)
    obs = np.ones(23, dtype=np.float32)
    obs[0] = 2.5 / 4.0  # ray 0
    obs[1] = 2.5 / 4.0  # ray 1 (left 30)
    obs[2] = 2.5 / 4.0  # ray 2 (right 30)
    obs[3] = 2.5 / 4.0  # ray 3 (left 60)
    obs[4] = 0.8 / 4.0  # ray 4 (right 60) - obstacle approaching!
    obs[5] = 1.8 / 4.0  # ray 5 (left 90 wall)
    obs[6] = 1.8 / 4.0  # ray 6 (right 90 wall)
    obs[12:15] = [5.0 / 2.0, 0.0, 0.0]  # goal far ahead

    raw_act = VelocityAction(vx=0.60, vy=0.0, vz=0.0, yaw_rate=0.0)
    guided, direction, c_left, c_right = arb.arbitrate(raw_act, obs)

    # Guidance should detect the obstacle via ray 4 and steer LEFT to evade
    assert direction in ["LEFT", "WALL_AVOID_LEFT"]
    assert guided.vy > 0.10, f"Expected positive vy (left evasion), got {guided.vy}"


def test_bc_scaled_v2_checkpoint_integrity_and_timing():
    """Verify bc_policy_scaled_v2.pt exists, loads, and infers within 3ms."""
    ckpt_path = Path("models/checkpoints/bc_policy_scaled_v2.pt")
    assert ckpt_path.exists(), "Checkpoint bc_policy_scaled_v2.pt must exist."

    # Verify sha256
    with open(ckpt_path, "rb") as f:
        sha = hashlib.sha256(f.read()).hexdigest()
    assert sha == "d02d5712a3e292c535c8ff78898c7c0a8d5758c8fd0a22738c998f67c8ba79d1"

    controller = BehaviorCloningPolicyController(ckpt_path)
    obs = np.ones(23, dtype=np.float32) * 0.5

    # Benchmark CPU inference latency over 100 iterations
    latencies = []
    for _ in range(100):
        t0 = time.perf_counter()
        act = controller.predict(obs)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        latencies.append(dt_ms)

    mean_lat = float(np.mean(latencies))
    p99_lat = float(np.percentile(latencies, 99))
    assert mean_lat < 1.0, f"Mean latency {mean_lat:.3f}ms exceeds 1.0ms limit."
    assert p99_lat < 3.0, f"P99 latency {p99_lat:.3f}ms exceeds 3.0ms real-time limit."
    assert -0.80 <= act.vx <= 0.80
    assert -0.80 <= act.vy <= 0.80


def test_immutable_checkpoints_preserved():
    """Verify that historical and baseline checkpoints are unmodified."""
    expected_hashes = {
        "models/checkpoints/bc_policy_best.pt": "93c4d7d1d8476b8d288a24fca1862d87133696f78b9603aa622bcc367afad44f",
        "models/checkpoints/bc_policy_scaled_v1.pt": "68470c642957594b6894a33d39747735a1c27223f856f7d7723842e68aa8dad0",
        "models/checkpoints/ppo_policy_best.pt": "33f6d8b705edcffa430c48b2a21ed33a3e00f81b1c8195323134dd8be2a00ee0",
    }
    for path_str, exp_hash in expected_hashes.items():
        p = Path(path_str)
        assert p.exists(), f"Expected checkpoint {p} does not exist."
        with open(p, "rb") as f:
            h = hashlib.sha256(f.read()).hexdigest()
        assert h == exp_hash, f"Checkpoint {p} hash mismatch! Expected {exp_hash}, got {h}"


def test_confirmation_results_exceed_90():
    """Verify that held-out confirmation campaign achieved >= 90% success with zero collisions."""
    conf_summary_path = Path("experiments/scaled_campaign/exceed_90_campaign/confirmation_summary.json")
    assert conf_summary_path.exists(), "confirmation_summary.json must exist."

    with open(conf_summary_path) as f:
        data = json.load(f)

    cand = data.get("best_candidate_scaled_guided_bc_v2")
    assert cand is not None, "best_candidate_scaled_guided_bc_v2 record missing."

    succ_pct = cand["feasible_success_pct"]
    cols = cand["total_collision_episodes"]
    ci_low, ci_high = cand["wilson_ci_95"]

    assert succ_pct >= 90.0, f"Success rate {succ_pct}% is below required 90.0% threshold."
    assert cols == 0, f"Collisions detected: {cols} (expected 0)."
    assert ci_low >= 80.0, f"95% CI lower bound {ci_low}% should be >= 80%."
