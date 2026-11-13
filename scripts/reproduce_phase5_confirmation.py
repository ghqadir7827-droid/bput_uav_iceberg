"""reproduce_phase5_confirmation.py - Freeze & Reproduce Level 2 Confirmation Baseline.

Strict Protocol:
- Controller: scaled_guided_bc_v2 (bc_policy_scaled_v2.pt + GuidanceConfig.for_65cm_uav_v2)
- Environment: Level 2 Standard (Corridor width=3.8m, height=2.5m, length=14m, R_drone=0.325m)
- Evaluation Seeds: 20050..20099 (N=50 held-out confirmation seeds)
- Episode Horizon: 1000 steps
- Verification: Compare reproduced metrics with confirmation_summary.json
"""

import csv
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time
import numpy as np

from controllers.bc_policy import BehaviorCloningPolicyController
from controllers.safety_supervisor import SafetySupervisor
from core.contracts import VelocityAction, compute_forward_clearance_m
from core.geometry import TerminationReason
from core.guidance import GuidanceConfig, StatelessClearanceGuidanceArbitrator
from core.physical_dimensions import (
    DifficultyLevel,
    PhysicalDroneModel,
    RouteFeasibilityChecker,
    ScaledEnvironmentFactory,
)
from sim.corridor_env import CorridorSimEnv
from scripts.run_exceed_90_experiments import EpisodeRecord, GenericGuidedController, summarize_episodes


def verify_checkpoint():
    ckpt_path = Path("models/checkpoints/bc_policy_scaled_v2.pt")
    assert ckpt_path.exists(), f"Missing checkpoint: {ckpt_path}"
    h = hashlib.sha256(ckpt_path.read_bytes()).hexdigest()
    expected_h = "d02d5712a3e292c535c8ff78898c7c0a8d5758c8fd0a22738c998f67c8ba79d1"
    print(f"[*] Checkpoint SHA-256: {h}")
    assert h == expected_h, f"Checkpoint mismatch! Expected {expected_h}, got {h}"
    return ckpt_path


def reproduce_confirmation():
    print("=" * 80)
    print(">>> PHASE 1: REPRODUCE LEVEL 2 HELD-OUT CONFIRMATION BASELINE <<<")
    print("=" * 80)

    ckpt_path = verify_checkpoint()
    drone_radius = 0.325

    g_cfg = GuidanceConfig.for_65cm_uav_v2(drone_radius=drone_radius)
    arb = StatelessClearanceGuidanceArbitrator(config=g_cfg)
    bc = BehaviorCloningPolicyController(str(ckpt_path))
    controller = GenericGuidedController(bc, arb, "reproduced_scaled_guided_bc_v2")

    cfg_l2 = ScaledEnvironmentFactory.get_environment_config(DifficultyLevel.LEVEL_2_STANDARD)
    env = CorridorSimEnv(scenario=cfg_l2, seed=20050)
    sup = SafetySupervisor(env.corridor)
    checker = RouteFeasibilityChecker(PhysicalDroneModel())

    confirmation_seeds = list(range(20050, 20100))
    records = []

    t0 = time.time()
    c_succ = 0
    c_col = 0
    c_env = 0

    from scripts.run_exceed_90_experiments import run_single_episode

    for s in confirmation_seeds:
        rec = run_single_episode(
            env=env,
            controller=controller,
            supervisor=sup,
            checker=checker,
            stage="reproduction",
            level=2,
            level_name="Level 2 (Standard)",
            seed=s,
            controller_id="reproduced_scaled_guided_bc_v2",
        )
        records.append(rec)
        if rec.success:
            c_succ += 1
        if rec.collision_episode:
            c_col += 1
        if rec.safety_envelope_violation:
            c_env += 1

    env.close()
    elapsed = time.time() - t0

    reproduced_summary = summarize_episodes(records)

    out_dir = Path("experiments/scaled_campaign/reproduction_level2")
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "reproduced_confirmation_episodes.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(records[0]).keys()))
        writer.writeheader()
        for r in records:
            writer.writerow(asdict(r))

    json_path = out_dir / "reproduced_confirmation_summary.json"
    with open(json_path, "w") as f:
        json.dump({"reproduced_scaled_guided_bc_v2": reproduced_summary}, f, indent=2)

    # Load original reported confirmation summary for comparison
    orig_path = Path("experiments/scaled_campaign/exceed_90_campaign/confirmation_summary.json")
    with open(orig_path) as f:
        orig_data = json.load(f)
    orig_cand = orig_data["best_candidate_scaled_guided_bc_v2"]

    print("\n" + "=" * 80)
    print("SIDE-BY-SIDE REPRODUCTION AUDIT (LEVEL 2, SEEDS 20050..20099, N=50)")
    print("=" * 80)
    print(f"{'Metric':<35} | {'Reported Baseline':<20} | {'Reproduced Baseline':<20} | Match")
    print("-" * 85)

    def comp(name, orig_val, rep_val, tolerance=1e-3):
        if isinstance(orig_val, float):
            m = abs(orig_val - rep_val) <= tolerance
            o_str = f"{orig_val:.4f}"
            r_str = f"{rep_val:.4f}"
        else:
            m = (orig_val == rep_val)
            o_str = str(orig_val)
            r_str = str(rep_val)
        print(f"{name:<35} | {o_str:<20} | {r_str:<20} | {'PASS' if m else 'DIFF'}")
        return m

    comp("Success Count", orig_cand["feasible_success"], reproduced_summary["feasible_success"])
    comp("Success Rate (%)", orig_cand["feasible_success_pct"], reproduced_summary["feasible_success_pct"])
    comp("95% Wilson CI Low", orig_cand["wilson_ci_95"][0], reproduced_summary["wilson_ci_95"][0])
    comp("95% Wilson CI High", orig_cand["wilson_ci_95"][1], reproduced_summary["wilson_ci_95"][1])
    comp("Total Collisions", orig_cand["total_collision_episodes"], reproduced_summary["total_collision_episodes"])
    comp("Envelope Violations", orig_cand["envelope_violation_episodes"], reproduced_summary["envelope_violation_episodes"])
    comp("Total Safety Interventions", orig_cand["total_safety_interventions"], reproduced_summary["total_safety_interventions"])
    comp("Min Clearance Overall (m)", orig_cand["min_clearance_across_all_m"], reproduced_summary["min_clearance_across_all_m"])
    comp("Mean Steps Successful", orig_cand["mean_steps_successful"], reproduced_summary["mean_steps_successful"])
    comp("Mean Flight Time (s)", orig_cand["mean_flight_time_s"], reproduced_summary["mean_flight_time_s"])
    comp("Mean Velocity (m/s)", orig_cand["mean_velocity_mps"], reproduced_summary["mean_velocity_mps"])

    print("-" * 85)
    print(f"Elapsed Time: {elapsed:.2f} s ({elapsed/50:.3f} s/episode)")
    print(f"Reproduction Artifacts written to: {out_dir.resolve()}")
    print("=" * 80)


if __name__ == "__main__":
    reproduce_confirmation()

