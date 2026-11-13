"""run_exceed_90_experiments.py - Controlled Development & Held-Out Confirmation Suite.

Executes Phases 4 and 5 of the Exceed 90% Navigation Campaign:
1. Controlled Development Experiments on Level 2 (Seeds 20000..20049, N=50):
   - Exp 1: Baseline scaled_guided_bc_v1 (bc_v1 + guidance_v1)
   - Exp 2: bc_v1_guidance_v2 (bc_v1 + harmonized guidance_v2)
   - Exp 3: bc_v2_raw (retrained bc_v2 unguided)
   - Exp 4: scaled_guided_bc_v2 (retrained bc_v2 + harmonized guidance_v2)
2. Held-Out Confirmation on Level 2 (Seeds 20050..20099, N=50 untouched seeds).
3. Generalization Benchmark across Levels 1, 3, and 4.
"""

import argparse
import csv
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from controllers.bc_policy import BehaviorCloningPolicyController
from controllers.registry import ReactiveBaselineController
from controllers.safety_supervisor import SafetySupervisor
from core.contracts import ActionLimits, VelocityAction, compute_forward_clearance_m
from core.geometry import Corridor3D, TerminationReason
from core.guidance import GuidanceConfig, StatelessClearanceGuidanceArbitrator
from core.physical_dimensions import (
    DifficultyLevel,
    PhysicalDroneModel,
    RouteFeasibilityChecker,
    ScaledEnvironmentFactory,
)
from sim.corridor_env import CorridorSimEnv


class GenericGuidedController:
    """Wraps a base policy with a guidance arbitrator."""
    def __init__(self, base_controller, arbitrator, name: str):
        self.base_controller = base_controller
        self.arbitrator = arbitrator
        self.name = name

    def reset(self):
        if hasattr(self.base_controller, "reset"):
            self.base_controller.reset()
        if hasattr(self.arbitrator, "reset"):
            self.arbitrator.reset()

    def predict(self, obs: np.ndarray) -> VelocityAction:
        raw_act = self.base_controller.predict(obs)
        guided_act, _, _, _ = self.arbitrator.arbitrate(raw_act, obs)
        return guided_act


@dataclass
class EpisodeRecord:
    experiment_stage: str # "dev", "confirmation", "generalization"
    level: int
    level_name: str
    seed: int
    controller_id: str
    is_feasible: bool
    status: str
    success: bool
    safe_rejection: bool
    obstacle_collision: bool
    wall_collision: bool
    collision_episode: bool
    safety_envelope_violation: bool
    total_steps: int
    flight_time_s: float
    flight_distance_m: float
    mean_speed_mps: float
    min_obstacle_clearance_m: float
    min_wall_clearance_m: float
    overall_min_clearance_m: float
    safety_interventions: int
    terminal_dist_to_goal_m: float


def compute_wilson_ci(k: int, n: int, confidence: float = 0.95) -> Tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    z = 1.959964
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    margin = (z / denom) * np.sqrt(p * (1.0 - p) / n + (z * z) / (4.0 * n * n))
    return float(max(0.0, center - margin) * 100.0), float(min(1.0, center + margin) * 100.0)


def run_single_episode(
    env: CorridorSimEnv,
    controller: Any,
    supervisor: SafetySupervisor,
    checker: RouteFeasibilityChecker,
    stage: str,
    level: int,
    level_name: str,
    seed: int,
    controller_id: str,
    max_steps: int = 1000,
    envelope_threshold_m: float = 0.15,
) -> EpisodeRecord:
    if hasattr(controller, "reset"):
        controller.reset()

    obs, _ = env.reset(seed=seed)
    corridor = env.corridor

    feas_res = checker.evaluate_feasibility(corridor, env.obstacles)

    total_steps = 0
    min_obs_clearance = float("inf")
    min_wall_clearance = float("inf")
    envelope_violation_count = 0
    safety_interventions = 0
    outcome = "TIMEOUT"
    success = False
    obstacle_collision = False
    wall_collision = False
    safe_rejection = False
    flight_distance = 0.0
    prev_pos = env._get_drone_position()

    dt = env.dt

    for step in range(max_steps):
        total_steps += 1
        pos = env._get_drone_position()
        vel = env._get_drone_velocity()

        flight_distance += float(np.linalg.norm(pos - prev_pos))
        prev_pos = pos

        d_obs = corridor.obstacle_clearance(pos, env.obstacles)
        d_wall = corridor.boundary_distance(pos)
        min_obs_clearance = min(min_obs_clearance, d_obs)
        min_wall_clearance = min(min_wall_clearance, d_wall)

        if d_obs < envelope_threshold_m or d_wall < envelope_threshold_m:
            envelope_violation_count += 1

        term_reason, _ = corridor.check_termination(pos, vel, env.obstacles, step_count=step, max_steps=max_steps)
        if term_reason == TerminationReason.SUCCESS:
            success = True
            outcome = "SUCCESS"
            break
        elif term_reason == TerminationReason.CRASH_OBSTACLE:
            obstacle_collision = True
            outcome = "CRASH_OBSTACLE"
            break
        elif term_reason == TerminationReason.CRASH_BOUNDARY:
            wall_collision = True
            outcome = "CRASH_BOUNDARY"
            break

        if not feas_res.is_feasible:
            if pos[0] >= 5.0 and np.linalg.norm(vel) < 0.06 and d_obs > envelope_threshold_m:
                if step > 200:
                    safe_rejection = True
                    outcome = "SAFE_REJECT"
                    break

        raw_act = controller.predict(obs)

        min_front = compute_forward_clearance_m(obs, max_range_m=env.sensor_config.max_range_m)
        safe_act, state, _ = supervisor.filter_action(
            proposed_action=raw_act,
            current_position=pos,
            current_velocity=vel,
            min_front_range=min_front,
            dt=dt,
            is_localization_healthy=True,
            telemetry_age_s=0.0,
            observation=obs,
        )

        if state.value != "nominal":
            safety_interventions += 1

        norm_act = safe_act.to_normalized(env.action_limits)
        obs, _, terminated, truncated, info = env.step(norm_act)

        if terminated or truncated:
            term_val = info.get("termination_reason", "")
            if term_val == "success":
                success = True
                outcome = "SUCCESS"
            elif term_val == "crash_obstacle":
                obstacle_collision = True
                outcome = "CRASH_OBSTACLE"
            elif term_val == "crash_boundary":
                wall_collision = True
                outcome = "CRASH_BOUNDARY"
            elif truncated:
                outcome = "TIMEOUT"
            break

    if not feas_res.is_feasible and not obstacle_collision and not wall_collision and min_obs_clearance >= 0.10:
        safe_rejection = True
        outcome = "SAFE_REJECT"

    final_pos = env._get_drone_position()
    term_dist_goal = float(np.linalg.norm(final_pos - corridor.goal_position))
    flight_time_s = float(total_steps * dt)
    mean_speed = float(flight_distance / max(1e-3, flight_time_s))

    has_collision = obstacle_collision or wall_collision
    overall_min = min(min_obs_clearance, min_wall_clearance)

    return EpisodeRecord(
        experiment_stage=stage,
        level=level,
        level_name=level_name,
        seed=seed,
        controller_id=controller_id,
        is_feasible=feas_res.is_feasible,
        status=outcome,
        success=success,
        safe_rejection=safe_rejection,
        obstacle_collision=obstacle_collision,
        wall_collision=wall_collision,
        collision_episode=has_collision,
        safety_envelope_violation=(envelope_violation_count > 0),
        total_steps=total_steps,
        flight_time_s=flight_time_s,
        flight_distance_m=round(flight_distance, 3),
        mean_speed_mps=round(mean_speed, 3),
        min_obstacle_clearance_m=float(min_obs_clearance),
        min_wall_clearance_m=float(min_wall_clearance),
        overall_min_clearance_m=float(overall_min),
        safety_interventions=safety_interventions,
        terminal_dist_to_goal_m=round(term_dist_goal, 3),
    )


def summarize_episodes(ep_list: List[EpisodeRecord]) -> Dict[str, Any]:
    feas = [r for r in ep_list if r.is_feasible]
    infeas = [r for r in ep_list if not r.is_feasible]
    succ = sum(1 for r in feas if r.success)
    ci_low, ci_high = compute_wilson_ci(succ, len(feas))
    succ_steps = [r.total_steps for r in feas if r.success]
    succ_times = [r.flight_time_s for r in feas if r.success]
    succ_speeds = [r.mean_speed_mps for r in feas if r.success]
    succ_min_clear = [r.overall_min_clearance_m for r in feas if r.success]

    tot_cols = sum(1 for r in ep_list if r.collision_episode)
    env_viol = sum(1 for r in ep_list if r.safety_envelope_violation)
    interventions = sum(r.safety_interventions for r in ep_list)
    infeas_rej = sum(1 for r in infeas if r.safe_rejection)

    return {
        "feasible_total": len(feas),
        "feasible_success": succ,
        "feasible_success_pct": round(succ / max(1, len(feas)) * 100.0, 2),
        "wilson_ci_95": [round(ci_low, 2), round(ci_high, 2)],
        "infeasible_total": len(infeas),
        "infeasible_safe_rejections": infeas_rej,
        "infeasible_rejection_pct": round(infeas_rej / max(1, len(infeas)) * 100.0, 2) if infeas else 100.0,
        "total_collision_episodes": tot_cols,
        "envelope_violation_episodes": env_viol,
        "total_safety_interventions": interventions,
        "min_clearance_across_all_m": round(float(min(r.overall_min_clearance_m for r in ep_list)), 4) if ep_list else 0.0,
        "min_clearance_successful_m": round(float(min(succ_min_clear)), 4) if succ_min_clear else 0.0,
        "mean_steps_successful": round(float(np.mean(succ_steps)), 1) if succ_steps else 1000.0,
        "mean_flight_time_s": round(float(np.mean(succ_times)), 2) if succ_times else 50.0,
        "mean_velocity_mps": round(float(np.mean(succ_speeds)), 3) if succ_speeds else 0.0,
    }


def run_campaign():
    out_dir = Path("experiments/scaled_campaign/exceed_90_campaign")
    out_dir.mkdir(parents=True, exist_ok=True)
    checker = RouteFeasibilityChecker(PhysicalDroneModel())

    # Controllers & Guidance
    drone_radius = 0.325
    g_cfg_v1 = GuidanceConfig.for_65cm_uav(drone_radius)
    g_cfg_v2 = GuidanceConfig.for_65cm_uav_v2(drone_radius)

    arb_v1 = StatelessClearanceGuidanceArbitrator(config=g_cfg_v1)
    arb_v2 = StatelessClearanceGuidanceArbitrator(config=g_cfg_v2)

    bc_v1 = BehaviorCloningPolicyController("models/checkpoints/bc_policy_scaled_v1.pt")
    bc_v2 = BehaviorCloningPolicyController("models/checkpoints/bc_policy_scaled_v2.pt")

    dev_controllers = {
        "scaled_guided_bc_v1": GenericGuidedController(bc_v1, arb_v1, "scaled_guided_bc_v1"),
        "bc_v1_guidance_v2": GenericGuidedController(bc_v1, arb_v2, "bc_v1_guidance_v2"),
        "bc_v2_raw": bc_v2,
        "scaled_guided_bc_v2": GenericGuidedController(bc_v2, arb_v2, "scaled_guided_bc_v2"),
    }

    # =========================================================================
    # Phase 4: Controlled Development Experiments (Level 2, Seeds 20000..20049)
    # =========================================================================
    dev_seeds = list(range(20000, 20050))
    print("=" * 80)
    print(">>> PHASE 4: CONTROLLED DEVELOPMENT EXPERIMENTS (N=50 SEEDS) <<<")
    print("=" * 80)

    cfg_l2 = ScaledEnvironmentFactory.get_environment_config(DifficultyLevel.LEVEL_2_STANDARD)
    env_l2 = CorridorSimEnv(scenario=cfg_l2, seed=dev_seeds[0])
    sup_l2 = SafetySupervisor(env_l2.corridor)

    dev_results: List[EpisodeRecord] = []
    dev_summary: Dict[str, Any] = {}

    for cid, ctrl in dev_controllers.items():
        t0 = time.time()
        c_succ = 0
        c_col = 0
        c_env = 0
        ep_recs = []

        for s in dev_seeds:
            rec = run_single_episode(
                env=env_l2,
                controller=ctrl,
                supervisor=sup_l2,
                checker=checker,
                stage="dev",
                level=2,
                level_name="Level 2 (Standard)",
                seed=s,
                controller_id=cid,
            )
            dev_results.append(rec)
            ep_recs.append(rec)
            if rec.success:
                c_succ += 1
            if rec.collision_episode:
                c_col += 1
            if rec.safety_envelope_violation:
                c_env += 1

        dev_summary[cid] = summarize_episodes(ep_recs)
        elapsed = time.time() - t0
        print(f"  {cid:<22} | Success: {c_succ:>2}/50 ({c_succ/50*100:.1f}%) [95% CI: {dev_summary[cid]['wilson_ci_95']}] | Cols: {c_col} | Envs: {c_env} | {elapsed:.1f}s")

    env_l2.close()

    dev_csv = out_dir / "development_episodes.csv"
    with open(dev_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(dev_results[0]).keys()))
        writer.writeheader()
        for r in dev_results:
            writer.writerow(asdict(r))

    with open(out_dir / "development_summary.json", "w") as f:
        json.dump(dev_summary, f, indent=2)

    # =========================================================================
    # Phase 5: Held-Out Confirmation on Level 2 (Seeds 20050..20099, N=50)
    # =========================================================================
    # Select the top candidate (scaled_guided_bc_v2) and verify against baseline
    confirmation_seeds = list(range(20050, 20100))
    print("\n" + "=" * 80)
    print(">>> PHASE 5: HELD-OUT CONFIRMATION (N=50 UNTOUCHED SEEDS) <<<")
    print("=" * 80)

    env_l2_conf = CorridorSimEnv(scenario=cfg_l2, seed=confirmation_seeds[0])
    sup_l2_conf = SafetySupervisor(env_l2_conf.corridor)

    conf_controllers = {
        "baseline_scaled_guided_bc_v1": GenericGuidedController(bc_v1, arb_v1, "baseline_scaled_guided_bc_v1"),
        "best_candidate_scaled_guided_bc_v2": GenericGuidedController(bc_v2, arb_v2, "best_candidate_scaled_guided_bc_v2"),
    }

    conf_results: List[EpisodeRecord] = []
    conf_summary: Dict[str, Any] = {}

    for cid, ctrl in conf_controllers.items():
        t0 = time.time()
        c_succ = 0
        c_col = 0
        c_env = 0
        ep_recs = []

        for s in confirmation_seeds:
            rec = run_single_episode(
                env=env_l2_conf,
                controller=ctrl,
                supervisor=sup_l2_conf,
                checker=checker,
                stage="confirmation",
                level=2,
                level_name="Level 2 (Standard)",
                seed=s,
                controller_id=cid,
            )
            conf_results.append(rec)
            ep_recs.append(rec)
            if rec.success:
                c_succ += 1
            if rec.collision_episode:
                c_col += 1
            if rec.safety_envelope_violation:
                c_env += 1

        conf_summary[cid] = summarize_episodes(ep_recs)
        elapsed = time.time() - t0
        print(f"  {cid:<34} | Success: {c_succ:>2}/50 ({c_succ/50*100:.1f}%) [95% CI: {conf_summary[cid]['wilson_ci_95']}] | Cols: {c_col} | Envs: {c_env} | {elapsed:.1f}s")

    env_l2_conf.close()

    conf_csv = out_dir / "confirmation_episodes.csv"
    with open(conf_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(conf_results[0]).keys()))
        writer.writeheader()
        for r in conf_results:
            writer.writerow(asdict(r))

    with open(out_dir / "confirmation_summary.json", "w") as f:
        json.dump(conf_summary, f, indent=2)

    # =========================================================================
    # Phase 5b: Multi-Tier Generalization of Winning Candidate (Levels 1, 3, 4)
    # =========================================================================
    print("\n" + "=" * 80)
    print(">>> MULTI-TIER GENERALIZATION (LEVELS 1, 3, 4) <<<")
    print("=" * 80)

    other_suites = [
        (1, "Level 1 (Basic)", DifficultyLevel.LEVEL_1_BASIC, list(range(20100, 20150))),
        (3, "Level 3 (Challenging)", DifficultyLevel.LEVEL_3_CHALLENGING, list(range(20200, 20250))),
        (4, "Level 4 (Boundary & Feasibility)", DifficultyLevel.LEVEL_4_FEASIBILITY_BOUNDARY, list(range(20300, 20350))),
    ]

    best_candidate = GenericGuidedController(bc_v2, arb_v2, "scaled_guided_bc_v2")
    reactive_ref = ReactiveBaselineController(cruise_speed=0.60, drone_radius=drone_radius)

    gen_results: List[EpisodeRecord] = []
    gen_summary: Dict[str, Any] = {}

    for lvl_idx, lvl_name, lvl_enum, seeds in other_suites:
        print(f"\n--- Generalization Evaluation: {lvl_name} (N={len(seeds)}) ---")
        cfg_other = ScaledEnvironmentFactory.get_environment_config(lvl_enum)
        env_other = CorridorSimEnv(scenario=cfg_other, seed=seeds[0])
        sup_other = SafetySupervisor(env_other.corridor)

        for cid, ctrl in [("scaled_guided_bc_v2", best_candidate), ("reactive_baseline", reactive_ref)]:
            t0 = time.time()
            c_succ = 0
            c_col = 0
            c_env = 0
            ep_recs = []

            for s in seeds:
                rec = run_single_episode(
                    env=env_other,
                    controller=ctrl,
                    supervisor=sup_other,
                    checker=checker,
                    stage="generalization",
                    level=lvl_idx,
                    level_name=lvl_name,
                    seed=s,
                    controller_id=cid,
                )
                gen_results.append(rec)
                ep_recs.append(rec)
                if rec.success:
                    c_succ += 1
                if rec.collision_episode:
                    c_col += 1
                if rec.safety_envelope_violation:
                    c_env += 1

            s_key = f"{lvl_name}_{cid}"
            gen_summary[s_key] = summarize_episodes(ep_recs)
            elapsed = time.time() - t0
            print(f"  {cid:<22} | Success: {c_succ:>2}/{len(seeds)} ({c_succ/len(seeds)*100:.1f}%) | Cols: {c_col} | Envs: {c_env} | {elapsed:.1f}s")

        env_other.close()

    gen_csv = out_dir / "generalization_episodes.csv"
    with open(gen_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(gen_results[0]).keys()))
        writer.writeheader()
        for r in gen_results:
            writer.writerow(asdict(r))

    with open(out_dir / "generalization_summary.json", "w") as f:
        json.dump(gen_summary, f, indent=2)

    print("\n" + "=" * 80)
    print(">>> EXCEED 90% EXPERIMENTAL CAMPAIGN COMPLETE <<<")
    print(f"All artifacts written to: {out_dir.resolve()}")
    print("=" * 80)


if __name__ == "__main__":
    run_campaign()

