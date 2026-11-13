"""run_exhaustive_v2_vs_v3_evaluation.py - Exhaustive Paired Regression Sweep.

Compares frozen baseline `scaled_guided_bc_v2` against directional-persistence candidate
`scaled_guided_bc_v3` across all 250 canonical evaluation seeds in the scaled workspace:
- Level 1 (Basic): Seeds 20100..20149 (N=50)
- Level 2 (Standard): Seeds 20000..20099 (N=100: 50 dev + 50 confirmation)
- Level 3 (Challenging): Seeds 20200..20249 (N=50)
- Level 4 (Feasibility & Boundary): Seeds 20300..20349 (N=50: 25 feasible, 25 infeasible)

Total = 500 episodes.
"""

import csv
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from controllers.bc_policy import BehaviorCloningPolicyController
from controllers.safety_supervisor import SafetySupervisor, SafetyState
from core.contracts import VelocityAction, compute_forward_clearance_m
from core.geometry import TerminationReason
from core.guidance import GuidanceConfig, StatelessClearanceGuidanceArbitrator
from core.physical_dimensions import (
    DifficultyLevel,
    PhysicalDroneModel,
    RouteFeasibilityChecker,
    ScaledEnvironmentFactory,
)
from scripts.run_exceed_90_experiments import GenericGuidedController, compute_wilson_ci
from sim.corridor_env import CorridorSimEnv


@dataclass
class ExhaustiveEpisodeRecord:
    tier: str
    level: int
    seed: int
    controller_id: str
    is_feasible: bool
    status: str
    success: bool
    safe_rejection: bool
    timeout: bool
    obstacle_collision: bool
    wall_collision: bool
    collision_episode: bool
    safety_envelope_violation: bool
    envelope_violation_steps: int
    total_steps: int
    flight_time_s: float
    flight_distance_m: float
    mean_speed_mps: float
    min_obstacle_clearance_m: float
    min_wall_clearance_m: float
    overall_min_clearance_m: float
    safety_interventions: int
    terminal_dist_to_goal_m: float
    bottleneck_gap_m: float
    guidance_activations: int
    tie_break_activations: int


def run_single_episode(
    env: CorridorSimEnv,
    controller: Any,
    supervisor: SafetySupervisor,
    checker: RouteFeasibilityChecker,
    tier_name: str,
    level_num: int,
    seed: int,
    controller_id: str,
    max_steps: int = 1000,
    envelope_threshold_m: float = 0.15,
) -> ExhaustiveEpisodeRecord:
    if hasattr(controller, "reset"):
        controller.reset()
    if hasattr(supervisor, "reset"):
        supervisor.reset()

    obs, _ = env.reset(seed=seed)
    corridor = env.corridor
    feas_res = checker.evaluate_feasibility(corridor, env.obstacles)

    total_steps = 0
    min_obs_clearance = float("inf")
    min_wall_clearance = float("inf")
    envelope_violation_steps = 0
    safety_interventions = 0
    guidance_activations = 0
    tie_break_activations = 0

    outcome = "TIMEOUT"
    success = False
    obstacle_collision = False
    wall_collision = False
    safe_rejection = False
    flight_distance = 0.0
    prev_pos = env._get_drone_position()
    dt = env.dt

    halt_counter = 0

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
            envelope_violation_steps += 1

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

        # Check for sustained safe halt in front of blocked barrier in infeasible corridors
        if not feas_res.is_feasible:
            if pos[0] >= 5.0 and abs(vel[0]) < 0.08 and d_obs > 0.05:
                halt_counter += 1
                if halt_counter >= 100:  # Safely held position in front of impassable barrier for 5.0s
                    safe_rejection = True
                    outcome = "SAFE_REJECT"
                    break
            else:
                halt_counter = max(0, halt_counter - 1)

        raw_act = controller.predict(obs)
        min_front = compute_forward_clearance_m(obs, max_range_m=env.sensor_config.max_range_m)

        # Guidance diagnostics
        if hasattr(controller, "arbitrator"):
            arb = controller.arbitrator
            if arb.last_dir != "NONE":
                guidance_activations += 1
                if "TIE_BREAK" in arb.last_dir:
                    tie_break_activations += 1

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

    # Final post-hoc evaluation for infeasible blocked passages that reached timeout safely
    has_collision = obstacle_collision or wall_collision
    final_pos = env._get_drone_position()
    final_d_obs = corridor.obstacle_clearance(final_pos, env.obstacles)

    if not feas_res.is_feasible and not has_collision:
        if final_pos[0] >= 5.0 and final_d_obs > 0.05:
            safe_rejection = True
            outcome = "SAFE_REJECT"

    dist_to_goal = float(corridor.distance_to_goal(final_pos))
    flight_time = round(total_steps * dt, 2)
    mean_speed = round(flight_distance / flight_time, 3) if flight_time > 0 else 0.0
    overall_min = min(min_obs_clearance, min_wall_clearance)

    return ExhaustiveEpisodeRecord(
        tier=tier_name,
        level=level_num,
        seed=seed,
        controller_id=controller_id,
        is_feasible=feas_res.is_feasible,
        status=outcome,
        success=success,
        safe_rejection=safe_rejection,
        timeout=(outcome == "TIMEOUT"),
        obstacle_collision=obstacle_collision,
        wall_collision=wall_collision,
        collision_episode=(obstacle_collision or wall_collision),
        safety_envelope_violation=(envelope_violation_steps > 0),
        envelope_violation_steps=envelope_violation_steps,
        total_steps=total_steps,
        flight_time_s=flight_time,
        flight_distance_m=round(flight_distance, 3),
        mean_speed_mps=mean_speed,
        min_obstacle_clearance_m=round(float(min_obs_clearance), 4),
        min_wall_clearance_m=round(float(min_wall_clearance), 4),
        overall_min_clearance_m=round(float(overall_min), 4),
        safety_interventions=safety_interventions,
        terminal_dist_to_goal_m=round(dist_to_goal, 3),
        bottleneck_gap_m=round(feas_res.min_bottleneck_gap_m, 4),
        guidance_activations=guidance_activations,
        tie_break_activations=tie_break_activations,
    )


def summarize_episodes(records: List[ExhaustiveEpisodeRecord]) -> Dict[str, Any]:
    n = len(records)
    if n == 0:
        return {}

    feas_recs = [r for r in records if r.is_feasible]
    infeas_recs = [r for r in records if not r.is_feasible]

    feas_succ = sum(1 for r in feas_recs if r.success)
    infeas_safe = sum(1 for r in infeas_recs if r.safe_rejection)

    total_cols = sum(1 for r in records if r.collision_episode)
    env_viol_eps = sum(1 for r in records if r.safety_envelope_violation)
    total_env_steps = sum(r.envelope_violation_steps for r in records)
    total_interventions = sum(r.safety_interventions for r in records)
    total_steps = sum(r.total_steps for r in records)

    succ_times = [r.flight_time_s for r in feas_recs if r.success]
    succ_speeds = [r.mean_speed_mps for r in feas_recs if r.success]
    succ_min_c = [r.overall_min_clearance_m for r in feas_recs if r.success]
    all_min_c = [r.overall_min_clearance_m for r in records]

    ci_feas = compute_wilson_ci(feas_succ, len(feas_recs)) if feas_recs else (0.0, 0.0)
    ci_env = compute_wilson_ci(env_viol_eps, n)

    return {
        "total_episodes": n,
        "feasible_total": len(feas_recs),
        "feasible_success": feas_succ,
        "feasible_success_pct": round(feas_succ / len(feas_recs) * 100.0, 2) if feas_recs else 0.0,
        "feasible_wilson_ci_95": [round(ci_feas[0] * 100.0, 2), round(ci_feas[1] * 100.0, 2)],
        "infeasible_total": len(infeas_recs),
        "infeasible_safe_rejections": infeas_safe,
        "infeasible_rejection_pct": round(infeas_safe / len(infeas_recs) * 100.0, 2) if infeas_recs else 0.0,
        "total_collision_episodes": total_cols,
        "collision_rate_pct": round(total_cols / n * 100.0, 2),
        "envelope_violation_episodes": env_viol_eps,
        "envelope_violation_rate_pct": round(env_viol_eps / n * 100.0, 2),
        "envelope_violation_wilson_ci_95": [round(ci_env[0] * 100.0, 2), round(ci_env[1] * 100.0, 2)],
        "total_envelope_violation_steps": total_env_steps,
        "total_safety_interventions": total_interventions,
        "intervention_rate_per_1000_steps": round(total_interventions / total_steps * 1000.0, 2) if total_steps > 0 else 0.0,
        "min_clearance_overall_m": round(float(min(all_min_c)), 4) if all_min_c else 0.0,
        "min_clearance_p05_m": round(float(np.percentile(all_min_c, 5)), 4) if all_min_c else 0.0,
        "min_clearance_median_m": round(float(np.median(all_min_c)), 4) if all_min_c else 0.0,
        "min_clearance_successful_m": round(float(min(succ_min_c)), 4) if succ_min_c else 0.0,
        "mean_flight_time_s": round(float(np.mean(succ_times)), 2) if succ_times else 50.0,
        "mean_velocity_mps": round(float(np.mean(succ_speeds)), 3) if succ_speeds else 0.0,
    }


def run_exhaustive_sweep():
    print("=" * 80)
    print(">>> EXHAUSTIVE V2 VS V3 REGRESSION EVALUATION SWEEP <<<")
    print("=" * 80)

    out_dir = Path("experiments/scaled_campaign/exhaustive_v2_v3")
    out_dir.mkdir(parents=True, exist_ok=True)
    drone_radius = 0.325
    checker = RouteFeasibilityChecker(PhysicalDroneModel())

    ckpt_path = "models/checkpoints/bc_policy_scaled_v2.pt"
    bc = BehaviorCloningPolicyController(ckpt_path)

    # Controller 1: Frozen v2 Baseline
    g_cfg_v2 = GuidanceConfig.for_65cm_uav_v2(drone_radius=drone_radius)
    arb_v2 = StatelessClearanceGuidanceArbitrator(config=g_cfg_v2)
    ctrl_v2 = GenericGuidedController(bc, arb_v2, "scaled_guided_bc_v2")

    # Controller 2: Candidate v3 with Directional Persistence
    g_cfg_v3 = GuidanceConfig.for_65cm_uav_v3(drone_radius=drone_radius)
    arb_v3 = StatelessClearanceGuidanceArbitrator(config=g_cfg_v3)
    ctrl_v3 = GenericGuidedController(bc, arb_v3, "scaled_guided_bc_v3")

    controllers = [
        ("scaled_guided_bc_v2", ctrl_v2),
        ("scaled_guided_bc_v3", ctrl_v3),
    ]

    tier_suites = [
        ("Level 1 (Basic)", 1, DifficultyLevel.LEVEL_1_BASIC, list(range(20100, 20150))),
        ("Level 2 Dev (Standard)", 2, DifficultyLevel.LEVEL_2_STANDARD, list(range(20000, 20050))),
        ("Level 2 Confirmation", 2, DifficultyLevel.LEVEL_2_STANDARD, list(range(20050, 20100))),
        ("Level 3 (Challenging)", 3, DifficultyLevel.LEVEL_3_CHALLENGING, list(range(20200, 20250))),
        ("Level 4 (Boundary & Feasibility)", 4, DifficultyLevel.LEVEL_4_FEASIBILITY_BOUNDARY, list(range(20300, 20350))),
    ]

    all_records: List[ExhaustiveEpisodeRecord] = []
    summary_dict: Dict[str, Any] = {}

    start_total = time.time()

    for tier_name, lvl_num, lvl_enum, seeds in tier_suites:
        print(f"\nEvaluating {tier_name} [N={len(seeds)} Seeds, Range: {seeds[0]}..{seeds[-1]}]")
        cfg = ScaledEnvironmentFactory.get_environment_config(lvl_enum)
        env = CorridorSimEnv(scenario=cfg, seed=seeds[0])
        sup = SafetySupervisor(env.corridor)

        for cid, ctrl in controllers:
            t0 = time.time()
            tier_records: List[ExhaustiveEpisodeRecord] = []
            for s in seeds:
                rec = run_single_episode(
                    env=env,
                    controller=ctrl,
                    supervisor=sup,
                    checker=checker,
                    tier_name=tier_name,
                    level_num=lvl_num,
                    seed=s,
                    controller_id=cid,
                )
                tier_records.append(rec)
                all_records.append(rec)

            elapsed = time.time() - t0
            summary_dict[f"{tier_name}_{cid}"] = summarize_episodes(tier_records)
            sm = summary_dict[f"{tier_name}_{cid}"]
            if sm["infeasible_total"] > 0:
                print(
                    f"  {cid:<22} | Feas: {sm['feasible_success']:>2}/{sm['feasible_total']} ({sm['feasible_success_pct']}%) "
                    f"[95% CI: {sm['feasible_wilson_ci_95']}] | Safe Rej: {sm['infeasible_safe_rejections']:>2}/{sm['infeasible_total']} "
                    f"| Cols: {sm['total_collision_episodes']} | Envs: {sm['envelope_violation_episodes']} | {elapsed:.1f}s"
                )
            else:
                print(
                    f"  {cid:<22} | Succ: {sm['feasible_success']:>2}/{sm['feasible_total']} ({sm['feasible_success_pct']}%) "
                    f"[95% CI: {sm['feasible_wilson_ci_95']}] | Cols: {sm['total_collision_episodes']} | Envs: {sm['envelope_violation_episodes']} | {elapsed:.1f}s"
                )

        env.close()

    total_time = time.time() - start_total
    print(f"\nAll 500 episodes completed in {total_time:.1f}s.")

    # Save CSV
    csv_file = out_dir / "exhaustive_episodes.csv"
    with open(csv_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(all_records[0]).keys()))
        writer.writeheader()
        for r in all_records:
            writer.writerow(asdict(r))
    print(f"Saved episode data to {csv_file}")

    # Compute Overall Controller Summaries
    for cid in ["scaled_guided_bc_v2", "scaled_guided_bc_v3"]:
        c_recs = [r for r in all_records if r.controller_id == cid]
        summary_dict[f"OVERALL_{cid}"] = summarize_episodes(c_recs)

    # Save Summary JSON
    json_file = out_dir / "exhaustive_summary.json"
    with open(json_file, "w") as f:
        json.dump(summary_dict, f, indent=2)
    print(f"Saved summary to {json_file}")


if __name__ == "__main__":
    run_exhaustive_sweep()
