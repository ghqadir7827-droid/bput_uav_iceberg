"""run_controlled_regression_campaign.py - Multi-Tier Controlled Regression Campaign.

Evaluates and compares:
1. Frozen baseline: scaled_guided_bc_v2 (GuidanceConfig.for_65cm_uav_v2)
2. New candidate: scaled_guided_bc_v3 (GuidanceConfig.for_65cm_uav_v3 with directional persistence)

Across paired development seeds:
- Level 1: Basic (Seeds 20100..20149, N=50)
- Level 2: Standard (Seeds 20000..20049, N=50)
- Level 3: Challenging (Seeds 20200..20249, N=50)
- Level 4: Feasibility & Boundary (Seeds 20300..20349, N=50: 25 feasible, 25 infeasible)

Confirmation seeds (20050..20099) are kept strictly separate and preserved.
"""

import csv
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time
from typing import Any, Dict, List, Tuple
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
from scripts.run_exceed_90_experiments import GenericGuidedController, compute_wilson_ci


@dataclass
class RegressionEpisodeRecord:
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
    total_steps: int
    flight_time_s: float
    flight_distance_m: float
    mean_speed_mps: float
    min_obstacle_clearance_m: float
    min_wall_clearance_m: float
    overall_min_clearance_m: float
    safety_interventions: int
    terminal_dist_to_goal_m: float


def run_episode(
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
) -> RegressionEpisodeRecord:
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

        # Check for sustained safe halt in front of blocked barrier in infeasible corridors
        if not feas_res.is_feasible:
            if pos[0] >= 5.0 and abs(vel[0]) < 0.08 and d_obs > 0.05:
                halt_counter += 1
                if halt_counter >= 100:  # Safely held position in front of impassable passage for 5.0s
                    safe_rejection = True
                    outcome = "SAFE_REJECT"
                    break
            else:
                halt_counter = max(0, halt_counter - 1)

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

    # Final post-hoc evaluation for infeasible blocked passages that reached timeout safely
    has_collision = obstacle_collision or wall_collision
    final_pos = env._get_drone_position()
    final_d_obs = corridor.obstacle_clearance(final_pos, env.obstacles)

    if not feas_res.is_feasible and not has_collision:
        # If the corridor is blocked and the drone safely held position in front of it without crashing
        if final_pos[0] >= 5.0 and final_d_obs > 0.05:
            safe_rejection = True
            outcome = "SAFE_REJECT"

    term_dist_goal = float(np.linalg.norm(final_pos - corridor.goal_position))
    flight_time_s = float(total_steps * dt)
    mean_speed = float(flight_distance / max(1e-3, flight_time_s))
    overall_min = min(min_obs_clearance, min_wall_clearance)

    return RegressionEpisodeRecord(
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


def summarize_tier(records: List[RegressionEpisodeRecord]) -> Dict[str, Any]:
    feas = [r for r in records if r.is_feasible]
    infeas = [r for r in records if not r.is_feasible]

    succ = sum(1 for r in feas if r.success)
    succ_rate = round(succ / max(1, len(feas)) * 100.0, 2)
    ci_low, ci_high = compute_wilson_ci(succ, len(feas))

    rej = sum(1 for r in infeas if r.safe_rejection)
    rej_rate = round(rej / max(1, len(infeas)) * 100.0, 2) if infeas else 100.0
    rej_ci_l, rej_ci_h = compute_wilson_ci(rej, len(infeas)) if infeas else (100.0, 100.0)

    tot_cols = sum(1 for r in records if r.collision_episode)
    env_viols = sum(1 for r in records if r.safety_envelope_violation)
    interventions = sum(r.safety_interventions for r in records)
    mean_interv_per_ep = round(interventions / max(1, len(records)), 1)
    timeouts = sum(1 for r in records if r.timeout)

    succ_recs = [r for r in feas if r.success]
    succ_steps = [r.total_steps for r in succ_recs]
    succ_times = [r.flight_time_s for r in succ_recs]
    succ_speeds = [r.mean_speed_mps for r in succ_recs]
    succ_min_c = [r.overall_min_clearance_m for r in succ_recs]

    return {
        "total_episodes": len(records),
        "feasible_total": len(feas),
        "feasible_success": succ,
        "feasible_success_pct": succ_rate,
        "feasible_wilson_ci_95": [round(ci_low, 2), round(ci_high, 2)],
        "infeasible_total": len(infeas),
        "infeasible_safe_rejections": rej,
        "infeasible_rejection_pct": rej_rate,
        "infeasible_wilson_ci_95": [round(rej_ci_l, 2), round(rej_ci_h, 2)] if infeas else [100.0, 100.0],
        "total_collision_episodes": tot_cols,
        "envelope_violation_episodes": env_viols,
        "envelope_violation_pct": round(env_viols / max(1, len(records)) * 100.0, 2),
        "total_timeouts": timeouts,
        "timeout_pct": round(timeouts / max(1, len(records)) * 100.0, 2),
        "total_safety_interventions": interventions,
        "mean_interventions_per_ep": mean_interv_per_ep,
        "min_clearance_overall_m": round(float(min(r.overall_min_clearance_m for r in records)), 4) if records else 0.0,
        "min_clearance_successful_m": round(float(min(succ_min_c)), 4) if succ_min_c else 0.0,
        "mean_steps_successful": round(float(np.mean(succ_steps)), 1) if succ_steps else 1000.0,
        "mean_flight_time_s": round(float(np.mean(succ_times)), 2) if succ_times else 50.0,
        "mean_velocity_mps": round(float(np.mean(succ_speeds)), 3) if succ_speeds else 0.0,
    }


def run_campaign():
    print("=" * 80)
    print(">>> PHASE 4: MULTI-TIER CONTROLLED REGRESSION CAMPAIGN <<<")
    print("=" * 80)

    out_dir = Path("experiments/scaled_campaign/regression_campaign")
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
        ("Level 2 (Standard)", 2, DifficultyLevel.LEVEL_2_STANDARD, list(range(20000, 20050))),
        ("Level 3 (Challenging)", 3, DifficultyLevel.LEVEL_3_CHALLENGING, list(range(20200, 20250))),
        ("Level 4 (Boundary & Feasibility)", 4, DifficultyLevel.LEVEL_4_FEASIBILITY_BOUNDARY, list(range(20300, 20350))),
    ]

    all_records: List[RegressionEpisodeRecord] = []
    campaign_summary: Dict[str, Any] = {}

    for tier_name, lvl_num, lvl_enum, seeds in tier_suites:
        print(f"\n{'='*30} Evaluating {tier_name} (N={len(seeds)} Seeds) {'='*30}")
        cfg = ScaledEnvironmentFactory.get_environment_config(lvl_enum)
        env = CorridorSimEnv(scenario=cfg, seed=seeds[0])
        sup = SafetySupervisor(env.corridor)

        for cid, ctrl in controllers:
            t0 = time.time()
            tier_recs: List[RegressionEpisodeRecord] = []

            for s in seeds:
                rec = run_episode(
                    env=env,
                    controller=ctrl,
                    supervisor=sup,
                    checker=checker,
                    tier_name=tier_name,
                    level_num=lvl_num,
                    seed=s,
                    controller_id=cid,
                )
                tier_recs.append(rec)
                all_records.append(rec)

            s_key = f"{lvl_num}_{cid}"
            summary = summarize_tier(tier_recs)
            campaign_summary[s_key] = summary
            elapsed = time.time() - t0

            if summary["infeasible_total"] > 0:
                print(
                    f"  {cid:<22} | Feas Succ: {summary['feasible_success']:>2}/{summary['feasible_total']} ({summary['feasible_success_pct']}%) "
                    f"| Safe Rej: {summary['infeasible_safe_rejections']:>2}/{summary['infeasible_total']} ({summary['infeasible_rejection_pct']}%) "
                    f"| Cols: {summary['total_collision_episodes']} | Envs: {summary['envelope_violation_episodes']} | {elapsed:.1f}s"
                )
            else:
                print(
                    f"  {cid:<22} | Success: {summary['feasible_success']:>2}/{summary['feasible_total']} ({summary['feasible_success_pct']}%) "
                    f"[95% CI: {summary['feasible_wilson_ci_95']}] | Cols: {summary['total_collision_episodes']} | Envs: {summary['envelope_violation_episodes']} | {elapsed:.1f}s"
                )

        env.close()

    # Save detailed CSV and summary JSON
    csv_file = out_dir / "regression_episodes.csv"
    with open(csv_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(all_records[0]).keys()))
        writer.writeheader()
        for r in all_records:
            writer.writerow(asdict(r))

    json_file = out_dir / "regression_summary.json"
    with open(json_file, "w") as f:
        json.dump(campaign_summary, f, indent=2)

    print("\n" + "=" * 80)
    print("REGRESSION CAMPAIGN COMPLETE")
    print(f"Results written to: {out_dir.resolve()}")
    print("=" * 80)


if __name__ == "__main__":
    run_campaign()

