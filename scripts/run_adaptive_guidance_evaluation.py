"""run_adaptive_guidance_evaluation.py - Comprehensive Paired Nominal Evaluation of v2 vs v3 vs v4.

Evaluates:
- scaled_guided_bc_v2 (Phase 5 frozen winner)
- scaled_guided_bc_v3 (Phase 6 persistent guidance)
- scaled_guided_bc_v4 (Corridor-width-adaptive guidance candidate)

Across 200 development seeds:
- Level 1: 20100..20149 (N=50)
- Level 2: 20000..20049 (N=50)
- Level 3: 20200..20249 (N=50)
- Level 4: 20300..20349 (N=50: 25 feasible, 25 infeasible), including deadlock seeds 20332, 20334, 20344

Generates:
- experiments/scaled_campaign/adaptive_guidance/nominal_episodes.csv
- experiments/scaled_campaign/adaptive_guidance/nominal_summary.json
- data/adaptive_guidance_evaluation_manifest.json
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
from controllers.registry import ControllerRegistry
from controllers.safety_supervisor import SafetySupervisor
from core.contracts import VelocityAction, compute_forward_clearance_m
from core.geometry import TerminationReason
from core.guidance import GuidanceConfig, StatelessClearanceGuidanceArbitrator
from core.physical_dimensions import DifficultyLevel, PhysicalDroneModel, RouteFeasibilityChecker, ScaledEnvironmentFactory
from scripts.comprehensive_statistical_audit import compute_wilson_ci, mcnemar_exact
from sim.corridor_env import CorridorSimEnv


@dataclass
class NominalEpisodeRecord:
    tier: str
    level: int
    seed: int
    controller_id: str
    is_feasible: bool
    status: str
    success: bool
    safe_rejection: bool
    timeout: bool
    collision_episode: bool
    obstacle_collision: bool
    wall_collision: bool
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
    guidance_activations: int
    tie_break_activations: int
    lateral_reversals: int
    terminal_dist_to_goal_m: float
    bottleneck_gap_m: float


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
) -> NominalEpisodeRecord:
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
    lateral_reversals = 0

    outcome = "TIMEOUT"
    success = False
    obstacle_collision = False
    wall_collision = False
    safe_rejection = False
    flight_distance = 0.0
    prev_pos = env._get_drone_position()
    dt = env.dt

    halt_counter = 0
    prev_vy_sign = 0

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

        # Corrected Safe Halt Check for Infeasible Corridors:
        # UAV is beyond start (x >= 1.80m), halted (speed < 0.05m/s), and maintaining safe clearance (> 0.05m)
        if not feas_res.is_feasible:
            speed_2d = float(np.linalg.norm(vel[:2]))
            if pos[0] >= 1.80 and speed_2d < 0.05 and d_obs > 0.05 and d_wall > 0.05:
                halt_counter += 1
                if halt_counter >= 60:  # 3.0s sustained safe hover in front of barrier
                    safe_rejection = True
                    outcome = "SAFE_REJECT"
                    break
            else:
                halt_counter = max(0, halt_counter - 1)

        raw_act = controller.predict(obs)
        min_front = compute_forward_clearance_m(obs, max_range_m=env.sensor_config.max_range_m)

        # Track lateral guidance actions and directional reversals (chatter)
        if hasattr(controller, "arbitrator"):
            arb = controller.arbitrator
            if arb.last_dir != "NONE":
                guidance_activations += 1
                if "TIE_BREAK" in arb.last_dir:
                    tie_break_activations += 1

        if abs(raw_act.vy) > 0.05:
            curr_sign = 1 if raw_act.vy > 0 else -1
            if prev_vy_sign != 0 and curr_sign != prev_vy_sign:
                lateral_reversals += 1
            prev_vy_sign = curr_sign

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
    final_vel = env._get_drone_velocity()
    final_d_obs = corridor.obstacle_clearance(final_pos, env.obstacles)
    final_d_wall = corridor.boundary_distance(final_pos)

    if not feas_res.is_feasible and not has_collision:
        speed_2d = float(np.linalg.norm(final_vel[:2]))
        if final_pos[0] >= 1.80 and final_d_obs > 0.05 and final_d_wall > 0.05 and speed_2d < 0.08:
            safe_rejection = True
            outcome = "SAFE_REJECT"

    dist_to_goal = float(corridor.distance_to_goal(final_pos))
    flight_time = round(total_steps * dt, 2)
    mean_speed = round(flight_distance / flight_time, 3) if flight_time > 0 else 0.0
    overall_min = min(min_obs_clearance, min_wall_clearance)

    return NominalEpisodeRecord(
        tier=tier_name,
        level=level_num,
        seed=seed,
        controller_id=controller_id,
        is_feasible=feas_res.is_feasible,
        status=outcome,
        success=success,
        safe_rejection=safe_rejection,
        timeout=(outcome == "TIMEOUT"),
        collision_episode=has_collision,
        obstacle_collision=obstacle_collision,
        wall_collision=wall_collision,
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
        guidance_activations=guidance_activations,
        tie_break_activations=tie_break_activations,
        lateral_reversals=lateral_reversals,
        terminal_dist_to_goal_m=round(dist_to_goal, 3),
        bottleneck_gap_m=round(float(feas_res.min_bottleneck_gap_m), 4),
    )


def run_evaluation():
    print("=" * 80)
    print("PHASE 3: PAIRED NOMINAL EVALUATION (v2 vs v3 vs v4)")
    print("=" * 80)

    out_dir = Path("experiments/scaled_campaign/adaptive_guidance")
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = Path("data/adaptive_guidance_evaluation_manifest.json")

    registry = ControllerRegistry()
    ctrl_v2 = registry.get_controller("scaled_guided_bc_v2")
    ctrl_v3 = registry.get_controller("scaled_guided_bc_v3")
    ctrl_v4 = registry.get_controller("scaled_guided_bc_v4")

    controllers = {
        "scaled_guided_bc_v2": ctrl_v2,
        "scaled_guided_bc_v3": ctrl_v3,
        "scaled_guided_bc_v4": ctrl_v4,
    }

    tier_specs = [
        ("Level 1 (Basic, W=4.5m)", 1, DifficultyLevel.LEVEL_1_BASIC, list(range(20100, 20150))),
        ("Level 2 (Standard, W=3.8m)", 2, DifficultyLevel.LEVEL_2_STANDARD, list(range(20000, 20050))),
        ("Level 3 (Challenging, W=3.2m)", 3, DifficultyLevel.LEVEL_3_CHALLENGING, list(range(20200, 20249 + 1))),
        ("Level 4 (Boundary, W=2.8m)", 4, DifficultyLevel.LEVEL_4_FEASIBILITY_BOUNDARY, list(range(20300, 20349 + 1))),
    ]

    drone_model = PhysicalDroneModel(tip_to_tip_diagonal_m=0.65, collision_radius_m=0.325)
    checker = RouteFeasibilityChecker(drone_model)

    records: List[NominalEpisodeRecord] = []
    total_evals = sum(len(seeds) * len(controllers) for _, _, _, seeds in tier_specs)
    print(f"Total planned runs: {total_evals} episodes across 4 tiers and 3 controllers.")

    start_time = time.time()
    count = 0

    for tier_name, level_num, diff_level, seeds in tier_specs:
        corridor_cfg = ScaledEnvironmentFactory.get_environment_config(diff_level)
        env = CorridorSimEnv(scenario=corridor_cfg, seed=seeds[0])
        supervisor = SafetySupervisor(corridor=corridor_cfg)

        print(f"\nEvaluating {tier_name} ({len(seeds)} seeds)...")
        for seed in seeds:
            for cid, ctrl in controllers.items():
                rec = run_episode(
                    env=env,
                    controller=ctrl,
                    supervisor=supervisor,
                    checker=checker,
                    tier_name=tier_name,
                    level_num=level_num,
                    seed=seed,
                    controller_id=cid,
                )
                records.append(rec)
                count += 1
                if count % 60 == 0:
                    elapsed = time.time() - start_time
                    print(f"  Completed {count}/{total_evals} episodes ({count / total_evals * 100:.1f}%) in {elapsed:.1f}s")
        env.close()

    total_time = time.time() - start_time
    print(f"\nAll {len(records)} episodes completed in {total_time:.2f}s!")

    # Write CSV
    csv_path = out_dir / "nominal_episodes.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[f.name for f in NominalEpisodeRecord.__dataclass_fields__.values()])
        writer.writeheader()
        for r in records:
            writer.writerow(asdict(r))
    print(f"Saved episode records to {csv_path}")

    # Compute Detailed Summaries
    summary = compute_comparative_summary(records, tier_specs, controllers)
    summary_path = out_dir / "nominal_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary metrics to {summary_path}")

    # Write Manifest
    manifest = {
        "title": "Corridor-Width-Adaptive Guidance Evaluation Manifest",
        "created_at_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "checkpoint": "models/checkpoints/bc_policy_scaled_v2.pt",
        "controllers": {
            "scaled_guided_bc_v2": ctrl_v2.metadata.description,
            "scaled_guided_bc_v3": ctrl_v3.metadata.description,
            "scaled_guided_bc_v4": ctrl_v4.metadata.description,
        },
        "tiers": [
            {"tier": t[0], "level": t[1], "seeds": [min(t[3]), max(t[3])], "count": len(t[3])}
            for t in tier_specs
        ],
        "deadlock_seeds_monitored": [20332, 20334, 20344],
        "total_episodes": len(records),
        "execution_time_seconds": round(total_time, 2),
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved evaluation manifest to {manifest_path}")

    return records, summary


def compute_comparative_summary(
    records: List[NominalEpisodeRecord],
    tier_specs: List[Tuple],
    controllers: Dict[str, Any],
) -> Dict[str, Any]:
    cids = list(controllers.keys())
    res: Dict[str, Any] = {"per_tier": {}, "overall": {}, "deadlock_case_studies": {}}

    # Group by tier and seed
    tier_names = [t[0] for t in tier_specs]
    for tier in tier_names:
        res["per_tier"][tier] = {}
        t_recs = [r for r in records if r.tier == tier]

        # Group by seed
        paired_by_seed: Dict[int, Dict[str, NominalEpisodeRecord]] = {}
        for r in t_recs:
            if r.seed not in paired_by_seed:
                paired_by_seed[r.seed] = {}
            paired_by_seed[r.seed][r.controller_id] = r

        for cid in cids:
            c_recs = [r for r in t_recs if r.controller_id == cid]
            n_tot = len(c_recs)
            n_feas = sum(1 for r in c_recs if r.is_feasible)
            n_infeas = n_tot - n_feas

            feas_succ = sum(1 for r in c_recs if r.is_feasible and r.success)
            infeas_safe = sum(1 for r in c_recs if not r.is_feasible and r.safe_rejection)
            timeouts = sum(1 for r in c_recs if r.timeout)
            collisions = sum(1 for r in c_recs if r.collision_episode)
            envelope_vios = sum(1 for r in c_recs if r.safety_envelope_violation)
            total_vio_steps = sum(r.envelope_violation_steps for r in c_recs)
            total_steps_tier = sum(r.total_steps for r in c_recs)

            overall_min_c = min(r.overall_min_clearance_m for r in c_recs) if c_recs else 0.0
            succ_recs = [r for r in c_recs if r.success or r.safe_rejection]
            succ_min_c = min(r.overall_min_clearance_m for r in succ_recs) if succ_recs else 0.0

            interventions = sum(r.safety_interventions for r in c_recs)
            reversals = sum(r.lateral_reversals for r in c_recs)
            mean_flight_time = float(np.mean([r.flight_time_s for r in succ_recs])) if succ_recs else 0.0
            mean_speed = float(np.mean([r.mean_speed_mps for r in succ_recs])) if succ_recs else 0.0

            res["per_tier"][tier][cid] = {
                "total_episodes": n_tot,
                "feasible_routes": n_feas,
                "infeasible_routes": n_infeas,
                "feasible_success_count": feas_succ,
                "feasible_success_rate": round(feas_succ / n_feas, 4) if n_feas > 0 else 1.0,
                "feasible_success_ci95": compute_wilson_ci(feas_succ, n_feas) if n_feas > 0 else [1.0, 1.0],
                "infeasible_safe_count": infeas_safe,
                "infeasible_safe_rate": round(infeas_safe / n_infeas, 4) if n_infeas > 0 else 1.0,
                "infeasible_safe_ci95": compute_wilson_ci(infeas_safe, n_infeas) if n_infeas > 0 else [1.0, 1.0],
                "timeouts": timeouts,
                "timeout_rate": round(timeouts / n_tot, 4),
                "collisions": collisions,
                "collision_rate": round(collisions / n_tot, 4),
                "collision_ci95": compute_wilson_ci(collisions, n_tot),
                "envelope_violation_episodes": envelope_vios,
                "envelope_violation_rate": round(envelope_vios / n_tot, 4),
                "envelope_violation_steps": total_vio_steps,
                "envelope_violation_step_pct": round(total_vio_steps / total_steps_tier * 100, 2) if total_steps_tier > 0 else 0.0,
                "overall_min_clearance_m": round(overall_min_c, 4),
                "successful_min_clearance_m": round(succ_min_c, 4),
                "total_interventions": interventions,
                "mean_interventions_per_ep": round(interventions / n_tot, 1),
                "total_lateral_reversals": reversals,
                "mean_reversals_per_ep": round(reversals / n_tot, 2),
                "mean_flight_time_s": round(mean_flight_time, 2),
                "mean_speed_mps": round(mean_speed, 3),
            }

        # Paired McNemar tests for this tier
        res["per_tier"][tier]["paired_comparisons"] = compute_paired_mcnemar_stats(paired_by_seed)

    # Overall Summary across all 200 seeds
    res["overall"] = {}
    paired_overall: Dict[int, Dict[str, NominalEpisodeRecord]] = {}
    for r in records:
        if r.seed not in paired_overall:
            paired_overall[r.seed] = {}
        paired_overall[r.seed][r.controller_id] = r

    for cid in cids:
        c_recs = [r for r in records if r.controller_id == cid]
        n_tot = len(c_recs)
        n_feas = sum(1 for r in c_recs if r.is_feasible)
        n_infeas = n_tot - n_feas

        feas_succ = sum(1 for r in c_recs if r.is_feasible and r.success)
        infeas_safe = sum(1 for r in c_recs if not r.is_feasible and r.safe_rejection)
        timeouts = sum(1 for r in c_recs if r.timeout)
        collisions = sum(1 for r in c_recs if r.collision_episode)
        envelope_vios = sum(1 for r in c_recs if r.safety_envelope_violation)
        total_vio_steps = sum(r.envelope_violation_steps for r in c_recs)
        total_steps_all = sum(r.total_steps for r in c_recs)

        overall_min_c = min(r.overall_min_clearance_m for r in c_recs) if c_recs else 0.0
        succ_recs = [r for r in c_recs if r.success or r.safe_rejection]
        succ_min_c = min(r.overall_min_clearance_m for r in succ_recs) if succ_recs else 0.0

        interventions = sum(r.safety_interventions for r in c_recs)
        reversals = sum(r.lateral_reversals for r in c_recs)
        mean_flight_time = float(np.mean([r.flight_time_s for r in succ_recs])) if succ_recs else 0.0
        mean_speed = float(np.mean([r.mean_speed_mps for r in succ_recs])) if succ_recs else 0.0

        res["overall"][cid] = {
            "total_episodes": n_tot,
            "feasible_routes": n_feas,
            "infeasible_routes": n_infeas,
            "feasible_success_count": feas_succ,
            "feasible_success_rate": round(feas_succ / n_feas, 4),
            "feasible_success_ci95": compute_wilson_ci(feas_succ, n_feas),
            "infeasible_safe_count": infeas_safe,
            "infeasible_safe_rate": round(infeas_safe / n_infeas, 4),
            "infeasible_safe_ci95": compute_wilson_ci(infeas_safe, n_infeas),
            "overall_effective_success_count": feas_succ + infeas_safe,
            "overall_effective_success_rate": round((feas_succ + infeas_safe) / n_tot, 4),
            "timeouts": timeouts,
            "timeout_rate": round(timeouts / n_tot, 4),
            "collisions": collisions,
            "collision_rate": round(collisions / n_tot, 4),
            "collision_ci95": compute_wilson_ci(collisions, n_tot),
            "envelope_violation_episodes": envelope_vios,
            "envelope_violation_rate": round(envelope_vios / n_tot, 4),
            "envelope_violation_steps": total_vio_steps,
            "envelope_violation_step_pct": round(total_vio_steps / total_steps_all * 100, 2),
            "overall_min_clearance_m": round(overall_min_c, 4),
            "successful_min_clearance_m": round(succ_min_c, 4),
            "total_interventions": interventions,
            "mean_interventions_per_ep": round(interventions / n_tot, 1),
            "total_lateral_reversals": reversals,
            "mean_reversals_per_ep": round(reversals / n_tot, 2),
            "mean_flight_time_s": round(mean_flight_time, 2),
            "mean_speed_mps": round(mean_speed, 3),
        }

    res["overall"]["paired_comparisons"] = compute_paired_mcnemar_stats(paired_overall)

    # Detailed Deadlock Seeds Analysis
    deadlock_seeds = [20332, 20334, 20344]
    for s in deadlock_seeds:
        res["deadlock_case_studies"][str(s)] = {}
        for cid in cids:
            matches = [r for r in records if r.seed == s and r.controller_id == cid]
            if matches:
                r = matches[0]
                res["deadlock_case_studies"][str(s)][cid] = {
                    "status": r.status,
                    "total_steps": r.total_steps,
                    "flight_time_s": r.flight_time_s,
                    "min_obstacle_clearance_m": r.min_obstacle_clearance_m,
                    "min_wall_clearance_m": r.min_wall_clearance_m,
                    "envelope_violation_steps": r.envelope_violation_steps,
                    "safety_interventions": r.safety_interventions,
                }

    return res


def compute_paired_mcnemar_stats(paired: Dict[int, Dict[str, NominalEpisodeRecord]]) -> Dict[str, Any]:
    def is_ok(r: NominalEpisodeRecord) -> bool:
        return r.status in ["SUCCESS", "SAFE_REJECT"]

    pairs = [
        ("v2_vs_v3", "scaled_guided_bc_v2", "scaled_guided_bc_v3"),
        ("v3_vs_v4", "scaled_guided_bc_v3", "scaled_guided_bc_v4"),
        ("v2_vs_v4", "scaled_guided_bc_v2", "scaled_guided_bc_v4"),
    ]
    out = {}
    for tag, c1, c2 in pairs:
        n_both_ok = 0
        n_c1_only = 0
        n_c2_only = 0
        n_neither = 0
        c1_wins = []
        c2_wins = []

        for s, p in paired.items():
            r1 = p.get(c1)
            r2 = p.get(c2)
            if not r1 or not r2:
                continue
            ok1 = is_ok(r1)
            ok2 = is_ok(r2)
            if ok1 and ok2:
                n_both_ok += 1
            elif ok1 and not ok2:
                n_c1_only += 1
                c1_wins.append(s)
            elif not ok1 and ok2:
                n_c2_only += 1
                c2_wins.append(s)
            else:
                n_neither += 1

        b = n_c1_only
        c = n_c2_only
        p_val = mcnemar_exact(b, c)
        out[tag] = {
            "n_evaluated": len(paired),
            "contingency_table": {
                "both_succeed": n_both_ok,
                f"{c1}_only": b,
                f"{c2}_only": c,
                "both_fail": n_neither,
            },
            "p_value": p_val,
            "statistically_significant": p_val < 0.05,
            f"{c1}_only_seeds": c1_wins,
            f"{c2}_only_seeds": c2_wins,
        }
    return out


if __name__ == "__main__":
    run_evaluation()

