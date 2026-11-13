"""run_phase4f_evaluation.py - Authoritative Phase 4F Paired Evaluation Harness.

Executes:
- Stage B: Development evaluation across 200 paired seeds (Levels 1-4) for v3, v4, v5.
- Stage C: Independent confirmation across 200 pristine held-out seeds (26100..26449).
- Evaluates against the authoritative 8-outcome taxonomy:
  CLEAN_SUCCESS, ENVELOPE_BREACH_ARRIVAL, COLLISION, TIMEOUT,
  SAFE_REJECTION, UNRESOLVED_OBSTACLE, UNSAFE_OR_UNSTABLE_ARREST, OTHER_FAILURE.
- Generates:
  - data/phase4f_development_metrics.json
  - data/phase4f_episode_outcomes.json
  - data/phase4f_independent_confirmation_metrics.json
"""

import csv
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from controllers.registry import ControllerRegistry
from controllers.safe_arrest import SafeArrestState, SafeArrestStateMachine
from controllers.safety_supervisor import SafetySupervisor
from core.contracts import SensorConfig, compute_forward_clearance_m
from core.geometry import Corridor3D, EpisodeOutcome, OutcomeClassifier, TerminationReason
from core.physical_dimensions import DifficultyLevel, PhysicalDroneModel, RouteFeasibilityChecker, ScaledEnvironmentFactory
from scripts.comprehensive_statistical_audit import compute_wilson_ci, mcnemar_exact
from sim.corridor_env import CorridorSimEnv


@dataclass
class Phase4FEpisodeRecord:
    stage: str
    tier: str
    level: int
    seed: int
    controller_id: str
    is_feasible: bool
    raw_termination: str
    outcome: str
    is_clean_success: bool
    is_envelope_breach_arrival: bool
    is_collision: bool
    is_timeout: bool
    is_safe_rejection: bool
    is_unresolved: bool
    is_unsafe_arrest: bool
    min_obstacle_clearance_m: float
    min_boundary_clearance_m: float
    overall_min_clearance_m: float
    envelope_violation_steps: int
    total_steps: int
    flight_time_s: float
    flight_distance_m: float
    mean_speed_mps: float
    safety_interventions: int
    lateral_reversals: int
    terminal_dist_to_goal_m: float
    arrest_state: str
    arrest_drift_m: float
    arrest_is_stable: bool
    classification_reason: str


def run_episode(
    controller_id: str,
    level: DifficultyLevel,
    seed: int,
    stage_name: str,
    max_steps: int = 1000,
    envelope_threshold_m: float = 0.150,
) -> Phase4FEpisodeRecord:
    cfg = ScaledEnvironmentFactory.get_environment_config(level)
    
    # Controller v5 uses 16-beam radial perception; v3 and v4 use 12-beam baseline
    use_16_beams = (controller_id == "scaled_guided_bc_v5")
    sensor_cfg = SensorConfig.radial_extended_16() if use_16_beams else SensorConfig()
    
    env = CorridorSimEnv(corridor=cfg, sensor_config=sensor_cfg)
    obs, _ = env.reset(seed=seed)
    
    reg = ControllerRegistry()
    controller = reg.get_controller(controller_id)
    if hasattr(controller, "reset"):
        controller.reset()
        
    supervisor = SafetySupervisor(corridor=cfg)
    if hasattr(supervisor, "reset"):
        supervisor.reset()
        
    checker = RouteFeasibilityChecker(PhysicalDroneModel(tip_to_tip_diagonal_m=0.65, collision_radius_m=0.325))
    feas = checker.evaluate_feasibility(env.corridor, env.obstacles)
    
    sm = SafeArrestStateMachine()
    
    total_steps = 0
    min_obs_c = float("inf")
    min_bnd_c = float("inf")
    envelope_violation_steps = 0
    safety_interventions = 0
    lateral_reversals = 0
    flight_distance = 0.0
    prev_pos = env._get_drone_position()
    dt = env.dt
    prev_vy_sign = 0
    
    raw_term = "TIMEOUT"
    final_arrest_state = "APPROACHING"
    final_arrest_drift = 0.0
    final_arrest_stable = False
    
    for step in range(max_steps):
        total_steps += 1
        pos = env._get_drone_position()
        vel = env._get_drone_velocity()
        
        flight_distance += float(np.linalg.norm(pos - prev_pos))
        prev_pos = pos
        
        d_obs = float(cfg.obstacle_clearance(pos, env.obstacles))
        d_bnd = float(cfg.boundary_distance(pos))
        min_obs_c = min(min_obs_c, d_obs)
        min_bnd_c = min(min_bnd_c, d_bnd)
        
        if d_obs < envelope_threshold_m or d_bnd < envelope_threshold_m:
            envelope_violation_steps += 1
            
        # Authoritative PyBullet termination check
        term_reason, _ = cfg.check_termination(pos, vel, env.obstacles, step_count=step, max_steps=max_steps)
        if term_reason == TerminationReason.SUCCESS:
            raw_term = "SUCCESS"
            break
        elif term_reason == TerminationReason.CRASH_OBSTACLE:
            raw_term = "CRASH_OBSTACLE"
            break
        elif term_reason == TerminationReason.CRASH_BOUNDARY:
            raw_term = "CRASH_BOUNDARY"
            break
            
        # Safe arrest state machine evaluation (active for v5; passive monitoring for v3/v4)
        arrest_state, should_arrest_term, arrest_diag = sm.update(
            current_position=pos,
            current_velocity=vel,
            observation=obs,
            min_obstacle_clearance=d_obs,
            min_boundary_clearance=d_bnd,
            is_known_infeasible_scenario=(not feas.is_feasible)
        )
        
        final_arrest_state = arrest_state.value
        final_arrest_drift = arrest_diag.position_drift_m
        final_arrest_stable = arrest_diag.is_stable
        
        if controller_id == "scaled_guided_bc_v5" and should_arrest_term:
            raw_term = "SAFE_ARREST"
            break
            
        # Predict action
        suppress_lat = arrest_diag.suppress_lateral_guidance if controller_id == "scaled_guided_bc_v5" else False
        if hasattr(controller, "arbitrator"):
            raw_act, sel_dir, _, _ = controller.arbitrator.arbitrate(
                raw_action=controller.predict(obs),
                observation=obs,
                suppress_guidance=suppress_lat
            )
        else:
            raw_act = controller.predict(obs)
            
        # Track lateral chatter
        if abs(raw_act.vy) > 0.05:
            curr_sign = 1 if raw_act.vy > 0 else -1
            if prev_vy_sign != 0 and curr_sign != prev_vy_sign:
                lateral_reversals += 1
            prev_vy_sign = curr_sign
            
        min_front = compute_forward_clearance_m(obs, max_range_m=env.sensor_config.max_range_m)
        safe_act, sup_state, _ = supervisor.filter_action(
            proposed_action=raw_act,
            current_position=pos,
            current_velocity=vel,
            min_front_range=min_front,
            dt=dt,
            observation=obs
        )
        
        if sup_state.value != "nominal":
            safety_interventions += 1
            
        obs, _, term, trunc, info = env.step(safe_act.to_normalized(env.action_limits))
        if term or trunc:
            term_val = info.get("termination_reason", "")
            if term_val == "success":
                raw_term = "SUCCESS"
            elif term_val == "crash_obstacle":
                raw_term = "CRASH_OBSTACLE"
            elif term_val == "crash_boundary":
                raw_term = "CRASH_BOUNDARY"
            elif trunc:
                raw_term = "TIMEOUT"
            break
            
    final_pos = env._get_drone_position()
    dist_to_goal = float(cfg.distance_to_goal(final_pos))
    flight_time = round(total_steps * dt, 2)
    mean_speed = round(flight_distance / flight_time, 3) if flight_time > 0 else 0.0
    overall_min = min(min_obs_c, min_bnd_c)
    
    env.close()
    
    # Classify episode outcome using authoritative 8-outcome taxonomy
    clf = OutcomeClassifier.classify(
        termination_reason=raw_term,
        min_obstacle_clearance_m=min_obs_c,
        min_boundary_clearance_m=min_bnd_c,
        safe_arrest_state=final_arrest_state,
        is_arrest_stable=final_arrest_stable,
    )
    
    return Phase4FEpisodeRecord(
        stage=stage_name,
        tier=level.name,
        level=level.value if isinstance(level.value, int) else int(level.name.split("_")[1]),
        seed=seed,
        controller_id=controller_id,
        is_feasible=feas.is_feasible,
        raw_termination=raw_term,
        outcome=clf.outcome.value,
        is_clean_success=(clf.outcome == EpisodeOutcome.CLEAN_SUCCESS),
        is_envelope_breach_arrival=(clf.outcome == EpisodeOutcome.ENVELOPE_BREACH_ARRIVAL),
        is_collision=(clf.outcome == EpisodeOutcome.COLLISION),
        is_timeout=(clf.outcome == EpisodeOutcome.TIMEOUT),
        is_safe_rejection=(clf.outcome == EpisodeOutcome.SAFE_REJECTION),
        is_unresolved=(clf.outcome == EpisodeOutcome.UNRESOLVED_OBSTACLE),
        is_unsafe_arrest=(clf.outcome == EpisodeOutcome.UNSAFE_OR_UNSTABLE_ARREST),
        min_obstacle_clearance_m=round(min_obs_c, 4),
        min_boundary_clearance_m=round(min_bnd_c, 4),
        overall_min_clearance_m=round(overall_min, 4),
        envelope_violation_steps=envelope_violation_steps,
        total_steps=total_steps,
        flight_time_s=flight_time,
        flight_distance_m=round(flight_distance, 3),
        mean_speed_mps=mean_speed,
        safety_interventions=safety_interventions,
        lateral_reversals=lateral_reversals,
        terminal_dist_to_goal_m=round(dist_to_goal, 3),
        arrest_state=final_arrest_state,
        arrest_drift_m=round(final_arrest_drift, 4),
        arrest_is_stable=final_arrest_stable,
        classification_reason=clf.reason,
    )


def run_stage(
    stage_name: str,
    tier_definitions: List[Dict[str, Any]],
    controllers: List[str],
) -> List[Phase4FEpisodeRecord]:
    records: List[Phase4FEpisodeRecord] = []
    total_episodes = sum(t["count"] for t in tier_definitions) * len(controllers)
    ep_idx = 0
    t0 = time.time()
    
    print(f"\n=======================================================")
    print(f"Starting {stage_name}: {total_episodes} Total Episodes")
    print(f"=======================================================")
    
    for t_def in tier_definitions:
        tier_level = t_def["level"]
        seed_start = t_def["seed_start"]
        seed_end = t_def["seed_end"]
        count = t_def["count"]
        seeds = list(range(seed_start, seed_end + 1))
        
        print(f"\n--- Tier: {tier_level.name} ({count} seeds: {seed_start}..{seed_end}) ---")
        
        for s in seeds:
            for cid in controllers:
                ep_idx += 1
                rec = run_episode(
                    controller_id=cid,
                    level=tier_level,
                    seed=s,
                    stage_name=stage_name,
                )
                records.append(rec)
                
            if ep_idx % 30 == 0 or ep_idx == total_episodes:
                elapsed = time.time() - t0
                rate = ep_idx / elapsed if elapsed > 0 else 0
                print(f"  Progress: {ep_idx}/{total_episodes} ({ep_idx/total_episodes*100:.1f}%) | {rate:.1f} ep/s")
                
    return records


def compute_stage_summary(records: List[Phase4FEpisodeRecord], controllers: List[str]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {"by_controller": {}, "by_tier": {}, "paired_comparisons": {}}
    
    tiers = sorted(list(set(r.tier for r in records)))
    
    # 1. Summary by Controller
    for cid in controllers:
        c_recs = [r for r in records if r.controller_id == cid]
        n_total = len(c_recs)
        feas_recs = [r for r in c_recs if r.is_feasible]
        infeas_recs = [r for r in c_recs if not r.is_feasible]
        
        n_feas = len(feas_recs)
        n_infeas = len(infeas_recs)
        
        clean_succ = sum(1 for r in feas_recs if r.is_clean_success)
        breach_arr = sum(1 for r in feas_recs if r.is_envelope_breach_arrival)
        collisions = sum(1 for r in c_recs if r.is_collision)
        timeouts = sum(1 for r in c_recs if r.is_timeout)
        safe_rejects = sum(1 for r in infeas_recs if r.is_safe_rejection)
        unresolved = sum(1 for r in c_recs if r.is_unresolved)
        unsafe_arrests = sum(1 for r in c_recs if r.is_unsafe_arrest)
        
        overall_c = min(r.overall_min_clearance_m for r in c_recs) if c_recs else 0.0
        succ_recs = [r for r in feas_recs if r.is_clean_success or r.is_envelope_breach_arrival]
        succ_c = min(r.overall_min_clearance_m for r in succ_recs) if succ_recs else 0.0
        
        summary["by_controller"][cid] = {
            "total_episodes": n_total,
            "feasible_count": n_feas,
            "infeasible_count": n_infeas,
            "clean_success_count": clean_succ,
            "clean_success_rate": round(clean_succ / n_feas, 4) if n_feas > 0 else 0.0,
            "clean_success_ci95": compute_wilson_ci(clean_succ, n_feas) if n_feas > 0 else [0, 0],
            "envelope_breach_arrival_count": breach_arr,
            "envelope_breach_arrival_rate": round(breach_arr / n_feas, 4) if n_feas > 0 else 0.0,
            "gross_goal_arrival_count": clean_succ + breach_arr,
            "gross_goal_arrival_rate": round((clean_succ + breach_arr) / n_feas, 4) if n_feas > 0 else 0.0,
            "collision_count": collisions,
            "collision_rate": round(collisions / n_total, 4) if n_total > 0 else 0.0,
            "timeout_count": timeouts,
            "timeout_rate": round(timeouts / n_total, 4) if n_total > 0 else 0.0,
            "safe_rejection_count": safe_rejects,
            "safe_rejection_rate": round(safe_rejects / n_infeas, 4) if n_infeas > 0 else 0.0,
            "safe_rejection_ci95": compute_wilson_ci(safe_rejects, n_infeas) if n_infeas > 0 else [0, 0],
            "unresolved_count": unresolved,
            "unsafe_arrest_count": unsafe_arrests,
            "overall_min_clearance_m": round(overall_c, 4),
            "successful_min_clearance_m": round(succ_c, 4),
            "mean_envelope_violation_steps": round(sum(r.envelope_violation_steps for r in c_recs) / n_total, 1),
            "mean_flight_time_s": round(sum(r.flight_time_s for r in c_recs) / n_total, 2),
            "mean_speed_mps": round(sum(r.mean_speed_mps for r in c_recs) / n_total, 3),
            "mean_interventions": round(sum(r.safety_interventions for r in c_recs) / n_total, 1),
            "mean_lateral_reversals": round(sum(r.lateral_reversals for r in c_recs) / n_total, 1),
        }
        
    # 2. Summary by Tier
    for tier in tiers:
        summary["by_tier"][tier] = {}
        t_recs = [r for r in records if r.tier == tier]
        for cid in controllers:
            tc_recs = [r for r in t_recs if r.controller_id == cid]
            n_tot = len(tc_recs)
            n_f = sum(1 for r in tc_recs if r.is_feasible)
            n_inf = sum(1 for r in tc_recs if not r.is_feasible)
            cl_succ = sum(1 for r in tc_recs if r.is_clean_success)
            br_arr = sum(1 for r in tc_recs if r.is_envelope_breach_arrival)
            sf_rej = sum(1 for r in tc_recs if r.is_safe_rejection)
            colls = sum(1 for r in tc_recs if r.is_collision)
            touts = sum(1 for r in tc_recs if r.is_timeout)
            min_c = min(r.overall_min_clearance_m for r in tc_recs) if tc_recs else 0.0
            
            summary["by_tier"][tier][cid] = {
                "n_episodes": n_tot,
                "n_feasible": n_f,
                "clean_success": cl_succ,
                "clean_rate": round(cl_succ / n_f, 4) if n_f > 0 else 0.0,
                "envelope_breach_arrivals": br_arr,
                "safe_rejections": sf_rej,
                "safe_rejection_rate": round(sf_rej / n_inf, 4) if n_inf > 0 else 0.0,
                "collisions": colls,
                "timeouts": touts,
                "min_clearance_m": round(min_c, 4),
            }
            
    # 3. Paired Comparisons (McNemar Exact Tests on Clean Success and Safe Rejection)
    pairs = [("scaled_guided_bc_v3", "scaled_guided_bc_v4"),
             ("scaled_guided_bc_v4", "scaled_guided_bc_v5"),
             ("scaled_guided_bc_v3", "scaled_guided_bc_v5")]
             
    for cA, cB in pairs:
        pair_key = f"{cA}_vs_{cB}"
        # Pair by (tier, seed)
        recsA = {(r.tier, r.seed): r for r in records if r.controller_id == cA}
        recsB = {(r.tier, r.seed): r for r in records if r.controller_id == cB}
        common_keys = sorted(list(set(recsA.keys()) & set(recsB.keys())))
        
        # Test 1: Mission Validity (Clean Success on feasible + Safe Rejection on infeasible)
        both_succ = 0
        A_only = 0
        B_only = 0
        both_fail = 0
        A_only_seeds = []
        B_only_seeds = []
        
        for k in common_keys:
            rA = recsA[k]
            rB = recsB[k]
            
            # Mission Valid = Clean Success (if feasible) or Safe Rejection (if infeasible)
            validA = (rA.is_clean_success if rA.is_feasible else rA.is_safe_rejection)
            validB = (rB.is_clean_success if rB.is_feasible else rB.is_safe_rejection)
            
            if validA and validB:
                both_succ += 1
            elif validA and not validB:
                A_only += 1
                A_only_seeds.append(rA.seed)
            elif not validA and validB:
                B_only += 1
                B_only_seeds.append(rB.seed)
            else:
                both_fail += 1
                
        p_val = mcnemar_exact(A_only, B_only)
        summary["paired_comparisons"][pair_key] = {
            "n_evaluated": len(common_keys),
            "contingency_table": {
                "both_mission_valid": both_succ,
                f"{cA}_only": A_only,
                f"{cB}_only": B_only,
                "both_fail": both_fail,
            },
            "p_value": p_val,
            "statistically_significant": (p_val < 0.05),
            f"{cA}_only_seeds": A_only_seeds,
            f"{cB}_only_seeds": B_only_seeds,
        }
        
    return summary


def main():
    controllers = [
        "scaled_guided_bc_v3",
        "scaled_guided_bc_v4",
        "scaled_guided_bc_v5",
    ]
    
    # -------------------------------------------------------------
    # Stage B: Development Seed Set (Levels 1-4, 200 seeds)
    # -------------------------------------------------------------
    stage_b_tiers = [
        {"level": DifficultyLevel.LEVEL_1_BASIC, "seed_start": 20100, "seed_end": 20149, "count": 50},
        {"level": DifficultyLevel.LEVEL_2_STANDARD, "seed_start": 20000, "seed_end": 20049, "count": 50},
        {"level": DifficultyLevel.LEVEL_3_CHALLENGING, "seed_start": 20200, "seed_end": 20249, "count": 50},
        {"level": DifficultyLevel.LEVEL_4_FEASIBILITY_BOUNDARY, "seed_start": 20300, "seed_end": 20349, "count": 50},
    ]
    
    stage_b_records = run_stage("Stage_B_Development", stage_b_tiers, controllers)
    stage_b_summary = compute_stage_summary(stage_b_records, controllers)
    
    # Save Stage B Results
    dev_metrics_path = Path("data/phase4f_development_metrics.json")
    with open(dev_metrics_path, "w") as f:
        json.dump(stage_b_summary, f, indent=2)
    print(f"\n[Saved] Development metrics -> {dev_metrics_path}")
    
    # Save Episode-level outcomes for Stage B
    outcomes_path = Path("data/phase4f_episode_outcomes.json")
    with open(outcomes_path, "w") as f:
        json.dump([asdict(r) for r in stage_b_records], f, indent=2)
    print(f"[Saved] Episode outcomes -> {outcomes_path}")
    
    # -------------------------------------------------------------
    # Stage C: Independent Confirmation Set (Disjoint Pristine Holdout)
    # -------------------------------------------------------------
    stage_c_tiers = [
        {"level": DifficultyLevel.LEVEL_1_BASIC, "seed_start": 26100, "seed_end": 26149, "count": 50},
        {"level": DifficultyLevel.LEVEL_2_STANDARD, "seed_start": 26200, "seed_end": 26249, "count": 50},
        {"level": DifficultyLevel.LEVEL_3_CHALLENGING, "seed_start": 26300, "seed_end": 26349, "count": 50},
        {"level": DifficultyLevel.LEVEL_4_FEASIBILITY_BOUNDARY, "seed_start": 26400, "seed_end": 26449, "count": 50},
    ]
    
    stage_c_records = run_stage("Stage_C_Independent_Confirmation", stage_c_tiers, controllers)
    stage_c_summary = compute_stage_summary(stage_c_records, controllers)
    
    # Save Stage C Results
    holdout_metrics_path = Path("data/phase4f_independent_confirmation_metrics.json")
    with open(holdout_metrics_path, "w") as f:
        json.dump(stage_c_summary, f, indent=2)
    print(f"[Saved] Independent confirmation metrics -> {holdout_metrics_path}")
    
    print("\n=======================================================")
    print("PHASE 4F EVALUATION COMPLETED SUCCESSFULLY")
    print("=======================================================")


if __name__ == "__main__":
    main()

