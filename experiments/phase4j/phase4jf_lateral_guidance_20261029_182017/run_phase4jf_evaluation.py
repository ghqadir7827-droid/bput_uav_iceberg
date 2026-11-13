"""run_phase4jf_evaluation.py - Phase 4J-F Controlled Paired-Seed Evaluation.

Evaluates the frozen Behavior Cloning (BC) baseline controller under two isolated configurations:
1. baseline: Existing canonical supervisor at d_caut = 0.50m.
2. lateral_guidance: Isolated lateral guidance supervisor with proactive bypass steering.

Seed Range: 2000..2049 (50 matched development seeds).
Episode Horizon: 500 steps (25.0s at dt = 0.05s).
"""

import csv
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch

from controllers.safety_supervisor import SafetyState
from core.contracts import ActionLimits, VelocityAction, compute_directional_clearances_m, compute_forward_clearance_m
from core.geometry import Corridor3D, TerminationReason
from experiments.phase4j.phase4jf_lateral_guidance_20261029_182017.experimental_supervisor import (
    GuidanceState,
    IsolatedLateralGuidanceSupervisor,
)
from models.policy_net import NavigationPolicyMLP
from sim.corridor_env import CorridorSimEnv

EVAL_SEEDS = list(range(2000, 2050))
CAUTION_DIST_M = 0.50
OBSTACLE_SAFETY_ENVELOPE_M = 0.080
BOUNDARY_SAFETY_ENVELOPE_M = 0.080

BC_CHECKPOINT_PATH = "models/checkpoints/bc_policy_best.pt"
EXPECTED_BC_SHA256 = "93c4d7d1d8476b8d288a24fca1862d87133696f78b9603aa622bcc367afad44f"
CANONICAL_SUPERVISOR_PATH = "controllers/safety_supervisor.py"
EXPECTED_SUPERVISOR_SHA256 = "9f84af22821f0f1473ea963b0f33a1924d1c4e7a487d4ed7893b7efc56fcd26f"


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        return super().default(obj)


def wilson_score_interval(successes: int, trials: int) -> Tuple[float, float]:
    if trials == 0:
        return 0.0, 0.0
    z = 1.959964
    p_hat = successes / trials
    denominator = 1 + z**2 / trials
    center = (p_hat + z**2 / (2 * trials)) / denominator
    spread = z * math.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * trials)) / trials) / denominator
    return round(max(0.0, center - spread) * 100.0, 1), round(min(1.0, center + spread) * 100.0, 1)


def mcnemar_exact_test(b: int, c: int) -> float:
    """Two-tailed McNemar exact test with binomial distribution."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    prob = sum(math.comb(n, i) * (0.5**n) for i in range(k + 1))
    return min(1.0, 2.0 * prob)


def verify_integrity():
    # Verify BC checkpoint
    p_bc = Path(BC_CHECKPOINT_PATH)
    if not p_bc.is_file():
        raise FileNotFoundError(f"BC checkpoint missing: {BC_CHECKPOINT_PATH}")
    h_bc = hashlib.sha256(p_bc.read_bytes()).hexdigest()
    if h_bc != EXPECTED_BC_SHA256:
        raise ValueError(f"BC checkpoint hash mismatch: expected {EXPECTED_BC_SHA256}, got {h_bc}")

    # Verify canonical supervisor
    p_sup = Path(CANONICAL_SUPERVISOR_PATH)
    if not p_sup.is_file():
        raise FileNotFoundError(f"Canonical supervisor missing: {CANONICAL_SUPERVISOR_PATH}")
    h_sup = hashlib.sha256(p_sup.read_bytes()).hexdigest()
    if h_sup != EXPECTED_SUPERVISOR_SHA256:
        raise ValueError(f"Canonical supervisor hash mismatch: expected {EXPECTED_SUPERVISOR_SHA256}, got {h_sup}")

    print("  [OK] BC Checkpoint verified:      ", h_bc)
    print("  [OK] Canonical Supervisor verified:", h_sup)


def run_episode(
    env: CorridorSimEnv,
    supervisor: IsolatedLateralGuidanceSupervisor,
    model: NavigationPolicyMLP,
    config_name: str,
    seed: int,
    device: torch.device,
    record_trace: bool = False,
) -> Dict[str, Any]:
    obs, info = env.reset(seed=seed)
    supervisor.reset_state()

    ep_start_pos = env._get_drone_position()
    steps = 0
    done = False

    min_obs_overall = float(info["privileged_state"].obstacle_clearance)
    min_wall_overall = float(info["privileged_state"].corridor_boundary_distance)
    per_obs_min = [10.0] * len(env.obstacles)

    interventions = 0
    emergency_brakes = 0
    retreats = 0
    throttled_steps = 0
    stationary_steps = 0

    raw_vx_list, raw_vy_list, raw_vz_list, raw_yr_list = [], [], [], []
    safe_vx_list, safe_vy_list, safe_vz_list, safe_yr_list = [], [], [], []

    step_trace = []
    current_info = info

    while not done and steps < env.max_steps:
        priv = current_info["privileged_state"]
        curr_pos = env._get_drone_position()
        curr_vel = priv.global_velocity
        goal_dist = float(priv.goal_distance)
        front_dist = compute_forward_clearance_m(obs, max_range_m=env.sensor_config.max_range_m)

        obs_c = float(priv.obstacle_clearance)
        wall_c = float(priv.corridor_boundary_distance)
        min_obs_overall = min(min_obs_overall, obs_c)
        min_wall_overall = min(min_wall_overall, wall_c)

        for i, o in enumerate(env.obstacles):
            c_i = o.distance_to_point(curr_pos) - env.corridor.drone_radius
            per_obs_min[i] = min(per_obs_min[i], float(c_i))

        speed = float(np.linalg.norm(curr_vel))
        if speed <= 0.05:
            stationary_steps += 1

        with torch.no_grad():
            t_obs = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            act_arr = model(t_obs, bounded=True).squeeze(0).cpu().numpy()

        raw_act = VelocityAction(
            vx=float(act_arr[0]),
            vy=float(act_arr[1]),
            vz=float(act_arr[2]),
            yaw_rate=float(act_arr[3]),
        )
        raw_vx_list.append(raw_act.vx)
        raw_vy_list.append(raw_act.vy)
        raw_vz_list.append(raw_act.vz)
        raw_yr_list.append(raw_act.yaw_rate)

        clamped_act = VelocityAction(
            vx=float(np.clip(raw_act.vx, env.action_limits.min_fwd_vel_mps, env.action_limits.max_fwd_vel_mps)),
            vy=float(np.clip(raw_act.vy, -env.action_limits.max_lat_vel_mps, env.action_limits.max_lat_vel_mps)),
            vz=float(np.clip(raw_act.vz, -env.action_limits.max_vert_vel_mps, env.action_limits.max_vert_vel_mps)),
            yaw_rate=float(np.clip(raw_act.yaw_rate, -env.action_limits.max_yaw_rate_radps, env.action_limits.max_yaw_rate_radps)),
        )

        safe_act, s_state, s_reason = supervisor.filter_action(
            proposed_action=clamped_act,
            current_position=priv.global_position,
            current_velocity=priv.global_velocity,
            min_front_range=front_dist,
            dt=env.dt,
            is_localization_healthy=True,
            telemetry_age_s=0.0,
            observation=obs,
        )

        safe_vx_list.append(safe_act.vx)
        safe_vy_list.append(safe_act.vy)
        safe_vz_list.append(safe_act.vz)
        safe_yr_list.append(safe_act.yaw_rate)

        norm_applied = safe_act.to_normalized(env.action_limits)

        if s_state != SafetyState.NOMINAL:
            interventions += 1
        if "Forward brake" in s_reason:
            emergency_brakes += 1
        if "Reversing" in s_reason or "Reverse retreat" in s_reason:
            retreats += 1
        if "Speed throttled" in s_reason:
            throttled_steps += 1

        if record_trace:
            step_trace.append({
                "step": steps,
                "time_s": round(steps * env.dt, 3),
                "pos": [round(float(x), 3) for x in curr_pos],
                "vel": [round(float(v), 3) for v in curr_vel],
                "speed": round(speed, 3),
                "goal_dist": round(goal_dist, 3),
                "front_dist": round(front_dist, 3),
                "obs_clearance": round(obs_c, 4),
                "wall_clearance": round(wall_c, 4),
                "raw_action": [round(raw_act.vx, 3), round(raw_act.vy, 3), round(raw_act.vz, 3), round(raw_act.yaw_rate, 3)],
                "safe_action": [round(safe_act.vx, 3), round(safe_act.vy, 3), round(safe_act.vz, 3), round(safe_act.yaw_rate, 3)],
                "state": s_state.value,
                "guidance_state": supervisor.guidance_state.value,
                "reason": s_reason,
            })

        obs, rew, term, trunc, current_info = env.step(norm_applied)
        steps += 1
        done = term or trunc

    # Final evaluations
    final_priv = current_info["privileged_state"]
    min_obs_overall = min(min_obs_overall, float(final_priv.obstacle_clearance))
    min_wall_overall = min(min_wall_overall, float(final_priv.corridor_boundary_distance))
    final_pos = env._get_drone_position()
    for i, o in enumerate(env.obstacles):
        per_obs_min[i] = min(per_obs_min[i], float(o.distance_to_point(final_pos) - env.corridor.drone_radius))

    final_vel = float(np.linalg.norm(final_priv.global_velocity))
    final_goal_dist = float(final_priv.goal_distance)
    term_reason = current_info.get("termination_reason", "unknown")

    is_success = (term_reason == "success") and (final_vel <= 0.35)
    goal_entered = (final_goal_dist <= env.corridor.goal_radius)
    is_obs_collision = (term_reason == "crash_obstacle") or (min_obs_overall <= 0.0)
    is_wall_collision = (term_reason == "crash_boundary") or (min_wall_overall <= 0.0)
    is_timeout = (steps >= env.max_steps and not is_success and not is_obs_collision and not is_wall_collision)

    is_obs_envelope_violation = (min_obs_overall < OBSTACLE_SAFETY_ENVELOPE_M)
    is_wall_envelope_violation = (min_wall_overall < BOUNDARY_SAFETY_ENVELOPE_M)

    # Detailed Failure Classification
    fail_category = "SUCCESS" if is_success else "UNKNOWN"
    if not is_success:
        if is_obs_collision:
            fail_category = "OBSTACLE_COLLISION"
        elif is_wall_collision:
            fail_category = "WALL_COLLISION"
        elif goal_entered and not is_success:
            fail_category = "GOAL_APPROACH_SPEED_EXCEEDED"
        elif final_goal_dist < 0.60:
            fail_category = "GOAL_ALIGNMENT_TIMEOUT"
        elif final_pos[0] < 4.0:
            fail_category = "PROGRESS_STALL_PILLAR_1"
        elif final_pos[0] < 6.0:
            fail_category = "PROGRESS_STALL_PILLAR_2"
        elif final_pos[0] < 8.0:
            fail_category = "PROGRESS_STALL_PILLAR_3"
        elif throttled_steps > 250:
            fail_category = "TIMEOUT_EXCESSIVE_THROTTLING"
        else:
            fail_category = "TIMEOUT_SLOW_TRANSIT"

    # State duration seconds
    state_durations_s = {
        k: round(v * env.dt, 2) for k, v in supervisor.state_step_counts.items()
    }

    res = {
        "seed": seed,
        "configuration": config_name,
        "goal_success": is_success,
        "termination_reason": term_reason,
        "final_position": [round(float(x), 3) for x in final_pos],
        "final_x": round(float(final_pos[0]), 3),
        "final_y": round(float(final_pos[1]), 3),
        "final_z": round(float(final_pos[2]), 3),
        "forward_progress_m": round(float(final_pos[0] - ep_start_pos[0]), 3),
        "episode_duration_s": round(steps * env.dt, 2),
        "steps_completed": steps,
        "obstacle_collisions": 1 if is_obs_collision else 0,
        "boundary_collisions": 1 if is_wall_collision else 0,
        "obstacle_safety_envelope_violations": 1 if is_obs_envelope_violation else 0,
        "boundary_safety_envelope_violations": 1 if is_wall_envelope_violation else 0,
        "min_obstacle_clearance_m": round(min_obs_overall, 4),
        "min_boundary_clearance_m": round(min_wall_overall, 4),
        "per_obstacle_min_clearance_m": [round(x, 4) for x in per_obs_min],
        "supervisor_intervention_rate": round(interventions / max(1, steps), 4),
        "fraction_steps_throttled": round(throttled_steps / max(1, steps), 4),
        "emergency_brakes": emergency_brakes,
        "retreats": retreats,
        "blocked_progress_detections": supervisor.total_deadlock_detections,
        "lateral_commitments": supervisor.total_commitments,
        "direction_changes": supervisor.total_direction_changes,
        "fallback_events": supervisor.total_fallback_events,
        "entered_fallback_state": bool(supervisor.total_fallback_events > 0),
        "time_spent_in_each_state_s": state_durations_s,
        "time_state_idle_s": state_durations_s.get("idle", 0.0),
        "time_state_guidance_left_s": state_durations_s.get("guidance_active_left", 0.0),
        "time_state_guidance_right_s": state_durations_s.get("guidance_active_right", 0.0),
        "time_state_fallback_blocked_s": state_durations_s.get("fallback_blocked", 0.0),
        "fail_category": fail_category,
    }

    if record_trace:
        res["trace"] = step_trace

    return res


def main():
    exp_dir = Path(__file__).resolve().parent
    traces_dir = exp_dir / "traces"
    traces_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 90)
    print(">>> PHASE 4J-F CONTROLLED PAIRED-SEED EVALUATION <<<")
    print("=" * 90)

    verify_integrity()

    device = torch.device("cpu")
    env = CorridorSimEnv(scenario="10m", gui=False, seed=42)

    model = NavigationPolicyMLP(obs_dim=23, act_dim=4).to(device)
    ckpt = torch.load(BC_CHECKPOINT_PATH, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    configs = ["baseline", "lateral_guidance"]
    all_results: Dict[str, List[Dict[str, Any]]] = {c: [] for c in configs}

    for config_name in configs:
        print(f"\n--- Running Configuration: {config_name.upper()} ({len(EVAL_SEEDS)} seeds) ---")
        sup = IsolatedLateralGuidanceSupervisor(
            corridor=env.corridor,
            mode=config_name,
            caution_distance_m=CAUTION_DIST_M,
        )

        for idx, seed in enumerate(EVAL_SEEDS, 1):
            res = run_episode(
                env=env,
                supervisor=sup,
                model=model,
                config_name=config_name,
                seed=seed,
                device=device,
                record_trace=False,
            )
            all_results[config_name].append(res)
            succ_str = "SUCCESS" if res["goal_success"] else "FAIL"
            print(
                f"[{idx:02d}/50] Seed {seed:4d} | {succ_str:<7} | "
                f"Fwd: {res['forward_progress_m']:4.2f}m | "
                f"MinObs: {res['min_obstacle_clearance_m']:.3f}m | "
                f"MinWall: {res['min_boundary_clearance_m']:.3f}m | "
                f"Steps: {res['steps_completed']:3d} | "
                f"Cat: {res['fail_category']} | "
                f"Deadlocks: {res['blocked_progress_detections']} | "
                f"Commits: {res['lateral_commitments']}"
            )

    # 1. Output paired_episode_results.csv
    csv_path = exp_dir / "paired_episode_results.csv"
    csv_fields = [
        "seed",
        "configuration",
        "goal_success",
        "termination_reason",
        "final_x",
        "final_y",
        "final_z",
        "forward_progress_m",
        "episode_duration_s",
        "steps_completed",
        "obstacle_collisions",
        "boundary_collisions",
        "obstacle_safety_envelope_violations",
        "boundary_safety_envelope_violations",
        "min_obstacle_clearance_m",
        "min_boundary_clearance_m",
        "supervisor_intervention_rate",
        "fraction_steps_throttled",
        "blocked_progress_detections",
        "lateral_commitments",
        "direction_changes",
        "time_state_idle_s",
        "time_state_guidance_left_s",
        "time_state_guidance_right_s",
        "time_state_fallback_blocked_s",
        "entered_fallback_state",
        "fail_category",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        for config_name in configs:
            for ep in all_results[config_name]:
                row = {k: ep[k] for k in csv_fields}
                writer.writerow(row)
    print(f"\n[OK] Saved paired CSV: {csv_path}")

    # 2. Paired McNemar and Contingency Analysis
    base_eps = all_results["baseline"]
    guid_eps = all_results["lateral_guidance"]

    n_both_succ = 0
    n_base_only = 0
    n_guid_only = 0
    n_both_fail = 0

    improved_seeds = []
    degraded_seeds = []
    matched_succ_seeds = []
    matched_fail_seeds = []

    for b_ep, g_ep in zip(base_eps, guid_eps):
        seed = b_ep["seed"]
        b_succ = b_ep["goal_success"]
        g_succ = g_ep["goal_success"]

        if b_succ and g_succ:
            n_both_succ += 1
            matched_succ_seeds.append(seed)
        elif b_succ and not g_succ:
            n_base_only += 1
            degraded_seeds.append(seed)
        elif not b_succ and g_succ:
            n_guid_only += 1
            improved_seeds.append(seed)
        else:
            n_both_fail += 1
            matched_fail_seeds.append(seed)

    mcnemar_p = mcnemar_exact_test(b=n_base_only, c=n_guid_only)

    print("\n" + "=" * 50)
    print("PAIRED CONTINGENCY TABLE (50 Matched Seeds):")
    print(f"  Both Succeed:             {n_both_succ:2d} (Seeds: {matched_succ_seeds})")
    print(f"  Baseline Only Succeeds:   {n_base_only:2d} (Degraded Seeds: {degraded_seeds})")
    print(f"  Guidance Only Succeeds:   {n_guid_only:2d} (Improved Seeds: {improved_seeds})")
    print(f"  Both Fail:                {n_both_fail:2d}")
    print(f"  McNemar exact test p-value: {mcnemar_p:.6f}")
    print("=" * 50)

    # 3. Capture Detailed Traces for Representative Pairs
    rep_seeds_to_trace = []
    if improved_seeds:
        rep_seeds_to_trace.append((improved_seeds[0], "GUIDANCE_IMPROVES"))
    if degraded_seeds:
        rep_seeds_to_trace.append((degraded_seeds[0], "GUIDANCE_DEGRADES"))
    if matched_succ_seeds:
        rep_seeds_to_trace.append((matched_succ_seeds[0], "BOTH_SUCCEED"))
    if matched_fail_seeds:
        rep_seeds_to_trace.append((matched_fail_seeds[0], "BOTH_FAIL"))

    # Also capture specific seeds if present
    for s_target in [2004, 2000]:
        if s_target not in [s for s, _ in rep_seeds_to_trace]:
            rep_seeds_to_trace.append((s_target, f"SEED_{s_target}_INSPECTION"))

    print(f"\nRecording detailed step traces for {len(rep_seeds_to_trace)} representative scenarios...")
    for s_target, cat_tag in rep_seeds_to_trace:
        matched_trace_data = {
            "seed": s_target,
            "category": cat_tag,
            "baseline": None,
            "lateral_guidance": None,
        }
        for cfg_name in ["baseline", "lateral_guidance"]:
            sup_t = IsolatedLateralGuidanceSupervisor(corridor=env.corridor, mode=cfg_name, caution_distance_m=CAUTION_DIST_M)
            ep_trace = run_episode(
                env=env,
                supervisor=sup_t,
                model=model,
                config_name=cfg_name,
                seed=s_target,
                device=device,
                record_trace=True,
            )
            matched_trace_data[cfg_name] = ep_trace

        t_filename = traces_dir / f"matched_trace_seed_{s_target}_{cat_tag.lower()}.json"
        with open(t_filename, "w", encoding="utf-8") as f:
            json.dump(matched_trace_data, f, indent=2, cls=NumpyEncoder)
        print(f"  [OK] Saved trace: {t_filename.name}")

    # 4. Summary metrics
    summary: Dict[str, Any] = {
        "evaluation_name": "Phase 4J-F Controlled BC Baseline vs Lateral Guidance Evaluation",
        "checkpoint": BC_CHECKPOINT_PATH,
        "checkpoint_sha256": EXPECTED_BC_SHA256,
        "canonical_supervisor_sha256": EXPECTED_SUPERVISOR_SHA256,
        "caution_distance_m": CAUTION_DIST_M,
        "num_seeds": len(EVAL_SEEDS),
        "seed_range": [EVAL_SEEDS[0], EVAL_SEEDS[-1]],
        "horizon_steps": 500,
        "dt_s": 0.05,
        "contingency_table": {
            "both_succeed": n_both_succ,
            "baseline_only": n_base_only,
            "guidance_only": n_guid_only,
            "both_fail": n_both_fail,
            "improved_seeds": improved_seeds,
            "degraded_seeds": degraded_seeds,
            "mcnemar_exact_p_value": round(mcnemar_p, 6),
            "is_statistically_significant": bool(mcnemar_p < 0.05),
        },
        "controllers": {},
    }

    for config_name in configs:
        eps = all_results[config_name]
        n_succ = sum(1 for e in eps if e["goal_success"])
        succ_rate = round((n_succ / len(eps)) * 100.0, 1)
        w_low, w_high = wilson_score_interval(n_succ, len(eps))

        tot_obs_coll = sum(e["obstacle_collisions"] for e in eps)
        tot_wall_coll = sum(e["boundary_collisions"] for e in eps)
        tot_obs_env = sum(e["obstacle_safety_envelope_violations"] for e in eps)
        tot_wall_env = sum(e["boundary_safety_envelope_violations"] for e in eps)

        fwd_progs = [e["forward_progress_m"] for e in eps]
        durations = [e["episode_duration_s"] for e in eps]
        obs_clears = [e["min_obstacle_clearance_m"] for e in eps]
        wall_clears = [e["min_boundary_clearance_m"] for e in eps]
        interv_rates = [e["supervisor_intervention_rate"] for e in eps]
        throttled_fracs = [e["fraction_steps_throttled"] for e in eps]

        summary["controllers"][config_name] = {
            "successes": n_succ,
            "total_episodes": len(eps),
            "success_rate_pct": succ_rate,
            "wilson_95_ci_pct": [w_low, w_high],
            "obstacle_collisions": tot_obs_coll,
            "boundary_collisions": tot_wall_coll,
            "obstacle_envelope_violations": tot_obs_env,
            "boundary_envelope_violations": tot_wall_env,
            "forward_progress_m": {
                "mean": round(float(np.mean(fwd_progs)), 3),
                "std": round(float(np.std(fwd_progs)), 3),
                "median": round(float(np.median(fwd_progs)), 3),
            },
            "duration_s": {
                "mean": round(float(np.mean(durations)), 2),
                "std": round(float(np.std(durations)), 2),
            },
            "min_obstacle_clearance_m": {
                "absolute_min": round(float(np.min(obs_clears)), 4),
                "mean": round(float(np.mean(obs_clears)), 4),
                "std": round(float(np.std(obs_clears)), 4),
            },
            "min_boundary_clearance_m": {
                "absolute_min": round(float(np.min(wall_clears)), 4),
                "mean": round(float(np.mean(wall_clears)), 4),
                "std": round(float(np.std(wall_clears)), 4),
            },
            "supervisor_intervention_rate": {
                "mean": round(float(np.mean(interv_rates)), 4),
                "std": round(float(np.std(interv_rates)), 4),
            },
            "fraction_steps_throttled": {
                "mean": round(float(np.mean(throttled_fracs)), 4),
                "std": round(float(np.std(throttled_fracs)), 4),
            },
            "guidance_metrics": {
                "total_deadlocks_detected": sum(e["blocked_progress_detections"] for e in eps),
                "total_lateral_commitments": sum(e["lateral_commitments"] for e in eps),
                "total_direction_changes": sum(e["direction_changes"] for e in eps),
                "total_fallback_events": sum(e["fallback_events"] for e in eps),
                "mean_time_state_idle_s": round(float(np.mean([e["time_state_idle_s"] for e in eps])), 2),
                "mean_time_guidance_left_s": round(float(np.mean([e["time_state_guidance_left_s"] for e in eps])), 2),
                "mean_time_guidance_right_s": round(float(np.mean([e["time_state_guidance_right_s"] for e in eps])), 2),
                "mean_time_fallback_blocked_s": round(float(np.mean([e["time_state_fallback_blocked_s"] for e in eps])), 2),
            },
        }

    summary_path = exp_dir / "controller_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, cls=NumpyEncoder)
    print(f"[OK] Saved controller summary: {summary_path}")

    # 5. Manifest
    manifest = {
        "experiment_name": "Phase 4J-F Proactive Lateral Guidance and Deadlock-Breaking Evaluation",
        "timestamp": "20261029_182017",
        "workspace": "/Users/ghulamqadir/.gemini/antigravity-ide/scratch/uav-bput-nav",
        "experiment_directory": str(exp_dir),
        "artifacts_generated": [
            "PHASE_4JF_PRE_RUN_AUDIT.md",
            "experimental_supervisor.py",
            "experimental_config.json",
            "run_phase4jf_evaluation.py",
            "paired_episode_results.csv",
            "controller_summary.json",
            "comparison_manifest.json",
            "failure_analysis.md",
            "PHASE_4JF_FINAL_REPORT.md",
        ],
        "cryptographic_hashes": {
            "bc_checkpoint_sha256": EXPECTED_BC_SHA256,
            "canonical_supervisor_sha256": EXPECTED_SUPERVISOR_SHA256,
            "experimental_supervisor_sha256": hashlib.sha256((exp_dir / "experimental_supervisor.py").read_bytes()).hexdigest(),
            "experimental_config_sha256": hashlib.sha256((exp_dir / "experimental_config.json").read_bytes()).hexdigest(),
            "evaluation_script_sha256": hashlib.sha256((exp_dir / "run_phase4jf_evaluation.py").read_bytes()).hexdigest(),
        },
        "environment_configuration": {
            "corridor": "10m canonical benchmark",
            "corridor_length_m": 10.0,
            "corridor_half_width_m": 1.0,
            "drone_radius_m": 0.15,
            "dt_s": 0.05,
            "max_steps": 500,
            "caution_distance_m": CAUTION_DIST_M,
            "safety_envelope_threshold_m": 0.08,
            "development_seeds": [EVAL_SEEDS[0], EVAL_SEEDS[-1]],
            "held_out_seeds_status": "SEALED AND UNTOUCHED (90000..90099)",
        },
        "python_environment": {
            "python_version": sys.version,
            "torch_version": torch.__version__,
            "numpy_version": np.__version__,
        },
    }
    manifest_path = exp_dir / "comparison_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, cls=NumpyEncoder)
    print(f"[OK] Saved comparison manifest: {manifest_path}")

    env.close()
    print("\nPhase 4J-F evaluation execution complete.")


if __name__ == "__main__":
    main()

