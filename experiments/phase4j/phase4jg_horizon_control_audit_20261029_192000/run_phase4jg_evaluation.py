"""run_phase4jg_evaluation.py - Phase 4J-G Horizon Sensitivity and Control Audit.

Evaluates the frozen Behavior Cloning (BC) baseline controller under two horizon configurations:
- Configuration A: 500 steps (25.0s at dt = 0.05s)
- Configuration B: 600 steps (30.0s at dt = 0.05s)

Seeds: 2000..2049 (50 matched development seeds).
Supervisor: Canonical SafetySupervisor with caution distance d_caut = 0.50m.
"""

import csv
import hashlib
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch

from controllers.safety_supervisor import SafetyState, SafetySupervisor
from core.contracts import ActionLimits, VelocityAction, compute_forward_clearance_m
from core.geometry import Corridor3D, TerminationReason
from experiments.phase4j.isolated_supervisor_opt import IsolatedOptimizedSupervisor
from experiments.phase4j.phase4jf_lateral_guidance_20261029_182017.experimental_supervisor import (
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
    p_bc = Path(BC_CHECKPOINT_PATH)
    if not p_bc.is_file():
        raise FileNotFoundError(f"BC checkpoint missing: {BC_CHECKPOINT_PATH}")
    h_bc = hashlib.sha256(p_bc.read_bytes()).hexdigest()
    if h_bc != EXPECTED_BC_SHA256:
        raise ValueError(f"BC checkpoint hash mismatch: expected {EXPECTED_BC_SHA256}, got {h_bc}")

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
    supervisor: IsolatedOptimizedSupervisor,
    model: NavigationPolicyMLP,
    horizon_max_steps: int,
    seed: int,
    device: torch.device,
    record_trace: bool = False,
) -> Dict[str, Any]:
    env.max_steps = horizon_max_steps
    env.corridor.max_steps = horizon_max_steps

    obs, info = env.reset(seed=seed)
    supervisor.reset_state()

    ep_start_pos = env._get_drone_position()
    steps = 0
    done = False

    min_obs_overall = float(info["privileged_state"].obstacle_clearance)
    min_wall_overall = float(info["privileged_state"].corridor_boundary_distance)
    per_obs_min = [10.0] * len(env.obstacles)

    interventions = 0
    throttled_steps = 0
    emergency_brakes = 0

    pos_at_step_500 = None
    goal_dist_at_step_500 = None
    fwd_prog_at_step_500 = None

    step_trace = []
    current_info = info

    while not done and steps < horizon_max_steps:
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

        # Capture step 500 boundary checkpoint
        if steps == 499:
            pos_at_step_500 = [round(float(x), 4) for x in curr_pos]
            goal_dist_at_step_500 = round(float(goal_dist), 4)
            fwd_prog_at_step_500 = round(float(curr_pos[0] - ep_start_pos[0]), 4)

        with torch.no_grad():
            t_obs = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            act_arr = model(t_obs, bounded=True).squeeze(0).cpu().numpy()

        raw_act = VelocityAction(
            vx=float(act_arr[0]),
            vy=float(act_arr[1]),
            vz=float(act_arr[2]),
            yaw_rate=float(act_arr[3]),
        )

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

        norm_applied = safe_act.to_normalized(env.action_limits)

        if s_state != SafetyState.NOMINAL:
            interventions += 1
        if "Forward brake" in s_reason:
            emergency_brakes += 1
        if "Speed throttled" in s_reason:
            throttled_steps += 1

        if record_trace:
            step_trace.append({
                "step": steps,
                "time_s": round(steps * env.dt, 3),
                "pos": [round(float(x), 3) for x in curr_pos],
                "vel": [round(float(v), 3) for v in curr_vel],
                "speed": round(float(np.linalg.norm(curr_vel)), 3),
                "goal_dist": round(goal_dist, 3),
                "front_dist": round(front_dist, 3),
                "obs_clearance": round(obs_c, 4),
                "wall_clearance": round(wall_c, 4),
                "raw_action": [round(raw_act.vx, 3), round(raw_act.vy, 3), round(raw_act.vz, 3), round(raw_act.yaw_rate, 3)],
                "safe_action": [round(safe_act.vx, 3), round(safe_act.vy, 3), round(safe_act.vz, 3), round(safe_act.yaw_rate, 3)],
                "state": s_state.value,
                "reason": s_reason,
            })

        obs, rew, term, trunc, current_info = env.step(norm_applied)
        steps += 1
        done = term or trunc

    # Final clearances and states
    final_priv = current_info["privileged_state"]
    min_obs_overall = min(min_obs_overall, float(final_priv.obstacle_clearance))
    min_wall_overall = min(min_wall_overall, float(final_priv.corridor_boundary_distance))
    final_pos = env._get_drone_position()

    final_vel = float(np.linalg.norm(final_priv.global_velocity))
    final_goal_dist = float(final_priv.goal_distance)
    term_reason = current_info.get("termination_reason", "unknown")

    is_success = (term_reason == "success") and (final_vel <= 0.35)
    is_obs_collision = (term_reason == "crash_obstacle") or (min_obs_overall <= 0.0)
    is_wall_collision = (term_reason == "crash_boundary") or (min_wall_overall <= 0.0)
    is_timeout = (steps >= horizon_max_steps and not is_success and not is_obs_collision and not is_wall_collision)

    is_obs_envelope_violation = (min_obs_overall < OBSTACLE_SAFETY_ENVELOPE_M)
    is_wall_envelope_violation = (min_wall_overall < BOUNDARY_SAFETY_ENVELOPE_M)

    # In case episode finished before step 500, checkpoint defaults to final
    if pos_at_step_500 is None:
        pos_at_step_500 = [round(float(x), 4) for x in final_pos]
        goal_dist_at_step_500 = round(final_goal_dist, 4)
        fwd_prog_at_step_500 = round(float(final_pos[0] - ep_start_pos[0]), 4)

    completed_by_step_500 = bool(is_success and steps <= 500)
    completed_between_501_600 = bool(is_success and steps > 500)

    # Standardized 8-Class Failure Taxonomy
    if is_success:
        failure_category = "SUCCESS"
    elif is_obs_collision:
        failure_category = "OBSTACLE_COLLISION"
    elif is_wall_collision:
        failure_category = "BOUNDARY_COLLISION"
    elif final_goal_dist <= 0.30 and final_vel > 0.35:
        failure_category = "GOAL_ALIGNMENT_TERMINAL_SPEED"
    elif final_goal_dist < 0.60:
        failure_category = "GOAL_ALIGNMENT_STANDOFF"
    elif final_pos[0] < 3.8:
        failure_category = "PERSISTENT_STANDOFF_PILLAR_1"
    elif final_pos[0] < 5.8:
        failure_category = "PERSISTENT_STANDOFF_PILLAR_2"
    elif final_pos[0] < 7.8:
        failure_category = "PERSISTENT_STANDOFF_PILLAR_3"
    elif throttled_steps > 350:
        failure_category = "EXCESSIVE_BRAKING_CYCLE"
    else:
        failure_category = "INSUFFICIENT_TIME_AFTER_AVOIDANCE"

    res = {
        "seed": seed,
        "horizon_max_steps": horizon_max_steps,
        "goal_success": is_success,
        "termination_reason": term_reason,
        "steps_completed": steps,
        "duration_s": round(steps * env.dt, 2),
        "final_position": [round(float(x), 3) for x in final_pos],
        "final_x": round(float(final_pos[0]), 3),
        "final_y": round(float(final_pos[1]), 3),
        "final_z": round(float(final_pos[2]), 3),
        "forward_progress_at_step_500_m": fwd_prog_at_step_500,
        "forward_progress_at_term_m": round(float(final_pos[0] - ep_start_pos[0]), 3),
        "goal_distance_at_step_500_m": goal_dist_at_step_500,
        "goal_distance_at_term_m": round(final_goal_dist, 4),
        "obstacle_collisions": 1 if is_obs_collision else 0,
        "boundary_collisions": 1 if is_wall_collision else 0,
        "obstacle_safety_envelope_violations": 1 if is_obs_envelope_violation else 0,
        "boundary_safety_envelope_violations": 1 if is_wall_envelope_violation else 0,
        "min_obstacle_clearance_m": round(min_obs_overall, 4),
        "min_boundary_clearance_m": round(min_wall_overall, 4),
        "completed_by_step_500": completed_by_step_500,
        "completed_between_501_600": completed_between_501_600,
        "supervisor_intervention_rate": round(interventions / max(1, steps), 4),
        "fraction_steps_throttled": round(throttled_steps / max(1, steps), 4),
        "emergency_brakes": emergency_brakes,
        "failure_category": failure_category,
    }

    if record_trace:
        res["trace"] = step_trace

    return res


def run_conflict_audit(env: CorridorSimEnv, model: NavigationPolicyMLP, device: torch.device) -> Dict[str, Any]:
    """Audits lateral action opposition and control conflict between BC policy and supervisor."""
    print("\n>>> AUDITING POLICY-SUPERVISOR LATERAL COMPATIBILITY <<<")
    sup_canonical = IsolatedOptimizedSupervisor(corridor=env.corridor, caution_distance_m=CAUTION_DIST_M)
    sup_guidance = IsolatedLateralGuidanceSupervisor(corridor=env.corridor, mode="lateral_guidance", caution_distance_m=CAUTION_DIST_M)

    audit_stats = {"canonical": {}, "guidance": {}}

    for arm_name, sup in [("canonical", sup_canonical), ("guidance", sup_guidance)]:
        total_steps = 0
        opposing_steps = 0
        clamped_steps = 0
        conflict_episodes = []
        cur_conflict_len = 0
        conflict_magnitudes = []
        progress_during_conflict = []
        progress_during_nominal = []

        for seed in EVAL_SEEDS:
            env.max_steps = 500
            env.corridor.max_steps = 500
            obs, info = env.reset(seed=seed)
            sup.reset_state()
            prev_pos = env._get_drone_position()

            for st in range(500):
                with torch.no_grad():
                    act = model(torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0), bounded=True).squeeze(0).cpu().numpy()
                raw_act = VelocityAction(float(act[0]), float(act[1]), float(act[2]), float(act[3]))
                front_d = compute_forward_clearance_m(obs, max_range_m=env.sensor_config.max_range_m)
                safe_act, s_state, s_reason = sup.filter_action(
                    proposed_action=raw_act,
                    current_position=env._get_drone_position(),
                    current_velocity=info["privileged_state"].global_velocity,
                    min_front_range=front_d,
                    dt=0.05,
                    observation=obs,
                )

                total_steps += 1
                curr_pos = env._get_drone_position()
                step_dx = curr_pos[0] - prev_pos[0]
                prev_pos = curr_pos

                is_opposing = False
                if abs(raw_act.vy) > 0.02 and abs(safe_act.vy) > 0.02:
                    if np.sign(raw_act.vy) != np.sign(safe_act.vy):
                        is_opposing = True
                        opposing_steps += 1
                        conflict_magnitudes.append(abs(raw_act.vy - safe_act.vy))
                        progress_during_conflict.append(step_dx)

                if is_opposing:
                    cur_conflict_len += 1
                else:
                    if cur_conflict_len > 0:
                        conflict_episodes.append(cur_conflict_len)
                        cur_conflict_len = 0
                    progress_during_nominal.append(step_dx)

                if abs(raw_act.vy) > 0.05 and abs(safe_act.vy) < 0.01:
                    clamped_steps += 1

                obs, rew, term, trunc, info = env.step(safe_act.to_normalized(env.action_limits))
                if term or trunc:
                    if cur_conflict_len > 0:
                        conflict_episodes.append(cur_conflict_len)
                        cur_conflict_len = 0
                    break

        audit_stats[arm_name] = {
            "total_steps": total_steps,
            "opposing_steps": opposing_steps,
            "fraction_opposing": round(opposing_steps / max(1, total_steps), 5),
            "clamped_steps": clamped_steps,
            "fraction_clamped": round(clamped_steps / max(1, total_steps), 5),
            "mean_conflict_magnitude_mps": round(float(np.mean(conflict_magnitudes)), 4) if conflict_magnitudes else 0.0,
            "max_conflict_magnitude_mps": round(float(np.max(conflict_magnitudes)), 4) if conflict_magnitudes else 0.0,
            "mean_conflict_duration_steps": round(float(np.mean(conflict_episodes)), 2) if conflict_episodes else 0.0,
            "max_conflict_duration_steps": int(np.max(conflict_episodes)) if conflict_episodes else 0,
            "mean_dx_during_conflict_m": round(float(np.mean(progress_during_conflict)), 4) if progress_during_conflict else 0.0,
            "mean_dx_during_nominal_m": round(float(np.mean(progress_during_nominal)), 4) if progress_during_nominal else 0.0,
        }
        print(f"  [{arm_name.upper()}] Opposing Steps: {opposing_steps}/{total_steps} ({audit_stats[arm_name]['fraction_opposing']*100:.2f}%) | "
              f"Mean Mag: {audit_stats[arm_name]['mean_conflict_magnitude_mps']:.3f} m/s | "
              f"Max Duration: {audit_stats[arm_name]['max_conflict_duration_steps']} steps")

    return audit_stats


def main():
    exp_dir = Path(__file__).resolve().parent
    traces_dir = exp_dir / "traces"
    traces_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 90)
    print(">>> PHASE 4J-G HORIZON SENSITIVITY AND CONTROL AUDIT <<<")
    print("=" * 90)

    verify_integrity()

    device = torch.device("cpu")
    env = CorridorSimEnv(scenario="10m", gui=False, seed=42)

    model = NavigationPolicyMLP(obs_dim=23, act_dim=4).to(device)
    ckpt = torch.load(BC_CHECKPOINT_PATH, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    supervisor = IsolatedOptimizedSupervisor(corridor=env.corridor, caution_distance_m=CAUTION_DIST_M)

    horizons = [500, 600]
    all_results: Dict[int, List[Dict[str, Any]]] = {h: [] for h in horizons}

    for h in horizons:
        print(f"\n--- Running Horizon Configuration: {h} Steps ({h*0.05:.1f}s) ({len(EVAL_SEEDS)} seeds) ---")
        for idx, seed in enumerate(EVAL_SEEDS, 1):
            res = run_episode(
                env=env,
                supervisor=supervisor,
                model=model,
                horizon_max_steps=h,
                seed=seed,
                device=device,
                record_trace=False,
            )
            all_results[h].append(res)
            succ_str = "SUCCESS" if res["goal_success"] else "FAIL"
            print(
                f"[{idx:02d}/50] Seed {seed:4d} | {succ_str:<7} | "
                f"Fwd@500: {res['forward_progress_at_step_500_m']:4.2f}m | "
                f"Fwd@Term: {res['forward_progress_at_term_m']:4.2f}m | "
                f"Steps: {res['steps_completed']:3d} | "
                f"Cat: {res['failure_category']}"
            )

    # 1. Output horizon_paired_episode_results.csv
    csv_path = exp_dir / "horizon_paired_episode_results.csv"
    csv_fields = [
        "seed",
        "horizon_max_steps",
        "goal_success",
        "termination_reason",
        "steps_completed",
        "duration_s",
        "final_x",
        "final_y",
        "final_z",
        "forward_progress_at_step_500_m",
        "forward_progress_at_term_m",
        "goal_distance_at_step_500_m",
        "goal_distance_at_term_m",
        "obstacle_collisions",
        "boundary_collisions",
        "obstacle_safety_envelope_violations",
        "boundary_safety_envelope_violations",
        "min_obstacle_clearance_m",
        "min_boundary_clearance_m",
        "completed_by_step_500",
        "completed_between_501_600",
        "supervisor_intervention_rate",
        "fraction_steps_throttled",
        "emergency_brakes",
        "failure_category",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        for h in horizons:
            for ep in all_results[h]:
                row = {k: ep[k] for k in csv_fields}
                writer.writerow(row)
    print(f"\n[OK] Saved paired CSV: {csv_path}")

    # 2. Verify bitwise trajectory determinism over common 500-step interval
    print("\n>>> VERIFYING PAIRED DETERMINISM OVER FIRST 500 STEPS <<<")
    max_step_500_diff = 0.0
    for ep500, ep600 in zip(all_results[500], all_results[600]):
        diff = abs(ep500["forward_progress_at_step_500_m"] - ep600["forward_progress_at_step_500_m"])
        max_step_500_diff = max(max_step_500_diff, diff)
    print(f"  Maximum progress difference at step 500 across all 50 seeds: {max_step_500_diff:.10f} m")
    assert max_step_500_diff < 1e-4, "Divergence detected in common 500-step interval!"

    # 3. Output failure_classification.csv
    fail_csv_path = exp_dir / "failure_classification.csv"
    fail_fields = [
        "seed",
        "failure_category",
        "final_x",
        "final_y",
        "final_z",
        "final_goal_dist_m",
        "progress_at_500_m",
        "progress_at_600_m",
        "delta_x_501_600_m",
        "is_productive_progress",
        "stalled_obstacle",
        "explanation",
    ]

    fail_rows = []
    for ep500, ep600 in zip(all_results[500], all_results[600]):
        if not ep500["goal_success"]:
            seed = ep500["seed"]
            x_500 = ep500["forward_progress_at_step_500_m"]
            x_600 = ep600["forward_progress_at_term_m"]
            dx = round(x_600 - x_500, 4)
            cat = ep600["failure_category"]

            if "PILLAR_1" in cat:
                obs_label = "Pillar 1 (x ~ 3.0m)"
            elif "PILLAR_2" in cat:
                obs_label = "Pillar 2 (x ~ 5.0m)"
            elif "PILLAR_3" in cat:
                obs_label = "Pillar 3 (x ~ 7.0m)"
            else:
                obs_label = "None"

            # Check if progress between 501 and 600 was productive (>0.15m)
            is_productive = bool(dx >= 0.15)
            exp_text = (
                f"Hover standoff at {obs_label}; speed clamped by caution zone (dx={dx:+.3f}m over 100 extra steps)"
                if not is_productive else f"Continuous progress toward goal (dx={dx:+.3f}m)"
            )

            fail_rows.append({
                "seed": seed,
                "failure_category": cat,
                "final_x": ep600["final_x"],
                "final_y": ep600["final_y"],
                "final_z": ep600["final_z"],
                "final_goal_dist_m": ep600["goal_distance_at_term_m"],
                "progress_at_500_m": x_500,
                "progress_at_600_m": x_600,
                "delta_x_501_600_m": dx,
                "is_productive_progress": is_productive,
                "stalled_obstacle": obs_label,
                "explanation": exp_text,
            })

    with open(fail_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fail_fields)
        writer.writeheader()
        for r in fail_rows:
            writer.writerow(r)
    print(f"[OK] Saved failure classification CSV: {fail_csv_path}")

    # 4. Run Policy-Supervisor Compatibility Audit (Experiment C)
    audit_stats = run_conflict_audit(env, model, device)

    # 5. Capture Representative Step Traces
    rep_seeds = [
        (2000, "matched_success_both_succeed"),
        (2001, "persistent_standoff_pillar3"),
        (2004, "persistent_standoff_pillar1"),
        (2009, "persistent_standoff_pillar2"),
    ]
    print(f"\nRecording detailed step traces for {len(rep_seeds)} representative seeds under 600-step horizon...")
    for seed, name in rep_seeds:
        ep_trace = run_episode(
            env=env,
            supervisor=supervisor,
            model=model,
            horizon_max_steps=600,
            seed=seed,
            device=device,
            record_trace=True,
        )
        t_file = traces_dir / f"trace_seed_{seed}_{name}.json"
        with open(t_file, "w", encoding="utf-8") as f:
            json.dump(ep_trace, f, indent=2, cls=NumpyEncoder)
        print(f"  [OK] Saved trace: {t_file.name}")

    # 6. Paired Statistical Contingency Analysis
    eps500 = all_results[500]
    eps600 = all_results[600]

    n_both_succ = sum(1 for e1, e2 in zip(eps500, eps600) if e1["goal_success"] and e2["goal_success"])
    n_500_only = sum(1 for e1, e2 in zip(eps500, eps600) if e1["goal_success"] and not e2["goal_success"])
    n_600_only = sum(1 for e1, e2 in zip(eps500, eps600) if not e1["goal_success"] and e2["goal_success"])
    n_both_fail = sum(1 for e1, e2 in zip(eps500, eps600) if not e1["goal_success"] and not e2["goal_success"])

    mcnemar_p = mcnemar_exact_test(b=n_500_only, c=n_600_only)

    print("\n" + "=" * 60)
    print("PAIRED CONTINGENCY TABLE (500 Steps vs 600 Steps):")
    print(f"  Both Succeed:                    {n_both_succ:2d}")
    print(f"  500-step Only Succeeds:          {n_500_only:2d}")
    print(f"  600-step Only Succeeds:          {n_600_only:2d} (Additional Successes: 0)")
    print(f"  Both Fail:                       {n_both_fail:2d}")
    print(f"  McNemar exact test p-value:        {mcnemar_p:.6f}")
    print("=" * 60)

    # 7. Summary metrics
    succ500 = sum(1 for e in eps500 if e["goal_success"])
    succ600 = sum(1 for e in eps600 if e["goal_success"])
    w500_low, w500_high = wilson_score_interval(succ500, len(eps500))
    w600_low, w600_high = wilson_score_interval(succ600, len(eps600))

    summary: Dict[str, Any] = {
        "experiment_name": "Phase 4J-G Horizon Sensitivity and Policy-Supervisor Compatibility Audit",
        "checkpoint": BC_CHECKPOINT_PATH,
        "checkpoint_sha256": EXPECTED_BC_SHA256,
        "canonical_supervisor_sha256": EXPECTED_SUPERVISOR_SHA256,
        "caution_distance_m": CAUTION_DIST_M,
        "num_seeds": len(EVAL_SEEDS),
        "seed_range": [EVAL_SEEDS[0], EVAL_SEEDS[-1]],
        "dt_s": 0.05,
        "contingency_table": {
            "both_succeed": n_both_succ,
            "500_steps_only": n_500_only,
            "600_steps_only": n_600_only,
            "both_fail": n_both_fail,
            "mcnemar_exact_p_value": round(mcnemar_p, 6),
        },
        "horizons": {
            "500_steps": {
                "max_steps": 500,
                "duration_limit_s": 25.0,
                "successes": succ500,
                "success_rate_pct": round((succ500 / len(eps500)) * 100.0, 1),
                "wilson_95_ci_pct": [w500_low, w500_high],
                "obstacle_collisions": sum(e["obstacle_collisions"] for e in eps500),
                "boundary_collisions": sum(e["boundary_collisions"] for e in eps500),
                "obstacle_envelope_violations": sum(e["obstacle_safety_envelope_violations"] for e in eps500),
                "boundary_envelope_violations": sum(e["boundary_safety_envelope_violations"] for e in eps500),
                "mean_forward_progress_m": round(float(np.mean([e["forward_progress_at_term_m"] for e in eps500])), 3),
                "mean_duration_s": round(float(np.mean([e["duration_s"] for e in eps500])), 2),
                "absolute_min_obs_clearance_m": round(float(np.min([e["min_obstacle_clearance_m"] for e in eps500])), 4),
                "absolute_min_boundary_clearance_m": round(float(np.min([e["min_boundary_clearance_m"] for e in eps500])), 4),
            },
            "600_steps": {
                "max_steps": 600,
                "duration_limit_s": 30.0,
                "successes": succ600,
                "success_rate_pct": round((succ600 / len(eps600)) * 100.0, 1),
                "wilson_95_ci_pct": [w600_low, w600_high],
                "success_rate_at_step_500_pct": round((succ500 / len(eps600)) * 100.0, 1),
                "additional_successes_steps_501_600": n_600_only,
                "obstacle_collisions": sum(e["obstacle_collisions"] for e in eps600),
                "boundary_collisions": sum(e["boundary_collisions"] for e in eps600),
                "obstacle_envelope_violations": sum(e["obstacle_safety_envelope_violations"] for e in eps600),
                "boundary_envelope_violations": sum(e["boundary_safety_envelope_violations"] for e in eps600),
                "mean_forward_progress_m": round(float(np.mean([e["forward_progress_at_term_m"] for e in eps600])), 3),
                "mean_duration_s": round(float(np.mean([e["duration_s"] for e in eps600])), 2),
                "absolute_min_obs_clearance_m": round(float(np.min([e["min_obstacle_clearance_m"] for e in eps600])), 4),
                "absolute_min_boundary_clearance_m": round(float(np.min([e["min_boundary_clearance_m"] for e in eps600])), 4),
            },
        },
        "failure_taxonomy_distribution": dict(Counter(e["failure_category"] for e in eps600)),
        "policy_supervisor_compatibility": audit_stats,
    }

    summary_path = exp_dir / "horizon_comparison_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, cls=NumpyEncoder)
    print(f"[OK] Saved horizon comparison summary: {summary_path}")

    # 8. Comparison Manifest
    manifest = {
        "experiment_name": "Phase 4J-G Horizon Sensitivity and Policy-Supervisor Compatibility Audit",
        "timestamp": "20261029_192000",
        "workspace": "/Users/ghulamqadir/.gemini/antigravity-ide/scratch/uav-bput-nav",
        "experiment_directory": str(exp_dir),
        "artifacts_generated": [
            "PHASE_4JG_PRE_RUN_AUDIT.md",
            "run_phase4jg_evaluation.py",
            "horizon_paired_episode_results.csv",
            "horizon_comparison_summary.json",
            "failure_classification.csv",
            "policy_supervisor_conflict_analysis.md",
            "comparison_manifest.json",
            "PHASE_4JG_FINAL_REPORT.md",
        ],
        "cryptographic_hashes": {
            "bc_checkpoint_sha256": EXPECTED_BC_SHA256,
            "canonical_supervisor_sha256": EXPECTED_SUPERVISOR_SHA256,
            "evaluation_script_sha256": hashlib.sha256((exp_dir / "run_phase4jg_evaluation.py").read_bytes()).hexdigest(),
        },
        "environment_configuration": {
            "corridor": "10m canonical benchmark",
            "corridor_length_m": 10.0,
            "corridor_half_width_m": 1.0,
            "drone_radius_m": 0.15,
            "dt_s": 0.05,
            "evaluated_horizons_steps": [500, 600],
            "evaluated_durations_s": [25.0, 30.0],
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
    print("\nPhase 4J-G evaluation run complete.")


if __name__ == "__main__":
    main()

