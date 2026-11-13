"""run_phase4k_evaluation.py - Phase 4K Paired Development Evaluation.

Evaluates:
  1. Frozen BC Baseline (models/checkpoints/bc_policy_best.pt)
  2. Isolated Curriculum Pilot Candidate (bc_curriculum_candidate.pt)

Across development seeds 2000..2049 (50 paired seeds) under canonical 500-step horizon
with IsolatedOptimizedSupervisor at d_caut = 0.50m.
"""

import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np
import torch

from controllers.safety_supervisor import SafetyState
from core.contracts import ActionLimits, VelocityAction, compute_forward_clearance_m
from core.geometry import Corridor3D
from experiments.phase4j.isolated_supervisor_opt import IsolatedOptimizedSupervisor
from models.policy_net import NavigationPolicyMLP
from sim.corridor_env import CorridorSimEnv


DEV_SEEDS = list(range(2000, 2050))
CAUTION_DIST_M = 0.50
BASELINE_CKPT = "models/checkpoints/bc_policy_best.pt"
CANDIDATE_CKPT = "experiments/phase4k/phase4k_symmetry_curriculum_20261029_210200/bc_curriculum_candidate.pt"
OUTPUT_DIR = Path("experiments/phase4k/phase4k_symmetry_curriculum_20261029_210200")
TRACES_DIR = OUTPUT_DIR / "traces"

OBSTACLE_SAFETY_ENVELOPE_M = 0.080
BOUNDARY_SAFETY_ENVELOPE_M = 0.080


def wilson_score_interval(successes: int, trials: int) -> Tuple[float, float]:
    if trials == 0:
        return 0.0, 0.0
    z = 1.959964
    p_hat = successes / trials
    denominator = 1 + z**2 / trials
    center = (p_hat + z**2 / (2 * trials)) / denominator
    spread = z * math.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * trials)) / trials) / denominator
    return round(max(0.0, center - spread) * 100.0, 1), round(min(1.0, center + spread) * 100.0, 1)


def run_episode(
    env: CorridorSimEnv,
    supervisor: IsolatedOptimizedSupervisor,
    model: NavigationPolicyMLP,
    seed: int,
    controller_name: str,
    device: torch.device = torch.device("cpu"),
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    obs, info = env.reset(seed=seed)
    supervisor.reset_state()

    start_pos = env._get_drone_position()
    steps = 0
    done = False

    min_obs_overall = float(info["privileged_state"].obstacle_clearance)
    min_wall_overall = float(info["privileged_state"].corridor_boundary_distance)

    interventions = 0
    throttled_steps = 0
    reversals = 0
    prev_vy_sign = 0

    trace = []
    current_info = info

    while not done and steps < env.max_steps:
        priv = current_info["privileged_state"]
        curr_pos = env._get_drone_position()
        curr_vel = priv.global_velocity
        speed = float(np.linalg.norm(curr_vel))
        obs_c = float(priv.obstacle_clearance)
        wall_c = float(priv.corridor_boundary_distance)
        front_dist = compute_forward_clearance_m(obs, max_range_m=env.sensor_config.max_range_m)

        min_obs_overall = min(min_obs_overall, obs_c)
        min_wall_overall = min(min_wall_overall, wall_c)

        with torch.no_grad():
            t_obs = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            act_arr = model(t_obs, bounded=True).squeeze(0).cpu().numpy()
        raw_act = VelocityAction(
            vx=float(act_arr[0]),
            vy=float(act_arr[1]),
            vz=float(act_arr[2]),
            yaw_rate=float(act_arr[3]),
        )

        curr_vy_sign = int(np.sign(raw_act.vy))
        if prev_vy_sign != 0 and curr_vy_sign != 0 and curr_vy_sign != prev_vy_sign:
            if obs_c < 0.60:
                reversals += 1
        prev_vy_sign = curr_vy_sign

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

        if s_state != SafetyState.NOMINAL:
            interventions += 1
        if "Speed throttled" in s_reason:
            throttled_steps += 1

        trace.append({
            "step": steps,
            "x": round(float(curr_pos[0]), 3),
            "y": round(float(curr_pos[1]), 3),
            "z": round(float(curr_pos[2]), 3),
            "speed": round(speed, 3),
            "obs_c": round(obs_c, 3),
            "wall_c": round(wall_c, 3),
            "raw_vx": round(raw_act.vx, 3),
            "raw_vy": round(raw_act.vy, 3),
            "safe_vx": round(safe_act.vx, 3),
            "safe_vy": round(safe_act.vy, 3),
            "state": s_state.value,
            "reason": s_reason,
        })

        norm_applied = safe_act.to_normalized(env.action_limits)
        obs, rew, term, trunc, current_info = env.step(norm_applied)
        steps += 1
        done = term or trunc

    final_priv = current_info["privileged_state"]
    min_obs_overall = min(min_obs_overall, float(final_priv.obstacle_clearance))
    min_wall_overall = min(min_wall_overall, float(final_priv.corridor_boundary_distance))
    final_pos = env._get_drone_position()
    final_vel = float(np.linalg.norm(final_priv.global_velocity))
    final_goal_dist = float(final_priv.goal_distance)
    term_reason = current_info.get("termination_reason", "unknown")

    is_success = (term_reason == "success") and (final_vel <= 0.35)
    goal_entered = (final_goal_dist <= env.corridor.goal_radius)
    is_obs_collision = (term_reason == "crash_obstacle") or (min_obs_overall <= 0.0)
    is_wall_collision = (term_reason == "crash_boundary") or (min_wall_overall <= 0.0)

    is_obs_envelope_violation = (min_obs_overall < OBSTACLE_SAFETY_ENVELOPE_M)
    is_wall_envelope_violation = (min_wall_overall < BOUNDARY_SAFETY_ENVELOPE_M)

    progress = float(final_pos[0] - start_pos[0])
    throttled_fraction_pct = round(100.0 * throttled_steps / max(1, steps), 1)

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
        elif reversals > 10:
            fail_category = "PROGRESS_LATERAL_OSCILLATION"
        elif throttled_steps > 250:
            fail_category = "TIMEOUT_EXCESSIVE_THROTTLING"
        else:
            fail_category = "TIMEOUT_SLOW_TRANSIT"

    summary = {
        "seed": seed,
        "controller": controller_name,
        "success": int(is_success),
        "steps": steps,
        "duration_s": round(steps * env.dt, 2),
        "progress_m": round(progress, 3),
        "min_obs_clearance_m": round(min_obs_overall, 4),
        "min_wall_clearance_m": round(min_wall_overall, 4),
        "obs_envelope_violations": int(is_obs_envelope_violation),
        "wall_envelope_violations": int(is_wall_envelope_violation),
        "interventions": interventions,
        "throttled_fraction_pct": throttled_fraction_pct,
        "reversals": reversals,
        "failure_category": fail_category,
    }

    return summary, trace


def main():
    print("Initializing environment...")
    env = CorridorSimEnv(gui=False)
    supervisor = IsolatedOptimizedSupervisor(
        corridor=env.corridor,
        caution_distance_m=CAUTION_DIST_M,
        limits=env.action_limits,
    )

    device = torch.device("cpu")

    print(f"Loading baseline from {BASELINE_CKPT}...")
    baseline_model = NavigationPolicyMLP()
    base_ckpt = torch.load(BASELINE_CKPT, map_location=device)
    baseline_model.load_state_dict(base_ckpt["model_state_dict"] if "model_state_dict" in base_ckpt else base_ckpt)
    baseline_model.eval()

    print(f"Loading candidate from {CANDIDATE_CKPT}...")
    candidate_model = NavigationPolicyMLP()
    cand_ckpt = torch.load(CANDIDATE_CKPT, map_location=device)
    candidate_model.load_state_dict(cand_ckpt["model_state_dict"] if "model_state_dict" in cand_ckpt else cand_ckpt)
    candidate_model.eval()

    results: List[Dict[str, Any]] = []
    TRACES_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Starting paired evaluation across {len(DEV_SEEDS)} seeds (2000..2049)...")
    for i, seed in enumerate(DEV_SEEDS):
        # Baseline
        base_res, base_trace = run_episode(env, supervisor, baseline_model, seed, "BC_Baseline", device)
        results.append(base_res)

        # Candidate
        cand_res, cand_trace = run_episode(env, supervisor, candidate_model, seed, "BC_Curriculum_Candidate", device)
        results.append(cand_res)

        # Save sample traces for seed 2000 and 2001 (or specific cases)
        if seed in [2000, 2001, 2002, 2005, 2010]:
            with open(TRACES_DIR / f"trace_seed{seed}_baseline.json", "w") as f:
                json.dump(base_trace, f, indent=2)
            with open(TRACES_DIR / f"trace_seed{seed}_candidate.json", "w") as f:
                json.dump(cand_trace, f, indent=2)

        if (i + 1) % 10 == 0:
            print(f"Completed {i + 1}/{len(DEV_SEEDS)} seeds...")

    env.close()

    # Save CSV
    csv_path = OUTPUT_DIR / "development_evaluation.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"Saved evaluation results to {csv_path}")

    # Compute comparative metrics
    base_evals = [r for r in results if r["controller"] == "BC_Baseline"]
    cand_evals = [r for r in results if r["controller"] == "BC_Curriculum_Candidate"]

    def summarize(evals: List[Dict[str, Any]], name: str):
        n = len(evals)
        succ = sum(r["success"] for r in evals)
        w_low, w_high = wilson_score_interval(succ, n)
        mean_prog = np.mean([r["progress_m"] for r in evals])
        mean_dur = np.mean([r["duration_s"] for r in evals])
        mean_throttled = np.mean([r["throttled_fraction_pct"] for r in evals])
        min_obs = min(r["min_obs_clearance_m"] for r in evals)
        min_wall = min(r["min_wall_clearance_m"] for r in evals)
        obs_viols = sum(r["obs_envelope_violations"] for r in evals)
        wall_viols = sum(r["wall_envelope_violations"] for r in evals)

        fail_counts = {}
        for r in evals:
            cat = r["failure_category"]
            fail_counts[cat] = fail_counts.get(cat, 0) + 1

        print(f"\n=================== {name} (N={n}) ===================")
        print(f"Successes: {succ}/{n} ({succ/n*100:.1f}%, 95% CI: [{w_low}%, {w_high}%])")
        print(f"Mean Progress: {mean_prog:.3f} m")
        print(f"Mean Duration: {mean_dur:.2f} s")
        print(f"Mean Throttled Fraction: {mean_throttled:.1f}%")
        print(f"Min Obstacle Clearance: {min_obs:.4f} m (Violations: {obs_viols})")
        print(f"Min Wall Clearance: {min_wall:.4f} m (Violations: {wall_viols})")
        print("Failure Categorization:")
        for cat, cnt in sorted(fail_counts.items(), key=lambda x: -x[1]):
            print(f"  - {cat}: {cnt} ({cnt/n*100:.1f}%)")

    summarize(base_evals, "Frozen BC Baseline")
    summarize(cand_evals, "BC Curriculum Candidate")

    # Paired McNemar / differences
    paired_wins = 0
    paired_losses = 0
    paired_ties = 0
    for b, c in zip(base_evals, cand_evals):
        if c["success"] > b["success"]:
            paired_wins += 1
        elif c["success"] < b["success"]:
            paired_losses += 1
        else:
            paired_ties += 1

    print("\n=================== Paired Comparison ===================")
    print(f"Candidate Wins (c=1, b=0): {paired_wins}")
    print(f"Candidate Losses (c=0, b=1): {paired_losses}")
    print(f"Ties: {paired_ties}")


if __name__ == "__main__":
    main()
