"""reproduce_phase4jc_baseline.py - Controlled Phase 4J-C BC Baseline Reproduction.

Evaluates the canonical BC baseline policy (models/checkpoints/bc_policy_best.pt)
at d_caut = 0.50m across development seeds 2000..2049 (50 episodes) under the
exact canonical 500-step horizon.

Records rich per-episode telemetry, per-obstacle clearances, direction reversals,
stationary durations, and failure mode classifications.
"""

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pybullet as p
import torch

from controllers.safety_supervisor import SafetyState
from core.contracts import ActionLimits, VelocityAction, compute_directional_clearances_m, compute_forward_clearance_m
from core.geometry import Corridor3D, TerminationReason
from experiments.phase4j.isolated_supervisor_opt import IsolatedOptimizedSupervisor
from models.policy_net import NavigationPolicyMLP
from sim.corridor_env import CorridorSimEnv


EVAL_SEEDS = list(range(2000, 2050)) # 50 paired seeds
CAUTION_DIST_M = 0.50
CHECKPOINT_PATH = "models/checkpoints/bc_policy_best.pt"
OBSTACLE_SAFETY_ENVELOPE_M = 0.080
BOUNDARY_SAFETY_ENVELOPE_M = 0.080


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


def run_reproduction_episode(
    env: CorridorSimEnv,
    supervisor: IsolatedOptimizedSupervisor,
    model: NavigationPolicyMLP,
    seed: int,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, Any]:
    obs, info = env.reset(seed=seed)
    supervisor.reset_state()

    ep_start_pos = env._get_drone_position()
    steps = 0
    done = False

    min_obs_overall = float(info["privileged_state"].obstacle_clearance)
    min_wall_overall = float(info["privileged_state"].corridor_boundary_distance)

    # Per-obstacle minimum clearance tracking
    per_obs_min = [10.0] * len(env.obstacles)

    interventions = 0
    emergency_brakes = 0
    retreats = 0
    throttled_steps = 0
    stationary_steps = 0
    direction_reversals = 0
    prev_vy_sign = 0

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

        # Stationary tracking (|v| <= 0.05 m/s)
        speed = float(np.linalg.norm(curr_vel))
        if speed <= 0.05:
            stationary_steps += 1

        # Policy inference (deterministic forward pass)
        with torch.no_grad():
            t_obs = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            act_arr = model(t_obs, bounded=True).squeeze(0).cpu().numpy()
        raw_act = VelocityAction(vx=float(act_arr[0]), vy=float(act_arr[1]), vz=float(act_arr[2]), yaw_rate=float(act_arr[3]))

        # Track lateral reversal near obstacles
        curr_vy_sign = int(np.sign(raw_act.vy))
        if prev_vy_sign != 0 and curr_vy_sign != 0 and curr_vy_sign != prev_vy_sign:
            if obs_c < 0.60:
                direction_reversals += 1
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
        if "Forward brake" in s_reason:
            emergency_brakes += 1
        if "Reversing" in s_reason or "Reverse retreat" in s_reason:
            retreats += 1
        if "Speed throttled" in s_reason:
            throttled_steps += 1

        step_trace.append({
            "step": steps,
            "x": round(float(curr_pos[0]), 3),
            "y": round(float(curr_pos[1]), 3),
            "z": round(float(curr_pos[2]), 3),
            "speed": round(speed, 3),
            "obs_clearance": round(obs_c, 4),
            "wall_clearance": round(wall_c, 4),
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

    # Incorporate final terminal state clearance
    final_priv = current_info["privileged_state"]
    min_obs_overall = min(min_obs_overall, float(final_priv.obstacle_clearance))
    min_wall_overall = min(min_wall_overall, float(final_priv.corridor_boundary_distance))
    for i, o in enumerate(env.obstacles):
        per_obs_min[i] = min(per_obs_min[i], float(o.distance_to_point(env._get_drone_position()) - env.corridor.drone_radius))

    final_pos = env._get_drone_position()
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

    # Classify failure category
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
        elif direction_reversals > 10:
            fail_category = "PROGRESS_LATERAL_OSCILLATION"
        elif throttled_steps > 250:
            fail_category = "TIMEOUT_EXCESSIVE_THROTTLING"
        else:
            fail_category = "TIMEOUT_SLOW_TRANSIT"

    return {
        "seed": seed,
        "controller_id": "Arm_B_BC_Baseline",
        "checkpoint": CHECKPOINT_PATH,
        "caution_distance_m": CAUTION_DIST_M,
        "steps": steps,
        "duration_s": round(steps * env.dt, 2),
        "status": term_reason,
        "is_success": is_success,
        "goal_entered": goal_entered,
        "is_timeout": is_timeout,
        "is_obstacle_collision": is_obs_collision,
        "is_boundary_collision": is_wall_collision,
        "is_obstacle_envelope_violation": is_obs_envelope_violation,
        "is_boundary_envelope_violation": is_wall_envelope_violation,
        "forward_progress_m": round(float(final_pos[0] - ep_start_pos[0]), 3),
        "final_position": [round(float(x), 3) for x in final_pos],
        "final_goal_distance_m": round(final_goal_dist, 3),
        "min_obstacle_clearance_m": round(min_obs_overall, 4),
        "min_boundary_clearance_m": round(min_wall_overall, 4),
        "per_obstacle_min_clearance_m": [round(x, 4) for x in per_obs_min],
        "terminal_speed_mps": round(final_vel, 3),
        "terminal_altitude_m": round(float(final_pos[2]), 3),
        "interventions": interventions,
        "intervention_rate": round(interventions / max(1, steps), 3),
        "emergency_brakes": emergency_brakes,
        "retreats": retreats,
        "throttled_steps": throttled_steps,
        "throttled_time_pct": round((throttled_steps / max(1, steps)) * 100.0, 1),
        "stationary_steps": stationary_steps,
        "stationary_time_pct": round((stationary_steps / max(1, steps)) * 100.0, 1),
        "direction_reversals": direction_reversals,
        "fail_category": fail_category,
        "trace": step_trace,
    }


def execute_reproduction():
    out_dir = Path("experiments/phase4j/phase4jc_results")
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cpu")
    model = NavigationPolicyMLP(obs_dim=23, act_dim=4).to(device)
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    env = CorridorSimEnv(scenario="10m", gui=False, seed=42)
    supervisor = IsolatedOptimizedSupervisor(corridor=env.corridor, caution_distance_m=CAUTION_DIST_M)

    print("=" * 90)
    print(">>> EXECUTING PHASE 4J-C BC BASELINE CONTROLLED REPRODUCTION <<<")
    print(f"Controller: Arm_B_BC_Baseline ({CHECKPOINT_PATH})")
    print(f"Caution distance: d_caut = {CAUTION_DIST_M} m")
    print(f"Seeds: {len(EVAL_SEEDS)} seeds ({EVAL_SEEDS[0]}..{EVAL_SEEDS[-1]})")
    print(f"Horizon: {env.max_steps} steps (25.0s)")
    print("=" * 90)

    episodes = []
    for s_idx, seed in enumerate(EVAL_SEEDS, 1):
        ep_res = run_reproduction_episode(env, supervisor, model, seed, device)
        episodes.append(ep_res)
        print(f"[{s_idx:02d}/50] Seed {seed:4d} | Status: {ep_res['status']:<10} | Succ: {str(ep_res['is_success']):<5} | Fwd: {ep_res['forward_progress_m']:4.2f}m | MinObs: {ep_res['min_obstacle_clearance_m']:.3f}m | Steps: {ep_res['steps']:3d} | Cat: {ep_res['fail_category']}")

    env.close()

    # Aggregate metrics
    succ_cnt = sum(1 for e in episodes if e["is_success"])
    obs_col_cnt = sum(1 for e in episodes if e["is_obstacle_collision"])
    wall_col_cnt = sum(1 for e in episodes if e["is_boundary_collision"])
    obs_viols = sum(1 for e in episodes if e["is_obstacle_envelope_violation"])
    wall_viols = sum(1 for e in episodes if e["is_boundary_envelope_violation"])
    timeouts = sum(1 for e in episodes if e["is_timeout"])

    fwd_prog = [e["forward_progress_m"] for e in episodes]
    durations = [e["duration_s"] for e in episodes]
    obs_clears = [e["min_obstacle_clearance_m"] for e in episodes]
    wall_clears = [e["min_boundary_clearance_m"] for e in episodes]
    throttled = [e["throttled_time_pct"] for e in episodes]
    intervs = [e["intervention_rate"] for e in episodes]

    succ_low, succ_high = wilson_score_interval(succ_cnt, len(EVAL_SEEDS))

    # Category breakdown
    cat_counts = {}
    for e in episodes:
        c = e["fail_category"]
        cat_counts[c] = cat_counts.get(c, 0) + 1

    summary = {
        "controller_id": "Arm_B_BC_Baseline",
        "checkpoint": CHECKPOINT_PATH,
        "caution_distance_m": CAUTION_DIST_M,
        "total_episodes": len(EVAL_SEEDS),
        "success_count": succ_cnt,
        "success_rate_pct": round((succ_cnt / len(EVAL_SEEDS)) * 100.0, 1),
        "success_ci95": [succ_low, succ_high],
        "obstacle_collision_count": obs_col_cnt,
        "obstacle_collision_pct": round((obs_col_cnt / len(EVAL_SEEDS)) * 100.0, 1),
        "boundary_collision_count": wall_col_cnt,
        "boundary_collision_pct": round((wall_col_cnt / len(EVAL_SEEDS)) * 100.0, 1),
        "obstacle_envelope_violation_count": obs_viols,
        "obstacle_envelope_violation_pct": round((obs_viols / len(EVAL_SEEDS)) * 100.0, 1),
        "boundary_envelope_violation_count": wall_viols,
        "timeout_count": timeouts,
        "timeout_pct": round((timeouts / len(EVAL_SEEDS)) * 100.0, 1),
        "mean_forward_progress_m": round(float(np.mean(fwd_prog)), 3),
        "median_forward_progress_m": round(float(np.median(fwd_prog)), 3),
        "mean_duration_s": round(float(np.mean(durations)), 2),
        "median_duration_s": round(float(np.median(durations)), 2),
        "mean_min_obstacle_clearance_m": round(float(np.mean(obs_clears)), 4),
        "global_min_obstacle_clearance_m": round(float(np.min(obs_clears)), 4),
        "mean_min_boundary_clearance_m": round(float(np.mean(wall_clears)), 4),
        "global_min_boundary_clearance_m": round(float(np.min(wall_clears)), 4),
        "mean_throttled_time_pct": round(float(np.mean(throttled)), 1),
        "mean_intervention_rate": round(float(np.mean(intervs)), 3),
        "category_breakdown": cat_counts,
    }

    # Save summary and episodes
    ep_file = out_dir / "phase4jc_baseline_episodes.json"
    with open(ep_file, "w") as f:
        json.dump(episodes, f, indent=2, cls=NumpyEncoder)

    sum_file = out_dir / "phase4jc_baseline_summary.json"
    with open(sum_file, "w") as f:
        json.dump(summary, f, indent=2, cls=NumpyEncoder)

    print("\n" + "=" * 90)
    print(">>> PHASE 4J-C REPRODUCTION SUMMARY <<<")
    print(f"Success Count: {succ_cnt}/50 ({summary['success_rate_pct']}% [{succ_low}-{succ_high}%])")
    print(f"Obstacle Collisions: {obs_col_cnt} | Boundary Collisions: {wall_col_cnt}")
    print(f"Obstacle Envelope Violations (<8cm): {obs_viols}")
    print(f"Global Min Obstacle Clearance: {summary['global_min_obstacle_clearance_m']:.4f} m")
    print(f"Global Min Boundary Clearance: {summary['global_min_boundary_clearance_m']:.4f} m")
    print(f"Mean Forward Progress: {summary['mean_forward_progress_m']:.2f} m")
    print(f"Mean Duration: {summary['mean_duration_s']:.2f} s")
    print(f"Mean Throttled Fraction: {summary['mean_throttled_time_pct']:.1f}%")
    print(f"Failure Categories: {cat_counts}")
    print("=" * 90)

    return summary, episodes


if __name__ == "__main__":
    execute_reproduction()

