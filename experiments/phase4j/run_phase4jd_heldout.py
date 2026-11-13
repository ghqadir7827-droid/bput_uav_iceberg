"""run_phase4jd_heldout.py - Phase 4J-D Sealed Held-Out Generalization Evaluation.

Evaluates the frozen Behavior-Cloning baseline policy (models/checkpoints/bc_policy_best.pt)
at d_caut = 0.50m across the 100 reserved held-out seeds 90000..90099.
Executes single-pass evaluation under the canonical 500-step horizon (25.0s at dt=0.05s).
"""

import hashlib
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


HELDOUT_SEEDS = list(range(90000, 90100))  # Exactly 100 seeds: 90000..90099 inclusive
CAUTION_DIST_M = 0.50
CHECKPOINT_PATH = "models/checkpoints/bc_policy_best.pt"
EXPECTED_CHECKPOINT_SHA256 = "93c4d7d1d8476b8d288a24fca1862d87133696f78b9603aa622bcc367afad44f"
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


def verify_checkpoint_integrity():
    p = Path(CHECKPOINT_PATH)
    if not p.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")
    h = hashlib.sha256(p.read_bytes()).hexdigest()
    if h != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError(f"Checkpoint hash mismatch! Expected {EXPECTED_CHECKPOINT_SHA256}, got {h}")
    print(f"[PRE-CHECK] Checkpoint verified: {h}")


def run_single_episode(
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
        raw_act = VelocityAction(
            vx=float(act_arr[0]),
            vy=float(act_arr[1]),
            vz=float(act_arr[2]),
            yaw_rate=float(act_arr[3]),
        )

        # Track lateral reversals near obstacles
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

        norm_applied = safe_act.to_normalized(env.action_limits)
        obs, rew, term, trunc, current_info = env.step(norm_applied)
        steps += 1
        done = term or trunc

    # Terminal state clearances
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

    # Count pillars cleared: drone X past pillar center X + radius
    pillars_cleared = sum(1 for o in env.obstacles if final_pos[0] > (o.center[0] + o.radius))

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
        "pillars_cleared": pillars_cleared,
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
    }


def execute_heldout_evaluation():
    verify_checkpoint_integrity()

    out_dir = Path("experiments/phase4j/phase4jd_results")
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cpu")
    model = NavigationPolicyMLP(obs_dim=23, act_dim=4).to(device)
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    env = CorridorSimEnv(scenario="10m", gui=False, seed=42)
    supervisor = IsolatedOptimizedSupervisor(corridor=env.corridor, caution_distance_m=CAUTION_DIST_M)

    print("=" * 95)
    print(">>> EXECUTING PHASE 4J-D SEALED HELD-OUT GENERALIZATION EVALUATION <<<")
    print(f"Controller: Arm_B_BC_Baseline ({CHECKPOINT_PATH})")
    print(f"Caution distance: d_caut = {CAUTION_DIST_M} m")
    print(f"Seeds: {len(HELDOUT_SEEDS)} seeds ({HELDOUT_SEEDS[0]}..{HELDOUT_SEEDS[-1]})")
    print(f"Horizon: {env.max_steps} steps (25.0s)")
    print("=" * 95)

    episodes = []
    for s_idx, seed in enumerate(HELDOUT_SEEDS, 1):
        ep_res = run_single_episode(env, supervisor, model, seed, device)
        episodes.append(ep_res)
        succ_tag = "SUCCESS" if ep_res["is_success"] else "FAIL"
        print(
            f"[{s_idx:03d}/100] Seed {seed:5d} | {succ_tag:<7} | Status: {ep_res['status']:<10} | "
            f"Fwd: {ep_res['forward_progress_m']:4.2f}m | MinObs: {ep_res['min_obstacle_clearance_m']:.3f}m | "
            f"Steps: {ep_res['steps']:3d} | Cat: {ep_res['fail_category']}"
        )

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
    stationary = [e["stationary_time_pct"] for e in episodes]
    reversals = [e["direction_reversals"] for e in episodes]

    succ_low, succ_high = wilson_score_interval(succ_cnt, len(HELDOUT_SEEDS))

    # Category breakdown
    cat_counts = {}
    for e in episodes:
        c = e["fail_category"]
        cat_counts[c] = cat_counts.get(c, 0) + 1

    summary = {
        "evaluation_milestone": "Phase 4J-D Sealed Held-Out Generalization Evaluation",
        "controller_id": "Arm_B_BC_Baseline",
        "checkpoint": CHECKPOINT_PATH,
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "caution_distance_m": CAUTION_DIST_M,
        "critical_distance_m": supervisor.config.critical_obstacle_dist_m,
        "reverse_distance_m": supervisor.config.emergency_reverse_dist_m,
        "boundary_buffer_m": supervisor.config.boundary_buffer_m,
        "horizon_steps": env.max_steps,
        "timestep_s": env.dt,
        "seed_range": [HELDOUT_SEEDS[0], HELDOUT_SEEDS[-1]],
        "total_episodes": len(HELDOUT_SEEDS),
        "success_count": succ_cnt,
        "success_rate_pct": round((succ_cnt / len(HELDOUT_SEEDS)) * 100.0, 1),
        "success_ci95": [succ_low, succ_high],
        "obstacle_collision_count": obs_col_cnt,
        "obstacle_collision_pct": round((obs_col_cnt / len(HELDOUT_SEEDS)) * 100.0, 1),
        "boundary_collision_count": wall_col_cnt,
        "boundary_collision_pct": round((wall_col_cnt / len(HELDOUT_SEEDS)) * 100.0, 1),
        "obstacle_envelope_violation_count": obs_viols,
        "obstacle_envelope_violation_pct": round((obs_viols / len(HELDOUT_SEEDS)) * 100.0, 1),
        "boundary_envelope_violation_count": wall_viols,
        "boundary_envelope_violation_pct": round((wall_viols / len(HELDOUT_SEEDS)) * 100.0, 1),
        "timeout_count": timeouts,
        "timeout_pct": round((timeouts / len(HELDOUT_SEEDS)) * 100.0, 1),
        "forward_progress_m": {
            "mean": round(float(np.mean(fwd_prog)), 3),
            "median": round(float(np.median(fwd_prog)), 3),
            "std": round(float(np.std(fwd_prog)), 3),
            "min": round(float(np.min(fwd_prog)), 3),
            "max": round(float(np.max(fwd_prog)), 3),
        },
        "episode_duration_s": {
            "mean": round(float(np.mean(durations)), 2),
            "median": round(float(np.median(durations)), 2),
            "min": round(float(np.min(durations)), 2),
            "max": round(float(np.max(durations)), 2),
        },
        "obstacle_clearance_m": {
            "mean": round(float(np.mean(obs_clears)), 4),
            "median": round(float(np.median(obs_clears)), 4),
            "global_min": round(float(np.min(obs_clears)), 4),
        },
        "boundary_clearance_m": {
            "mean": round(float(np.mean(wall_clears)), 4),
            "median": round(float(np.median(wall_clears)), 4),
            "global_min": round(float(np.min(wall_clears)), 4),
        },
        "supervisor_metrics": {
            "mean_throttled_time_pct": round(float(np.mean(throttled)), 1),
            "mean_intervention_rate": round(float(np.mean(intervs)), 3),
            "mean_stationary_time_pct": round(float(np.mean(stationary)), 1),
            "mean_direction_reversals": round(float(np.mean(reversals)), 2),
        },
        "category_breakdown": cat_counts,
        "development_reference_comparison": {
            "phase4jc_dev_seeds": "2000..2049 (50 seeds)",
            "dev_success_rate_pct": 48.0,
            "dev_success_ci95": [34.8, 61.5],
            "dev_obstacle_collisions": 0,
            "dev_boundary_collisions": 0,
            "dev_envelope_violations": 0,
            "dev_global_min_obs_clearance_m": 0.1073,
            "dev_global_min_boundary_clearance_m": 0.1873,
            "dev_mean_forward_progress_m": 6.53,
            "dev_mean_duration_s": 23.0,
            "dev_mean_throttled_time_pct": 27.6,
        },
    }

    # Save summary and episodes
    ep_file = out_dir / "heldout_episodes.json"
    with open(ep_file, "w") as f:
        json.dump(episodes, f, indent=2, cls=NumpyEncoder)

    sum_file = out_dir / "heldout_summary.json"
    with open(sum_file, "w") as f:
        json.dump(summary, f, indent=2, cls=NumpyEncoder)

    print("\n" + "=" * 95)
    print(">>> PHASE 4J-D SEALED HELD-OUT EVALUATION SUMMARY <<<")
    print(f"Total Episodes: {len(HELDOUT_SEEDS)}")
    print(f"Success Count: {succ_cnt}/100 ({summary['success_rate_pct']}% [95% CI: {succ_low}% - {succ_high}%])")
    print(f"Obstacle Collisions: {obs_col_cnt} ({summary['obstacle_collision_pct']}%)")
    print(f"Boundary Collisions: {wall_col_cnt} ({summary['boundary_collision_pct']}%)")
    print(f"Obstacle Envelope Violations (<8cm): {obs_viols} ({summary['obstacle_envelope_violation_pct']}%)")
    print(f"Boundary Envelope Violations (<8cm): {wall_viols} ({summary['boundary_envelope_violation_pct']}%)")
    print(f"Timeouts: {timeouts} ({summary['timeout_pct']}%)")
    print(f"Global Min Obstacle Clearance: {summary['obstacle_clearance_m']['global_min']:.4f} m")
    print(f"Global Min Boundary Clearance: {summary['boundary_clearance_m']['global_min']:.4f} m")
    print(f"Mean Forward Progress: {summary['forward_progress_m']['mean']:.2f} m (Median: {summary['forward_progress_m']['median']:.2f} m)")
    print(f"Mean Duration: {summary['episode_duration_s']['mean']:.2f} s")
    print(f"Mean Throttled Fraction: {summary['supervisor_metrics']['mean_throttled_time_pct']:.1f}%")
    print(f"Mean Intervention Rate: {summary['supervisor_metrics']['mean_intervention_rate']:.3f}")
    print(f"Failure Category Breakdown: {cat_counts}")
    print("=" * 95)

    return summary, episodes


if __name__ == "__main__":
    execute_heldout_evaluation()

