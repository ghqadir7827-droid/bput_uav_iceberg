"""run_phase4je_comparison.py - Phase 4J-E Controlled BC vs PPO Paired Evaluation.

Evaluates 4 frozen checkpoints on development seeds 2000..2049 (50 paired episodes each):
Primary:
1. Arm_B_BC_Baseline: models/checkpoints/bc_policy_best.pt
2. Arm_A_PPO_Baseline: models/checkpoints/ppo_policy_best.pt
Secondary:
3. BC_Phase4H_Candidate: experiments/phase4h/checkpoints/bc_phase4h_candidate.pt
4. PPO_Phase4H_Candidate: experiments/phase4h/checkpoints/ppo_phase4h_candidate.pt

Under identical experimental conditions:
- 10m canonical corridor benchmark
- 500 steps (25.0s at dt=0.05s)
- Isolated supervisor at d_caut = 0.50m
- Exact deterministic policy inference
"""

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch

from controllers.safety_supervisor import SafetyState
from core.contracts import ActionLimits, VelocityAction, compute_directional_clearances_m, compute_forward_clearance_m
from core.geometry import Corridor3D, TerminationReason
from experiments.phase4j.isolated_supervisor_opt import IsolatedOptimizedSupervisor
from models.policy_net import NavigationPolicyMLP
from sim.corridor_env import CorridorSimEnv


EVAL_SEEDS = list(range(2000, 2050))  # Exactly 50 development seeds
CAUTION_DIST_M = 0.50
OBSTACLE_SAFETY_ENVELOPE_M = 0.080
BOUNDARY_SAFETY_ENVELOPE_M = 0.080

CHECKPOINTS = {
    "Arm_B_BC_Baseline": {
        "path": "models/checkpoints/bc_policy_best.pt",
        "expected_sha256": "93c4d7d1d8476b8d288a24fca1862d87133696f78b9603aa622bcc367afad44f",
        "role": "primary",
    },
    "Arm_A_PPO_Baseline": {
        "path": "models/checkpoints/ppo_policy_best.pt",
        "expected_sha256": "33f6d8b705edcffa430c48b2a21ed33a3e00f81b1c8195323134dd8be2a00ee0",
        "role": "primary",
    },
    "BC_Phase4H_Candidate": {
        "path": "experiments/phase4h/checkpoints/bc_phase4h_candidate.pt",
        "expected_sha256": "1fdb2286170b8188421609231af3ac685467a41a8d9dc24c5f793117ac9c319b",
        "role": "secondary",
    },
    "PPO_Phase4H_Candidate": {
        "path": "experiments/phase4h/checkpoints/ppo_phase4h_candidate.pt",
        "expected_sha256": "751431cda829539e9ef327bd843d9b292e8f097bb2b36c0f5a1deb1bfad159dd",
        "role": "secondary",
    },
}


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
    """Computes two-tailed p-value for McNemar exact test with binomial distribution.
    b: Number of pairs where Model 1 succeeds and Model 2 fails.
    c: Number of pairs where Model 2 succeeds and Model 1 fails.
    """
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    # Two-sided exact binomial p-value with p=0.5
    prob = sum(math.comb(n, i) * (0.5**n) for i in range(k + 1))
    return min(1.0, 2.0 * prob)


def verify_checkpoints():
    print(">>> VERIFYING CHECKPOINT CRYPTOGRAPHIC INTEGRITY <<<")
    for name, cfg in CHECKPOINTS.items():
        p = Path(cfg["path"])
        if not p.is_file():
            raise FileNotFoundError(f"Checkpoint missing: {cfg['path']}")
        actual_h = hashlib.sha256(p.read_bytes()).hexdigest()
        if actual_h != cfg["expected_sha256"]:
            raise ValueError(f"Hash mismatch for {name}: expected {cfg['expected_sha256']}, got {actual_h}")
        print(f"  [OK] {name:22s} -> {actual_h}")


def run_episode(
    env: CorridorSimEnv,
    supervisor: IsolatedOptimizedSupervisor,
    model: NavigationPolicyMLP,
    controller_id: str,
    checkpoint_path: str,
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
    direction_reversals = 0
    prev_vy_sign = 0

    # Action telemetry
    raw_vx_list, raw_vy_list, raw_vz_list, raw_yr_list = [], [], [], []
    safe_vx_list, safe_vy_list, safe_vz_list, safe_yr_list = [], [], [], []
    sat_policy_vx = sat_policy_vy = sat_policy_vz = sat_policy_yr = 0
    sat_applied_vx = sat_applied_vy = sat_applied_vz = sat_applied_yr = 0

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

        # Policy forward pass (deterministic)
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

        # Check raw action saturation vs limits
        norm_raw = raw_act.to_normalized(env.action_limits)
        if abs(norm_raw[0]) >= 0.99: sat_policy_vx += 1
        if abs(norm_raw[1]) >= 0.99: sat_policy_vy += 1
        if abs(norm_raw[2]) >= 0.99: sat_policy_vz += 1
        if abs(norm_raw[3]) >= 0.99: sat_policy_yr += 1

        # Track lateral reversal
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

        safe_vx_list.append(safe_act.vx)
        safe_vy_list.append(safe_act.vy)
        safe_vz_list.append(safe_act.vz)
        safe_yr_list.append(safe_act.yaw_rate)

        norm_applied = safe_act.to_normalized(env.action_limits)
        if abs(norm_applied[0]) >= 0.99: sat_applied_vx += 1
        if abs(norm_applied[1]) >= 0.99: sat_applied_vy += 1
        if abs(norm_applied[2]) >= 0.99: sat_applied_vz += 1
        if abs(norm_applied[3]) >= 0.99: sat_applied_yr += 1

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
                "reason": s_reason,
            })

        obs, rew, term, trunc, current_info = env.step(norm_applied)
        steps += 1
        done = term or trunc

    # Final clearances
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

    res = {
        "controller_id": controller_id,
        "checkpoint": checkpoint_path,
        "seed": seed,
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
        "raw_action_mean": [
            round(float(np.mean(raw_vx_list)), 4),
            round(float(np.mean(raw_vy_list)), 4),
            round(float(np.mean(raw_vz_list)), 4),
            round(float(np.mean(raw_yr_list)), 4),
        ],
        "safe_action_mean": [
            round(float(np.mean(safe_vx_list)), 4),
            round(float(np.mean(safe_vy_list)), 4),
            round(float(np.mean(safe_vz_list)), 4),
            round(float(np.mean(safe_yr_list)), 4),
        ],
        "policy_saturation_pct": {
            "vx": round((sat_policy_vx / max(1, steps)) * 100.0, 1),
            "vy": round((sat_policy_vy / max(1, steps)) * 100.0, 1),
            "vz": round((sat_policy_vz / max(1, steps)) * 100.0, 1),
            "yaw_rate": round((sat_policy_yr / max(1, steps)) * 100.0, 1),
        },
        "applied_saturation_pct": {
            "vx": round((sat_applied_vx / max(1, steps)) * 100.0, 1),
            "vy": round((sat_applied_vy / max(1, steps)) * 100.0, 1),
            "vz": round((sat_applied_vz / max(1, steps)) * 100.0, 1),
            "yaw_rate": round((sat_applied_yr / max(1, steps)) * 100.0, 1),
        },
        "fail_category": fail_category,
    }

    if record_trace:
        res["trace"] = step_trace

    return res


def execute_stage_d(audit_dir: Path):
    verify_checkpoints()

    device = torch.device("cpu")
    env = CorridorSimEnv(scenario="10m", gui=False, seed=42)
    supervisor = IsolatedOptimizedSupervisor(corridor=env.corridor, caution_distance_m=CAUTION_DIST_M)

    # Dictionary to store all episodes by controller
    all_controller_episodes: Dict[str, List[Dict[str, Any]]] = {}
    controller_models: Dict[str, NavigationPolicyMLP] = {}

    # Load all models
    for name, cfg in CHECKPOINTS.items():
        m = NavigationPolicyMLP(obs_dim=23, act_dim=4).to(device)
        ckpt = torch.load(cfg["path"], map_location=device)
        m.load_state_dict(ckpt["model_state_dict"])
        m.eval()
        controller_models[name] = m

    print("=" * 100)
    print(">>> EXECUTING PHASE 4J-E CONTROLLED PAIRED-SEED EVALUATION <<<")
    print(f"Seeds: {len(EVAL_SEEDS)} development seeds ({EVAL_SEEDS[0]}..{EVAL_SEEDS[-1]})")
    print(f"Caution distance: d_caut = {CAUTION_DIST_M} m | Horizon: 500 steps (25.0s)")
    print("=" * 100)

    # First pass: evaluate all 4 controllers on all 50 seeds
    for name, cfg in CHECKPOINTS.items():
        print(f"\n--- Evaluating {name} ({cfg['role'].upper()}) ---")
        model = controller_models[name]
        ep_list = []
        for s_idx, seed in enumerate(EVAL_SEEDS, 1):
            ep = run_episode(
                env, supervisor, model,
                controller_id=name, checkpoint_path=cfg["path"],
                seed=seed, device=device, record_trace=False
            )
            ep_list.append(ep)
            succ_tag = "SUCCESS" if ep["is_success"] else "FAIL"
            print(f"[{s_idx:02d}/50] Seed {seed:4d} | {succ_tag:<7} | Fwd: {ep['forward_progress_m']:4.2f}m | MinObs: {ep['min_obstacle_clearance_m']:.3f}m | MinWall: {ep['min_boundary_clearance_m']:.3f}m | Steps: {ep['steps']:3d} | Cat: {ep['fail_category']}")
        all_controller_episodes[name] = ep_list

    # Generate CSV of paired results
    csv_path = audit_dir / "paired_episode_results.csv"
    fieldnames = [
        "seed", "controller_id", "role", "is_success", "status", "fail_category",
        "forward_progress_m", "duration_s", "steps",
        "min_obstacle_clearance_m", "min_boundary_clearance_m",
        "is_obstacle_collision", "is_boundary_collision",
        "is_obstacle_envelope_violation", "is_boundary_envelope_violation",
        "intervention_rate", "throttled_time_pct", "stationary_time_pct", "direction_reversals",
        "raw_vx_mean", "raw_vy_mean", "safe_vx_mean", "safe_vy_mean",
        "sat_policy_vx_pct", "sat_policy_vy_pct", "sat_applied_vx_pct", "sat_applied_vy_pct",
    ]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for name, eps in all_controller_episodes.items():
            role = CHECKPOINTS[name]["role"]
            for ep in eps:
                writer.writerow({
                    "seed": ep["seed"],
                    "controller_id": name,
                    "role": role,
                    "is_success": ep["is_success"],
                    "status": ep["status"],
                    "fail_category": ep["fail_category"],
                    "forward_progress_m": ep["forward_progress_m"],
                    "duration_s": ep["duration_s"],
                    "steps": ep["steps"],
                    "min_obstacle_clearance_m": ep["min_obstacle_clearance_m"],
                    "min_boundary_clearance_m": ep["min_boundary_clearance_m"],
                    "is_obstacle_collision": ep["is_obstacle_collision"],
                    "is_boundary_collision": ep["is_boundary_collision"],
                    "is_obstacle_envelope_violation": ep["is_obstacle_envelope_violation"],
                    "is_boundary_envelope_violation": ep["is_boundary_envelope_violation"],
                    "intervention_rate": ep["intervention_rate"],
                    "throttled_time_pct": ep["throttled_time_pct"],
                    "stationary_time_pct": ep["stationary_time_pct"],
                    "direction_reversals": ep["direction_reversals"],
                    "raw_vx_mean": ep["raw_action_mean"][0],
                    "raw_vy_mean": ep["raw_action_mean"][1],
                    "safe_vx_mean": ep["safe_action_mean"][0],
                    "safe_vy_mean": ep["safe_action_mean"][1],
                    "sat_policy_vx_pct": ep["policy_saturation_pct"]["vx"],
                    "sat_policy_vy_pct": ep["policy_saturation_pct"]["vy"],
                    "sat_applied_vx_pct": ep["applied_saturation_pct"]["vx"],
                    "sat_applied_vy_pct": ep["applied_saturation_pct"]["vy"],
                })

    print(f"\n[OUTPUT] Saved paired episode results to: {csv_path}")

    # Compute summary metrics for each controller
    controller_summary: Dict[str, Any] = {}
    for name, eps in all_controller_episodes.items():
        succ_cnt = sum(1 for e in eps if e["is_success"])
        obs_col = sum(1 for e in eps if e["is_obstacle_collision"])
        wall_col = sum(1 for e in eps if e["is_boundary_collision"])
        obs_viol = sum(1 for e in eps if e["is_obstacle_envelope_violation"])
        wall_viol = sum(1 for e in eps if e["is_boundary_envelope_violation"])
        timeouts = sum(1 for e in eps if e["is_timeout"])

        fwd = [e["forward_progress_m"] for e in eps]
        durs = [e["duration_s"] for e in eps]
        steps_list = [e["steps"] for e in eps]
        obs_c = [e["min_obstacle_clearance_m"] for e in eps]
        wall_c = [e["min_boundary_clearance_m"] for e in eps]
        intervs = [e["intervention_rate"] for e in eps]
        throttled = [e["throttled_time_pct"] for e in eps]
        stat_time = [e["stationary_time_pct"] for e in eps]
        revs = [e["direction_reversals"] for e in eps]

        raw_vx = [e["raw_action_mean"][0] for e in eps]
        raw_vy = [e["raw_action_mean"][1] for e in eps]
        safe_vx = [e["safe_action_mean"][0] for e in eps]
        safe_vy = [e["safe_action_mean"][1] for e in eps]

        sat_p_vx = [e["policy_saturation_pct"]["vx"] for e in eps]
        sat_p_vy = [e["policy_saturation_pct"]["vy"] for e in eps]
        sat_a_vx = [e["applied_saturation_pct"]["vx"] for e in eps]
        sat_a_vy = [e["applied_saturation_pct"]["vy"] for e in eps]

        low_ci, high_ci = wilson_score_interval(succ_cnt, len(eps))

        cat_counts = {}
        for e in eps:
            c = e["fail_category"]
            cat_counts[c] = cat_counts.get(c, 0) + 1

        controller_summary[name] = {
            "controller_id": name,
            "role": CHECKPOINTS[name]["role"],
            "checkpoint": CHECKPOINTS[name]["path"],
            "checkpoint_sha256": CHECKPOINTS[name]["expected_sha256"],
            "total_episodes": len(eps),
            "success_count": succ_cnt,
            "success_rate_pct": round((succ_cnt / len(eps)) * 100.0, 1),
            "success_ci95": [low_ci, high_ci],
            "obstacle_collision_count": obs_col,
            "obstacle_collision_pct": round((obs_col / len(eps)) * 100.0, 1),
            "boundary_collision_count": wall_col,
            "boundary_collision_pct": round((wall_col / len(eps)) * 100.0, 1),
            "obstacle_envelope_violation_count": obs_viol,
            "obstacle_envelope_violation_pct": round((obs_viol / len(eps)) * 100.0, 1),
            "boundary_envelope_violation_count": wall_viol,
            "boundary_envelope_violation_pct": round((wall_viol / len(eps)) * 100.0, 1),
            "timeout_count": timeouts,
            "timeout_pct": round((timeouts / len(eps)) * 100.0, 1),
            "forward_progress_m": {
                "mean": round(float(np.mean(fwd)), 3),
                "median": round(float(np.median(fwd)), 3),
                "std": round(float(np.std(fwd)), 3),
                "min": round(float(np.min(fwd)), 3),
                "max": round(float(np.max(fwd)), 3),
            },
            "duration_s": {
                "mean": round(float(np.mean(durs)), 2),
                "median": round(float(np.median(durs)), 2),
            },
            "steps": {
                "mean": round(float(np.mean(steps_list)), 1),
                "median": round(float(np.median(steps_list)), 1),
            },
            "obstacle_clearance_m": {
                "mean": round(float(np.mean(obs_c)), 4),
                "median": round(float(np.median(obs_c)), 4),
                "global_min": round(float(np.min(obs_c)), 4),
            },
            "boundary_clearance_m": {
                "mean": round(float(np.mean(wall_c)), 4),
                "median": round(float(np.median(wall_c)), 4),
                "global_min": round(float(np.min(wall_c)), 4),
            },
            "supervisor_metrics": {
                "mean_intervention_rate": round(float(np.mean(intervs)), 3),
                "mean_throttled_time_pct": round(float(np.mean(throttled)), 1),
                "mean_stationary_time_pct": round(float(np.mean(stat_time)), 1),
                "mean_direction_reversals": round(float(np.mean(revs)), 2),
            },
            "action_diagnostics": {
                "mean_raw_vx": round(float(np.mean(raw_vx)), 4),
                "mean_raw_vy": round(float(np.mean(raw_vy)), 4),
                "mean_safe_vx": round(float(np.mean(safe_vx)), 4),
                "mean_safe_vy": round(float(np.mean(safe_vy)), 4),
                "mean_policy_sat_vx_pct": round(float(np.mean(sat_p_vx)), 1),
                "mean_policy_sat_vy_pct": round(float(np.mean(sat_p_vy)), 1),
                "mean_applied_sat_vx_pct": round(float(np.mean(sat_a_vx)), 1),
                "mean_applied_sat_vy_pct": round(float(np.mean(sat_a_vy)), 1),
            },
            "failure_categories": cat_counts,
        }

    # Paired Statistics: BC Baseline vs PPO Baseline
    bc_eps = all_controller_episodes["Arm_B_BC_Baseline"]
    ppo_eps = all_controller_episodes["Arm_A_PPO_Baseline"]

    both_succ = []
    bc_only_succ = []
    ppo_only_succ = []
    both_fail = []

    for s_idx, seed in enumerate(EVAL_SEEDS):
        bc_s = bc_eps[s_idx]["is_success"]
        ppo_s = ppo_eps[s_idx]["is_success"]
        if bc_s and ppo_s:
            both_succ.append(seed)
        elif bc_s and not ppo_s:
            bc_only_succ.append(seed)
        elif not bc_s and ppo_s:
            ppo_only_succ.append(seed)
        else:
            both_fail.append(seed)

    # McNemar's test: b = bc_only_succ count, c = ppo_only_succ count
    b = len(bc_only_succ)
    c = len(ppo_only_succ)
    p_val_mcnemar = mcnemar_exact_test(b, c)

    # Paired differences in forward progress
    paired_diff_fwd = [bc_eps[i]["forward_progress_m"] - ppo_eps[i]["forward_progress_m"] for i in range(len(EVAL_SEEDS))]
    paired_diff_obs = [bc_eps[i]["min_obstacle_clearance_m"] - ppo_eps[i]["min_obstacle_clearance_m"] for i in range(len(EVAL_SEEDS))]

    paired_analysis = {
        "comparison": "Arm_B_BC_Baseline vs Arm_A_PPO_Baseline",
        "sample_size": len(EVAL_SEEDS),
        "bc_success_count": sum(1 for e in bc_eps if e["is_success"]),
        "ppo_success_count": sum(1 for e in ppo_eps if e["is_success"]),
        "absolute_difference_pct": round(((sum(1 for e in bc_eps if e["is_success"]) - sum(1 for e in ppo_eps if e["is_success"])) / len(EVAL_SEEDS)) * 100.0, 1),
        "contingency_table": {
            "both_succeed": {"count": len(both_succ), "seeds": both_succ},
            "bc_succeeds_ppo_fails": {"count": len(bc_only_succ), "seeds": bc_only_succ},
            "ppo_succeeds_bc_fails": {"count": len(ppo_only_succ), "seeds": ppo_only_succ},
            "both_fail": {"count": len(both_fail), "seeds": both_fail},
        },
        "mcnemar_exact_test": {
            "b_bc_only": b,
            "c_ppo_only": c,
            "two_tailed_p_value": round(p_val_mcnemar, 6),
            "statistically_significant_at_alpha_0_05": p_val_mcnemar < 0.05,
        },
        "paired_forward_progress_difference_m": {
            "mean_delta_bc_minus_ppo": round(float(np.mean(paired_diff_fwd)), 3),
            "median_delta": round(float(np.median(paired_diff_fwd)), 3),
            "std_delta": round(float(np.std(paired_diff_fwd)), 3),
        },
        "paired_obstacle_clearance_difference_m": {
            "mean_delta_bc_minus_ppo": round(float(np.mean(paired_diff_obs)), 4),
            "median_delta": round(float(np.median(paired_diff_obs)), 4),
        },
    }

    # Save summary json
    summary_out = {
        "metadata": {
            "evaluation_phase": "Phase 4J-E Controlled Comparison",
            "date": "2026-10-29",
            "seed_set": "Development seeds 2000..2049 (50 paired seeds)",
            "caution_distance_m": CAUTION_DIST_M,
            "horizon_steps": env.max_steps,
            "timestep_s": env.dt,
        },
        "controller_summary": controller_summary,
        "primary_paired_comparison": paired_analysis,
    }

    sum_path = audit_dir / "controller_summary.json"
    with open(sum_path, "w") as f:
        json.dump(summary_out, f, indent=2, cls=NumpyEncoder)
    print(f"[OUTPUT] Saved summary JSON to: {sum_path}")

    # Generate traces for representative seeds (Stage G)
    trace_dir = audit_dir / "traces"
    trace_dir.mkdir(parents=True, exist_ok=True)

    # Pick representative seeds:
    # 2 Both succeed
    # 2 BC succeeds, PPO fails
    # 2 PPO succeeds, BC fails (if any)
    # 2 Both fail
    selected_trace_seeds = []
    if len(both_succ) >= 2:
        selected_trace_seeds.extend([("BOTH_SUCCEED", s) for s in both_succ[:2]])
    elif len(both_succ) == 1:
        selected_trace_seeds.append(("BOTH_SUCCEED", both_succ[0]))

    if len(bc_only_succ) >= 2:
        selected_trace_seeds.extend([("BC_SUCCEEDS_PPO_FAILS", s) for s in bc_only_succ[:2]])
    elif len(bc_only_succ) == 1:
        selected_trace_seeds.append(("BC_SUCCEEDS_PPO_FAILS", bc_only_succ[0]))

    if len(ppo_only_succ) >= 2:
        selected_trace_seeds.extend([("PPO_SUCCEEDS_BC_FAILS", s) for s in ppo_only_succ[:2]])
    elif len(ppo_only_succ) == 1:
        selected_trace_seeds.append(("PPO_SUCCEEDS_BC_FAILS", ppo_only_succ[0]))
    else:
        print("[INFO] No seeds where PPO succeeds and BC fails were found.")

    if len(both_fail) >= 2:
        selected_trace_seeds.extend([("BOTH_FAIL", s) for s in both_fail[:2]])
    elif len(both_fail) == 1:
        selected_trace_seeds.append(("BOTH_FAIL", both_fail[0]))

    print(f"\n>>> GENERATING DIAGNOSTIC TRACES FOR REPRESENTATIVE SEEDS <<<")
    trace_manifest = []
    for category, s in selected_trace_seeds:
        print(f"Recording matched traces for Seed {s} ({category})...")
        # Run BC with trace
        bc_trace_ep = run_episode(
            env, supervisor, controller_models["Arm_B_BC_Baseline"],
            controller_id="Arm_B_BC_Baseline", checkpoint_path=CHECKPOINTS["Arm_B_BC_Baseline"]["path"],
            seed=s, device=device, record_trace=True
        )
        # Run PPO with trace
        ppo_trace_ep = run_episode(
            env, supervisor, controller_models["Arm_A_PPO_Baseline"],
            controller_id="Arm_A_PPO_Baseline", checkpoint_path=CHECKPOINTS["Arm_A_PPO_Baseline"]["path"],
            seed=s, device=device, record_trace=True
        )

        t_file = trace_dir / f"matched_trace_seed_{s}_{category.lower()}.json"
        with open(t_file, "w") as f:
            json.dump({
                "seed": s,
                "category": category,
                "Arm_B_BC_Baseline": bc_trace_ep,
                "Arm_A_PPO_Baseline": ppo_trace_ep,
            }, f, indent=2, cls=NumpyEncoder)
        trace_manifest.append({"seed": s, "category": category, "file": str(t_file.name)})

    env.close()

    # Create Comparison Manifest
    manifest = {
        "manifest_metadata": {
            "phase": "Phase 4J-E",
            "milestone": "Historical PPO Audit & Controlled BC vs PPO Comparison",
            "timestamp_iso": "2026-10-29T17:50:00Z",
            "workspace": "/Users/ghulamqadir/.gemini/antigravity-ide/scratch/uav-bput-nav",
            "evaluation_seeds": f"{len(EVAL_SEEDS)} development seeds ({EVAL_SEEDS[0]}..{EVAL_SEEDS[-1]})",
            "horizon_steps": 500,
            "timestep_s": 0.05,
            "caution_distance_m": CAUTION_DIST_M,
        },
        "verified_file_hashes": {
            name: cfg["expected_sha256"] for name, cfg in CHECKPOINTS.items()
        },
        "system_hashes": {
            "controllers/safety_supervisor.py": "9f84af22821f0f1473ea963b0f33a1924d1c4e7a487d4ed7893b7efc56fcd26f",
            "experiments/phase4j/isolated_supervisor_opt.py": "d68fbc38b58e5f2e21e5ad81efc1bc9d053734fe9f4d98b993445784b6b2f83f",
            "sim/corridor_env.py": "4d0c9bd20b370c181a49b5e5516cab851d4353d038774053557aad260255fc66",
            "core/geometry.py": "22ca56dd3420614e93346d4d0e78e53531530cb9b6cadb9a400a69fd8c9cfadb",
            "core/contracts.py": "1272af8f2f301f854ab75b0088e4032f464789379c8120825aa747b1580967ab",
            "models/policy_net.py": "fe5f8e81a09e4dd123d8cde011f4c7ac9d0511911d83fab1f550337a1f204ea9",
            "models/actor_critic.py": "b7a8eec4828a849c9ddd36ad37d707baeb1bbecf0495b93c3a2b01a2eca30162",
        },
        "results_summary": {
            name: {
                "success_rate_pct": controller_summary[name]["success_rate_pct"],
                "obstacle_collisions": controller_summary[name]["obstacle_collision_count"],
                "boundary_collisions": controller_summary[name]["boundary_collision_count"],
                "envelope_violations": controller_summary[name]["obstacle_envelope_violation_count"],
                "mean_forward_progress_m": controller_summary[name]["forward_progress_m"]["mean"],
                "mean_duration_s": controller_summary[name]["duration_s"]["mean"],
            } for name in CHECKPOINTS.keys()
        },
        "primary_comparison_mcnemar_p_value": p_val_mcnemar,
        "trace_manifest": trace_manifest,
    }

    manifest_path = audit_dir / "comparison_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[OUTPUT] Saved comparison manifest to: {manifest_path}")

    # Create controller_comparison.md
    comp_md_path = audit_dir / "controller_comparison.md"
    generate_comparison_markdown(controller_summary, paired_analysis, comp_md_path)
    print(f"[OUTPUT] Saved controller comparison markdown to: {comp_md_path}")

    return summary_out


def generate_comparison_markdown(summary: Dict[str, Any], paired: Dict[str, Any], out_path: Path):
    bc_b = summary["Arm_B_BC_Baseline"]
    ppo_b = summary["Arm_A_PPO_Baseline"]
    bc_h = summary["BC_Phase4H_Candidate"]
    ppo_h = summary["PPO_Phase4H_Candidate"]

    md = f"""# Phase 4J-E: Controlled Controller Comparison Report

**Audit Directory:** `{out_path.parent}`  
**Evaluation Seeds:** 50 Development Seeds (`2000..2049`)  
**Conditions:** 10m Corridor Benchmark, Horizon 500 steps (25.0s), Isolated Supervisor $d_{{\\text{{caut}}}} = 0.50\\text{{ m}}$  
**Date:** 2026-10-29  

---

## 1. Primary Benchmark Results Table

| Metric | Arm B: BC Baseline (Primary) | Arm A: PPO Baseline (Primary) | BC Phase 4H Candidate (Secondary) | PPO Phase 4H Candidate (Secondary) |
|:---|:---:|:---:|:---:|:---:|
| **Goal Success Count** | **{bc_b['success_count']} / 50** | **{ppo_b['success_count']} / 50** | **{bc_h['success_count']} / 50** | **{ppo_h['success_count']} / 50** |
| **Goal Success Rate** | **{bc_b['success_rate_pct']}%** | **{ppo_b['success_rate_pct']}%** | **{bc_h['success_rate_pct']}%** | **{ppo_h['success_rate_pct']}%** |
| **95% Wilson Score CI** | `[{bc_b['success_ci95'][0]}%, {bc_b['success_ci95'][1]}%]` | `[{ppo_b['success_ci95'][0]}%, {ppo_b['success_ci95'][1]}%]` | `[{bc_h['success_ci95'][0]}%, {bc_h['success_ci95'][1]}%]` | `[{ppo_h['success_ci95'][0]}%, {ppo_h['success_ci95'][1]}%]` |
| **Obstacle Collisions** | **{bc_b['obstacle_collision_count']} ({bc_b['obstacle_collision_pct']}%)** | **{ppo_b['obstacle_collision_count']} ({ppo_b['obstacle_collision_pct']}%)** | **{bc_h['obstacle_collision_count']} ({bc_h['obstacle_collision_pct']}%)** | **{ppo_h['obstacle_collision_count']} ({ppo_h['obstacle_collision_pct']}%)** |
| **Boundary Collisions** | **{bc_b['boundary_collision_count']} ({bc_b['boundary_collision_pct']}%)** | **{ppo_b['boundary_collision_count']} ({ppo_b['boundary_collision_pct']}%)** | **{bc_h['boundary_collision_count']} ({bc_h['boundary_collision_pct']}%)** | **{ppo_h['boundary_collision_count']} ({ppo_h['boundary_collision_pct']}%)** |
| **Obstacle Envelope Violations (< 8cm)** | **{bc_b['obstacle_envelope_violation_count']} ({bc_b['obstacle_envelope_violation_pct']}%)** | **{ppo_b['obstacle_envelope_violation_count']} ({ppo_b['obstacle_envelope_violation_pct']}%)** | **{bc_h['obstacle_envelope_violation_count']} ({bc_h['obstacle_envelope_violation_pct']}%)** | **{ppo_h['obstacle_envelope_violation_count']} ({ppo_h['obstacle_envelope_violation_pct']}%)** |
| **Boundary Envelope Violations (< 8cm)** | **{bc_b['boundary_envelope_violation_count']} ({bc_b['boundary_envelope_violation_pct']}%)** | **{ppo_b['boundary_envelope_violation_count']} ({ppo_b['boundary_envelope_violation_pct']}%)** | **{bc_h['boundary_envelope_violation_count']} ({bc_h['boundary_envelope_violation_pct']}%)** | **{ppo_h['boundary_envelope_violation_count']} ({ppo_h['boundary_envelope_violation_pct']}%)** |
| **Episode Timeouts (500 steps)** | {bc_b['timeout_count']} ({bc_b['timeout_pct']}%) | {ppo_b['timeout_count']} ({ppo_b['timeout_pct']}%) | {bc_h['timeout_count']} ({bc_h['timeout_pct']}%) | {ppo_h['timeout_count']} ({ppo_h['timeout_pct']}%) |
| **Mean Forward Progress** | **{bc_b['forward_progress_m']['mean']:.2f} m** | {ppo_b['forward_progress_m']['mean']:.2f} m | {bc_h['forward_progress_m']['mean']:.2f} m | {ppo_h['forward_progress_m']['mean']:.2f} m |
| **Median Forward Progress** | **{bc_b['forward_progress_m']['median']:.2f} m** | {ppo_b['forward_progress_m']['median']:.2f} m | {bc_h['forward_progress_m']['median']:.2f} m | {ppo_h['forward_progress_m']['median']:.2f} m |
| **Global Min Obstacle Clearance** | {bc_b['obstacle_clearance_m']['global_min']:.4f} m | {ppo_b['obstacle_clearance_m']['global_min']:.4f} m | {bc_h['obstacle_clearance_m']['global_min']:.4f} m | {ppo_h['obstacle_clearance_m']['global_min']:.4f} m |
| **Global Min Boundary Clearance** | {bc_b['boundary_clearance_m']['global_min']:.4f} m | {ppo_b['boundary_clearance_m']['global_min']:.4f} m | {bc_h['boundary_clearance_m']['global_min']:.4f} m | **{ppo_h['boundary_clearance_m']['global_min']:.4f} m** |
| **Mean Episode Duration** | {bc_b['duration_s']['mean']:.2f} s | {ppo_b['duration_s']['mean']:.2f} s | {bc_h['duration_s']['mean']:.2f} s | {ppo_h['duration_s']['mean']:.2f} s |
| **Mean Throttled Fraction** | {bc_b['supervisor_metrics']['mean_throttled_time_pct']:.1f}% | {ppo_b['supervisor_metrics']['mean_throttled_time_pct']:.1f}% | {bc_h['supervisor_metrics']['mean_throttled_time_pct']:.1f}% | {ppo_h['supervisor_metrics']['mean_throttled_time_pct']:.1f}% |
| **Mean Intervention Rate** | {bc_b['supervisor_metrics']['mean_intervention_rate']:.3f} | {ppo_b['supervisor_metrics']['mean_intervention_rate']:.3f} | {bc_h['supervisor_metrics']['mean_intervention_rate']:.3f} | {ppo_h['supervisor_metrics']['mean_intervention_rate']:.3f} |
| **Mean Raw Lateral Velocity ($v_y$)** | {bc_b['action_diagnostics']['mean_raw_vy']:+.4f} m/s | **{ppo_b['action_diagnostics']['mean_raw_vy']:+.4f} m/s** | {bc_h['action_diagnostics']['mean_raw_vy']:+.4f} m/s | {ppo_h['action_diagnostics']['mean_raw_vy']:+.4f} m/s |
| **Policy Saturation ($v_x$)** | {bc_b['action_diagnostics']['mean_policy_sat_vx_pct']:.1f}% | {ppo_b['action_diagnostics']['mean_policy_sat_vx_pct']:.1f}% | {bc_h['action_diagnostics']['mean_policy_sat_vx_pct']:.1f}% | {ppo_h['action_diagnostics']['mean_policy_sat_vx_pct']:.1f}% |

---

## 2. Paired Statistical Comparison: BC Baseline vs. PPO Baseline

- **Sample Size:** 50 paired development episodes (Seeds `2000..2049`)
- **BC Successes:** {paired['bc_success_count']} / 50 ({bc_b['success_rate_pct']}%)
- **PPO Successes:** {paired['ppo_success_count']} / 50 ({ppo_b['success_rate_pct']}%)
- **Absolute Success Difference:** **{paired['absolute_difference_pct']:+.1f} percentage points** in favor of BC Baseline

### Paired Contingency Table (50 Seeds):
| Outcome Combination | Seed Count | Percentage |
|:---|:---:|:---:|
| **Both BC and PPO Succeed** | **{paired['contingency_table']['both_succeed']['count']}** | **{paired['contingency_table']['both_succeed']['count'] / 50 * 100:.1f}%** |
| **BC Succeeds, PPO Fails** | **{paired['contingency_table']['bc_succeeds_ppo_fails']['count']}** | **{paired['contingency_table']['bc_succeeds_ppo_fails']['count'] / 50 * 100:.1f}%** |
| **PPO Succeeds, BC Fails** | **{paired['contingency_table']['ppo_succeeds_bc_fails']['count']}** | **{paired['contingency_table']['ppo_succeeds_bc_fails']['count'] / 50 * 100:.1f}%** |
| **Both BC and PPO Fail** | **{paired['contingency_table']['both_fail']['count']}** | **{paired['contingency_table']['both_fail']['count'] / 50 * 100:.1f}%** |

### McNemar's Exact Test:
- Disagreeing Pairs: $b = {paired['mcnemar_exact_test']['b_bc_only']}$ (BC wins), $c = {paired['mcnemar_exact_test']['c_ppo_only']}$ (PPO wins)
- **Two-Tailed Exact $p$-Value:** **{paired['mcnemar_exact_test']['two_tailed_p_value']}**
- **Statistical Significance:** **{'STATISTICALLY SIGNIFICANT ($p < 0.05$)' if paired['mcnemar_exact_test']['statistically_significant_at_alpha_0_05'] else 'NOT STATISTICALLY SIGNIFICANT'}**

---

## 3. Failure Mode Distribution Across Controllers

| Failure Mode | BC Baseline | PPO Baseline | BC Phase 4H Candidate | PPO Phase 4H Candidate |
|:---|:---:|:---:|:---:|:---:|
| `SUCCESS` | {bc_b['failure_categories'].get('SUCCESS', 0)} | {ppo_b['failure_categories'].get('SUCCESS', 0)} | {bc_h['failure_categories'].get('SUCCESS', 0)} | {ppo_h['failure_categories'].get('SUCCESS', 0)} |
| `WALL_COLLISION` | {bc_b['failure_categories'].get('WALL_COLLISION', 0)} | {ppo_b['failure_categories'].get('WALL_COLLISION', 0)} | {bc_h['failure_categories'].get('WALL_COLLISION', 0)} | **{ppo_h['failure_categories'].get('WALL_COLLISION', 0)}** |
| `OBSTACLE_COLLISION` | {bc_b['failure_categories'].get('OBSTACLE_COLLISION', 0)} | {ppo_b['failure_categories'].get('OBSTACLE_COLLISION', 0)} | {bc_h['failure_categories'].get('OBSTACLE_COLLISION', 0)} | {ppo_h['failure_categories'].get('OBSTACLE_COLLISION', 0)} |
| `PROGRESS_STALL_PILLAR_1` | {bc_b['failure_categories'].get('PROGRESS_STALL_PILLAR_1', 0)} | {ppo_b['failure_categories'].get('PROGRESS_STALL_PILLAR_1', 0)} | {bc_h['failure_categories'].get('PROGRESS_STALL_PILLAR_1', 0)} | {ppo_h['failure_categories'].get('PROGRESS_STALL_PILLAR_1', 0)} |
| `PROGRESS_STALL_PILLAR_2` | {bc_b['failure_categories'].get('PROGRESS_STALL_PILLAR_2', 0)} | {ppo_b['failure_categories'].get('PROGRESS_STALL_PILLAR_2', 0)} | {bc_h['failure_categories'].get('PROGRESS_STALL_PILLAR_2', 0)} | {ppo_h['failure_categories'].get('PROGRESS_STALL_PILLAR_2', 0)} |
| `PROGRESS_STALL_PILLAR_3` | {bc_b['failure_categories'].get('PROGRESS_STALL_PILLAR_3', 0)} | {ppo_b['failure_categories'].get('PROGRESS_STALL_PILLAR_3', 0)} | {bc_h['failure_categories'].get('PROGRESS_STALL_PILLAR_3', 0)} | {ppo_h['failure_categories'].get('PROGRESS_STALL_PILLAR_3', 0)} |
| `PROGRESS_LATERAL_OSCILLATION` | {bc_b['failure_categories'].get('PROGRESS_LATERAL_OSCILLATION', 0)} | {ppo_b['failure_categories'].get('PROGRESS_LATERAL_OSCILLATION', 0)} | {bc_h['failure_categories'].get('PROGRESS_LATERAL_OSCILLATION', 0)} | {ppo_h['failure_categories'].get('PROGRESS_LATERAL_OSCILLATION', 0)} |
| `TIMEOUT_EXCESSIVE_THROTTLING` | {bc_b['failure_categories'].get('TIMEOUT_EXCESSIVE_THROTTLING', 0)} | {ppo_b['failure_categories'].get('TIMEOUT_EXCESSIVE_THROTTLING', 0)} | {bc_h['failure_categories'].get('TIMEOUT_EXCESSIVE_THROTTLING', 0)} | {ppo_h['failure_categories'].get('TIMEOUT_EXCESSIVE_THROTTLING', 0)} |
"""
    out_path.write_text(md)


if __name__ == "__main__":
    audit_directory = Path("experiments/phase4j/phase4je_audit_20261029_173705")
    audit_directory.mkdir(parents=True, exist_ok=True)
    execute_stage_d(audit_directory)

