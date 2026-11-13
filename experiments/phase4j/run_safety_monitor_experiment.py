"""run_safety_monitor_experiment.py - Compare Reference vs Oracle-Assisted Configuration.

Evaluates:
- Config A (Reference): BC Baseline at d_caut=0.50m (No oracle monitor).
- Config B (Experimental): BC Baseline at d_caut=0.50m WITH OracleGeometricSafetyMonitor.
- Validation Test: Runs Seed 1007 on BC Candidate with Oracle Monitor to verify prevention of 3mm anomaly.

Across seeds 2000..2049 (50 paired seeds).
Saves results to experiments/phase4j/phase4jc_results/safety_monitor_comparison.json.
"""

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch

from core.contracts import ActionLimits, VelocityAction, compute_forward_clearance_m
from core.geometry import Corridor3D, TerminationReason
from experiments.phase4j.isolated_supervisor_opt import IsolatedOptimizedSupervisor
from experiments.phase4j.isolated_geometric_safety_monitor import OracleGeometricSafetyMonitor
from models.policy_net import NavigationPolicyMLP
from sim.corridor_env import CorridorSimEnv


EVAL_SEEDS = list(range(2000, 2050))
CAUTION_DIST_M = 0.50
CHECKPOINT_PATH = "models/checkpoints/bc_policy_best.pt"


def run_episode(
    env: CorridorSimEnv,
    supervisor: IsolatedOptimizedSupervisor,
    model: NavigationPolicyMLP,
    seed: int,
    oracle_monitor: Optional[OracleGeometricSafetyMonitor] = None,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, Any]:
    obs, info = env.reset(seed=seed)
    supervisor.reset_state()
    if oracle_monitor is not None:
        oracle_monitor.reset_state()

    ep_start_pos = env._get_drone_position()
    steps = 0
    done = False

    min_obs_clear = float(info["privileged_state"].obstacle_clearance)
    min_wall_clear = float(info["privileged_state"].corridor_boundary_distance)

    supervisor_interventions = 0
    oracle_interventions = 0
    throttled_steps = 0
    stationary_steps = 0
    current_info = info

    while not done and steps < env.max_steps:
        priv = current_info["privileged_state"]
        curr_pos = env._get_drone_position()
        curr_vel = priv.global_velocity
        _, _, _, curr_yaw = env._get_drone_orientation()

        obs_c = float(priv.obstacle_clearance)
        wall_c = float(priv.corridor_boundary_distance)
        min_obs_clear = min(min_obs_clear, obs_c)
        min_wall_clear = min(min_wall_clear, wall_c)

        speed = float(np.linalg.norm(curr_vel))
        if speed <= 0.05:
            stationary_steps += 1

        front_dist = compute_forward_clearance_m(obs, max_range_m=env.sensor_config.max_range_m)

        # Policy forward pass
        with torch.no_grad():
            t_obs = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            act_arr = model(t_obs, bounded=True).squeeze(0).cpu().numpy()
        raw_act = VelocityAction(vx=float(act_arr[0]), vy=float(act_arr[1]), vz=float(act_arr[2]), yaw_rate=float(act_arr[3]))

        clamped_act = VelocityAction(
            vx=float(np.clip(raw_act.vx, env.action_limits.min_fwd_vel_mps, env.action_limits.max_fwd_vel_mps)),
            vy=float(np.clip(raw_act.vy, -env.action_limits.max_lat_vel_mps, env.action_limits.max_lat_vel_mps)),
            vz=float(np.clip(raw_act.vz, -env.action_limits.max_vert_vel_mps, env.action_limits.max_vert_vel_mps)),
            yaw_rate=float(np.clip(raw_act.yaw_rate, -env.action_limits.max_yaw_rate_radps, env.action_limits.max_yaw_rate_radps)),
        )

        # 1. Existing Supervisor Filter
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

        if s_state.value != "nominal":
            supervisor_interventions += 1
        if "Speed throttled" in s_reason:
            throttled_steps += 1

        # 2. Optional Oracle Geometric Safety Monitor Filter
        applied_act = safe_act
        if oracle_monitor is not None:
            applied_act, intervened, o_reason = oracle_monitor.evaluate_and_filter(
                proposed_action=safe_act,
                current_position=curr_pos,
                current_velocity=curr_vel,
                current_yaw=curr_yaw,
                obstacles=env.obstacles,
            )
            if intervened:
                oracle_interventions += 1

        norm_applied = applied_act.to_normalized(env.action_limits)
        obs, rew, term, trunc, current_info = env.step(norm_applied)
        steps += 1
        done = term or trunc

    # Terminal state clearances
    final_priv = current_info["privileged_state"]
    min_obs_clear = min(min_obs_clear, float(final_priv.obstacle_clearance))
    min_wall_clear = min(min_wall_clear, float(final_priv.corridor_boundary_distance))

    final_pos = env._get_drone_position()
    final_vel = float(np.linalg.norm(final_priv.global_velocity))
    final_goal_dist = float(final_priv.goal_distance)
    term_reason = current_info.get("termination_reason", "unknown")

    is_success = (term_reason == "success") and (final_vel <= 0.35)
    is_obs_collision = (term_reason == "crash_obstacle") or (min_obs_clear <= 0.0)
    is_wall_collision = (term_reason == "crash_boundary") or (min_wall_clear <= 0.0)
    is_timeout = (steps >= env.max_steps and not is_success and not is_obs_collision and not is_wall_collision)

    return {
        "seed": seed,
        "steps": steps,
        "duration_s": round(steps * env.dt, 2),
        "status": term_reason,
        "is_success": is_success,
        "is_timeout": is_timeout,
        "is_obstacle_collision": is_obs_collision,
        "is_boundary_collision": is_wall_collision,
        "is_obstacle_envelope_violation": (min_obs_clear < 0.080),
        "forward_progress_m": round(float(final_pos[0] - ep_start_pos[0]), 3),
        "min_obstacle_clearance_m": round(min_obs_clear, 4),
        "min_boundary_clearance_m": round(min_wall_clear, 4),
        "terminal_speed_mps": round(final_vel, 3),
        "supervisor_interventions": supervisor_interventions,
        "oracle_interventions": oracle_interventions,
        "throttled_steps": throttled_steps,
        "throttled_time_pct": round((throttled_steps / max(1, steps)) * 100.0, 1),
        "stationary_steps": stationary_steps,
    }


def run_experiment():
    device = torch.device("cpu")
    model = NavigationPolicyMLP(obs_dim=23, act_dim=4).to(device)
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    env = CorridorSimEnv(scenario="10m", gui=False, seed=42)
    supervisor = IsolatedOptimizedSupervisor(corridor=env.corridor, caution_distance_m=CAUTION_DIST_M)
    oracle_monitor = OracleGeometricSafetyMonitor(corridor=env.corridor, guard_envelope_m=0.080)

    print("=" * 90)
    print(">>> RUNNING CONFIG A: REFERENCE (SUPERVISOR ONLY, d_caut=0.50m) <<<")
    print("=" * 90)
    results_a = []
    for s in EVAL_SEEDS:
        r = run_episode(env, supervisor, model, seed=s, oracle_monitor=None, device=device)
        results_a.append(r)

    print("=" * 90)
    print(">>> RUNNING CONFIG B: EXPERIMENTAL (SUPERVISOR + ORACLE GEOMETRIC MONITOR) <<<")
    print("=" * 90)
    results_b = []
    for s in EVAL_SEEDS:
        r = run_episode(env, supervisor, model, seed=s, oracle_monitor=oracle_monitor, device=device)
        results_b.append(r)

    # Validation on Seed 1007 with BC Candidate
    print("=" * 90)
    print(">>> VALIDATION ON SEED 1007 (BC CANDIDATE ANOMALY SCENARIO) <<<")
    print("=" * 90)
    cand_model = NavigationPolicyMLP(obs_dim=23, act_dim=4).to(device)
    cand_ckpt = torch.load("experiments/phase4h/checkpoints/bc_phase4h_candidate.pt", map_location=device)
    cand_model.load_state_dict(cand_ckpt["model_state_dict"])
    cand_model.eval()

    res_1007_unmonitored = run_episode(env, supervisor, cand_model, seed=1007, oracle_monitor=None, device=device)
    res_1007_monitored = run_episode(env, supervisor, cand_model, seed=1007, oracle_monitor=oracle_monitor, device=device)

    env.close()

    def summarize(res_list, name):
        succ = sum(1 for r in res_list if r["is_success"])
        obs_col = sum(1 for r in res_list if r["is_obstacle_collision"])
        wall_col = sum(1 for r in res_list if r["is_boundary_collision"])
        viols = sum(1 for r in res_list if r["is_obstacle_envelope_violation"])
        prog = float(np.mean([r["forward_progress_m"] for r in res_list]))
        dur = float(np.mean([r["duration_s"] for r in res_list]))
        min_c = float(np.min([r["min_obstacle_clearance_m"] for r in res_list]))
        th = float(np.mean([r["throttled_time_pct"] for r in res_list]))
        o_int = sum(r.get("oracle_interventions", 0) for r in res_list)
        return {
            "name": name,
            "success_count": succ,
            "success_rate_pct": round(succ / len(res_list) * 100.0, 1),
            "obstacle_collisions": obs_col,
            "boundary_collisions": wall_col,
            "envelope_violations": viols,
            "global_min_obs_clearance_m": round(min_c, 4),
            "mean_forward_progress_m": round(prog, 2),
            "mean_duration_s": round(dur, 2),
            "mean_throttled_time_pct": round(th, 1),
            "total_oracle_interventions": o_int,
        }

    sum_a = summarize(results_a, "Config_A_Reference")
    sum_b = summarize(results_b, "Config_B_Oracle_Monitor")

    print(f"\n--- COMPARISON SUMMARY ---")
    print(f"Config A (Reference): Succ={sum_a['success_count']}/50 ({sum_a['success_rate_pct']}%) | Violations={sum_a['envelope_violations']} | MinClear={sum_a['global_min_obs_clearance_m']}m | Fwd={sum_a['mean_forward_progress_m']}m")
    print(f"Config B (Oracle)   : Succ={sum_b['success_count']}/50 ({sum_b['success_rate_pct']}%) | Violations={sum_b['envelope_violations']} | MinClear={sum_b['global_min_obs_clearance_m']}m | Fwd={sum_b['mean_forward_progress_m']}m | OracleInterv={sum_b['total_oracle_interventions']}")
    print(f"\nSeed 1007 BC Candidate:")
    print(f"  Unmonitored: MinClear={res_1007_unmonitored['min_obstacle_clearance_m']}m (Viol={res_1007_unmonitored['is_obstacle_envelope_violation']})")
    print(f"  Monitored  : MinClear={res_1007_monitored['min_obstacle_clearance_m']}m (Viol={res_1007_monitored['is_obstacle_envelope_violation']}) | OracleInterv={res_1007_monitored['oracle_interventions']}")

    out_file = Path("experiments/phase4j/phase4jc_results/safety_monitor_comparison.json")
    with open(out_file, "w") as f:
        json.dump({
            "summary_a": sum_a,
            "summary_b": sum_b,
            "episodes_a": results_a,
            "episodes_b": results_b,
            "seed_1007_unmonitored": res_1007_unmonitored,
            "seed_1007_monitored": res_1007_monitored,
        }, f, indent=2)

    return sum_a, sum_b


if __name__ == "__main__":
    run_experiment()

