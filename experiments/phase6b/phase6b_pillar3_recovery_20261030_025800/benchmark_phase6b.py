"""benchmark_phase6b.py - Phase 6B Comprehensive Evaluation Benchmark.

Runs controlled paired comparison between:
1. Phase 6A Baseline (Stateless Clearance Guidance Controller)
2. Phase 6B Candidate (Post-Evasion Recovery Controller)
across:
- Diagnostic scenarios (Scenario A, B_Left, B_Right, C_Wide_Left, C_Wide_Right)
- All 50 canonical development seeds (2000-2049)

Reports exact paired metrics, clearance envelopes, collision counts, progress, and failure mode shifts.
"""

import csv
import json
import math
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pybullet as p
import torch

from controllers.guidance_controller import StatelessClearanceGuidanceController
from controllers.safety_supervisor import SafetyState, SafetySupervisor, SafetySupervisorConfig
from core.contracts import ActionLimits, DeployableObservation, VelocityAction, compute_forward_clearance_m
from core.geometry import Corridor3D, Obstacle3D
from experiments.phase6b.phase6b_pillar3_recovery_20261030_025800.diagnose_phase6a_failures import (
    NumpyEncoder,
    run_episode,
)
from experiments.phase6b.phase6b_pillar3_recovery_20261030_025800.pillar3_recovery_controller import (
    PostEvasionRecoveryController,
)
from sim.corridor_env import CorridorSimEnv

WORKSPACE_ROOT = Path("/Users/ghulamqadir/.gemini/antigravity-ide/scratch/uav-bput-nav")
EXP_DIR = WORKSPACE_ROOT / "experiments/phase6b/phase6b_pillar3_recovery_20261030_025800"
TRACES_DIR = EXP_DIR / "traces"

CAUTION_DIST_M = 0.50
CANONICAL_HORIZON_STEPS = 500
OBSTACLE_SAFETY_ENVELOPE_M = 0.080
BOUNDARY_SAFETY_ENVELOPE_M = 0.080
DEV_SEEDS = list(range(2000, 2050))
TARGET_FAILURE_SEEDS = [2001, 2005, 2006, 2022, 2042]


def setup_scenario_obstacles(env: CorridorSimEnv, scenario_id: str) -> List[Tuple[float, float, float]]:
    """Configures deterministic pillar layouts for diagnostic evaluation scenarios."""
    for o_id in env.obstacle_ids:
        p.removeBody(o_id, physicsClientId=env.client_id)
    env.obstacle_ids.clear()
    env.obstacles.clear()

    pillar_configs: List[Tuple[float, float, float]] = []
    if scenario_id == "Scenario_A_Centered":
        pillar_configs = [(3.0, 0.0, 0.15)]
    elif scenario_id == "Scenario_B_Left_Mirrored":
        pillar_configs = [(3.0, 0.10, 0.15)]
    elif scenario_id == "Scenario_B_Right_Mirrored":
        pillar_configs = [(3.0, -0.10, 0.15)]
    elif scenario_id == "Scenario_C_Wide_Right":
        pillar_configs = [(3.0, 0.25, 0.15)]
    elif scenario_id == "Scenario_C_Wide_Left":
        pillar_configs = [(3.0, -0.25, 0.15)]
    elif scenario_id == "Scenario_D_Standoff":
        pillar_configs = [(3.0, 0.0, 0.15)]
    elif scenario_id == "Scenario_E_Multi_Obstacle":
        pillar_configs = [
            (3.0, 0.0, 0.15),
            (5.5, 0.25, 0.15),
            (8.0, -0.25, 0.15),
        ]

    c = env.corridor
    for x_c, y_c, rad in pillar_configs:
        height = c.height
        col_shape = p.createCollisionShape(
            p.GEOM_CYLINDER, radius=rad, height=height, physicsClientId=env.client_id
        )
        vis_shape = p.createVisualShape(
            p.GEOM_CYLINDER, radius=rad, length=height, rgbaColor=[0.85, 0.35, 0.25, 1.0],
            physicsClientId=env.client_id
        )
        body_id = p.createMultiBody(
            baseMass=0, baseCollisionShapeIndex=col_shape, baseVisualShapeIndex=vis_shape,
            basePosition=[x_c, y_c, (c.z_floor + c.z_ceiling) / 2.0],
            physicsClientId=env.client_id
        )
        env.obstacle_ids.append(body_id)
        env.obstacles.append(Obstacle3D(center=np.array([x_c, y_c, (c.z_floor + c.z_ceiling) / 2.0]), radius=rad, height=height))

    return pillar_configs


def run_scenario_episode(
    env: CorridorSimEnv,
    supervisor: SafetySupervisor,
    controller: Any,
    scenario_id: str,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    obs, info = env.reset(seed=42)
    supervisor.loss_timer_s = 0.0
    supervisor.stalled_timer_s = 0.0
    supervisor.last_safe_action = VelocityAction(0.0, 0.0, 0.0, 0.0)
    if hasattr(controller, "reset"):
        controller.reset()

    pillar_configs = setup_scenario_obstacles(env, scenario_id)
    if scenario_id == "Scenario_D_Standoff":
        start_x, start_y, start_z = 2.45, 0.0, 1.0
        start_vx, start_vy, start_vz = 0.0, 0.0, 0.0
    else:
        start_x, start_y, start_z = 1.0, 0.0, 1.0
        start_vx, start_vy, start_vz = 0.40, 0.0, 0.0

    p.resetBasePositionAndOrientation(env.drone_id, [start_x, start_y, start_z], [0, 0, 0, 1], physicsClientId=env.client_id)
    p.resetBaseVelocity(env.drone_id, [start_vx, start_vy, start_vz], [0, 0, 0], physicsClientId=env.client_id)
    p.stepSimulation(physicsClientId=env.client_id)
    d_obs, _ = env._get_observations()
    obs = d_obs.to_vector()

    steps = 0
    done = False
    min_obs_overall = 10.0
    min_wall_overall = 10.0
    standoff_steps = 0
    throttled_steps = 0
    interventions = 0
    trace: List[Dict[str, Any]] = []

    while not done and steps < CANONICAL_HORIZON_STEPS:
        curr_pos = env._get_drone_position()
        curr_vel = env._get_drone_velocity()
        speed = float(np.linalg.norm(curr_vel))

        obs_c = min([o.distance_to_point(curr_pos) - env.corridor.drone_radius for o in env.obstacles]) if env.obstacles else 10.0
        wall_c = min(
            env.corridor.y_max - curr_pos[1] - env.corridor.drone_radius,
            curr_pos[1] - env.corridor.y_min - env.corridor.drone_radius,
            env.corridor.z_ceiling - curr_pos[2] - env.corridor.drone_radius,
            curr_pos[2] - env.corridor.z_floor - env.corridor.drone_radius,
        )
        min_obs_overall = min(min_obs_overall, obs_c)
        min_wall_overall = min(min_wall_overall, wall_c)

        rays = obs[0:12] * 4.0
        c_front = float(min(rays[0], rays[1], rays[2]))
        c_left = float(min(rays[1], rays[3], rays[5]))
        c_right = float(min(rays[2], rays[4], rays[6]))
        diff = float(c_left - c_right)
        wall_l = float(rays[5])
        wall_r = float(rays[6])

        raw_act, s_dir, _, _ = controller.predict_with_diagnostics(obs)

        if speed <= 0.08 and obs_c < 0.50:
            standoff_steps += 1

        clamped_act = VelocityAction(
            vx=float(np.clip(raw_act.vx, env.action_limits.min_fwd_vel_mps, env.action_limits.max_fwd_vel_mps)),
            vy=float(np.clip(raw_act.vy, -env.action_limits.max_lat_vel_mps, env.action_limits.max_lat_vel_mps)),
            vz=float(np.clip(raw_act.vz, -env.action_limits.max_vert_vel_mps, env.action_limits.max_vert_vel_mps)),
            yaw_rate=float(np.clip(raw_act.yaw_rate, -env.action_limits.max_yaw_rate_radps, env.action_limits.max_yaw_rate_radps)),
        )

        front_dist = compute_forward_clearance_m(obs, max_range_m=env.sensor_config.max_range_m)
        safe_act, s_state, s_reason = supervisor.filter_action(
            proposed_action=clamped_act,
            current_position=curr_pos,
            current_velocity=curr_vel,
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
            "speed_mps": round(speed, 3),
            "obs_c": round(obs_c, 3),
            "wall_c": round(wall_c, 3),
            "c_front": round(c_front, 3),
            "c_left": round(c_left, 3),
            "c_right": round(c_right, 3),
            "diff": round(diff, 3),
            "wall_l": round(wall_l, 3),
            "wall_r": round(wall_r, 3),
            "raw_vx": round(raw_act.vx, 3),
            "raw_vy": round(raw_act.vy, 3),
            "safe_vx": round(safe_act.vx, 3),
            "safe_vy": round(safe_act.vy, 3),
            "guidance_dir": s_dir,
            "safety_state": s_state.value,
            "safety_reason": s_reason,
        })

        norm_applied = safe_act.to_normalized(env.action_limits)
        obs, rew, term, trunc, info = env.step(norm_applied)
        steps += 1
        done = term or trunc

    final_pos = env._get_drone_position()
    final_vel = float(np.linalg.norm(env._get_drone_velocity()))
    term_reason = info.get("termination_reason", "unknown")
    progress = float(final_pos[0] - start_x)
    goal_dist = float(np.linalg.norm(final_pos - env.corridor.goal_position))

    is_success = (term_reason == "success") and (final_vel <= 0.35)
    is_obs_collision = (term_reason == "crash_obstacle") or (min_obs_overall <= 0.0)
    is_wall_collision = (term_reason == "crash_boundary") or (min_wall_overall <= 0.0)
    is_obs_viol = (min_obs_overall < OBSTACLE_SAFETY_ENVELOPE_M)
    is_wall_viol = (min_wall_overall < BOUNDARY_SAFETY_ENVELOPE_M)

    p_cleared = [int(final_pos[0] > (px + pr)) for (px, py, pr) in pillar_configs]

    summary = {
        "scenario_id": scenario_id,
        "success": int(is_success),
        "term_reason": term_reason,
        "steps": steps,
        "progress_m": round(progress, 3),
        "goal_dist_m": round(goal_dist, 3),
        "final_speed_mps": round(final_vel, 3),
        "min_obs_clearance_m": round(min_obs_overall, 4),
        "min_wall_clearance_m": round(min_wall_overall, 4),
        "obs_violations": int(is_obs_viol),
        "wall_violations": int(is_wall_viol),
        "obs_collisions": int(is_obs_collision),
        "wall_collisions": int(is_wall_collision),
        "interventions": interventions,
        "throttled_fraction_pct": round(throttled_steps / max(1, steps) * 100.0, 1),
        "standoff_duration_s": round(standoff_steps * env.dt, 2),
        "pillars_cleared": sum(p_cleared),
    }
    return summary, trace


def main():
    print("=================== PHASE 6B BENCHMARK EXECUTION ===================")
    corridor = Corridor3D.get_10m_benchmark()
    env = CorridorSimEnv(corridor=corridor)
    sup_config = SafetySupervisorConfig(obstacle_caution_dist_m=CAUTION_DIST_M)
    supervisor = SafetySupervisor(corridor=corridor, limits=env.action_limits, config=sup_config)

    # 1. Controllers
    baseline_controller = StatelessClearanceGuidanceController(
        base_model=str(WORKSPACE_ROOT / "models/checkpoints/bc_policy_best.pt"),
        activation_threshold_m=0.85,
        default_tie_breaker="LEFT",
    )
    candidate_controller = PostEvasionRecoveryController(
        base_model=str(WORKSPACE_ROOT / "models/checkpoints/bc_policy_best.pt"),
    )

    # Diagnostic Scenarios
    scenarios = [
        "Scenario_A_Centered",
        "Scenario_B_Left_Mirrored",
        "Scenario_B_Right_Mirrored",
        "Scenario_C_Wide_Right",
        "Scenario_C_Wide_Left",
    ]

    print("\n--- Diagnostic Scenarios ---")
    diag_results = []
    for sc in scenarios:
        b_sum, b_tr = run_scenario_episode(env, supervisor, baseline_controller, sc)
        c_sum, c_tr = run_scenario_episode(env, supervisor, candidate_controller, sc)
        diag_results.append({
            "scenario": sc,
            "baseline_success": b_sum["success"],
            "candidate_success": c_sum["success"],
            "baseline_steps": b_sum["steps"],
            "candidate_steps": c_sum["steps"],
            "baseline_min_obs": b_sum["min_obs_clearance_m"],
            "candidate_min_obs": c_sum["min_obs_clearance_m"],
            "baseline_min_wall": b_sum["min_wall_clearance_m"],
            "candidate_min_wall": c_sum["min_wall_clearance_m"],
        })
        print(f"{sc:26s}: Baseline Succ={b_sum['success']} (steps={b_sum['steps']}, obs={b_sum['min_obs_clearance_m']:.3f}m) | Candidate Succ={c_sum['success']} (steps={c_sum['steps']}, obs={c_sum['min_obs_clearance_m']:.3f}m)")

    # Save diagnostic CSV
    diag_csv_path = EXP_DIR / "diagnostic_comparison.csv"
    with open(diag_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(diag_results[0].keys()))
        writer.writeheader()
        writer.writerows(diag_results)

    # 2. Canonical Development Benchmark (Seeds 2000-2049)
    print("\n--- Canonical Development Seeds Benchmark (2000-2049) ---")
    paired_rows = []
    baseline_successes = 0
    candidate_successes = 0

    both_succeed = 0
    candidate_only = 0
    baseline_only = 0
    both_fail = 0

    b_min_obs = 10.0
    c_min_obs = 10.0
    b_min_wall = 10.0
    c_min_wall = 10.0

    b_progresses = []
    c_progresses = []
    b_durations = []
    c_durations = []
    b_throttled = []
    c_throttled = []

    b_collisions = 0
    c_collisions = 0
    b_violations = 0
    c_violations = 0

    b_p1_cleared = 0
    b_p2_cleared = 0
    b_p3_cleared = 0
    c_p1_cleared = 0
    c_p2_cleared = 0
    c_p3_cleared = 0

    for s in DEV_SEEDS:
        record_trace = (s in TARGET_FAILURE_SEEDS)
        b_sum, b_tr = run_episode(env, supervisor, baseline_controller, s, record_trace=record_trace)
        c_sum, c_tr = run_episode(env, supervisor, candidate_controller, s, record_trace=record_trace)

        if record_trace:
            with open(TRACES_DIR / f"seed_{s}_candidate_phase6b_telemetry.json", "w") as tf:
                json.dump({"summary": c_sum, "trace": c_tr}, tf, indent=2, cls=NumpyEncoder)

        if b_sum["success"] == 1:
            baseline_successes += 1
        if c_sum["success"] == 1:
            candidate_successes += 1

        if b_sum["success"] == 1 and c_sum["success"] == 1:
            both_succeed += 1
        elif b_sum["success"] == 0 and c_sum["success"] == 1:
            candidate_only += 1
        elif b_sum["success"] == 1 and c_sum["success"] == 0:
            baseline_only += 1
        else:
            both_fail += 1

        b_min_obs = min(b_min_obs, b_sum["min_obs_clearance_m"])
        c_min_obs = min(c_min_obs, c_sum["min_obs_clearance_m"])
        b_min_wall = min(b_min_wall, b_sum["min_wall_clearance_m"])
        c_min_wall = min(c_min_wall, c_sum["min_wall_clearance_m"])

        b_progresses.append(b_sum["progress_m"])
        c_progresses.append(c_sum["progress_m"])
        b_durations.append(b_sum["steps"] * env.dt)
        c_durations.append(c_sum["steps"] * env.dt)
        b_throttled.append(b_sum["throttled_fraction_pct"])
        c_throttled.append(c_sum["throttled_fraction_pct"])

        b_collisions += (b_sum["obs_collisions"] + b_sum["wall_collisions"])
        c_collisions += (c_sum["obs_collisions"] + c_sum["wall_collisions"])
        b_violations += (b_sum["obs_violations"] + b_sum["wall_violations"])
        c_violations += (c_sum["obs_violations"] + c_sum["wall_violations"])

        b_p1_cleared += b_sum["pillar_1_cleared"]
        b_p2_cleared += b_sum["pillar_2_cleared"]
        b_p3_cleared += b_sum["pillar_3_cleared"]

        c_p1_cleared += c_sum["pillar_1_cleared"]
        c_p2_cleared += c_sum["pillar_2_cleared"]
        c_p3_cleared += c_sum["pillar_3_cleared"]

        paired_rows.append({
            "seed": s,
            "baseline_outcome": b_sum["outcome"],
            "candidate_outcome": c_sum["outcome"],
            "baseline_success": b_sum["success"],
            "candidate_success": c_sum["success"],
            "baseline_steps": b_sum["steps"],
            "candidate_steps": c_sum["steps"],
            "baseline_progress_m": b_sum["progress_m"],
            "candidate_progress_m": c_sum["progress_m"],
            "baseline_goal_dist_m": b_sum["goal_dist_m"],
            "candidate_goal_dist_m": c_sum["goal_dist_m"],
            "baseline_min_obs_m": b_sum["min_obs_clearance_m"],
            "candidate_min_obs_m": c_sum["min_obs_clearance_m"],
            "baseline_min_wall_m": b_sum["min_wall_clearance_m"],
            "candidate_min_wall_m": c_sum["min_wall_clearance_m"],
            "baseline_pillars_cleared": b_sum["pillars_cleared"],
            "candidate_pillars_cleared": c_sum["pillars_cleared"],
        })

    env.close()

    # Save paired comparison CSV
    paired_csv_path = EXP_DIR / "paired_development_results_phase6b.csv"
    with open(paired_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(paired_rows[0].keys()))
        writer.writeheader()
        writer.writerows(paired_rows)

    summary_metrics = {
        "benchmark_seeds": "2000-2049",
        "num_seeds": len(DEV_SEEDS),
        "baseline_successes": baseline_successes,
        "baseline_success_rate": baseline_successes / len(DEV_SEEDS),
        "candidate_successes": candidate_successes,
        "candidate_success_rate": candidate_successes / len(DEV_SEEDS),
        "paired_contingency": {
            "both_succeed": both_succeed,
            "candidate_only_succeeds": candidate_only,
            "baseline_only_succeeds": baseline_only,
            "both_fail": both_fail,
        },
        "safety_metrics": {
            "baseline_collisions": b_collisions,
            "candidate_collisions": c_collisions,
            "baseline_envelope_violations": b_violations,
            "candidate_envelope_violations": c_violations,
            "baseline_min_obs_clearance_m": round(b_min_obs, 4),
            "candidate_min_obs_clearance_m": round(c_min_obs, 4),
            "baseline_min_wall_clearance_m": round(b_min_wall, 4),
            "candidate_min_wall_clearance_m": round(c_min_wall, 4),
        },
        "efficiency_metrics": {
            "baseline_mean_progress_m": round(float(np.mean(b_progresses)), 3),
            "candidate_mean_progress_m": round(float(np.mean(c_progresses)), 3),
            "baseline_mean_duration_s": round(float(np.mean(b_durations)), 2),
            "candidate_mean_duration_s": round(float(np.mean(c_durations)), 2),
            "baseline_mean_throttled_pct": round(float(np.mean(b_throttled)), 1),
            "candidate_mean_throttled_pct": round(float(np.mean(c_throttled)), 1),
        },
        "pillar_passage_rates": {
            "pillar_1": {
                "baseline": f"{b_p1_cleared}/{len(DEV_SEEDS)} ({b_p1_cleared/len(DEV_SEEDS)*100:.1f}%)",
                "candidate": f"{c_p1_cleared}/{len(DEV_SEEDS)} ({c_p1_cleared/len(DEV_SEEDS)*100:.1f}%)",
            },
            "pillar_2": {
                "baseline": f"{b_p2_cleared}/{len(DEV_SEEDS)} ({b_p2_cleared/len(DEV_SEEDS)*100:.1f}%)",
                "candidate": f"{c_p2_cleared}/{len(DEV_SEEDS)} ({c_p2_cleared/len(DEV_SEEDS)*100:.1f}%)",
            },
            "pillar_3": {
                "baseline": f"{b_p3_cleared}/{len(DEV_SEEDS)} ({b_p3_cleared/len(DEV_SEEDS)*100:.1f}%)",
                "candidate": f"{c_p3_cleared}/{len(DEV_SEEDS)} ({c_p3_cleared/len(DEV_SEEDS)*100:.1f}%)",
            },
        },
        "target_failures_outcomes": {
            s: {
                "baseline": [r for r in paired_rows if r["seed"] == s][0]["baseline_outcome"],
                "candidate": [r for r in paired_rows if r["seed"] == s][0]["candidate_outcome"],
                "baseline_prog_m": [r for r in paired_rows if r["seed"] == s][0]["baseline_progress_m"],
                "candidate_prog_m": [r for r in paired_rows if r["seed"] == s][0]["candidate_progress_m"],
                "candidate_p_cleared": [r for r in paired_rows if r["seed"] == s][0]["candidate_pillars_cleared"],
            }
            for s in TARGET_FAILURE_SEEDS
        }
    }

    summary_json_path = EXP_DIR / "summary.json"
    with open(summary_json_path, "w") as f:
        json.dump(summary_metrics, f, indent=2, cls=NumpyEncoder)

    print("\n=================== BENCHMARK SUMMARY ===================")
    print(f"Phase 6A Baseline Success:  {baseline_successes}/50 ({baseline_successes*2.0}%)")
    print(f"Phase 6B Candidate Success: {candidate_successes}/50 ({candidate_successes*2.0}%)")
    print(f"Paired outcomes: Both={both_succeed}, Cand-Only={candidate_only}, Base-Only={baseline_only}, Neither={both_fail}")
    print(f"Collisions: Baseline={b_collisions}, Candidate={c_collisions}")
    print(f"Envelope Violations: Baseline={b_violations}, Candidate={c_violations}")
    print(f"Min Obstacle Clearance: Baseline={b_min_obs:.4f}m, Candidate={c_min_obs:.4f}m")
    print(f"Min Wall Clearance:     Baseline={b_min_wall:.4f}m, Candidate={c_min_wall:.4f}m")
    print(f"Mean Progress:          Baseline={summary_metrics['efficiency_metrics']['baseline_mean_progress_m']}m, Candidate={summary_metrics['efficiency_metrics']['candidate_mean_progress_m']}m")
    print(f"Pillar 3 Passage Rate:  Baseline={summary_metrics['pillar_passage_rates']['pillar_3']['baseline']} -> Candidate={summary_metrics['pillar_passage_rates']['pillar_3']['candidate']}")
    print("\nTarget 5 Failure Seeds Comparison:")
    for s, info in summary_metrics["target_failures_outcomes"].items():
        print(f"  Seed {s}: {info['baseline']} (prog={info['baseline_prog_m']:.2f}m) -> {info['candidate']} (prog={info['candidate_prog_m']:.2f}m, p_cleared={info['candidate_p_cleared']})")


if __name__ == "__main__":
    main()

