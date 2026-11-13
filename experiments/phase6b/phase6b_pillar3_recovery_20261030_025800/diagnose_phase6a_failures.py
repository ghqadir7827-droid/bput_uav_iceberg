"""diagnose_phase6a_failures.py - Reconstruct and diagnose the 5 Pillar 3 failures.

Runs Phase 6A baseline on seeds 2000-2049, verifies exact reproduction of 42/50,
and captures comprehensive per-step telemetry for seeds 2001, 2005, 2006, 2022, 2042.
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
from core.guidance import GuidanceConfig, StatelessClearanceGuidanceArbitrator
from models.policy_net import NavigationPolicyMLP
from sim.corridor_env import CorridorSimEnv

WORKSPACE_ROOT = Path("/Users/ghulamqadir/.gemini/antigravity-ide/scratch/uav-bput-nav")
EXP_DIR = WORKSPACE_ROOT / "experiments/phase6b/phase6b_pillar3_recovery_20261030_025800"
TRACES_DIR = EXP_DIR / "traces"
TRACES_DIR.mkdir(parents=True, exist_ok=True)

CAUTION_DIST_M = 0.50
CANONICAL_HORIZON_STEPS = 500
OBSTACLE_SAFETY_ENVELOPE_M = 0.080
BOUNDARY_SAFETY_ENVELOPE_M = 0.080
DEV_SEEDS = list(range(2000, 2050))
TARGET_FAILURE_SEEDS = [2001, 2005, 2006, 2022, 2042]


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def run_episode(
    env: CorridorSimEnv,
    supervisor: SafetySupervisor,
    controller: StatelessClearanceGuidanceController,
    seed: int,
    max_steps: int = CANONICAL_HORIZON_STEPS,
    record_trace: bool = False,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    obs, info = env.reset(seed=seed)
    # Reset supervisor state (clear loss timer, stalled timer)
    supervisor.loss_timer_s = 0.0
    supervisor.stalled_timer_s = 0.0
    supervisor.last_safe_action = VelocityAction(0.0, 0.0, 0.0, 0.0)
    controller.reset()

    start_pos = env._get_drone_position()
    start_x = float(start_pos[0])

    pillar_configs = [(float(o.center[0]), float(o.center[1]), float(o.radius)) for o in env.obstacles]

    steps = 0
    done = False
    min_obs_overall = 10.0
    min_wall_overall = 10.0
    standoff_steps = 0
    throttled_steps = 0
    interventions = 0
    trace: List[Dict[str, Any]] = []

    pos_history = []

    while not done and steps < max_steps:
        curr_pos = env._get_drone_position()
        curr_vel = env._get_drone_velocity()
        speed = float(np.linalg.norm(curr_vel))

        pos_history.append((curr_pos[0], curr_pos[1]))

        # Rolling 50-step progress
        if len(pos_history) > 50:
            rolling_prog = float(curr_pos[0] - pos_history[-50][0])
        else:
            rolling_prog = float(curr_pos[0] - pos_history[0][0])

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

        # Obstacle geometry relative to drone
        obs_rel = []
        for idx, (px, py, pr) in enumerate(pillar_configs):
            rel_x = px - curr_pos[0]
            rel_y = py - curr_pos[1]
            dist_center = math.hypot(rel_x, rel_y)
            surface_dist = dist_center - pr - env.corridor.drone_radius
            passed = bool(curr_pos[0] > (px + pr))
            obs_rel.append({
                "pillar_idx": idx + 1,
                "center_x": px,
                "center_y": py,
                "rel_x": rel_x,
                "rel_y": rel_y,
                "surface_dist": surface_dist,
                "passed": passed,
            })

        if record_trace:
            trace.append({
                "step": steps,
                "elapsed_time_s": round(steps * env.dt, 3),
                "position": [round(float(curr_pos[0]), 3), round(float(curr_pos[1]), 3), round(float(curr_pos[2]), 3)],
                "lateral_displacement_y": round(float(curr_pos[1]), 3),
                "distance_to_centerline": round(abs(float(curr_pos[1])), 3),
                "velocity": [round(float(curr_vel[0]), 3), round(float(curr_vel[1]), 3), round(float(curr_vel[2]), 3)],
                "speed_mps": round(speed, 3),
                "rolling_progress_50step_m": round(rolling_prog, 3),
                "obs_clearance": round(obs_c, 3),
                "wall_clearance": round(wall_c, 3),
                "c_front": round(c_front, 3),
                "c_left": round(c_left, 3),
                "c_right": round(c_right, 3),
                "diff": round(diff, 3),
                "wall_l": round(wall_l, 3),
                "wall_r": round(wall_r, 3),
                "raw_action": [round(raw_act.vx, 3), round(raw_act.vy, 3), round(raw_act.vz, 3), round(raw_act.yaw_rate, 3)],
                "clamped_action": [round(clamped_act.vx, 3), round(clamped_act.vy, 3), round(clamped_act.vz, 3), round(clamped_act.yaw_rate, 3)],
                "safe_action": [round(safe_act.vx, 3), round(safe_act.vy, 3), round(safe_act.vz, 3), round(safe_act.yaw_rate, 3)],
                "guidance_dir": s_dir,
                "safety_state": s_state.value,
                "safety_reason": s_reason,
                "obstacles": obs_rel,
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

    if is_success:
        outcome = "SUCCESS"
    elif is_obs_collision:
        outcome = "CRASH_OBSTACLE"
    elif is_wall_collision:
        outcome = "CRASH_BOUNDARY"
    elif is_obs_viol or is_wall_viol:
        outcome = "SAFETY_VIOLATION"
    elif p_cleared == [1, 1, 1]:
        if goal_dist < 1.0:
            outcome = "GOAL_ALIGNMENT_TIMEOUT"
        else:
            outcome = "TIMEOUT_SLOW_TRANSIT"
    elif p_cleared == [1, 1, 0]:
        outcome = "PROGRESS_STALL_PILLAR_3"
    elif p_cleared == [1, 0, 0]:
        outcome = "PROGRESS_STALL_PILLAR_2"
    else:
        outcome = "PROGRESS_STALL_PILLAR_1"

    summary = {
        "seed": seed,
        "success": int(is_success),
        "outcome": outcome,
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
        "pillar_1_cleared": p_cleared[0] if len(p_cleared) > 0 else 0,
        "pillar_2_cleared": p_cleared[1] if len(p_cleared) > 1 else 0,
        "pillar_3_cleared": p_cleared[2] if len(p_cleared) > 2 else 0,
        "pillar_configs": pillar_configs,
    }

    return summary, trace


def main():
    print("=== Phase 6A Baseline Reproduction & Failure Telemetry Capture ===")
    corridor = Corridor3D.get_10m_benchmark()
    env = CorridorSimEnv(corridor=corridor)

    sup_config = SafetySupervisorConfig(obstacle_caution_dist_m=CAUTION_DIST_M)
    supervisor = SafetySupervisor(corridor=corridor, limits=env.action_limits, config=sup_config)

    controller = StatelessClearanceGuidanceController(
        base_model=str(WORKSPACE_ROOT / "models/checkpoints/bc_policy_best.pt"),
        activation_threshold_m=0.85,
        default_tie_breaker="LEFT",
    )

    results = []
    successes = 0

    for s in DEV_SEEDS:
        record_trace = (s in TARGET_FAILURE_SEEDS)
        summary, trace = run_episode(
            env=env,
            supervisor=supervisor,
            controller=controller,
            seed=s,
            record_trace=record_trace,
        )
        results.append(summary)
        if summary["success"] == 1:
            successes += 1

        if record_trace:
            trace_path = TRACES_DIR / f"seed_{s}_phase6a_telemetry.json"
            with open(trace_path, "w") as tf:
                json.dump({"summary": summary, "trace": trace}, tf, indent=2, cls=NumpyEncoder)
            print(f"Captured detailed telemetry for failure Seed {s}: outcome={summary['outcome']}, progress={summary['progress_m']}m, steps={summary['steps']}")

    env.close()

    print(f"\nPhase 6A Development Benchmark Result: {successes}/50 ({successes*2.0}%)")
    assert successes == 42, f"Baseline reproduction failed! Expected 42/50, got {successes}/50"
    print("Baseline reproduction verified: exactly 42/50 successes (84.0%)!")

    # Check the 5 target failure seeds
    for s in TARGET_FAILURE_SEEDS:
        matching = [r for r in results if r["seed"] == s][0]
        print(f"Seed {s}: outcome={matching['outcome']}, progress={matching['progress_m']}m, pillars_cleared={matching['pillars_cleared']}")


if __name__ == "__main__":
    main()
