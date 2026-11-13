"""diagnose_phase4f0_seeds.py - Phase 4F-0 Seed Diversity & Repeated Route Diagnostic.

Performs controlled experiments on seeds [1, 20, 100, 2000, 8000] (plus seed 1 replay control)
under the exact interactive viewer PPO configuration.
Captures full trajectory, raw policy action proposals, supervisor filtering,
rangefinder readings, and obstacle bypass decisions.
"""

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np
import torch

from controllers.ppo_policy import PPOPolicyController
from controllers.safety_supervisor import SafetyState, SafetySupervisor
from core.contracts import ActionLimits, VelocityAction, compute_forward_clearance_m
from core.geometry import Corridor3D, Obstacle3D, TerminationReason
from sim.corridor_env import CorridorSimEnv


def get_file_sha256(filepath: str) -> str:
    p = Path(filepath)
    if not p.exists():
        return "NOT_FOUND"
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def run_episode(
    seed: int,
    controller: PPOPolicyController,
    scenario_id: str = "10m",
) -> Dict[str, Any]:
    env = CorridorSimEnv(scenario=scenario_id, gui=False, seed=seed)
    supervisor = SafetySupervisor(corridor=env.corridor)

    obs_vec, info = env.reset(seed=seed)
    priv_state = info["privileged_state"]

    # Initial state recording
    init_drone_pos = priv_state.global_position.tolist()
    init_drone_vel = priv_state.global_velocity.tolist()
    init_drone_quat = priv_state.global_orientation_quat.tolist()
    _, roll, pitch, yaw = env._get_drone_orientation()
    init_drone_rpy = [float(roll), float(pitch), float(yaw)]

    obstacles_info = []
    for idx, obs in enumerate(env.obstacles):
        # Gap analysis
        y_c = float(obs.center[1])
        r = float(obs.radius)
        left_gap = float(env.corridor.y_max - (y_c + r))
        right_gap = float((y_c - r) - env.corridor.y_min)
        obstacles_info.append({
            "id": idx + 1,
            "center": [float(x) for x in obs.center],
            "radius": r,
            "height": float(obs.height),
            "y_offset": y_c,
            "left_gap_m": round(left_gap, 4),
            "right_gap_m": round(right_gap, 4),
            "wider_side": "left" if left_gap > right_gap else ("right" if right_gap > left_gap else "equal")
        })

    initial_rays = [float(x) * 4.0 for x in obs_vec[0:12]]
    initial_obs_23 = [float(x) for x in obs_vec]

    # Stepping through episode
    steps_data = []
    trajectory = [init_drone_pos]
    done = False
    step_count = 0
    interventions = 0
    termination_reason = "timeout"

    current_info = info
    while not done and step_count < env.corridor.max_steps:
        priv = current_info["privileged_state"]
        pos = priv.global_position.copy()
        vel = priv.global_velocity.copy()

        # 1. Raw Policy Action Proposal
        raw_action = controller.predict(obs_vec)

        # Also get raw unbounded network logits if available
        with torch.no_grad():
            t_obs = torch.as_tensor(obs_vec, dtype=torch.float32).unsqueeze(0)
            raw_unbounded = controller.model.forward(t_obs, bounded=False).squeeze(0).numpy().tolist()
            raw_bounded = controller.model.forward(t_obs, bounded=True).squeeze(0).numpy().tolist()

        # 2. Multi-beam front clearance
        front_dist = compute_forward_clearance_m(obs_vec, max_range_m=env.sensor_config.max_range_m)

        # 3. Safety Supervisor Filtering
        safe_action, s_state, s_reason = supervisor.filter_action(
            proposed_action=raw_action,
            current_position=pos,
            current_velocity=vel,
            min_front_range=front_dist,
            dt=env.dt,
            is_localization_healthy=True,
            telemetry_age_s=0.0,
            observation=obs_vec,
        )

        if s_state != SafetyState.NOMINAL:
            interventions += 1

        # 4. Step physics
        norm_action = safe_action.to_normalized(env.action_limits)
        obs_vec, reward, terminated, truncated, current_info = env.step(norm_action)
        step_count += 1

        new_pos = current_info["privileged_state"].global_position.tolist()
        trajectory.append(new_pos)

        steps_data.append({
            "step": step_count,
            "pos": [round(x, 4) for x in pos.tolist()],
            "vel": [round(x, 4) for x in vel.tolist()],
            "front_clearance_m": round(float(front_dist), 4),
            "rays_m": [round(float(x) * 4.0, 3) for x in obs_vec[0:12]],
            "raw_action": {
                "vx": round(float(raw_action.vx), 4),
                "vy": round(float(raw_action.vy), 4),
                "vz": round(float(raw_action.vz), 4),
                "yaw_rate": round(float(raw_action.yaw_rate), 4),
            },
            "raw_unbounded_head": [round(x, 4) for x in raw_unbounded],
            "raw_bounded_head": [round(x, 4) for x in raw_bounded],
            "safe_action": {
                "vx": round(float(safe_action.vx), 4),
                "vy": round(float(safe_action.vy), 4),
                "vz": round(float(safe_action.vz), 4),
                "yaw_rate": round(float(safe_action.yaw_rate), 4),
            },
            "safety_state": s_state.value,
            "safety_reason": s_reason,
            "obs_clearance_m": round(float(current_info["privileged_state"].obstacle_clearance), 4),
            "wall_clearance_m": round(float(current_info["privileged_state"].corridor_boundary_distance), 4),
        })

        if terminated or truncated:
            done = True
            termination_reason = current_info.get("termination_reason", "unknown")

    env.close()

    # Analyze route around each obstacle
    # Longitudinal trigger ranges:
    # Obstacle 1: x in [2.5, 3.5]
    # Obstacle 2: x in [4.5, 5.5]
    # Obstacle 3: x in [6.5, 7.5]
    encounters = []
    for obs in obstacles_info:
        obs_id = obs["id"]
        obs_x = obs["center"][0]
        obs_y = obs["center"][1]
        r = obs["radius"]

        # Points in trajectory near this obstacle
        near_pts = [p for p in trajectory if abs(p[0] - obs_x) <= 0.4]
        if near_pts:
            # Average Y position when passing or encountering
            avg_y = float(np.mean([p[1] for p in near_pts]))
            min_y = float(min(p[1] for p in near_pts))
            max_y = float(max(p[1] for p in near_pts))

            # Selected side relative to obstacle center
            # If avg_y > obs_y -> Left bypass (+Y is left)
            # If avg_y < obs_y -> Right bypass (-Y is right)
            chosen_side = "left" if avg_y > obs_y else "right"
            
            # Check feasibility
            # Drone radius = 0.15m, diameter = 0.30m
            # Buffer required = 0.25m wall, 0.30m obstacle -> nominal clearance
            left_feasible = obs["left_gap_m"] >= 0.45
            right_feasible = obs["right_gap_m"] >= 0.45

            if left_feasible and right_feasible:
                feasibility = "both"
            elif left_feasible and not right_feasible:
                feasibility = "left_only"
            elif not left_feasible and right_feasible:
                feasibility = "right_only"
            else:
                feasibility = "neither"

            # Check if drone chose constrained side
            chose_constrained = (chosen_side == "left" and not left_feasible) or (chosen_side == "right" and not right_feasible)

            # Check supervisor intervention during encounter
            encounter_steps = [s for s in steps_data if abs(s["pos"][0] - obs_x) <= 0.8]
            sup_active = any(s["safety_state"] != "nominal" for s in encounter_steps)
            sup_altered_route = any(
                abs(s["raw_action"]["vy"] - s["safe_action"]["vy"]) > 0.05
                for s in encounter_steps
            )

            encounters.append({
                "obstacle_id": obs_id,
                "obs_center": obs["center"],
                "obs_y": obs_y,
                "left_gap_m": obs["left_gap_m"],
                "right_gap_m": obs["right_gap_m"],
                "wider_side": obs["wider_side"],
                "feasibility": feasibility,
                "chosen_side": chosen_side,
                "chose_constrained": chose_constrained,
                "avg_drone_y": round(avg_y, 4),
                "supervisor_intervened": sup_active,
                "supervisor_altered_lateral": sup_altered_route,
            })
        else:
            encounters.append({
                "obstacle_id": obs_id,
                "obs_center": obs["center"],
                "obs_y": obs_y,
                "status": "not_reached"
            })

    return {
        "seed": seed,
        "scenario": scenario_id,
        "termination_reason": termination_reason,
        "total_steps": step_count,
        "interventions_count": interventions,
        "initial_drone_pos": init_drone_pos,
        "initial_drone_vel": init_drone_vel,
        "initial_drone_rpy": init_drone_rpy,
        "obstacles": obstacles_info,
        "initial_rays_m": [round(x, 3) for x in initial_rays],
        "initial_obs_23": [round(x, 4) for x in initial_obs_23],
        "trajectory": [[round(c, 4) for c in pt] for pt in trajectory],
        "encounters": encounters,
        "steps": steps_data,
    }


def main():
    checkpoint_path = "models/checkpoints/ppo_policy_best.pt"
    sha = get_file_sha256(checkpoint_path)
    print(f"Loading PPO controller from {checkpoint_path} (SHA-256: {sha})...")
    ctrl = PPOPolicyController(checkpoint_path=checkpoint_path)
    assert ctrl.metadata.is_implemented, "PPO controller failed to load!"

    test_seeds = [1, 1, 20, 100, 2000, 8000] # Seed 1 twice for replay control
    runs = []

    for i, s in enumerate(test_seeds):
        tag = f"seed_{s}_replay" if i == 1 else f"seed_{s}"
        print(f"Running simulation for {tag}...")
        res = run_episode(s, ctrl, scenario_id="10m")
        res["tag"] = tag
        res["checkpoint_sha256"] = sha
        res["checkpoint_path"] = checkpoint_path
        runs.append(res)

    # Save detailed JSON
    out_json = Path("reports/phase4f0_seed_diversity_metrics.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    
    # Filter step details for lightweight metrics JSON, keep full data in memory for analysis
    metrics_summary = {
        "checkpoint_path": checkpoint_path,
        "checkpoint_sha256": sha,
        "runs": [
            {
                "tag": r["tag"],
                "seed": r["seed"],
                "termination_reason": r["termination_reason"],
                "total_steps": r["total_steps"],
                "interventions_count": r["interventions_count"],
                "initial_drone_pos": r["initial_drone_pos"],
                "obstacles": r["obstacles"],
                "initial_rays_m": r["initial_rays_m"],
                "encounters": r["encounters"],
                "trajectory_sample": r["trajectory"][::10], # 10x subsample for compactness
            }
            for r in runs
        ]
    }

    with open(out_json, "w") as f:
        json.dump(metrics_summary, f, indent=2)
    print(f"Wrote metrics summary to {out_json}")

    # Build CSV for route comparison
    csv_path = Path("reports/phase4f0_route_comparison.csv")
    with open(csv_path, "w") as f:
        f.write("run_tag,seed,obstacle_id,obs_x,obs_y,left_gap_m,right_gap_m,wider_side,feasibility,chosen_side,chose_constrained,avg_drone_y,sup_intervened,sup_altered_lat\n")
        for r in runs:
            for enc in r["encounters"]:
                if "status" in enc and enc["status"] == "not_reached":
                    f.write(f"{r['tag']},{r['seed']},{enc['obstacle_id']},{enc['obs_center'][0]},{enc['obs_y']},,,,not_reached,,,,,\n")
                else:
                    f.write(
                        f"{r['tag']},{r['seed']},{enc['obstacle_id']},"
                        f"{enc['obs_center'][0]},{enc['obs_y']:.3f},"
                        f"{enc['left_gap_m']:.3f},{enc['right_gap_m']:.3f},"
                        f"{enc['wider_side']},{enc['feasibility']},"
                        f"{enc['chosen_side']},{enc['chose_constrained']},"
                        f"{enc['avg_drone_y']:.3f},{enc['supervisor_intervened']},"
                        f"{enc['supervisor_altered_lateral']}\n"
                    )
    print(f"Wrote route comparison CSV to {csv_path}")

    # Print summary table to console
    print("\n" + "="*80)
    print("PHASE 4F-0 ROUTE SELECTION SUMMARY")
    print("="*80)
    for r in runs:
        print(f"\nRun: {r['tag']} (Outcome: {r['termination_reason']}, Steps: {r['total_steps']}, Interventions: {r['interventions_count']})")
        print(f"  Initial Pos: {r['initial_drone_pos']}")
        for enc in r["encounters"]:
            if "status" in enc and enc["status"] == "not_reached":
                print(f"  Obstacle {enc['obstacle_id']} (X={enc['obs_center'][0]:.1f}): NOT REACHED")
            else:
                warn = " [CONSTRAINED!]" if enc["chose_constrained"] else ""
                print(f"  Obstacle {enc['obstacle_id']} (X={enc['obs_center'][0]:.1f}, Y_obs={enc['obs_y']:+.3f}): "
                      f"LeftGap={enc['left_gap_m']:.2f}m, RightGap={enc['right_gap_m']:.2f}m (Wider: {enc['wider_side']}) -> "
                      f"Chosen: {enc['chosen_side'].upper()}{warn} (Drone Y={enc['avg_drone_y']:+.3f}, SupInterv={enc['supervisor_intervened']})")

    # Policy action analysis: check raw output bias
    print("\n" + "="*80)
    print("RAW PPO LATERAL ACTION ANALYSIS (vy)")
    print("="*80)
    for r in runs:
        raw_vys = [s["raw_action"]["vy"] for s in r["steps"]]
        mean_vy = float(np.mean(raw_vys))
        pos_vy_pct = float(np.mean([1 if vy > 0.05 else 0 for vy in raw_vys]) * 100)
        neg_vy_pct = float(np.mean([1 if vy < -0.05 else 0 for vy in raw_vys]) * 100)
        neutral_pct = float(np.mean([1 if abs(vy) <= 0.05 else 0 for vy in raw_vys]) * 100)
        print(f"Run {r['tag']:18s}: Mean vy = {mean_vy:+.4f} m/s | Propose Left (+vy): {pos_vy_pct:5.1f}% | Propose Right (-vy): {neg_vy_pct:5.1f}% | Neutral: {neutral_pct:5.1f}%")


if __name__ == "__main__":
    main()
