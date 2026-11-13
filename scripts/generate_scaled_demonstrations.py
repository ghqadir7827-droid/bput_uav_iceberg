"""generate_scaled_demonstrations.py - Scaled 65cm UAV Demonstration Dataset Generator.

Generates verified state-action trajectories from the scripted APF expert controller in the
Level 2 standard scaled environment (Corridor 3.8m x 2.5m x 14.0m, Drone R=0.325m).

Strict quality filter:
- Trajectory must reach goal successfully.
- Surface-to-surface obstacle clearance >= 0.12 m at every step.
- Surface-to-surface wall clearance >= 0.12 m at every step.
- Zero collisions with obstacles or corridor boundaries.
- No timeout.

Records deployable observations (23 floats) and normalized/physical actions (4 floats).
Saves separate train (200 episodes) and validation (50 episodes) datasets.
"""

import argparse
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Dict, List
import numpy as np

from controllers.apf_expert import ScriptedAPFExpertController
from core.contracts import ActionLimits, VelocityAction
from core.physical_dimensions import DifficultyLevel, ScaledEnvironmentFactory
from sim.corridor_env import CorridorSimEnv


def generate_split(
    scenario_cfg: Any,
    target_episodes: int,
    start_seed: int,
    max_steps_per_episode: int = 1000,
    min_obstacle_clearance_thresh_m: float = 0.12,
    min_boundary_distance_thresh_m: float = 0.12,
    output_npz: str = "data/demonstrations/bc_train_scaled.npz",
    audit_json: str = "data/demonstrations/bc_train_scaled_audit.json",
    split_name: str = "train",
) -> Dict[str, Any]:
    print("=" * 80)
    print(f">>> GENERATING SCALED 65cm UAV DEMONSTRATIONS [{split_name.upper()}] <<<")
    print(f"Target Episodes:      {target_episodes}")
    print(f"Start Seed:           {start_seed}")
    print(f"Safety Filter Thresh: Obs Clearance >= {min_obstacle_clearance_thresh_m}m, Wall Clearance >= {min_boundary_distance_thresh_m}m")
    print(f"Output File:          {output_npz}")
    print("=" * 80)

    env = CorridorSimEnv(scenario=scenario_cfg, gui=False, seed=start_seed)
    expert = ScriptedAPFExpertController(
        corridor=env.corridor,
        cruise_speed=0.60,
        k_att=1.0,
        k_rep_obs=0.50,
        d_rep_obs=1.40,
        k_circ=0.60,
        k_wall=1.8,
        d_rep_wall=0.55,
    )

    collected_obs: List[np.ndarray] = []
    collected_acts: List[np.ndarray] = []
    collected_starts: List[bool] = []
    collected_ep_ids: List[int] = []
    collected_step_idx: List[int] = []

    successful_episodes_meta: List[Dict[str, Any]] = []
    rejected_episodes_meta: List[Dict[str, Any]] = []

    current_seed = start_seed
    accepted_count = 0
    attempted_count = 0

    print(f"\n{'Seed':<7} | {'Status':<12} | {'Steps':<6} | {'Min Obs (m)':<12} | {'Min Wall (m)':<12} | {'Decision':<12}")
    print("-" * 75)

    while accepted_count < target_episodes:
        attempted_count += 1
        s = current_seed
        current_seed += 1

        obs, info = env.reset(seed=s)
        expert.update_environment(env.corridor, env.obstacles)

        ep_obs: List[np.ndarray] = []
        ep_acts: List[np.ndarray] = []

        min_obs_clear = float("inf")
        min_wall_clear = float("inf")
        flight_distance = 0.0
        prev_pos = env._get_drone_position()

        episode_done = False
        termination_status = "unknown"

        for step in range(1, max_steps_per_episode + 1):
            priv = info["privileged_state"]
            expert.set_state(priv.global_position, priv.global_velocity)

            action = expert.compute_privileged_expert_action(
                position=priv.global_position,
                velocity=priv.global_velocity,
                obstacles=env.obstacles,
            )

            deployable_obs_vec = obs.astype(np.float32)
            action_vec = np.array([action.vx, action.vy, action.vz, action.yaw_rate], dtype=np.float32)

            ep_obs.append(deployable_obs_vec)
            ep_acts.append(action_vec)

            norm_action = action.to_normalized(env.action_limits)
            obs, rew, term, trunc, info = env.step(norm_action)

            curr_pos = env._get_drone_position()
            flight_distance += float(np.linalg.norm(curr_pos - prev_pos))
            prev_pos = curr_pos

            d_obs = float(info.get("obstacle_clearance", 10.0))
            d_wall = float(info.get("boundary_distance", 10.0))
            min_obs_clear = min(min_obs_clear, d_obs)
            min_wall_clear = min(min_wall_clear, d_wall)

            if term or trunc:
                termination_status = info.get("termination_reason", "unknown")
                episode_done = True
                break

        is_success = (termination_status == "success")
        is_safe_obs = (min_obs_clear >= min_obstacle_clearance_thresh_m)
        is_safe_wall = (min_wall_clear >= min_boundary_distance_thresh_m)
        is_valid_trajectory = (len(ep_obs) >= 80) and (len(ep_obs) <= max_steps_per_episode)

        passed_quality_filter = is_success and is_safe_obs and is_safe_wall and is_valid_trajectory

        ep_meta = {
            "seed": s,
            "episode_id": attempted_count,
            "status": termination_status,
            "steps": len(ep_obs),
            "flight_time_s": round(len(ep_obs) * env.dt, 2),
            "distance_m": round(flight_distance, 2),
            "min_obstacle_clearance_m": round(min_obs_clear, 4),
            "min_boundary_distance_m": round(min_wall_clear, 4),
            "final_position": [round(float(x), 3) for x in curr_pos],
            "passed_filter": passed_quality_filter,
        }

        if passed_quality_filter:
            accepted_count += 1
            decision = f"ACCEPTED ({accepted_count}/{target_episodes})"
            successful_episodes_meta.append(ep_meta)

            for idx, (o, a) in enumerate(zip(ep_obs, ep_acts)):
                collected_obs.append(o)
                collected_acts.append(a)
                collected_starts.append(idx == 0)
                collected_ep_ids.append(accepted_count)
                collected_step_idx.append(idx)
        else:
            decision = "REJECTED"
            ep_meta["rejection_reason"] = (
                f"status={termination_status}, obs_clear={min_obs_clear:.4f}m, wall_clear={min_wall_clear:.4f}m"
            )
            rejected_episodes_meta.append(ep_meta)

        if accepted_count % 20 == 0 or not passed_quality_filter or accepted_count <= 5 or accepted_count == target_episodes:
            print(f"{s:<7} | {termination_status:<12} | {len(ep_obs):<6} | {min_obs_clear:<12.4f} | {min_wall_clear:<12.4f} | {decision:<12}")

    env.close()

    obs_arr = np.array(collected_obs, dtype=np.float32)
    act_arr = np.array(collected_acts, dtype=np.float32)
    starts_arr = np.array(collected_starts, dtype=bool)
    ep_ids_arr = np.array(collected_ep_ids, dtype=np.int32)
    step_idx_arr = np.array(collected_step_idx, dtype=np.int32)

    output_path = Path(output_npz)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        observations=obs_arr,
        actions=act_arr,
        episode_starts=starts_arr,
        episode_ids=ep_ids_arr,
        step_indices=step_idx_arr,
    )

    with open(output_path, "rb") as f:
        file_sha256 = hashlib.sha256(f.read()).hexdigest()

    metadata = {
        "dataset_name": f"scaled_expert_apf_level_2_{split_name}",
        "scenario": "level_2_standard",
        "split": split_name,
        "drone_radius_m": 0.325,
        "drone_tip_to_tip_diagonal_m": 0.65,
        "corridor_dimensions": [env.corridor.x_end, env.corridor.width, env.corridor.height],
        "start_position": list(env.corridor.start_position),
        "goal_position": list(env.corridor.goal_position),
        "teacher_controller_id": expert.metadata.controller_id,
        "total_episodes_accepted": accepted_count,
        "total_episodes_attempted": attempted_count,
        "total_transitions": int(len(obs_arr)),
        "observation_dim": int(obs_arr.shape[1]),
        "action_dim": int(act_arr.shape[1]),
        "seeds_used": [m["seed"] for m in successful_episodes_meta],
        "action_limits": {
            "vx": [-0.2, 0.8],
            "vy": [-0.6, 0.6],
            "vz": [-0.4, 0.4],
            "yaw_rate": [-0.6, 0.6],
        },
        "quality_filter_criteria": {
            "min_obstacle_clearance_m": min_obstacle_clearance_thresh_m,
            "min_boundary_distance_m": min_boundary_distance_thresh_m,
            "max_steps": max_steps_per_episode,
        },
        "file_sha256": file_sha256,
        "generation_timestamp_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    audit_path = Path(audit_json)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_data = {
        "metadata": metadata,
        "summary": {
            "attempted_episodes": attempted_count,
            "accepted_episodes": accepted_count,
            "rejected_episodes": len(rejected_episodes_meta),
            "filter_acceptance_rate_pct": round((accepted_count / attempted_count) * 100.0, 1),
            "mean_steps_per_episode": round(float(np.mean([m["steps"] for m in successful_episodes_meta])), 1),
            "mean_flight_time_s": round(float(np.mean([m["flight_time_s"] for m in successful_episodes_meta])), 2),
            "mean_distance_m": round(float(np.mean([m["distance_m"] for m in successful_episodes_meta])), 2),
            "min_obstacle_clearance_overall_m": round(float(min(m["min_obstacle_clearance_m"] for m in successful_episodes_meta)), 4),
            "min_wall_clearance_overall_m": round(float(min(m["min_boundary_distance_m"] for m in successful_episodes_meta)), 4),
            "action_vx_mean": round(float(np.mean(act_arr[:, 0])), 4),
            "action_vx_std": round(float(np.std(act_arr[:, 0])), 4),
            "action_vy_mean": round(float(np.mean(act_arr[:, 1])), 4),
            "action_vy_std": round(float(np.std(act_arr[:, 1])), 4),
        },
        "accepted_episodes": successful_episodes_meta,
        "rejected_episodes": rejected_episodes_meta,
    }

    def _json_serial(o):
        if isinstance(o, (np.floating, np.float32, np.float64)):
            return float(o)
        if isinstance(o, (np.integer, np.int32, np.int64)):
            return int(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)

    with open(audit_path, "w") as f:
        json.dump(audit_data, f, indent=2, default=_json_serial)

    print("\n" + "=" * 80)
    print(f">>> {split_name.upper()} DATASET GENERATION COMPLETE <<<")
    print(f"Accepted Clean Episodes:     {accepted_count} / {attempted_count} ({audit_data['summary']['filter_acceptance_rate_pct']}%)")
    print(f"Total Transitions:           {len(obs_arr)}")
    print(f"Mean Steps per Episode:      {audit_data['summary']['mean_steps_per_episode']}")
    print(f"Overall Min Obs Clearance:   {audit_data['summary']['min_obstacle_clearance_overall_m']} m")
    print(f"Overall Min Wall Clearance:  {audit_data['summary']['min_wall_clearance_overall_m']} m")
    print(f"SHA-256 Checksum:            {file_sha256}")
    print(f"Dataset Archive:             {output_path.resolve()}")
    print("=" * 80)

    return audit_data


def main():
    parser = argparse.ArgumentParser(description="Generate Scaled 65cm UAV Demonstrations")
    parser.add_argument("--num-train", type=int, default=200, help="Number of training episodes")
    parser.add_argument("--num-val", type=int, default=50, help="Number of validation episodes")
    parser.add_argument("--train-start-seed", type=int, default=30000, help="Start seed for training split")
    parser.add_argument("--val-start-seed", type=int, default=30200, help="Start seed for val split")
    parser.add_argument("--min-clearance", type=float, default=0.12, help="Min surface-to-surface clearance threshold (m)")
    args = parser.parse_args()

    cfg = ScaledEnvironmentFactory.get_environment_config(DifficultyLevel.LEVEL_2_STANDARD)

    # 1. Train Split
    generate_split(
        scenario_cfg=cfg,
        target_episodes=args.num_train,
        start_seed=args.train_start_seed,
        min_obstacle_clearance_thresh_m=args.min_clearance,
        min_boundary_distance_thresh_m=args.min_clearance,
        output_npz="data/demonstrations/bc_train_scaled.npz",
        audit_json="data/demonstrations/bc_train_scaled_audit.json",
        split_name="train",
    )

    # 2. Validation Split
    generate_split(
        scenario_cfg=cfg,
        target_episodes=args.num_val,
        start_seed=args.val_start_seed,
        min_obstacle_clearance_thresh_m=args.min_clearance,
        min_boundary_distance_thresh_m=args.min_clearance,
        output_npz="data/demonstrations/bc_val_scaled.npz",
        audit_json="data/demonstrations/bc_val_scaled_audit.json",
        split_name="val",
    )


if __name__ == "__main__":
    main()

