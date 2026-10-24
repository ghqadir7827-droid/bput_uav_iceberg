"""generate_demonstrations.py - Pilot Expert Demonstration Dataset Generator.

Generates verified state-action trajectories from the scripted APF expert controller.
Strictly separates privileged teacher geometric guidance from deployable student observations:
- Teacher uses: exact obstacle centers and corridor boundaries for potential gradient calculation.
- Demonstrations record ONLY: the 23-float DeployableObservation vector (rangefinder rays, relative
  goal in body FLU, body velocity, attitude trigonometry, and yaw rate).
- Quality Filtering: rigorously rejects any collisions, boundary infractions, or abnormal rollouts.
"""

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List
import numpy as np

from controllers.apf_expert import ScriptedAPFExpertController
from core.contracts import ActionLimits, DeployableObservation, VelocityAction
from sim.corridor_env import CorridorSimEnv


def generate_pilot_demonstrations(
    scenario: str = "10m",
    target_episodes: int = 50,
    start_seed: int = 1000,
    max_steps_per_episode: int = 500,
    min_obstacle_clearance_thresh_m: float = 0.08,
    min_boundary_distance_thresh_m: float = 0.08,
    output_npz: str = "data/demonstrations/pilot_demonstrations_10m.npz",
    audit_json: str = "data/demonstrations/pilot_demonstrations_audit.json",
) -> Dict[str, Any]:
    print("=" * 80)
    print(">>> GENERATING PILOT EXPERT DEMONSTRATION DATASET <<<")
    print(f"Scenario:             {scenario}")
    print(f"Target Episodes:      {target_episodes}")
    print(f"Start Seed:           {start_seed}")
    print(f"Safety Filter Thresh: Obs Clearance >= {min_obstacle_clearance_thresh_m}m, Wall Clearance >= {min_boundary_distance_thresh_m}m")
    print(f"Output File:          {output_npz}")
    print("=" * 80)

    env = CorridorSimEnv(scenario=scenario, gui=False, seed=start_seed)
    expert = ScriptedAPFExpertController(corridor=env.corridor)

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

    print(f"\n{'Seed':<6} | {'Status':<12} | {'Steps':<6} | {'Min Obs Clear':<14} | {'Min Wall Clear':<15} | {'Decision':<10}")
    print("-" * 75)

    while accepted_count < target_episodes:
        attempted_count += 1
        s = current_seed
        current_seed += 1

        obs, info = env.reset(seed=s)
        expert.update_environment(env.corridor, env.obstacles)

        ep_obs: List[np.ndarray] = []
        ep_acts: List[np.ndarray] = []

        min_obs_clear = 10.0
        min_wall_clear = 10.0
        flight_distance = 0.0
        prev_pos = env._get_drone_position()

        episode_done = False
        termination_status = "unknown"

        for step in range(1, max_steps_per_episode + 1):
            priv = info["privileged_state"]
            expert.set_state(priv.global_position, priv.global_velocity)

            # Compute expert action (analytical potential field)
            action = expert.compute_privileged_expert_action(
                position=priv.global_position,
                velocity=priv.global_velocity,
                obstacles=env.obstacles
            )

            # Record strictly the deployable student observation
            deployable_obs_vec = obs.astype(np.float32)
            action_vec = np.array([action.vx, action.vy, action.vz, action.yaw_rate], dtype=np.float32)

            ep_obs.append(deployable_obs_vec)
            ep_acts.append(action_vec)

            # Step physics
            norm_action = action.to_normalized(env.action_limits)
            obs, rew, term, trunc, info = env.step(norm_action)

            curr_pos = env._get_drone_position()
            flight_distance += float(np.linalg.norm(curr_pos - prev_pos))
            prev_pos = curr_pos

            min_obs_clear = min(min_obs_clear, float(info.get("obstacle_clearance", 10.0)))
            min_wall_clear = min(min_wall_clear, float(info.get("boundary_distance", 10.0)))

            if term or trunc:
                termination_status = info.get("termination_reason", "unknown")
                episode_done = True
                break

        # Rigorous Quality Gate Validation
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
            "min_obstacle_clearance_m": round(min_obs_clear, 3),
            "min_boundary_distance_m": round(min_wall_clear, 3),
            "final_position": [round(float(x), 2) for x in curr_pos],
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
                f"status={termination_status}, obs_clear={min_obs_clear:.3f}m, wall_clear={min_wall_clear:.3f}m"
            )
            rejected_episodes_meta.append(ep_meta)

        print(f"{s:<6} | {termination_status:<12} | {len(ep_obs):<6} | {min_obs_clear:<14.3f} | {min_wall_clear:<15.3f} | {decision:<10}")

    env.close()

    obs_arr = np.array(collected_obs, dtype=np.float32)
    act_arr = np.array(collected_acts, dtype=np.float32)
    starts_arr = np.array(collected_starts, dtype=bool)
    ep_ids_arr = np.array(collected_ep_ids, dtype=np.int32)
    step_idx_arr = np.array(collected_step_idx, dtype=np.int32)

    metadata = {
        "dataset_name": f"pilot_expert_apf_{scenario}",
        "scenario": scenario,
        "corridor_dimensions": [env.corridor.x_end, env.corridor.width, env.corridor.height],
        "start_position": list(env.corridor.start_position),
        "goal_position": list(env.corridor.goal_position),
        "teacher_controller_id": expert.metadata.controller_id,
        "teacher_checkpoint_id": expert.metadata.checkpoint_id,
        "total_episodes_accepted": accepted_count,
        "total_episodes_attempted": attempted_count,
        "total_transitions": int(len(obs_arr)),
        "observation_dim": int(obs_arr.shape[1]),
        "action_dim": int(act_arr.shape[1]),
        "feature_spec": {
            "indices_0_11": "multi_beam_rangefinder_rays_normalized_0_to_1",
            "indices_12_14": "unit_relative_goal_vector_body_flu",
            "indices_15_17": "body_linear_velocity_flu_mps",
            "indices_18_21": "attitude_trigonometry_cos_sin_roll_pitch",
            "index_22": "body_yaw_rate_radps",
        },
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
        "generation_timestamp_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    def _json_serial(o):
        if isinstance(o, (np.floating, np.float32, np.float64)):
            return float(o)
        if isinstance(o, (np.integer, np.int32, np.int64)):
            return int(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)

    # Save NPZ dataset
    output_path = Path(output_npz)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        observations=obs_arr,
        actions=act_arr,
        episode_starts=starts_arr,
        episode_ids=ep_ids_arr,
        step_indices=step_idx_arr,
        metadata_json=json.dumps(metadata, indent=2, default=_json_serial),
    )

    # Save Audit Log
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
        },
        "accepted_episodes": successful_episodes_meta,
        "rejected_episodes": rejected_episodes_meta,
    }
    with open(audit_path, "w") as f:
        json.dump(audit_data, f, indent=2, default=_json_serial)

    print("\n" + "=" * 80)
    print(">>> PILOT DEMONSTRATION GENERATION COMPLETE <<<")
    print(f"Accepted Clean Episodes:     {accepted_count} / {attempted_count} ({audit_data['summary']['filter_acceptance_rate_pct']}%)")
    print(f"Total Transition Pairs:      {len(obs_arr)}")
    print(f"Mean Steps per Episode:      {audit_data['summary']['mean_steps_per_episode']}")
    print(f"Dataset Archive Written to:  {output_path.resolve()}")
    print(f"Quality Audit Log Saved to:  {audit_path.resolve()}")
    print("=" * 80)

    return audit_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pilot Expert Demonstration Generator")
    parser.add_argument("--scenario", type=str, default="10m", choices=["10m", "20m"], help="Corridor scenario")
    parser.add_argument("--episodes", type=int, default=50, help="Number of certified demonstrations to collect")
    parser.add_argument("--start-seed", type=int, default=1000, help="Starting random seed")
    parser.add_argument("--output", type=str, default="data/demonstrations/pilot_demonstrations_10m.npz", help="Output .npz file")
    parser.add_argument("--audit", type=str, default="data/demonstrations/pilot_demonstrations_audit.json", help="Audit .json log")
    args = parser.parse_args()

    generate_pilot_demonstrations(
        scenario=args.scenario,
        target_episodes=args.episodes,
        start_seed=args.start_seed,
        output_npz=args.output,
        audit_json=args.audit,
    )
