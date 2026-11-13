"""generate_phase4h_bc_demos.py - Phase 4H Targeted Demonstration Generator for BC.

Generates verified expert demonstrations targeting confirmed failure modes:
1. Centered single obstacles (Y_obs = 0.00m) with consistent, decisive bypass.
2. Left-shifted single obstacles (Y_obs in [0.10, 0.30]m) with clean right bypass.
3. Right-shifted single obstacles (Y_obs in [-0.30, -0.10]m) with clean left bypass.
4. Multi-pillar sequential configurations.
5. Clear corridor flights with terminal arrival deceleration (v_term <= 0.35 m/s).

Quality gates:
- Must reach goal (termination_reason == 'success')
- Min obstacle clearance >= 0.08m
- Min wall clearance >= 0.08m
- Terminal speed <= 0.35 m/s
- Step count <= 500
"""

import json
from pathlib import Path
import random
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pybullet as p

from controllers.apf_expert import ScriptedAPFExpertController
from core.contracts import ActionLimits, VelocityAction
from core.geometry import Corridor3D, Obstacle3D
from sim.corridor_env import CorridorSimEnv


# Use canonical ScriptedAPFExpertController directly
# ScriptedAPFExpertController has canonical potential field and smooth landing deceleration



def generate_phase4h_dataset(
    output_dir: str = "experiments/phase4h/data",
    total_target_episodes: int = 120,
) -> Dict[str, Any]:
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    env = CorridorSimEnv(scenario="10m", gui=False, seed=1200)
    expert = ScriptedAPFExpertController(corridor=env.corridor)

    collected_obs = []
    collected_acts = []
    collected_starts = []
    collected_ep_ids = []
    collected_step_idx = []
    episodes_meta = []

    # Scenario distributions:
    # 1. Centered obstacle (Y=0.00m) at X=3.0m (25 episodes)
    # 2. Left-shifted obstacles (Y=0.15m, 0.25m) (25 episodes)
    # 3. Right-shifted obstacles (Y=-0.15m, -0.25m) (25 episodes)
    # 4. Clear corridor (15 episodes)
    # 5. Multi-pillar diverse random corridors (30 episodes)

    scenario_plan = []
    # Centered:
    for i in range(25):
        scenario_plan.append({"type": "single", "y_obs": 0.00, "seed": 2000 + i})
    # Left-shifted:
    for i in range(25):
        y = 0.15 if i % 2 == 0 else 0.25
        scenario_plan.append({"type": "single", "y_obs": y, "seed": 2100 + i})
    # Right-shifted:
    for i in range(25):
        y = -0.15 if i % 2 == 0 else -0.25
        scenario_plan.append({"type": "single", "y_obs": y, "seed": 2200 + i})
    # Clear corridor:
    for i in range(15):
        scenario_plan.append({"type": "clear", "seed": 2300 + i})
    # Multi-pillar:
    for i in range(30):
        scenario_plan.append({"type": "multi", "seed": 2400 + i})

    print(f"Generating Phase 4H expert demonstrations ({len(scenario_plan)} planned episodes)...")
    accepted_count = 0

    for item in scenario_plan:
        s = item["seed"]
        obs, info = env.reset(seed=s)

        # Apply scenario configuration
        if item["type"] == "clear":
            for obs_id in env.obstacle_ids:
                p.resetBasePositionAndOrientation(obs_id, [50.0, 50.0, -10.0], [0, 0, 0, 1], physicsClientId=env.client_id)
            env.obstacles = []
        elif item["type"] == "single":
            y_obs = item["y_obs"]
            for idx, obs_id in enumerate(env.obstacle_ids):
                if idx == 0:
                    p.resetBasePositionAndOrientation(obs_id, [3.0, y_obs, 1.0], [0, 0, 0, 1], physicsClientId=env.client_id)
                else:
                    p.resetBasePositionAndOrientation(obs_id, [50.0, 50.0, -10.0], [0, 0, 0, 1], physicsClientId=env.client_id)
            env.obstacles = [Obstacle3D(center=np.array([3.0, y_obs, 1.0]), radius=0.20, height=2.0)]
        # For multi, leave env.obstacles as spawned by reset(seed)

        expert.update_environment(env.corridor, env.obstacles)

        ep_obs = []
        ep_acts = []
        min_obs = 10.0
        min_wall = 10.0

        for step in range(500):
            priv = info["privileged_state"]
            action = expert.compute_privileged_expert_action(
                position=priv.global_position,
                velocity=priv.global_velocity,
                obstacles=env.obstacles,
            )

            ep_obs.append(obs.astype(np.float32))
            ep_acts.append(np.array([action.vx, action.vy, action.vz, action.yaw_rate], dtype=np.float32))

            norm_action = action.to_normalized(env.action_limits)
            obs, rew, term, trunc, info = env.step(norm_action)

            min_obs = min(min_obs, float(info.get("obstacle_clearance", 10.0)))
            min_wall = min(min_wall, float(info.get("boundary_distance", 10.0)))

            if term or trunc:
                break

        term_reason = info.get("termination_reason", "unknown")
        final_vel = float(np.linalg.norm(info["privileged_state"].global_velocity))
        passed = (
            term_reason == "success"
            and min_obs >= 0.08
            and min_wall >= 0.08
            and final_vel <= 0.35
            and len(ep_obs) >= 80
        )

        if passed:
            accepted_count += 1
            episodes_meta.append({
                "episode_id": accepted_count,
                "type": item["type"],
                "seed": s,
                "steps": len(ep_obs),
                "min_obs_clearance": round(min_obs, 3),
                "min_wall_clearance": round(min_wall, 3),
                "terminal_speed": round(final_vel, 3),
            })
            for idx, (o, a) in enumerate(zip(ep_obs, ep_acts)):
                collected_obs.append(o)
                collected_acts.append(a)
                collected_starts.append(idx == 0)
                collected_ep_ids.append(accepted_count)
                collected_step_idx.append(idx)
            print(f"Accepted {accepted_count}/{len(scenario_plan)}: {item['type']} (seed {s}) - steps: {len(ep_obs)}, v_term: {final_vel:.3f} m/s")
        else:
            print(f"REJECTED: {item['type']} (seed {s}) - term: {term_reason}, obs: {min_obs:.3f}, wall: {min_wall:.3f}, v_term: {final_vel:.3f}")

    env.close()

    total_transitions = len(collected_obs)
    print(f"\nTotal Accepted Episodes: {accepted_count}")
    print(f"Total Transitions: {total_transitions}")

    # Split into Train (80%) and Val (20%) at episode level
    unique_ep_ids = list(range(1, accepted_count + 1))
    random.seed(42)
    random.shuffle(unique_ep_ids)
    val_cutoff = int(0.20 * len(unique_ep_ids))
    val_ep_set = set(unique_ep_ids[:val_cutoff])
    train_ep_set = set(unique_ep_ids[val_cutoff:])

    train_indices = [i for i, ep_id in enumerate(collected_ep_ids) if ep_id in train_ep_set]
    val_indices = [i for i, ep_id in enumerate(collected_ep_ids) if ep_id in val_ep_set]

    obs_np = np.array(collected_obs, dtype=np.float32)
    act_np = np.array(collected_acts, dtype=np.float32)
    starts_np = np.array(collected_starts, dtype=bool)
    ep_ids_np = np.array(collected_ep_ids, dtype=np.int32)
    steps_np = np.array(collected_step_idx, dtype=np.int32)

    train_file = out_path / "bc_train_phase4h.npz"
    val_file = out_path / "bc_val_phase4h.npz"

    meta_dict = {
        "dataset_name": "phase4h_targeted_bc_demonstrations",
        "total_episodes": accepted_count,
        "train_episodes": len(train_ep_set),
        "val_episodes": len(val_ep_set),
        "train_transitions": len(train_indices),
        "val_transitions": len(val_indices),
        "timestamp_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    np.savez_compressed(
        train_file,
        observations=obs_np[train_indices],
        actions=act_np[train_indices],
        episode_starts=starts_np[train_indices],
        episode_ids=ep_ids_np[train_indices],
        step_indices=steps_np[train_indices],
        metadata_json=json.dumps(meta_dict, indent=2),
    )

    np.savez_compressed(
        val_file,
        observations=obs_np[val_indices],
        actions=act_np[val_indices],
        episode_starts=starts_np[val_indices],
        episode_ids=ep_ids_np[val_indices],
        step_indices=steps_np[val_indices],
        metadata_json=json.dumps(meta_dict, indent=2),
    )

    print(f"Saved train dataset: {train_file} ({len(train_indices):,} samples)")
    print(f"Saved val dataset:   {val_file} ({len(val_indices):,} samples)")

    with open(out_path / "bc_demonstrations_meta.json", "w") as f:
        json.dump({"meta": meta_dict, "episodes": episodes_meta}, f, indent=2)

    return meta_dict


if __name__ == "__main__":
    generate_phase4h_dataset()
