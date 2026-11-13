"""generate_curriculum_data.py - Phase 4K Isolated Curriculum Dataset Generator.

Augments baseline demonstrations with balanced near-obstacle recovery demonstrations:
- Stage 3 near-obstacle standstill recoveries: drone starts at x in [2.40, 2.50]m facing Pillar 1.
- Exactly 50% left bypasses (evade_dir = +1.0) and 50% right bypasses (evade_dir = -1.0).
- Strictly preserves student observation contract (23 floats).
- Saves isolated datasets in experiments/phase4k/.../data/.
"""

import json
from pathlib import Path
from typing import Any, Dict, List
import numpy as np
import pybullet as p

from controllers.apf_expert import ScriptedAPFExpertController
from core.contracts import ActionLimits, VelocityAction
from sim.corridor_env import CorridorSimEnv


def generate_curriculum_dataset():
    exp_dir = Path(__file__).resolve().parent
    data_dir = exp_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    base_train_path = Path("data/demonstrations/bc_train_10m.npz")
    base_val_path = Path("data/demonstrations/bc_val_10m.npz")

    d_tr = np.load(base_train_path, allow_pickle=True)
    d_val = np.load(base_val_path, allow_pickle=True)

    train_obs = list(d_tr["observations"])
    train_acts = list(d_tr["actions"])
    train_starts = list(d_tr["episode_starts"])
    train_ep_ids = list(d_tr["episode_ids"])

    val_obs = list(d_val["observations"])
    val_acts = list(d_val["actions"])
    val_starts = list(d_val["episode_starts"])
    val_ep_ids = list(d_val["episode_ids"])

    current_ep_id = int(max(np.max(train_ep_ids), np.max(val_ep_ids))) + 1

    print("=" * 80)
    print(">>> GENERATING ISOLATED NEAR-OBSTACLE CURRICULUM DEMONSTRATIONS <<<")
    print(f"Base Train Transitions: {len(train_obs):,}")
    print(f"Base Val Transitions:   {len(val_obs):,}")
    print("=" * 80)

    env = CorridorSimEnv(scenario="10m", gui=False, seed=42)
    expert = ScriptedAPFExpertController(corridor=env.corridor)

    # 40 recovery episodes: 20 left (+1), 20 right (-1)
    # 32 for train, 8 for val
    evade_dirs = [+1.0] * 20 + [-1.0] * 20
    np.random.seed(1337)
    np.random.shuffle(evade_dirs)

    accepted_recovery = 0

    for idx, e_dir in enumerate(evade_dirs, 1):
        # Reset scene with deterministic seed
        seed_val = 5000 + idx
        obs, info = env.reset(seed=seed_val)

        # Reposition drone to standoff state: x in [2.40, 2.45], y in [-0.02, 0.02]
        start_x = float(np.random.uniform(2.40, 2.46))
        start_y = float(np.random.uniform(-0.02, 0.02))
        p.resetBasePositionAndOrientation(env.drone_id, [start_x, start_y, 1.0], [0, 0, 0, 1], physicsClientId=env.client_id)
        p.resetBaseVelocity(env.drone_id, [0, 0, 0], [0, 0, 0], physicsClientId=env.client_id)
        p.stepSimulation(physicsClientId=env.client_id)

        expert.update_environment(env.corridor, env.obstacles)

        # Run episode with forced evasion direction
        ep_obs = []
        ep_acts = []
        min_obs = 10.0
        min_wall = 10.0

        for st in range(500):
            curr_pos = env._get_drone_position()
            curr_vel = env._get_drone_velocity()

            # Privileged calculation with forced evasion direction
            p_curr = np.asarray(curr_pos, dtype=np.float64)
            goal = np.asarray(env.corridor.goal_position, dtype=np.float64)
            d_goal_vec = goal - p_curr
            d_goal = float(np.linalg.norm(d_goal_vec))
            u_att = d_goal_vec / d_goal if d_goal > 1e-4 else np.zeros(3)

            vx_des = 0.50 if d_goal >= 1.5 else float(0.50 * (d_goal / 1.5))
            vy_des = float(0.70 * u_att[1])
            vz_des = float(0.80 * u_att[2])

            for obs_obj in env.obstacles:
                obs_c = np.asarray(obs_obj.center, dtype=np.float64)
                delta_xy = p_curr[:2] - obs_c[:2]
                dist_center = float(np.linalg.norm(delta_xy))
                clearance = dist_center - obs_obj.radius - env.corridor.drone_radius
                dx = obs_c[0] - p_curr[0]

                if dx > -0.60 and clearance < 1.30:
                    n_rep = delta_xy / dist_center if dist_center > 1e-4 else np.array([0.0, e_dir])
                    rep_mag = float(np.clip(0.45 * ((1.0 / max(0.04, clearance)) - (1.0 / 1.30)), 0.0, 1.2))
                    circ_mag = float(0.55 * np.exp(-max(0.0, clearance) / 0.35))
                    vy_des += float(rep_mag * n_rep[1] + circ_mag * e_dir)
                    if abs(delta_xy[1]) < (obs_obj.radius + env.corridor.drone_radius + 0.15) and dx > 0:
                        vx_des *= float(np.clip(clearance / 1.30, 0.40, 1.0))

            # Wall buffers
            d_left = float(env.corridor.y_max - p_curr[1] - env.corridor.drone_radius)
            d_right = float(p_curr[1] - env.corridor.y_min - env.corridor.drone_radius)
            if d_left < 0.45:
                vy_des -= float(0.35 * ((1.0 / max(0.02, d_left)) - (1.0 / 0.45)))
            if d_right < 0.45:
                vy_des += float(0.35 * ((1.0 / max(0.02, d_right)) - (1.0 / 0.45)))

            action = VelocityAction(
                vx=float(np.clip(vx_des, -0.15, 0.70)),
                vy=float(np.clip(vy_des, -0.60, 0.60)),
                vz=float(np.clip(vz_des, -0.30, 0.30)),
                yaw_rate=float(np.clip(1.2 * vy_des, -0.5, 0.5)),
            )

            deployable_obs, priv_st = env._get_observations()
            ep_obs.append(deployable_obs.to_vector())
            ep_acts.append(np.array([action.vx, action.vy, action.vz, action.yaw_rate], dtype=np.float32))

            obs, rew, term, trunc, info = env.step(action.to_normalized(env.action_limits))
            min_obs = min(min_obs, float(info["privileged_state"].obstacle_clearance))
            min_wall = min(min_wall, float(info["privileged_state"].corridor_boundary_distance))

            if term or trunc:
                break

        t_reason = info.get("termination_reason")
        is_succ = (t_reason == "success") and (min_obs >= 0.08) and (min_wall >= 0.08)
        dir_tag = "LEFT" if e_dir > 0 else "RIGHT"

        if is_succ and len(ep_obs) > 40:
            accepted_recovery += 1
            is_val_split = (accepted_recovery % 5 == 0) # 20% to val

            target_obs = val_obs if is_val_split else train_obs
            target_acts = val_acts if is_val_split else train_acts
            target_starts = val_starts if is_val_split else train_starts
            target_ep_ids = val_ep_ids if is_val_split else train_ep_ids

            split_tag = "VAL" if is_val_split else "TRAIN"

            for s_i, (o_v, a_v) in enumerate(zip(ep_obs, ep_acts)):
                target_obs.append(o_v)
                target_acts.append(a_v)
                target_starts.append(bool(s_i == 0))
                target_ep_ids.append(current_ep_id)

            current_ep_id += 1
            print(f"[{accepted_recovery:02d}/40] Recovery Ep {idx:2d} | Evasion: {dir_tag:<5} | Steps: {len(ep_obs):3d} | MinObs: {min_obs:.3f}m | MinWall: {min_wall:.3f}m | Split: {split_tag}")

    env.close()

    train_out_path = data_dir / "bc_curriculum_train.npz"
    val_out_path = data_dir / "bc_curriculum_val.npz"

    np.savez_compressed(
        train_out_path,
        observations=np.array(train_obs, dtype=np.float32),
        actions=np.array(train_acts, dtype=np.float32),
        episode_starts=np.array(train_starts, dtype=bool),
        episode_ids=np.array(train_ep_ids, dtype=np.int32),
    )

    np.savez_compressed(
        val_out_path,
        observations=np.array(val_obs, dtype=np.float32),
        actions=np.array(val_acts, dtype=np.float32),
        episode_starts=np.array(val_starts, dtype=bool),
        episode_ids=np.array(val_ep_ids, dtype=np.int32),
    )

    print("\n" + "=" * 80)
    print(">>> CURRICULUM DATASET GENERATION COMPLETE <<<")
    print(f"Final Train Transitions: {len(train_obs):,} -> Saved: {train_out_path}")
    print(f"Final Val Transitions:   {len(val_obs):,} -> Saved: {val_out_path}")
    print("=" * 80)


if __name__ == "__main__":
    generate_curriculum_dataset()

