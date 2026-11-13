"""run_phase4g_validation.py - Controlled Validation of Arms A, B, C, and D.

Compares:
- Arm A: Frozen PPO Baseline (models/checkpoints/ppo_policy_best.pt)
- Arm B: Frozen BC Baseline (models/checkpoints/bc_policy_best.pt)
- Arm C: PPO without reflection augmentation (experiments/phase4g/checkpoints/ppo_arm_c_baseline.pt)
- Arm D: PPO with reflection augmentation (experiments/phase4g/checkpoints/ppo_arm_d_reflection.pt)

Validation Protocol:
1. Open-space symmetry probe (lateral sweep Y in [-0.4, +0.4] m at X=1.0 m, Z=1.0 m).
2. Paired mirrored obstacle bypass tests at X=3.0 m (Y_obs = +/-0.25, +/-0.15, 0.0 m).
3. 10-seed fixed validation pool (seeds 100..109) with identical initial conditions.
4. Predeclared candidate selection rule:
   - Primary: Highest end-to-end goal success rate on validation seeds.
   - Secondary: Lowest open-space lateral bias |vy| and yaw-rate bias |yaw_rate|.
   - Tertiary: Lowest supervisor intervention rate and highest obstacle clearance.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np
import pybullet as p
import torch

from controllers.bc_policy import BehaviorCloningPolicyController
from controllers.ppo_policy import PPOPolicyController
from controllers.safety_supervisor import SafetyState, SafetySupervisor
from core.contracts import ActionLimits, DeployableObservation, VelocityAction, compute_forward_clearance_m
from models.train_ppo import VALIDATION_SEEDS
from sim.corridor_env import CorridorSimEnv


def evaluate_controller_on_val_seeds(
    ctrl_name: str,
    ctrl: Any,
    seeds: List[int] = VALIDATION_SEEDS,
    scenario: str = "10m",
) -> Dict[str, Any]:
    """Runs deterministic evaluation over validation seeds with full telemetry."""
    env = CorridorSimEnv(scenario=scenario, gui=False, seed=seeds[0])
    supervisor = SafetySupervisor(corridor=env.corridor)

    total_episodes = len(seeds)
    successes = 0
    collisions = 0
    wall_collisions = 0
    timeouts = 0
    total_steps = 0
    total_interventions = 0
    min_obs_clearances = []
    min_wall_clearances = []
    forward_displacements = []
    final_goal_distances = []

    per_seed_results = []

    for seed in seeds:
        obs, info = env.reset(seed=seed)
        supervisor.loss_timer_s = 0.0
        supervisor.stalled_timer_s = 0.0
        supervisor.last_safe_action = VelocityAction(0.0, 0.0, 0.0, 0.0)

        ep_start_pos = env._get_drone_position()
        done = False
        steps = 0
        min_obs = 10.0
        min_wall = 10.0
        ep_interventions = 0

        current_info = info
        while not done and steps < 200:
            priv = current_info["privileged_state"]
            min_obs = min(min_obs, float(priv.obstacle_clearance))
            min_wall = min(min_wall, float(priv.corridor_boundary_distance))

            raw_act = ctrl.predict(obs)
            front_dist = compute_forward_clearance_m(obs, max_range_m=env.sensor_config.max_range_m)

            safe_act, s_state, s_reason = supervisor.filter_action(
                proposed_action=raw_act,
                current_position=priv.global_position,
                current_velocity=priv.global_velocity,
                min_front_range=front_dist,
                dt=env.dt,
                is_localization_healthy=True,
                telemetry_age_s=0.0,
                observation=obs,
            )

            if s_state != SafetyState.NOMINAL:
                ep_interventions += 1
                total_interventions += 1

            norm_act = safe_act.to_normalized(env.action_limits)
            obs, rew, term, trunc, current_info = env.step(norm_act)
            steps += 1
            done = term or trunc

        term_reason = current_info.get("termination_reason", "")
        if term_reason == "success":
            successes += 1
        elif "crash" in term_reason:
            collisions += 1
            if "wall" in term_reason:
                wall_collisions += 1
        elif term_reason == "timeout":
            timeouts += 1

        total_steps += steps
        final_pos = env._get_drone_position()
        fwd_disp = float(final_pos[0] - ep_start_pos[0])
        goal_dist = float(current_info["privileged_state"].goal_distance)

        forward_displacements.append(fwd_disp)
        final_goal_distances.append(goal_dist)
        min_obs_clearances.append(min_obs)
        min_wall_clearances.append(min_wall)

        per_seed_results.append({
            "seed": seed,
            "outcome": term_reason,
            "steps": steps,
            "forward_displacement_m": round(fwd_disp, 3),
            "final_goal_dist_m": round(goal_dist, 3),
            "min_obs_clearance_m": round(min_obs, 3),
            "min_wall_clearance_m": round(min_wall, 3),
            "interventions": ep_interventions,
        })

    env.close()
    return {
        "controller": ctrl_name,
        "total_episodes": total_episodes,
        "success_count": successes,
        "success_rate_pct": round((successes / max(1, total_episodes)) * 100.0, 1),
        "collision_count": collisions,
        "wall_collision_count": wall_collisions,
        "timeout_count": timeouts,
        "mean_steps": round(total_steps / max(1, total_episodes), 1),
        "mean_forward_displacement_m": round(float(np.mean(forward_displacements)), 3),
        "mean_final_goal_dist_m": round(float(np.mean(final_goal_distances)), 3),
        "mean_min_obs_clearance_m": round(float(np.mean(min_obs_clearances)), 3),
        "mean_min_wall_clearance_m": round(float(np.mean(min_wall_clearances)), 3),
        "intervention_rate_pct": round((total_interventions / max(1, total_steps)) * 100.0, 2),
        "per_seed_results": per_seed_results,
    }


def evaluate_controller_open_space_bias(ctrl: Any) -> Dict[str, Any]:
    """Measures lateral velocity and yaw rate across lateral sweep Y in [-0.4, +0.4] m."""
    env = CorridorSimEnv(scenario="10m", gui=False, seed=1)
    env.reset(seed=1)
    y_grid = [-0.40, -0.30, -0.20, -0.10, 0.00, 0.10, 0.20, 0.30, 0.40]
    sweep_results = []

    for y_pos in y_grid:
        p.resetBasePositionAndOrientation(env.drone_id, [1.0, y_pos, 1.0], [0, 0, 0, 1], physicsClientId=env.client_id)
        p.resetBaseVelocity(env.drone_id, [0.60, 0.0, 0.0], [0, 0, 0], physicsClientId=env.client_id)

        obs, _ = env._get_observations()
        act = ctrl.predict(obs.to_vector())

        sweep_results.append({
            "y": y_pos,
            "vx": round(float(act.vx), 4),
            "vy": round(float(act.vy), 4),
            "vz": round(float(act.vz), 4),
            "yaw_rate": round(float(act.yaw_rate), 4),
        })

    center_sample = [r for r in sweep_results if r["y"] == 0.0][0]
    env.close()
    return {
        "center_vy": center_sample["vy"],
        "center_yaw_rate": center_sample["yaw_rate"],
        "lateral_sweep": sweep_results,
    }


def evaluate_controller_mirrored_obstacles(ctrl: Any) -> List[Dict[str, Any]]:
    """Evaluates bypass side and clearance on mirrored obstacle configurations."""
    test_y_offsets = [-0.25, -0.15, 0.00, +0.15, +0.25]
    results = []

    for y_obs in test_y_offsets:
        env = CorridorSimEnv(scenario="10m", gui=False, seed=1)
        supervisor = SafetySupervisor(corridor=env.corridor)
        obs, info = env.reset(seed=1)

        p.resetBasePositionAndOrientation(
            env.obstacle_ids[0], [3.0, y_obs, 1.0], [0, 0, 0, 1], physicsClientId=env.client_id
        )
        env.obstacles[0].center = np.array([3.0, y_obs, 1.0])

        current_info = info
        done = False
        steps = 0
        min_obs = 10.0
        min_wall = 10.0
        interventions = 0

        while not done and steps < 200:
            priv = current_info["privileged_state"]
            min_obs = min(min_obs, float(priv.obstacle_clearance))
            min_wall = min(min_wall, float(priv.corridor_boundary_distance))

            raw_act = ctrl.predict(obs)
            front_dist = compute_forward_clearance_m(obs, max_range_m=env.sensor_config.max_range_m)

            safe_act, s_state, _ = supervisor.filter_action(
                proposed_action=raw_act,
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

            norm_act = safe_act.to_normalized(env.action_limits)
            obs, rew, term, trunc, current_info = env.step(norm_act)
            steps += 1
            done = term or trunc
            if priv.global_position[0] >= 3.8:
                break

        final_pos = current_info["privileged_state"].global_position
        bypass = "left" if final_pos[1] > y_obs else "right"
        passed = bool(final_pos[0] >= 3.8)

        left_gap = float(1.0 - (y_obs + 0.20))
        right_gap = float((y_obs - 0.20) - (-1.0))
        wider = "left" if left_gap > right_gap else ("right" if right_gap > left_gap else "symmetric")
        feasible_side = "left" if left_gap >= 0.85 else ("right" if right_gap >= 0.85 else "neither")

        results.append({
            "y_obs": y_obs,
            "wider_side": wider,
            "feasible_side": feasible_side,
            "bypass_side": bypass,
            "passed_obstacle": passed,
            "min_obs_clearance_m": round(min_obs, 3),
            "min_wall_clearance_m": round(min_wall, 3),
            "interventions": interventions,
            "final_x": round(float(final_pos[0]), 3),
            "final_y": round(float(final_pos[1]), 3),
        })
        env.close()

    return results


def run_full_validation_suite() -> Dict[str, Any]:
    """Runs validation comparison across Arms A, B, C, and D."""
    controllers = {
        "Arm_A_Frozen_PPO": PPOPolicyController("models/checkpoints/ppo_policy_best.pt"),
        "Arm_B_Frozen_BC": BehaviorCloningPolicyController("models/checkpoints/bc_policy_best.pt"),
        "Arm_C_PPO_NoAug": PPOPolicyController("experiments/phase4g/checkpoints/ppo_arm_c_baseline.pt"),
        "Arm_D_PPO_Reflection": PPOPolicyController("experiments/phase4g/checkpoints/ppo_arm_d_reflection.pt"),
    }

    suite_results = {}

    for name, ctrl in controllers.items():
        print(f"\n>>> Running Validation for {name} <<<")
        assert ctrl._is_loaded, f"Failed to load controller {name}!"
        val_eval = evaluate_controller_on_val_seeds(name, ctrl)
        open_space = evaluate_controller_open_space_bias(ctrl)
        obs_pairs = evaluate_controller_mirrored_obstacles(ctrl)

        suite_results[name] = {
            "validation_metrics": val_eval,
            "open_space_bias": open_space,
            "mirrored_obstacle_tests": obs_pairs,
        }
        print(f"  Success Rate: {val_eval['success_rate_pct']}% ({val_eval['success_count']}/{val_eval['total_episodes']})")
        print(f"  Open-Space vy: {open_space['center_vy']:+.4f} m/s, yaw_rate: {open_space['center_yaw_rate']:+.4f} rad/s")

    out_file = Path("experiments/phase4g/logs/phase4g_validation_results.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(suite_results, f, indent=2)
    print(f"\nAll validation results written to {out_file}")
    return suite_results


if __name__ == "__main__":
    run_full_validation_suite()
