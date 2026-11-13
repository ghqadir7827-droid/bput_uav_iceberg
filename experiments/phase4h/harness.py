"""harness.py - Isolated Shared Evaluation Harness for Phase 4H.

Evaluates:
- Arm B: Frozen BC Baseline (models/checkpoints/bc_policy_best.pt)
- Arm A: Frozen PPO Baseline (models/checkpoints/ppo_policy_best.pt)
- Arm C: Phase 4G Standard PPO Candidate (experiments/phase4g/checkpoints/ppo_arm_c_baseline.pt)
- Arm D: Phase 4G Reflection PPO Candidate (experiments/phase4g/checkpoints/ppo_arm_d_reflection.pt)

Supported Scenarios:
1. clear_corridor: No obstacles (evaluates nominal cruise, terminal deceleration, arrival speed).
2. obstacle_sweep: Single obstacle at X=3.0m with lateral offsets Y in [-0.25, -0.15, 0.00, +0.15, +0.25]m.
3. narrow_passage: Two obstacles at X=3.0m (Y = +/-0.55m) leaving a constrained central opening.
4. sequential_obstacles: Standard 3-pillar corridor across fixed validation seeds (100, 101, 102).
5. goal_approach_test: Drone placed at X=7.5m with vx=0.50 m/s approaching goal at X=9.5m (tests terminal braking).

Outputs:
- experiments/phase4h/logs/baseline_harness_records.json
- reports/phase4h_baseline_evaluation.md
- reports/phase4h_baseline_evaluation_metrics.json
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pybullet as p
import torch

from controllers.bc_policy import BehaviorCloningPolicyController
from controllers.ppo_policy import PPOPolicyController
from controllers.safety_supervisor import SafetyState, SafetySupervisor
from core.contracts import ActionLimits, DeployableObservation, VelocityAction, compute_forward_clearance_m
from core.geometry import Obstacle3D
from sim.corridor_env import CorridorSimEnv

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        if isinstance(obj, (np.integer, int)):
            return int(obj)
        if isinstance(obj, (np.floating, float)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def run_scenario_episode(
    ctrl_name: str,
    ctrl: Any,
    scenario_type: str,
    scenario_params: Dict[str, Any],
    max_steps: int = 500,
    seed: int = 1,
    record_step_telemetry: bool = False,
) -> Dict[str, Any]:
    """Runs a single controlled scenario episode with full telemetry logging."""
    env = CorridorSimEnv(scenario="10m", gui=False, seed=seed)
    supervisor = SafetySupervisor(corridor=env.corridor)
    obs, info = env.reset(seed=seed)

    supervisor.loss_timer_s = 0.0
    supervisor.stalled_timer_s = 0.0
    supervisor.last_safe_action = VelocityAction(0.0, 0.0, 0.0, 0.0)

    # Apply scenario-specific obstacle and drone setup
    if scenario_type == "clear_corridor":
        for oid in env.obstacle_ids:
            p.resetBasePositionAndOrientation(oid, [100.0, 100.0, 100.0], [0, 0, 0, 1], physicsClientId=env.client_id)
        env.obstacles = []

    elif scenario_type == "single_obstacle":
        y_obs = float(scenario_params["y_obs"])
        x_obs = float(scenario_params.get("x_obs", 3.0))
        p.resetBasePositionAndOrientation(env.obstacle_ids[0], [x_obs, y_obs, 1.0], [0, 0, 0, 1], physicsClientId=env.client_id)
        env.obstacles[0].center = np.array([x_obs, y_obs, 1.0])
        for oid in env.obstacle_ids[1:]:
            p.resetBasePositionAndOrientation(oid, [100.0, 100.0, 100.0], [0, 0, 0, 1], physicsClientId=env.client_id)
        env.obstacles = [env.obstacles[0]]

    elif scenario_type == "narrow_passage":
        # Two pillars at X=3.0: Y = -0.55m and Y = +0.55m
        p.resetBasePositionAndOrientation(env.obstacle_ids[0], [3.0, -0.55, 1.0], [0, 0, 0, 1], physicsClientId=env.client_id)
        env.obstacles[0].center = np.array([3.0, -0.55, 1.0])
        p.resetBasePositionAndOrientation(env.obstacle_ids[1], [3.0, +0.55, 1.0], [0, 0, 0, 1], physicsClientId=env.client_id)
        env.obstacles[1].center = np.array([3.0, +0.55, 1.0])
        if len(env.obstacle_ids) > 2:
            for oid in env.obstacle_ids[2:]:
                p.resetBasePositionAndOrientation(oid, [100.0, 100.0, 100.0], [0, 0, 0, 1], physicsClientId=env.client_id)
            env.obstacles = env.obstacles[:2]

    elif scenario_type == "goal_approach":
        # Clear obstacles and initialize drone close to goal
        for oid in env.obstacle_ids:
            p.resetBasePositionAndOrientation(oid, [100.0, 100.0, 100.0], [0, 0, 0, 1], physicsClientId=env.client_id)
        env.obstacles = []
        p.resetBasePositionAndOrientation(env.drone_id, [7.5, 0.0, 1.0], [0, 0, 0, 1], physicsClientId=env.client_id)
        p.resetBaseVelocity(env.drone_id, [0.50, 0.0, 0.0], [0, 0, 0], physicsClientId=env.client_id)
        deploy_obs, priv = env._get_observations()
        obs = deploy_obs.to_vector()
        info["privileged_state"] = priv

    ep_start_pos = env._get_drone_position()
    done = False
    step = 0
    interventions = 0
    min_obs_clear = 10.0
    min_wall_clear = 10.0
    stalled_steps = 0
    passed_obs1 = False
    trajectory = []
    step_telemetry = []

    current_info = info
    while not done and step < max_steps:
        step += 1
        priv = current_info["privileged_state"]
        pos = priv.global_position.copy()
        vel = priv.global_velocity.copy()
        trajectory.append([round(float(v), 3) for v in pos])

        min_obs_clear = min(min_obs_clear, float(priv.obstacle_clearance))
        min_wall_clear = min(min_wall_clear, float(priv.corridor_boundary_distance))

        speed = float(np.linalg.norm(vel))
        if speed < 0.05:
            stalled_steps += 1

        if pos[0] >= 3.8 and not passed_obs1:
            passed_obs1 = True

        raw_act = ctrl.predict(obs)
        front_dist = compute_forward_clearance_m(obs, max_range_m=env.sensor_config.max_range_m)

        safe_act, s_state, s_reason = supervisor.filter_action(
            proposed_action=raw_act,
            current_position=pos,
            current_velocity=vel,
            min_front_range=front_dist,
            dt=env.dt,
            is_localization_healthy=True,
            telemetry_age_s=0.0,
            observation=obs,
        )

        if s_state != SafetyState.NOMINAL:
            interventions += 1

        if record_step_telemetry:
            step_telemetry.append({
                "step": step,
                "x": round(float(pos[0]), 3),
                "y": round(float(pos[1]), 3),
                "z": round(float(pos[2]), 3),
                "speed": round(speed, 3),
                "raw_vx": round(float(raw_act.vx), 3),
                "raw_vy": round(float(raw_act.vy), 3),
                "raw_yaw_rate": round(float(raw_act.yaw_rate), 3),
                "safe_vx": round(float(safe_act.vx), 3),
                "safe_vy": round(float(safe_act.vy), 3),
                "safe_yaw_rate": round(float(safe_act.yaw_rate), 3),
                "supervisor_state": s_state.value,
                "supervisor_reason": s_reason,
                "front_clearance_m": round(float(front_dist), 3),
                "min_obs_clear_m": round(float(priv.obstacle_clearance), 3),
            })

        norm_act = safe_act.to_normalized(env.action_limits)
        obs, rew, term, trunc, current_info = env.step(norm_act)
        done = term or trunc

    final_pos = current_info["privileged_state"].global_position
    final_vel = current_info["privileged_state"].global_velocity
    final_speed = float(np.linalg.norm(final_vel))
    final_goal_dist = float(current_info["privileged_state"].goal_distance)
    term_reason = current_info.get("termination_reason", "timeout" if step >= max_steps else "unknown")

    # Bypass side classification for obstacle scenarios
    y_target_obs = scenario_params.get("y_obs", 0.0)
    bypass_side = "left" if final_pos[1] > y_target_obs else "right"

    is_success = (term_reason == "success")
    is_crash = ("crash" in term_reason)
    is_deadlock = (step >= max_steps and final_pos[0] < 5.0 and stalled_steps > 150)

    env.close()

    return {
        "controller": ctrl_name,
        "scenario": scenario_type,
        "params": scenario_params,
        "max_steps": max_steps,
        "steps_taken": step,
        "outcome": term_reason,
        "is_success": is_success,
        "is_crash": is_crash,
        "is_deadlock": is_deadlock,
        "passed_first_obstacle": passed_obs1,
        "bypass_side": bypass_side,
        "forward_displacement_m": round(float(final_pos[0] - ep_start_pos[0]), 3),
        "final_position": [round(float(v), 3) for v in final_pos],
        "final_speed_mps": round(final_speed, 3),
        "final_goal_distance_m": round(final_goal_dist, 3),
        "min_obstacle_clearance_m": round(min_obs_clear, 3),
        "min_wall_clearance_m": round(min_wall_clear, 3),
        "supervisor_interventions": interventions,
        "intervention_rate_pct": round((interventions / max(1, step)) * 100.0, 1),
        "stalled_steps": stalled_steps,
        "trajectory_sample": trajectory[::10],
        "step_telemetry": step_telemetry if record_step_telemetry else [],
    }


def execute_full_harness() -> Dict[str, Any]:
    """Runs all 4 controllers across all test scenarios under 500-step and 200-step horizons."""
    controllers = {
        "Arm_B_BC": BehaviorCloningPolicyController("models/checkpoints/bc_policy_best.pt"),
        "Arm_A_PPO": PPOPolicyController("models/checkpoints/ppo_policy_best.pt"),
        "Arm_C_PPO_NoAug": PPOPolicyController("experiments/phase4g/checkpoints/ppo_arm_c_baseline.pt"),
        "Arm_D_PPO_Reflection": PPOPolicyController("experiments/phase4g/checkpoints/ppo_arm_d_reflection.pt"),
    }

    all_records = []

    print("=" * 80)
    print(">>> EXECUTING PHASE 4H SHARED CONTROLLED EVALUATION HARNESS <<<")
    print("=" * 80)

    # 1. Clear Corridor Scenario (under 500 steps and 200 steps)
    print("\n--- Test 1: Clear Corridor ---")
    for max_steps in [500, 200]:
        for name, ctrl in controllers.items():
            rec = run_scenario_episode(
                ctrl_name=name, ctrl=ctrl, scenario_type="clear_corridor",
                scenario_params={"horizon_label": f"{max_steps}_steps"}, max_steps=max_steps, seed=1
            )
            all_records.append(rec)
            print(f"[{max_steps}s] {name:<22} | outcome: {rec['outcome']:<14} | steps: {rec['steps_taken']:3d} | final_x: {rec['final_position'][0]:.2f}m | speed: {rec['final_speed_mps']:.3f}m/s")

    # 2. Lateral Obstacle Sweep at X=3.0m (500 steps)
    print("\n--- Test 2: Lateral Obstacle Sweep (500 steps) ---")
    y_offsets = [-0.25, -0.15, 0.00, +0.15, +0.25]
    for y_obs in y_offsets:
        for name, ctrl in controllers.items():
            rec = run_scenario_episode(
                ctrl_name=name, ctrl=ctrl, scenario_type="single_obstacle",
                scenario_params={"y_obs": y_obs, "horizon_label": "500_steps"}, max_steps=500, seed=1,
                record_step_telemetry=(y_obs == 0.00) # Full telemetry on centered obstacle for failure diagnosis
            )
            all_records.append(rec)
            print(f"y={y_obs:+.2f} | {name:<22} | pass_obs: {rec['passed_first_obstacle']!s:<5} | outcome: {rec['outcome']:<14} | bypass: {rec['bypass_side']:<5} | final_x: {rec['final_position'][0]:.2f}m")

    # 3. Narrow Passage Scenario (X=3.0m, Y=+/-0.55m)
    print("\n--- Test 3: Narrow Passage (500 steps) ---")
    for name, ctrl in controllers.items():
        rec = run_scenario_episode(
            ctrl_name=name, ctrl=ctrl, scenario_type="narrow_passage",
            scenario_params={"gap_width_m": 0.70}, max_steps=500, seed=1
        )
        all_records.append(rec)
        print(f"Narrow | {name:<22} | pass_obs: {rec['passed_first_obstacle']!s:<5} | outcome: {rec['outcome']:<14} | final_x: {rec['final_position'][0]:.2f}m")

    # 4. Goal Approach Braking Test
    print("\n--- Test 4: Goal Approach Braking (X_init=7.5m, vx=0.50m/s) ---")
    for name, ctrl in controllers.items():
        rec = run_scenario_episode(
            ctrl_name=name, ctrl=ctrl, scenario_type="goal_approach",
            scenario_params={"init_x": 7.5, "init_vx": 0.50}, max_steps=150, seed=1
        )
        all_records.append(rec)
        print(f"Landing | {name:<22} | outcome: {rec['outcome']:<14} | steps: {rec['steps_taken']:3d} | final_speed: {rec['final_speed_mps']:.3f}m/s | final_dist: {rec['final_goal_distance_m']:.3f}m")

    # 5. Sequential Obstacles (Validation seeds 100, 101, 102 under 500 steps)
    print("\n--- Test 5: Sequential 3-Pillar Corridor (500 steps) ---")
    for seed in [100, 101, 102]:
        for name, ctrl in controllers.items():
            rec = run_scenario_episode(
                ctrl_name=name, ctrl=ctrl, scenario_type="sequential_obstacles",
                scenario_params={"seed": seed}, max_steps=500, seed=seed
            )
            all_records.append(rec)
            print(f"Seed {seed} | {name:<22} | outcome: {rec['outcome']:<14} | steps: {rec['steps_taken']:3d} | final_x: {rec['final_position'][0]:.2f}m | obs_clear: {rec['min_obstacle_clearance_m']:.3f}m")
    output_path = Path("reports/phase4h_baseline_evaluation_metrics.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_records, f, indent=2, cls=NumpyEncoder)
    print(f"\nAll harness metrics written to {output_path}")

    return {"records": all_records}


if __name__ == "__main__":
    execute_full_harness()
