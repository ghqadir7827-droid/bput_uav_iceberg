"""run_symmetry_diagnostic.py - Phase 4K Experiment B Symmetry Diagnostic Suite.

Systematically probes the frozen BC baseline policy under controlled symmetric and asymmetric
obstacle configurations to measure:
1. Centered obstacle standoff behavior.
2. Mirrored obstacle response symmetry (left vs right).
3. Lateral velocity inertia reliance (sensitivity to initial vy).
4. Closed-loop trajectory evolution in isolated PyBullet simulation.
"""

import csv
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np
import pybullet as p
import torch

from controllers.safety_supervisor import SafetyState
from core.contracts import ActionLimits, VelocityAction, compute_forward_clearance_m
from core.geometry import Corridor3D, Obstacle3D
from experiments.phase4j.isolated_supervisor_opt import IsolatedOptimizedSupervisor
from models.policy_net import NavigationPolicyMLP
from sim.corridor_env import CorridorSimEnv

BC_CHECKPOINT_PATH = "models/checkpoints/bc_policy_best.pt"
CAUTION_DIST_M = 0.50


def run_controlled_simulation_test(
    env: CorridorSimEnv,
    supervisor: IsolatedOptimizedSupervisor,
    model: NavigationPolicyMLP,
    test_id: str,
    test_description: str,
    obstacle_y_offsets: List[float],
    start_pos: Tuple[float, float, float] = (0.5, 0.0, 1.0),
    start_vel: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    max_steps: int = 500,
) -> Dict[str, Any]:
    # Reset environment to basic scene
    obs, info = env.reset(seed=42)
    supervisor.reset_state()

    # Reposition obstacles according to test specification
    # Default benchmark has 3 pillars at x = 3.0, 5.0, 7.0
    for i, o_id in enumerate(env.obstacle_ids):
        if i < len(obstacle_y_offsets):
            y_val = obstacle_y_offsets[i]
            x_val = env.obstacles[i].center[0]
            p.resetBasePositionAndOrientation(o_id, [x_val, y_val, 1.0], [0, 0, 0, 1], physicsClientId=env.client_id)
            env.obstacles[i].center = np.array([x_val, y_val, 1.0], dtype=np.float32)

    # Set custom start position and velocity
    p.resetBasePositionAndOrientation(env.drone_id, list(start_pos), [0, 0, 0, 1], physicsClientId=env.client_id)
    p.resetBaseVelocity(env.drone_id, list(start_vel), [0, 0, 0], physicsClientId=env.client_id)
    p.stepSimulation(physicsClientId=env.client_id)

    # Re-sample initial observation
    deployable_obs, priv_state = env._get_observations()
    obs = deployable_obs.to_vector()
    info = {"privileged_state": priv_state}

    min_obs_c = 10.0
    min_wall_c = 10.0
    stalled_steps = 0
    raw_vy_list = []
    safe_vy_list = []
    traj_x = []
    traj_y = []

    steps = 0
    done = False
    term_reason = "timeout"

    current_info = info
    while not done and steps < max_steps:
        priv = current_info["privileged_state"]
        curr_pos = env._get_drone_position()
        curr_vel = priv.global_velocity
        front_d = compute_forward_clearance_m(obs, max_range_m=env.sensor_config.max_range_m)

        min_obs_c = min(min_obs_c, float(priv.obstacle_clearance))
        min_wall_c = min(min_wall_c, float(priv.corridor_boundary_distance))

        speed = float(np.linalg.norm(curr_vel))
        if speed < 0.05:
            stalled_steps += 1

        with torch.no_grad():
            act = model(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0), bounded=True).squeeze(0).numpy()

        raw_act = VelocityAction(float(act[0]), float(act[1]), float(act[2]), float(act[3]))
        clamped_act = VelocityAction(
            vx=float(np.clip(raw_act.vx, env.action_limits.min_fwd_vel_mps, env.action_limits.max_fwd_vel_mps)),
            vy=float(np.clip(raw_act.vy, -env.action_limits.max_lat_vel_mps, env.action_limits.max_lat_vel_mps)),
            vz=float(np.clip(raw_act.vz, -env.action_limits.max_vert_vel_mps, env.action_limits.max_vert_vel_mps)),
            yaw_rate=float(np.clip(raw_act.yaw_rate, -env.action_limits.max_yaw_rate_radps, env.action_limits.max_yaw_rate_radps)),
        )

        safe_act, s_state, s_reason = supervisor.filter_action(
            proposed_action=clamped_act,
            current_position=curr_pos,
            current_velocity=curr_vel,
            min_front_range=front_d,
            dt=env.dt,
            observation=obs,
        )

        raw_vy_list.append(raw_act.vy)
        safe_vy_list.append(safe_act.vy)
        traj_x.append(float(curr_pos[0]))
        traj_y.append(float(curr_pos[1]))

        obs, rew, term, trunc, current_info = env.step(safe_act.to_normalized(env.action_limits))
        steps += 1
        if term or trunc:
            term_reason = current_info.get("termination_reason", "unknown")
            done = True

    final_pos = env._get_drone_position()
    fwd_prog = float(final_pos[0] - start_pos[0])
    is_success = (term_reason == "success") and (float(np.linalg.norm(env._get_drone_velocity())) <= 0.35)

    return {
        "test_id": test_id,
        "description": test_description,
        "obstacle_1_y": obstacle_y_offsets[0],
        "start_pos_x": start_pos[0],
        "start_pos_y": start_pos[1],
        "start_vel_y": start_vel[1],
        "steps_completed": steps,
        "status": "SUCCESS" if is_success else term_reason,
        "is_success": is_success,
        "final_x": round(float(final_pos[0]), 3),
        "final_y": round(float(final_pos[1]), 3),
        "forward_progress_m": round(fwd_prog, 3),
        "min_obstacle_clearance_m": round(min_obs_c, 4),
        "min_boundary_clearance_m": round(min_wall_c, 4),
        "stalled_steps": stalled_steps,
        "mean_raw_vy": round(float(np.mean(raw_vy_list)), 4),
        "mean_safe_vy": round(float(np.mean(safe_vy_list)), 4),
        "outcome_classification": (
            "SUCCESS" if is_success
            else ("COLLISION" if min_obs_c <= 0 or min_wall_c <= 0
            else ("CENTERED_STANDOFF" if fwd_prog < 3.0 and stalled_steps > 200
            else "TIMEOUT_DELAY"))
        ),
    }


def main():
    exp_dir = Path(__file__).resolve().parent
    out_csv = exp_dir / "symmetry_diagnostic_results.csv"

    print("=" * 80)
    print(">>> RUNNING PHASE 4K EXPERIMENT B: SYMMETRY DIAGNOSTIC SUITE <<<")
    print("=" * 80)

    device = torch.device("cpu")
    env = CorridorSimEnv(scenario="10m", gui=False, seed=42)
    supervisor = IsolatedOptimizedSupervisor(corridor=env.corridor, caution_distance_m=CAUTION_DIST_M)

    model = NavigationPolicyMLP(obs_dim=23, act_dim=4).to(device)
    ckpt = torch.load(BC_CHECKPOINT_PATH, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    test_cases = []

    # 1. Perfectly centered obstacle (y = 0.0)
    test_cases.append(("TEST_01_CENTERED", "Perfectly centered single pillar at y=0.0m", [0.0, 0.5, -0.5], (0.5, 0.0, 1.0), (0.0, 0.0, 0.0)))

    # 2. Mirrored lateral offsets: +/- 0.05m, +/- 0.10m, +/- 0.20m, +/- 0.35m
    for y_off in [0.05, 0.10, 0.20, 0.35]:
        test_cases.append((f"TEST_LEFT_Y_{int(y_off*100):02d}", f"Pillar 1 offset to Left by +{y_off:.2f}m", [+y_off, 0.5, -0.5], (0.5, 0.0, 1.0), (0.0, 0.0, 0.0)))
        test_cases.append((f"TEST_RIGHT_Y_{int(y_off*100):02d}", f"Pillar 1 offset to Right by -{y_off:.2f}m", [-y_off, 0.5, -0.5], (0.5, 0.0, 1.0), (0.0, 0.0, 0.0)))

    # 3. Initial lateral velocity perturbations at rest in front of centered pillar (x = 2.40m)
    for vy_init in [-0.20, -0.10, 0.00, +0.10, +0.20]:
        sign_str = "NEG" if vy_init < 0 else ("POS" if vy_init > 0 else "ZERO")
        test_cases.append((
            f"TEST_STANDOFF_VY_{sign_str}_{abs(int(vy_init*100)):02d}",
            f"Drone placed at x=2.40m facing centered pillar with initial vy={vy_init:+.2f}m/s",
            [0.0, 0.5, -0.5],
            (2.40, 0.0, 1.0),
            (0.0, vy_init, 0.0)
        ))

    results = []
    print(f"\n{'Test ID':<26} | {'Pillar 1 Y':<10} | {'Init vy':<8} | {'Status':<10} | {'Fwd Prog':<9} | {'Min Obs C':<10} | {'Outcome':<18}")
    print("-" * 105)

    for tid, desc, obs_y, s_pos, s_vel in test_cases:
        res = run_controlled_simulation_test(
            env=env,
            supervisor=supervisor,
            model=model,
            test_id=tid,
            test_description=desc,
            obstacle_y_offsets=obs_y,
            start_pos=s_pos,
            start_vel=s_vel,
            max_steps=500,
        )
        results.append(res)
        print(f"{res['test_id']:<26} | {res['obstacle_1_y']:+8.2f}m  | {res['start_vel_y']:+6.2f}   | {res['status']:<10} | {res['forward_progress_m']:6.2f}m   | {res['min_obstacle_clearance_m']:8.4f}m  | {res['outcome_classification']:<18}")

    # Write CSV
    csv_fields = list(results[0].keys())
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        for r in results:
            writer.writerow(r)

    print(f"\n[OK] Saved symmetry diagnostic results to: {out_csv}")
    env.close()


if __name__ == "__main__":
    main()
