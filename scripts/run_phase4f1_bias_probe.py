"""run_phase4f1_bias_probe.py - Phase 4F-1 Controlled PPO vs BC Policy Bias & Route Experiment.

Executes:
1. Equivalence check for PPO and BC inference pipelines.
2. Controlled open-space initial state probes across lateral sweep Y in [-0.4, +0.4]m.
3. Controlled shifted/mirrored obstacle pairs at X=3.0m (Y_obs = +/-0.25m, +/-0.15m, 0.0m).
4. Full trajectory evaluations on diagnostic seeds [1, 20, 100, 2000, 8000] with same-seed replay control.
5. Exact geometric clearance and supervisor feasibility accounting.
Outputs:
- reports/phase4f1_policy_probe_metrics.json
- reports/phase4f1_mirrored_route_comparison.csv
"""

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pybullet as p
import torch

from controllers.bc_policy import BehaviorCloningPolicyController
from controllers.ppo_policy import PPOPolicyController
from controllers.safety_supervisor import SafetyState, SafetySupervisor, VelocityAction
from core.contracts import ActionLimits, DeployableObservation, compute_forward_clearance_m
from core.geometry import Corridor3D, Obstacle3D
from sim.corridor_env import CorridorSimEnv


def get_file_sha256(filepath: str) -> str:
    path = Path(filepath)
    if not path.exists():
        return "NOT_FOUND"
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def probe_open_space(
    ppo_ctrl: PPOPolicyController,
    bc_ctrl: BehaviorCloningPolicyController,
) -> Dict[str, Any]:
    """Evaluates raw network outputs and supervisor actions in open corridor space."""
    env = CorridorSimEnv(scenario="10m", gui=False, seed=1)
    supervisor = SafetySupervisor(corridor=env.corridor)
    env.reset(seed=1)

    y_grid = [-0.40, -0.30, -0.20, -0.10, 0.00, 0.10, 0.20, 0.30, 0.40]
    vel_modes = [
        ("static_hover", [0.0, 0.0, 0.0]),
        ("forward_cruise", [0.60, 0.0, 0.0]),
    ]

    probe_records = []

    for v_name, v_vec in vel_modes:
        for y_pos in y_grid:
            # Place drone at X=1.0m (well before first obstacle at X=3.0m)
            p.resetBasePositionAndOrientation(
                env.drone_id, [1.0, y_pos, 1.0], [0, 0, 0, 1], physicsClientId=env.client_id
            )
            p.resetBaseVelocity(
                env.drone_id, v_vec, [0, 0, 0], physicsClientId=env.client_id
            )

            deploy_obs, priv = env._get_observations()
            obs_vec = deploy_obs.to_vector()

            # PPO prediction
            raw_ppo = ppo_ctrl.predict(obs_vec)
            with torch.no_grad():
                t_obs = torch.as_tensor(obs_vec, dtype=torch.float32).unsqueeze(0)
                ppo_head_logits = ppo_ctrl.model.head(ppo_ctrl.model.backbone(t_obs)).squeeze(0).numpy().tolist()

            # BC prediction
            raw_bc = bc_ctrl.predict(obs_vec)
            with torch.no_grad():
                bc_head_logits = bc_ctrl.model.head(bc_ctrl.model.backbone(t_obs)).squeeze(0).numpy().tolist()

            # Supervisor filtering for PPO
            front_dist = compute_forward_clearance_m(obs_vec, max_range_m=env.sensor_config.max_range_m)
            safe_ppo, s_state_ppo, s_reason_ppo = supervisor.filter_action(
                proposed_action=raw_ppo,
                current_position=priv.global_position,
                current_velocity=priv.global_velocity,
                min_front_range=front_dist,
                dt=env.dt,
                is_localization_healthy=True,
                telemetry_age_s=0.0,
                observation=obs_vec,
            )

            # Supervisor filtering for BC
            safe_bc, s_state_bc, s_reason_bc = supervisor.filter_action(
                proposed_action=raw_bc,
                current_position=priv.global_position,
                current_velocity=priv.global_velocity,
                min_front_range=front_dist,
                dt=env.dt,
                is_localization_healthy=True,
                telemetry_age_s=0.0,
                observation=obs_vec,
            )

            probe_records.append({
                "mode": v_name,
                "x": 1.0,
                "y": y_pos,
                "z": 1.0,
                "vx_init": v_vec[0],
                "left_wall_dist_m": round(1.0 - y_pos, 4),
                "right_wall_dist_m": round(1.0 + y_pos, 4),
                "observation_23": [round(float(x), 4) for x in obs_vec],
                "ppo": {
                    "head_logits": [round(float(x), 4) for x in ppo_head_logits],
                    "action": {
                        "vx": round(float(raw_ppo.vx), 4),
                        "vy": round(float(raw_ppo.vy), 4),
                        "vz": round(float(raw_ppo.vz), 4),
                        "yaw_rate": round(float(raw_ppo.yaw_rate), 4),
                    },
                    "safe_action": {
                        "vx": round(float(safe_ppo.vx), 4),
                        "vy": round(float(safe_ppo.vy), 4),
                        "vz": round(float(safe_ppo.vz), 4),
                        "yaw_rate": round(float(safe_ppo.yaw_rate), 4),
                    },
                    "safety_state": s_state_ppo.value,
                },
                "bc": {
                    "head_logits": [round(float(x), 4) for x in bc_head_logits],
                    "action": {
                        "vx": round(float(raw_bc.vx), 4),
                        "vy": round(float(raw_bc.vy), 4),
                        "vz": round(float(raw_bc.vz), 4),
                        "yaw_rate": round(float(raw_bc.yaw_rate), 4),
                    },
                    "safe_action": {
                        "vx": round(float(safe_bc.vx), 4),
                        "vy": round(float(safe_bc.vy), 4),
                        "vz": round(float(safe_bc.vz), 4),
                        "yaw_rate": round(float(safe_bc.yaw_rate), 4),
                    },
                    "safety_state": s_state_bc.value,
                }
            })

    env.close()
    return {"probe_records": probe_records}


def evaluate_mirrored_obstacle_pairs(
    ppo_ctrl: PPOPolicyController,
    bc_ctrl: BehaviorCloningPolicyController,
) -> List[Dict[str, Any]]:
    """Evaluates PPO and BC approaching mirrored obstacle positions at X=3.0m."""
    test_y_offsets = [-0.25, -0.15, 0.00, +0.15, +0.25]
    results = []

    for y_obs in test_y_offsets:
        for ctrl_name, ctrl in [("ppo", ppo_ctrl), ("bc", bc_ctrl)]:
            env = CorridorSimEnv(scenario="10m", gui=False, seed=1)
            supervisor = SafetySupervisor(corridor=env.corridor)
            obs_vec, info = env.reset(seed=1)

            # Move Obstacle 1 to (3.0, y_obs, 1.0)
            p.resetBasePositionAndOrientation(
                env.obstacle_ids[0], [3.0, y_obs, 1.0], [0, 0, 0, 1], physicsClientId=env.client_id
            )
            env.obstacles[0].center = np.array([3.0, y_obs, 1.0])

            # Measure geometric gaps
            r_obs = 0.20
            r_drone = 0.15
            left_gap = float(1.0 - (y_obs + r_obs))
            right_gap = float((y_obs - r_obs) - (-1.0))
            wider_side = "left" if left_gap > right_gap else ("right" if right_gap > left_gap else "symmetric")
            
            # Supervisor buffer rules: wall buffer = 0.25m, obstacle buffer = 0.30m, drone diameter = 0.30m
            # Required gap for nominal cruise: 0.25 + 0.30 + 0.30 = 0.85m
            req_gap_nominal = 0.85
            left_feasible = left_gap >= req_gap_nominal
            right_feasible = right_gap >= req_gap_nominal
            if left_feasible and right_feasible:
                feasibility = "both_feasible"
            elif left_feasible:
                feasibility = "left_only"
            elif right_feasible:
                feasibility = "right_only"
            else:
                feasibility = "neither_nominal"

            # Rollout until drone passes Obstacle 1 (X >= 3.8m) or crashes/terminates
            trajectory = []
            steps = 0
            done = False
            interventions = 0
            term_reason = "passed_obs1"
            min_obs_clear = 10.0
            min_wall_clear = 10.0

            # Record initial approach action at X=2.0m
            approach_sample = None

            current_info = info
            while not done and steps < 200:
                priv = current_info["privileged_state"]
                pos = priv.global_position.copy()
                vel = priv.global_velocity.copy()
                trajectory.append(pos.tolist())

                min_obs_clear = min(min_obs_clear, float(priv.obstacle_clearance))
                min_wall_clear = min(min_wall_clear, float(priv.corridor_boundary_distance))

                raw_act = ctrl.predict(obs_vec)
                front_dist = compute_forward_clearance_m(obs_vec, max_range_m=env.sensor_config.max_range_m)

                safe_act, s_state, s_reason = supervisor.filter_action(
                    proposed_action=raw_act,
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

                # Capture approach probe when drone is 1.0m before obstacle (X in [1.9, 2.1])
                if approach_sample is None and 1.95 <= pos[0] <= 2.15:
                    with torch.no_grad():
                        t_obs = torch.as_tensor(obs_vec, dtype=torch.float32).unsqueeze(0)
                        h_logits = ctrl.model.head(ctrl.model.backbone(t_obs)).squeeze(0).numpy().tolist()
                    approach_sample = {
                        "drone_x": round(float(pos[0]), 3),
                        "drone_y": round(float(pos[1]), 3),
                        "front_clearance_m": round(float(front_dist), 3),
                        "rays_m": [round(float(x) * 4.0, 3) for x in obs_vec[0:12]],
                        "raw_head_logits": [round(float(x), 4) for x in h_logits],
                        "raw_vy": round(float(raw_act.vy), 4),
                        "raw_yaw_rate": round(float(raw_act.yaw_rate), 4),
                        "safe_vy": round(float(safe_act.vy), 4),
                    }

                norm_action = safe_act.to_normalized(env.action_limits)
                obs_vec, reward, terminated, truncated, current_info = env.step(norm_action)
                steps += 1

                if terminated or truncated:
                    done = True
                    term_reason = current_info.get("termination_reason", "unknown")
                elif pos[0] >= 3.8:
                    done = True
                    term_reason = "passed_obs1"

            env.close()

            # Determine bypass side
            pts_near_obs = [p for p in trajectory if 2.7 <= p[0] <= 3.3]
            if pts_near_obs:
                avg_y_near = float(np.mean([p[1] for p in pts_near_obs]))
                chosen_side = "left" if avg_y_near > y_obs else "right"
            else:
                avg_y_near = float(trajectory[-1][1])
                chosen_side = "left" if avg_y_near > y_obs else "right"

            chose_constrained = (chosen_side == "left" and not left_feasible) or (chosen_side == "right" and not right_feasible)

            results.append({
                "scenario_type": "controlled_y_shift",
                "controller": ctrl_name,
                "obstacle_y": y_obs,
                "left_gap_m": round(left_gap, 4),
                "right_gap_m": round(right_gap, 4),
                "wider_side": wider_side,
                "feasibility": feasibility,
                "chosen_side": chosen_side,
                "chose_constrained": chose_constrained,
                "avg_drone_y_at_obs": round(avg_y_near, 4),
                "steps": steps,
                "interventions": interventions,
                "min_obs_clearance_m": round(min_obs_clear, 4),
                "min_wall_clearance_m": round(min_wall_clear, 4),
                "outcome": term_reason,
                "approach_sample": approach_sample,
            })

    return results


def run_full_episodes_on_seeds(
    seeds: List[int],
    ppo_ctrl: PPOPolicyController,
    bc_ctrl: BehaviorCloningPolicyController,
) -> List[Dict[str, Any]]:
    """Runs full episodes for PPO and BC on diagnostic seeds."""
    full_records = []

    for s in seeds:
        for ctrl_name, ctrl in [("ppo", ppo_ctrl), ("bc", bc_ctrl)]:
            env = CorridorSimEnv(scenario="10m", gui=False, seed=s)
            supervisor = SafetySupervisor(corridor=env.corridor)
            obs_vec, info = env.reset(seed=s)

            # Record obstacle placement
            obs_specs = []
            for idx, o in enumerate(env.obstacles):
                y_c = float(o.center[1])
                r = float(o.radius)
                lg = round(float(1.0 - (y_c + r)), 4)
                rg = round(float((y_c - r) - (-1.0)), 4)
                obs_specs.append({
                    "id": idx + 1,
                    "center": [float(x) for x in o.center],
                    "radius": r,
                    "left_gap_m": lg,
                    "right_gap_m": rg,
                    "wider_side": "left" if lg > rg else ("right" if rg > lg else "symmetric"),
                    "feasibility": "both_feasible" if (lg >= 0.85 and rg >= 0.85) else ("left_only" if lg >= 0.85 else ("right_only" if rg >= 0.85 else "neither_nominal")),
                })

            trajectory = [info["privileged_state"].global_position.tolist()]
            steps = 0
            done = False
            interventions = 0
            term_reason = "timeout"
            min_obs_clear = 10.0
            min_wall_clear = 10.0

            current_info = info
            while not done and steps < env.corridor.max_steps:
                priv = current_info["privileged_state"]
                pos = priv.global_position.copy()
                vel = priv.global_velocity.copy()

                min_obs_clear = min(min_obs_clear, float(priv.obstacle_clearance))
                min_wall_clear = min(min_wall_clear, float(priv.corridor_boundary_distance))

                raw_act = ctrl.predict(obs_vec)
                front_dist = compute_forward_clearance_m(obs_vec, max_range_m=env.sensor_config.max_range_m)

                safe_act, s_state, s_reason = supervisor.filter_action(
                    proposed_action=raw_act,
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

                norm_action = safe_act.to_normalized(env.action_limits)
                obs_vec, reward, terminated, truncated, current_info = env.step(norm_action)
                steps += 1
                trajectory.append(current_info["privileged_state"].global_position.tolist())

                if terminated or truncated:
                    done = True
                    term_reason = current_info.get("termination_reason", "unknown")

            env.close()

            # Analyze route choices per obstacle
            encounters = []
            for o in obs_specs:
                obs_x = o["center"][0]
                obs_y = o["center"][1]
                near_pts = [p for p in trajectory if abs(p[0] - obs_x) <= 0.4]
                if near_pts:
                    avg_y = float(np.mean([p[1] for p in near_pts]))
                    chosen_side = "left" if avg_y > obs_y else "right"
                    chose_constrained = (chosen_side == "left" and o["feasibility"] == "right_only") or (chosen_side == "right" and o["feasibility"] == "left_only")
                    encounters.append({
                        "obstacle_id": o["id"],
                        "obs_x": obs_x,
                        "obs_y": obs_y,
                        "left_gap_m": o["left_gap_m"],
                        "right_gap_m": o["right_gap_m"],
                        "wider_side": o["wider_side"],
                        "feasibility": o["feasibility"],
                        "chosen_side": chosen_side,
                        "chose_constrained": chose_constrained,
                        "avg_drone_y": round(avg_y, 4),
                    })
                else:
                    encounters.append({
                        "obstacle_id": o["id"],
                        "obs_x": obs_x,
                        "obs_y": obs_y,
                        "status": "not_reached",
                    })

            full_records.append({
                "seed": s,
                "controller": ctrl_name,
                "outcome": term_reason,
                "total_steps": steps,
                "interventions": interventions,
                "min_obs_clearance_m": round(min_obs_clear, 4),
                "min_wall_clearance_m": round(min_wall_clear, 4),
                "initial_pos": trajectory[0],
                "obstacles": obs_specs,
                "encounters": encounters,
                "final_pos": trajectory[-1],
                "trajectory_sample": trajectory[::10],
            })

    return full_records


def main():
    ppo_path = "models/checkpoints/ppo_policy_best.pt"
    bc_path = "models/checkpoints/bc_policy_best.pt"
    ppo_sha = get_file_sha256(ppo_path)
    bc_sha = get_file_sha256(bc_path)

    print(f"Loading PPO Controller ({ppo_path}) | SHA-256: {ppo_sha}")
    ppo_ctrl = PPOPolicyController(checkpoint_path=ppo_path)
    assert ppo_ctrl.metadata.is_implemented, "PPO failed to load"

    print(f"Loading BC Controller ({bc_path}) | SHA-256: {bc_sha}")
    bc_ctrl = BehaviorCloningPolicyController(checkpoint_path=bc_path)
    assert bc_ctrl.metadata.is_implemented, "BC failed to load"

    # Step A & B: Open space probe
    print("\n[Phase 4F-1] Running Open Space Lateral Probes...")
    open_space_data = probe_open_space(ppo_ctrl, bc_ctrl)

    # Step C: Controlled mirrored/shifted obstacle pairs
    print("[Phase 4F-1] Running Controlled Mirrored Obstacle Tests...")
    mirrored_data = evaluate_mirrored_obstacle_pairs(ppo_ctrl, bc_ctrl)

    # Step D: Diagnostic seeds (with replay control for seed 1)
    diagnostic_seeds = [1, 1, 20, 100, 2000, 8000]
    print("[Phase 4F-1] Running Full Seed Evaluations across [1 (replay), 20, 100, 2000, 8000]...")
    seed_records = run_full_episodes_on_seeds(diagnostic_seeds, ppo_ctrl, bc_ctrl)

    # Combine into metrics JSON
    metrics_json = {
        "checkpoints": {
            "ppo_policy_best": {"path": ppo_path, "sha256": ppo_sha},
            "bc_policy_best": {"path": bc_path, "sha256": bc_sha},
        },
        "open_space_probes": open_space_data["probe_records"],
        "mirrored_obstacle_tests": mirrored_data,
        "seed_evaluation_records": seed_records,
    }

    out_json_path = Path("reports/phase4f1_policy_probe_metrics.json")
    out_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json_path, "w") as f:
        json.dump(metrics_json, f, indent=2)
    print(f"Wrote metrics JSON to: {out_json_path}")

    # Build CSV for mirrored and diagnostic route comparison
    csv_path = Path("reports/phase4f1_mirrored_route_comparison.csv")
    with open(csv_path, "w") as f:
        f.write(
            "test_type,scenario_or_seed,controller,checkpoint_sha256,obstacle_id,obs_x,obs_y,"
            "left_gap_m,right_gap_m,wider_side,feasibility,raw_vy,raw_yaw_rate,safe_vy,"
            "chosen_side,chose_constrained,min_obs_clearance_m,min_wall_clearance_m,outcome\n"
        )
        
        # Write mirrored obstacle records
        for m in mirrored_data:
            app = m.get("approach_sample")
            raw_vy = app["raw_vy"] if app else ""
            raw_yr = app["raw_yaw_rate"] if app else ""
            safe_vy = app["safe_vy"] if app else ""
            sha = ppo_sha if m["controller"] == "ppo" else bc_sha
            f.write(
                f"mirrored_shift,Y_obs={m['obstacle_y']:+.2f},{m['controller']},{sha},1,3.0,{m['obstacle_y']:.3f},"
                f"{m['left_gap_m']:.3f},{m['right_gap_m']:.3f},{m['wider_side']},{m['feasibility']},"
                f"{raw_vy},{raw_yr},{safe_vy},{m['chosen_side']},{m['chose_constrained']},"
                f"{m['min_obs_clearance_m']:.3f},{m['min_wall_clearance_m']:.3f},{m['outcome']}\n"
            )

        # Write seed records per obstacle
        for sr in seed_records:
            sha = ppo_sha if sr["controller"] == "ppo" else bc_sha
            for enc in sr["encounters"]:
                if "status" in enc and enc["status"] == "not_reached":
                    f.write(
                        f"seed_eval,seed_{sr['seed']},{sr['controller']},{sha},{enc['obstacle_id']},{enc['obs_x']},{enc['obs_y']:.3f},"
                        f",,,,,,,,not_reached,,,{sr['outcome']}\n"
                    )
                else:
                    f.write(
                        f"seed_eval,seed_{sr['seed']},{sr['controller']},{sha},{enc['obstacle_id']},{enc['obs_x']},{enc['obs_y']:.3f},"
                        f"{enc['left_gap_m']:.3f},{enc['right_gap_m']:.3f},{enc['wider_side']},{enc['feasibility']},"
                        f",,,{enc['chosen_side']},{enc['chose_constrained']},"
                        f"{sr['min_obs_clearance_m']:.3f},{sr['min_wall_clearance_m']:.3f},{sr['outcome']}\n"
                    )

    print(f"Wrote route comparison CSV to: {csv_path}")

    # Print summary highlights
    print("\n" + "="*80)
    print("PHASE 4F-1: OPEN-SPACE CENTERLINE (Y=0.0m, Vx=0.6m/s) COMPARISON")
    print("="*80)
    center_probes = [r for r in open_space_data["probe_records"] if abs(r["y"]) < 1e-4 and r["mode"] == "forward_cruise"]
    for cp in center_probes:
        print(f"PPO Action: vx={cp['ppo']['action']['vx']:+.3f}, vy={cp['ppo']['action']['vy']:+.3f} (Safe: {cp['ppo']['safe_action']['vy']:+.3f}), yr={cp['ppo']['action']['yaw_rate']:+.3f}")
        print(f"BC  Action: vx={cp['bc']['action']['vx']:+.3f}, vy={cp['bc']['action']['vy']:+.3f} (Safe: {cp['bc']['safe_action']['vy']:+.3f}), yr={cp['bc']['action']['yaw_rate']:+.3f}")

    print("\n" + "="*80)
    print("PHASE 4F-1: CONTROLLED MIRRORED OBSTACLE ENCOUNTER SUMMARY")
    print("="*80)
    for m in mirrored_data:
        warn = " [CHOSE CONSTRAINED!]" if m["chose_constrained"] else ""
        print(
            f"Obs Y={m['obstacle_y']:+.2f} ({m['controller'].upper():3s}): "
            f"LeftGap={m['left_gap_m']:.2f}m, RightGap={m['right_gap_m']:.2f}m (Wider: {m['wider_side']:5s}, Feas: {m['feasibility']:13s}) -> "
            f"Chosen: {m['chosen_side'].upper():5s}{warn} | Outcome: {m['outcome']}"
        )


if __name__ == "__main__":
    main()
