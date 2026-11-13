"""run_phase4e_comparison.py - Comparative Evaluation of Canonical vs Experimental Hysteresis Supervisor.

Phase 4E-A Isolated Comparative Investigation:
Compares the canonical SafetySupervisor against the ExperimentalHysteresisSupervisor
under strictly equivalent initial conditions:
1. Synthetic Obstacle Benchmark (Head-on, Tight Corridors, Dead-End, Repeated Forward Push).
2. Diagnostic Corridor Simulation Episodes (Seeds 50000, 50001, 100, 101, 102).

Outputs metrics to reports/phase4e_hysteresis_comparative_metrics.json.
"""

import json
import time
from typing import Any, Dict, List
import numpy as np
import torch

from core.contracts import ActionLimits, VelocityAction, compute_forward_clearance_m
from core.geometry import Corridor3D
from controllers.safety_supervisor import SafetySupervisor, SafetySupervisorConfig
from controllers.experimental_hysteresis_supervisor import (
    ExperimentalHysteresisSupervisor,
    HysteresisSupervisorConfig,
    HysteresisSafetyState,
)
from models.actor_critic import ActorCriticPPO
from sim.corridor_env import CorridorSimEnv


def run_synthetic_chattering_benchmark(steps: int = 200, dt: float = 0.05) -> Dict[str, Any]:
    """Tests canonical vs hysteresis supervisor under constant forward policy push into an obstacle."""
    corridor = Corridor3D.get_10m_benchmark()
    limits = ActionLimits()

    canonical_sup = SafetySupervisor(corridor=corridor, limits=limits, config=SafetySupervisorConfig())
    hysteresis_sup = ExperimentalHysteresisSupervisor(corridor=corridor, limits=limits, config=HysteresisSupervisorConfig())

    results = {}

    for name, sup in [("canonical", canonical_sup), ("hysteresis", hysteresis_sup)]:
        pos = np.array([2.5, 0.0, 1.0], dtype=np.float32)
        vel = np.array([0.5, 0.0, 0.0], dtype=np.float32)
        prop = VelocityAction(vx=0.6, vy=0.0, vz=0.0, yaw_rate=0.0)

        sign_reversals = 0
        prev_vx = 0.0
        stationary_steps = 0
        interventions = 0
        states = []
        min_clear = 10.0
        trajectory = []

        for step in range(steps):
            front_clear = max(0.01, 2.65 - pos[0])
            min_clear = min(min_clear, front_clear)

            # Rays: center ray is front_clear, left has 0.6m, right has 0.2m, rear has 1.0m
            rays = np.full(12, fill_value=4.0, dtype=np.float32)
            rays[0] = front_clear
            rays[1] = 0.60; rays[3] = 0.60; rays[5] = 0.60
            rays[2] = 0.20; rays[4] = 0.20; rays[6] = 0.20
            rays[7] = 1.0

            act, state, reason = sup.filter_action(
                proposed_action=prop,
                current_position=pos,
                current_velocity=vel,
                min_front_range=front_clear,
                dt=dt,
                range_rays=rays,
            )

            # Check velocity sign reversals
            if (prev_vx > 1e-4 and act.vx < -1e-4) or (prev_vx < -1e-4 and act.vx > 1e-4):
                sign_reversals += 1
            prev_vx = act.vx

            if abs(act.vx) <= 0.05 and abs(act.vy) <= 0.05:
                stationary_steps += 1

            if abs(act.vx - prop.vx) > 1e-4 or abs(act.vy - prop.vy) > 1e-4:
                interventions += 1

            state_val = state.value if hasattr(state, "value") else str(state)
            states.append(state_val)

            # Update kinematics
            pos[0] += act.vx * dt
            pos[1] += act.vy * dt
            vel[0] = act.vx
            vel[1] = act.vy
            trajectory.append((step, float(pos[0]), float(pos[1]), float(act.vx), float(act.vy), state_val))

        results[name] = {
            "sign_reversals": sign_reversals,
            "stationary_duration_s": round(stationary_steps * dt, 2),
            "intervention_steps": interventions,
            "intervention_rate_pct": round((interventions / steps) * 100.0, 1),
            "final_position": [round(float(pos[0]), 3), round(float(pos[1]), 3), round(float(pos[2]), 3)],
            "net_displacement_m": round(float(pos[0] - 2.5), 3),
            "min_obstacle_clearance_m": round(float(min_clear), 3),
            "sample_trajectory": trajectory[:20],
        }

    return results


def run_diagnostic_corridor_simulation(
    seeds: List[int],
    ckpt_path: str = "models/checkpoints/ppo_phase4b_candidate.pt",
    scenario: str = "10m",
    max_steps: int = 500,
    dt: float = 0.05,
) -> Dict[str, Any]:
    """Runs deterministic simulation on candidate policy with canonical vs hysteresis supervisor."""
    device = torch.device("cpu")
    model = ActorCriticPPO(obs_dim=23, act_dim=4).to(device)
    model.load_ppo_checkpoint(ckpt_path, device=device)
    model.eval()

    corridor = Corridor3D.get_10m_benchmark()
    limits = ActionLimits()

    comparison_results = {}

    for sup_type in ["canonical", "hysteresis"]:
        env = CorridorSimEnv(scenario=scenario, gui=False, seed=seeds[0])
        sup_config_can = SafetySupervisorConfig()
        sup_config_hys = HysteresisSupervisorConfig()

        seed_records = []

        for s in seeds:
            obs, info = env.reset(seed=s)
            if sup_type == "canonical":
                supervisor = SafetySupervisor(corridor=env.corridor, limits=limits, config=sup_config_can)
            else:
                supervisor = ExperimentalHysteresisSupervisor(corridor=env.corridor, limits=limits, config=sup_config_hys)

            done = False
            step = 0
            ep_rew = 0.0
            ep_dist = 0.0
            sign_reversals = 0
            prev_vx = 0.0
            stationary_steps = 0
            interventions = 0
            start_pos = env._get_drone_position().copy()
            prev_pos = start_pos.copy()
            min_obs_clear = 10.0
            min_wall_clear = 10.0

            while not done and step < max_steps:
                step += 1
                with torch.no_grad():
                    t_obs = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                    act, _, _, _ = model.get_action_and_value(t_obs, deterministic=True)
                    act_np = act.squeeze(0).cpu().numpy()

                clamped_act = VelocityAction(
                    vx=float(np.clip(act_np[0], limits.min_fwd_vel_mps, limits.max_fwd_vel_mps)),
                    vy=float(np.clip(act_np[1], -limits.max_lat_vel_mps, limits.max_lat_vel_mps)),
                    vz=float(np.clip(act_np[2], -limits.max_vert_vel_mps, limits.max_vert_vel_mps)),
                    yaw_rate=float(np.clip(act_np[3], -limits.max_yaw_rate_radps, limits.max_yaw_rate_radps))
                )

                priv = info["privileged_state"]
                front_dist = compute_forward_clearance_m(obs, max_range_m=env.sensor_config.max_range_m)

                safe_act, s_state, s_reason = supervisor.filter_action(
                    proposed_action=clamped_act,
                    current_position=priv.global_position,
                    current_velocity=priv.global_velocity,
                    min_front_range=front_dist,
                    dt=dt,
                    observation=obs,
                )

                if (prev_vx > 1e-4 and safe_act.vx < -1e-4) or (prev_vx < -1e-4 and safe_act.vx > 1e-4):
                    sign_reversals += 1
                prev_vx = safe_act.vx

                if abs(safe_act.vx) <= 0.05 and abs(safe_act.vy) <= 0.05:
                    stationary_steps += 1

                if abs(safe_act.vx - clamped_act.vx) > 1e-4 or abs(safe_act.vy - clamped_act.vy) > 1e-4:
                    interventions += 1

                norm_applied = safe_act.to_normalized(limits)
                obs, rew, term, trunc, info = env.step(norm_applied)
                done = term or trunc
                ep_rew += float(rew)

                curr_pos = env._get_drone_position()
                ep_dist += float(np.linalg.norm(curr_pos - prev_pos))
                prev_pos = curr_pos

                b_dist = float(info.get("boundary_distance", 10.0))
                o_dist = float(info.get("obstacle_clearance", 10.0))
                min_wall_clear = min(min_wall_clear, b_dist)
                min_obs_clear = min(min_obs_clear, o_dist)

            final_pos = env._get_drone_position()
            net_fwd = float(final_pos[0] - start_pos[0])
            final_goal_dist = float(np.linalg.norm(final_pos - corridor.goal_position))
            status = info.get("termination_reason", "unknown")

            seed_records.append({
                "seed": s,
                "status": status,
                "steps": step,
                "episode_return": round(ep_rew, 2),
                "distance_travelled_m": round(ep_dist, 2),
                "net_forward_displacement_m": round(net_fwd, 2),
                "final_distance_to_goal_m": round(final_goal_dist, 2),
                "sign_reversals": sign_reversals,
                "stationary_duration_s": round(stationary_steps * dt, 2),
                "interventions": interventions,
                "min_obstacle_clearance_m": round(min_obs_clear, 3),
                "min_wall_clearance_m": round(min_wall_clear, 3),
                "final_position": [round(float(x), 2) for x in final_pos],
            })

        comparison_results[sup_type] = seed_records

    return comparison_results


def main():
    print("=" * 80)
    print(">>> PHASE 4E-A: ISOLATED SUPERVISOR HYSTERESIS COMPARATIVE EVALUATION <<<")
    print("=" * 80)

    # 1. Run Synthetic Benchmark
    print("\n[1/2] Running isolated synthetic chattering benchmark (200 steps)...")
    synthetic_res = run_synthetic_chattering_benchmark(steps=200, dt=0.05)
    print(f"Canonical Supervisor:  {synthetic_res['canonical']['sign_reversals']} sign reversals, "
          f"stationary={synthetic_res['canonical']['stationary_duration_s']}s, "
          f"net_disp={synthetic_res['canonical']['net_displacement_m']}m")
    print(f"Hysteresis Supervisor: {synthetic_res['hysteresis']['sign_reversals']} sign reversals, "
          f"stationary={synthetic_res['hysteresis']['stationary_duration_s']}s, "
          f"net_disp={synthetic_res['hysteresis']['net_displacement_m']}m")

    # 2. Run Diagnostic Corridor Simulation
    # Diagnostic seeds: 50000 (Obs 1 stall), 50001 (Obs 3 stall), and non-heldout 100, 101, 102
    diagnostic_seeds = [50000, 50001, 100, 101, 102]
    print(f"\n[2/2] Running controlled simulation across diagnostic seeds {diagnostic_seeds}...")
    sim_res = run_diagnostic_corridor_simulation(seeds=diagnostic_seeds)

    for i, s in enumerate(diagnostic_seeds):
        can = sim_res["canonical"][i]
        hys = sim_res["hysteresis"][i]
        print(f"\n--- Seed {s} Comparison ---")
        print(f"  Canonical:  status={can['status']}, steps={can['steps']}, "
              f"disp={can['net_forward_displacement_m']}m, reversals={can['sign_reversals']}, "
              f"stationary={can['stationary_duration_s']}s, final_pos={can['final_position']}")
        print(f"  Hysteresis: status={hys['status']}, steps={hys['steps']}, "
              f"disp={hys['net_forward_displacement_m']}m, reversals={hys['sign_reversals']}, "
              f"stationary={hys['stationary_duration_s']}s, final_pos={hys['final_position']}")

    # Save to JSON
    report_data = {
        "benchmark_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "diagnostic_seeds": diagnostic_seeds,
        "synthetic_benchmark": synthetic_res,
        "simulation_benchmark": sim_res,
    }

    out_path = "reports/phase4e_hysteresis_comparative_metrics.json"
    with open(out_path, "w") as f:
        json.dump(report_data, f, indent=2)
    print(f"\n[Saved] Metrics successfully exported to: {out_path}")


if __name__ == "__main__":
    main()
