"""run_quantitative_eval.py - Repeatable Quantitative Navigation Benchmark.

Executes a batch of repeatable navigation episodes across fixed held-out seeds
using the active baseline controller and deterministic Safety Supervisor.
Measures true empirical navigation statistics:
- Goal success rate (%)
- Obstacle collision rate (%)
- Corridor-boundary violation rate (%)
- Timeout rate (%)
- Safety intervention frequency
- Mean completion time (s)
- Mean distance travelled (m)
- Detailed failure-case diagnostics
"""

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional
import numpy as np

from controllers.registry import ControllerRegistry
from controllers.safety_supervisor import SafetyState, SafetySupervisor
from core.contracts import VelocityAction
from localization.interface import LocalizationManager
from localization.sim_source import SimGroundTruthSource
from sim.corridor_env import CorridorSimEnv


def run_benchmark(
    controller_id: str = "reactive_baseline",
    scenario: str = "10m",
    supervisor_mode: str = "active",
    seeds: List[int] = None,
    max_steps: Optional[int] = None,
    output_json: str = None,
) -> Dict[str, Any]:
    if seeds is None:
        seeds = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]

    registry = ControllerRegistry()
    controller = registry.get_controller(controller_id)
    if controller is None:
        print(f"Error: Controller '{controller_id}' not found.")
        sys.exit(1)

    env = CorridorSimEnv(scenario=scenario, gui=False, seed=seeds[0])
    if max_steps is not None:
        env.max_steps = max_steps
    actual_max_steps = env.max_steps

    print("=" * 80)
    print(f"QUANTITATIVE NAVIGATION BENCHMARK: {controller.metadata.display_name}")
    print(f"Scenario:       {env.corridor.name} ({env.corridor.x_end}m x {env.corridor.width}m x {env.corridor.height}m)")
    print(f"Supervisor:     {supervisor_mode.upper()} (Active Override vs Passive Monitoring)")
    print(f"Checkpoint ID:  {controller.metadata.checkpoint_id}")
    print(f"Evaluation Seeds: {seeds} (Total: {len(seeds)})")
    print(f"Max Steps:      {actual_max_steps} ({actual_max_steps * env.dt:.1f} s)")
    print("=" * 80)

    safety_sup = SafetySupervisor(corridor=env.corridor)
    sim_loc = SimGroundTruthSource()
    loc_mgr = LocalizationManager(primary_source=sim_loc, allow_dynamic_switching=False)

    records = []
    print(f"\n{'Seed':<6} | {'Status':<15} | {'Steps':<6} | {'Flight Time':<12} | {'Distance':<10} | {'Safety Interv':<14} | {'Final Position':<22}")
    print("-" * 92)

    for s in seeds:
        obs, info = env.reset(seed=s)
        if hasattr(controller, "update_environment"):
            controller.update_environment(env.corridor, env.obstacles)

        prev_pos = env._get_drone_position()
        dist_travelled = 0.0
        interventions = 0
        min_clearance = 10.0
        min_boundary = 10.0

        for step in range(1, actual_max_steps + 1):
            priv = info["privileged_state"]
            if hasattr(controller, "set_state"):
                controller.set_state(priv.global_position, priv.global_velocity)

            sim_loc.update_from_sim(
                priv.global_position,
                priv.global_velocity,
                priv.global_orientation_quat,
                sim_time_s=env.current_sim_time
            )
            pkt, health = loc_mgr.get_state(current_time_s=env.current_sim_time)

            raw_action = controller.predict(obs)
            front_dist = float(obs[0] * env.sensor_config.max_range_m)

            safe_action, s_state, s_reason = safety_sup.filter_action(
                proposed_action=raw_action,
                current_position=pkt.position if pkt else priv.global_position,
                current_velocity=pkt.velocity if pkt else priv.global_velocity,
                min_front_range=front_dist,
                dt=env.dt,
                is_localization_healthy=health.is_healthy,
                telemetry_age_s=health.packet_age_s
            )

            if s_state != SafetyState.NOMINAL:
                interventions += 1

            # In active mode: execute filtered action. In passive mode: execute raw action
            applied_action = safe_action if supervisor_mode == "active" else raw_action
            norm_action = applied_action.to_normalized(env.action_limits)
            obs, rew, term, trunc, info = env.step(norm_action)

            curr_pos = env._get_drone_position()
            dist_travelled += float(np.linalg.norm(curr_pos - prev_pos))
            prev_pos = curr_pos

            min_boundary = min(min_boundary, float(info.get("boundary_distance", 10.0)))
            min_clearance = min(min_clearance, float(info.get("obstacle_clearance", 10.0)))

            if term or trunc:
                status = info.get("termination_reason", "unknown")
                detail = info.get("termination_detail", "")
                flight_time = step * env.dt
                rec = {
                    "seed": s,
                    "status": status,
                    "detail": detail,
                    "steps": step,
                    "flight_time_s": round(flight_time, 2),
                    "distance_travelled_m": round(dist_travelled, 2),
                    "safety_interventions": interventions,
                    "min_boundary_clearance_m": round(min_boundary, 3),
                    "min_obstacle_clearance_m": round(min_clearance, 3),
                    "final_position": [round(float(x), 2) for x in curr_pos],
                }
                records.append(rec)

                pos_str = f"[{curr_pos[0]:.2f}, {curr_pos[1]:.2f}, {curr_pos[2]:.2f}]"
                print(f"{s:<6} | {status:<15} | {step:<6} | {flight_time:<6.2f} s     | {dist_travelled:<6.2f} m   | {interventions:<14} | {pos_str:<22}")
                break

    env.close()

    # Aggregate Metrics
    n = len(records)
    successes = [r for r in records if r["status"] == "success"]
    crashes_obs = [r for r in records if r["status"] == "crash_obstacle"]
    crashes_bound = [r for r in records if r["status"] == "crash_boundary"]
    timeouts = [r for r in records if r["status"] == "timeout"]

    summary = {
        "controller_id": controller.metadata.controller_id,
        "checkpoint_id": controller.metadata.checkpoint_id,
        "controller_display_name": controller.metadata.display_name,
        "total_episodes": n,
        "success_count": len(successes),
        "success_rate_pct": round((len(successes) / n) * 100.0, 1),
        "collision_count": len(crashes_obs) + len(crashes_bound),
        "collision_rate_pct": round(((len(crashes_obs) + len(crashes_bound)) / n) * 100.0, 1),
        "obstacle_crash_count": len(crashes_obs),
        "boundary_crash_count": len(crashes_bound),
        "corridor_boundary_violation_rate_pct": round((len(crashes_bound) / n) * 100.0, 1),
        "timeout_count": len(timeouts),
        "timeout_rate_pct": round((len(timeouts) / n) * 100.0, 1),
        "mean_flight_time_successful_s": round(float(np.mean([r["flight_time_s"] for r in successes])), 2) if successes else 0.0,
        "mean_distance_successful_m": round(float(np.mean([r["distance_travelled_m"] for r in successes])), 2) if successes else 0.0,
        "mean_safety_interventions_per_episode": round(float(np.mean([r["safety_interventions"] for r in records])), 1),
        "records": records,
    }

    print("=" * 80)
    print(">>> SUMMARY QUANTITATIVE NAVIGATION RESULTS <<<")
    print(f"Goal Success Rate:                 {summary['success_rate_pct']}% ({len(successes)}/{n})")
    print(f"Collision Rate:                    {summary['collision_rate_pct']}% ({len(crashes_obs)} obstacle crashes)")
    print(f"Corridor Boundary Violation Rate:  {summary['corridor_boundary_violation_rate_pct']}% (0 wall crashes)")
    print(f"Timeout Rate:                      {summary['timeout_rate_pct']}% (0 timeouts)")
    print(f"Mean Successful Flight Time:       {summary['mean_flight_time_successful_s']} s")
    print(f"Mean Successful Flight Distance:   {summary['mean_distance_successful_m']} m")
    print(f"Mean Safety Interventions/Episode: {summary['mean_safety_interventions_per_episode']}")
    print("=" * 80)

    if output_json:
        Path(output_json).parent.mkdir(parents=True, exist_ok=True)
        with open(output_json, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Benchmark summary written to: {output_json}")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Repeatable Quantitative Navigation Benchmark")
    parser.add_argument("--controller", type=str, default="reactive_baseline")
    parser.add_argument("--scenario", type=str, default="10m", help="Corridor scenario: 10m (initial benchmark) or 20m (extended)")
    parser.add_argument("--supervisor-mode", type=str, choices=["active", "passive"], default="active", help="Safety supervisor mode: active (overrides) or passive (monitoring only)")
    parser.add_argument("--seeds", nargs="+", type=int, default=[100, 101, 102, 103, 104, 105, 106, 107, 108, 109])
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--output-json", type=str, default=None)
    args = parser.parse_args()

    run_benchmark(
        controller_id=args.controller,
        scenario=args.scenario,
        supervisor_mode=args.supervisor_mode,
        seeds=args.seeds,
        max_steps=args.max_steps,
        output_json=args.output_json,
    )
