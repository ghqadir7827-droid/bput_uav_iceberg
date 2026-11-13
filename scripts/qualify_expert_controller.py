"""qualify_expert_controller.py - Rigorous Multi-Tier Qualification of Scripted APF Expert.

Evaluates candidate expert across:
- Level 1 (Basic, N=50)
- Level 2 (Standard, N=100)
- Level 3 (Challenging, N=50)
- Level 4 (Feasible Boundary, N=25)
Measures success rate, clearances, collisions, timeouts, and supervisor intervention rate.
"""

import json
from pathlib import Path
import time
from typing import Any, Dict, List
import numpy as np

from controllers.apf_expert import ScriptedAPFExpertController
from controllers.safety_supervisor import SafetySupervisor
from core.contracts import compute_forward_clearance_m
from core.physical_dimensions import DifficultyLevel, ScaledEnvironmentFactory
from sim.corridor_env import CorridorSimEnv


def qualify():
    out_dir = Path("experiments/scaled_campaign/exceed_90_campaign")
    out_dir.mkdir(parents=True, exist_ok=True)

    suites = [
        ("Level 1 (Basic)", DifficultyLevel.LEVEL_1_BASIC, list(range(20100, 20150))),
        ("Level 2 (Standard)", DifficultyLevel.LEVEL_2_STANDARD, list(range(20000, 20100))),
        ("Level 3 (Challenging)", DifficultyLevel.LEVEL_3_CHALLENGING, list(range(20200, 20250))),
        ("Level 4 (Feasible)", DifficultyLevel.LEVEL_4_FEASIBILITY_BOUNDARY, [s for s in range(20300, 20350) if s % 2 == 0]),
    ]

    qualification_summary: Dict[str, Any] = {}
    print("=" * 80)
    print(">>> QUALIFYING SCRIPTED APF EXPERT CONTROLLER ACROSS TIERS <<<")
    print("=" * 80)

    for name, lvl_enum, seeds in suites:
        print(f"\n--- Evaluating Expert on {name} (N={len(seeds)}) ---")
        cfg = ScaledEnvironmentFactory.get_environment_config(lvl_enum)
        env = CorridorSimEnv(scenario=cfg, seed=seeds[0])
        supervisor = SafetySupervisor(env.corridor)

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

        succ_count = 0
        timeouts = 0
        collisions = 0
        obs_clears = []
        wall_clears = []
        steps_list = []
        interventions = 0
        total_steps_all = 0

        t0 = time.time()
        for s in seeds:
            obs, info = env.reset(seed=s)
            expert.update_environment(env.corridor, env.obstacles)

            min_obs = float("inf")
            min_wall = float("inf")

            for step in range(1, 1001):
                total_steps_all += 1
                priv = info["privileged_state"]
                pos = priv.global_position
                vel = priv.global_velocity
                expert.set_state(pos, vel)

                action = expert.compute_privileged_expert_action(pos, vel, env.obstacles)

                # Supervisor monitoring
                min_front = compute_forward_clearance_m(obs, max_range_m=4.0)
                safe_act, state, _ = supervisor.filter_action(
                    proposed_action=action,
                    current_position=pos,
                    current_velocity=vel,
                    min_front_range=min_front,
                    dt=env.dt,
                    is_localization_healthy=True,
                    telemetry_age_s=0.0,
                    observation=obs,
                )
                if state.value != "nominal":
                    interventions += 1

                obs, _, term, trunc, info = env.step(safe_act.to_normalized(env.action_limits))

                d_o = float(info.get("obstacle_clearance", 10.0))
                d_w = float(info.get("boundary_distance", 10.0))
                min_obs = min(min_obs, d_o)
                min_wall = min(min_wall, d_w)

                if term or trunc:
                    reason = info.get("termination_reason", "unknown")
                    if reason == "success":
                        succ_count += 1
                        steps_list.append(step)
                        obs_clears.append(min_obs)
                        wall_clears.append(min_wall)
                    elif "crash" in reason:
                        collisions += 1
                    else:
                        timeouts += 1
                    break

        env.close()
        elapsed = time.time() - t0

        succ_pct = (succ_count / len(seeds)) * 100.0
        interv_rate = (interventions / max(1, total_steps_all)) * 100.0

        qualification_summary[name] = {
            "total_episodes": len(seeds),
            "successes": succ_count,
            "success_rate_pct": succ_pct,
            "collisions": collisions,
            "timeouts": timeouts,
            "mean_steps_successful": float(np.mean(steps_list)) if steps_list else 1000.0,
            "min_obs_clearance_m": float(min(obs_clears)) if obs_clears else 0.0,
            "min_wall_clearance_m": float(min(wall_clears)) if wall_clears else 0.0,
            "mean_obs_clearance_m": float(np.mean(obs_clears)) if obs_clears else 0.0,
            "supervisor_intervention_rate_pct": round(interv_rate, 2),
            "elapsed_time_s": round(elapsed, 1),
        }

        print(f"  Success Rate: {succ_count}/{len(seeds)} ({succ_pct:.1f}%) | Collisions: {collisions} | Timeouts: {timeouts} | Interventions: {interv_rate:.2f}% | Elapsed: {elapsed:.1f}s")

    out_file = out_dir / "expert_controller_qualification.json"
    with open(out_file, "w") as f:
        json.dump(qualification_summary, f, indent=2)

    print(f"\nSaved expert qualification metrics to: {out_file}")
    return qualification_summary


if __name__ == "__main__":
    qualify()

