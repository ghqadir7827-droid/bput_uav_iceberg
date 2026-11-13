"""reproduce_expert.py - Phase 4J-A3 Reproducible Expert Comparison.

Evaluates ScriptedAPFExpertController across 10 development seeds (1000..1009):
1. Config A: Canonical SafetySupervisor Enabled (obstacle_caution_dist_m = 0.80m)
2. Config B: SafetySupervisor Disabled (pure diagnostic counterfactual)

Verifies and logs:
- Success rate, steps, duration (s)
- Min obstacle clearance, min wall clearance
- Terminal speed and altitude
- Collisions, deadlocks, timeouts
"""

import json
from pathlib import Path
import time
from typing import Any, Dict, List, Tuple
import numpy as np

from controllers.apf_expert import ScriptedAPFExpertController
from controllers.safety_supervisor import SafetyState, SafetySupervisor
from core.contracts import VelocityAction, compute_forward_clearance_m
from sim.corridor_env import CorridorSimEnv


DEVELOPMENT_SEEDS = list(range(1000, 1010))


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        return super().default(obj)


def evaluate_expert_configuration(
    use_supervisor: bool,
    seeds: List[int] = DEVELOPMENT_SEEDS,
    max_steps: int = 500,
) -> Dict[str, Any]:
    config_tag = "WITH_CANONICAL_SUPERVISOR" if use_supervisor else "SUPERVISOR_DISABLED_COUNTERFACTUAL"
    print(f"\nEvaluating Expert Controller [{config_tag}] across {len(seeds)} seeds...")

    env = CorridorSimEnv(scenario="10m", gui=False, seed=seeds[0])
    expert = ScriptedAPFExpertController(corridor=env.corridor)
    supervisor = SafetySupervisor(corridor=env.corridor) if use_supervisor else None

    episodes = []
    successes = 0
    collisions = 0
    timeouts = 0

    for seed in seeds:
        obs, info = env.reset(seed=seed)
        expert.update_environment(env.corridor, env.obstacles)
        if supervisor is not None:
            supervisor.loss_timer_s = 0.0
            supervisor.stalled_timer_s = 0.0
            supervisor.last_safe_action = VelocityAction(0.0, 0.0, 0.0, 0.0)

        ep_start_pos = env._get_drone_position()
        done = False
        steps = 0
        min_obs = 10.0
        min_wall = 10.0
        interventions = 0

        current_info = info
        while not done and steps < max_steps:
            priv = current_info["privileged_state"]
            min_obs = min(min_obs, float(priv.obstacle_clearance))
            min_wall = min(min_wall, float(priv.corridor_boundary_distance))

            raw_action = expert.compute_privileged_expert_action(
                position=priv.global_position,
                velocity=priv.global_velocity,
                obstacles=env.obstacles,
            )

            clamped_action = VelocityAction(
                vx=float(np.clip(raw_action.vx, env.action_limits.min_fwd_vel_mps, env.action_limits.max_fwd_vel_mps)),
                vy=float(np.clip(raw_action.vy, -env.action_limits.max_lat_vel_mps, env.action_limits.max_lat_vel_mps)),
                vz=float(np.clip(raw_action.vz, -env.action_limits.max_vert_vel_mps, env.action_limits.max_vert_vel_mps)),
                yaw_rate=float(np.clip(raw_action.yaw_rate, -env.action_limits.max_yaw_rate_radps, env.action_limits.max_yaw_rate_radps)),
            )

            if supervisor is not None:
                front_dist = compute_forward_clearance_m(obs, max_range_m=env.sensor_config.max_range_m)
                safe_action, s_state, s_reason = supervisor.filter_action(
                    proposed_action=clamped_action,
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
            else:
                safe_action = clamped_action

            norm_action = safe_action.to_normalized(env.action_limits)
            obs, rew, term, trunc, current_info = env.step(norm_action)
            steps += 1
            done = term or trunc

        final_pos = env._get_drone_position()
        final_vel = float(np.linalg.norm(current_info["privileged_state"].global_velocity))
        final_alt = float(final_pos[2])
        final_goal_dist = float(current_info["privileged_state"].goal_distance)
        fwd_prog = float(final_pos[0] - ep_start_pos[0])
        term_reason = current_info.get("termination_reason", "unknown")

        is_success = (term_reason == "success") and (final_vel <= 0.35)
        is_collision = ("crash" in term_reason)
        is_timeout = (steps >= max_steps and not is_success and not is_collision)

        if is_success:
            successes += 1
        elif is_collision:
            collisions += 1
        elif is_timeout:
            timeouts += 1

        episodes.append({
            "seed": seed,
            "steps": steps,
            "duration_s": round(steps * env.dt, 2),
            "status": term_reason,
            "is_success": is_success,
            "forward_progress_m": round(fwd_prog, 3),
            "final_goal_distance_m": round(final_goal_dist, 3),
            "min_obs_clearance_m": round(min_obs, 3),
            "min_wall_clearance_m": round(min_wall, 3),
            "terminal_speed_mps": round(final_vel, 3),
            "terminal_altitude_m": round(final_alt, 3),
            "interventions": interventions,
            "intervention_rate": round(interventions / max(1, steps), 3),
        })

        print(f"  Seed {seed}: Steps={steps:3d} ({steps*env.dt:5.2f}s) | Status={term_reason:<8} | Success={is_success!s:<5} | Fwd={fwd_prog:4.2f}m | MinObs={min_obs:.3f}m | MinWall={min_wall:.3f}m | v_term={final_vel:.3f}m/s")

    env.close()

    total = len(seeds)
    summary = {
        "configuration": config_tag,
        "use_supervisor": use_supervisor,
        "total_episodes": total,
        "success_count": successes,
        "success_rate_pct": round((successes / total) * 100.0, 1),
        "collision_count": collisions,
        "collision_rate_pct": round((collisions / total) * 100.0, 1),
        "timeout_count": timeouts,
        "timeout_rate_pct": round((timeouts / total) * 100.0, 1),
        "mean_steps": round(float(np.mean([e["steps"] for e in episodes])), 1),
        "mean_duration_s": round(float(np.mean([e["duration_s"] for e in episodes])), 2),
        "mean_forward_progress_m": round(float(np.mean([e["forward_progress_m"] for e in episodes])), 3),
        "mean_min_obs_clearance_m": round(float(np.mean([e["min_obs_clearance_m"] for e in episodes])), 3),
        "mean_min_wall_clearance_m": round(float(np.mean([e["min_wall_clearance_m"] for e in episodes])), 3),
        "mean_terminal_speed_mps": round(float(np.mean([e["terminal_speed_mps"] for e in episodes])), 3),
        "mean_terminal_altitude_m": round(float(np.mean([e["terminal_altitude_m"] for e in episodes])), 3),
        "mean_intervention_rate": round(float(np.mean([e["intervention_rate"] for e in episodes])), 3) if use_supervisor else 0.0,
    }

    return {"summary": summary, "episodes": episodes}


def run_reproduction_suite():
    out_dir = Path("experiments/phase4j")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. With Canonical Supervisor
    res_with_sup = evaluate_expert_configuration(use_supervisor=True)

    # 2. Without Supervisor (Diagnostic Counterfactual)
    res_no_sup = evaluate_expert_configuration(use_supervisor=False)

    full_report = {
        "with_canonical_supervisor": res_with_sup,
        "supervisor_disabled_counterfactual": res_no_sup,
        "timestamp_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    report_file = out_dir / "expert_reproduction_metrics.json"
    with open(report_file, "w") as f:
        json.dump(full_report, f, indent=2, cls=NumpyEncoder)

    print("\n" + "=" * 80)
    print(">>> EXPERT REPRODUCTION SUMMARY <<<")
    print(f"Config A (With Canonical Supervisor): Success: {res_with_sup['summary']['success_rate_pct']}% | Mean Steps: {res_with_sup['summary']['mean_steps']} | Progress: {res_with_sup['summary']['mean_forward_progress_m']:.2f}m")
    print(f"Config B (Supervisor Disabled):       Success: {res_no_sup['summary']['success_rate_pct']}% | Mean Steps: {res_no_sup['summary']['mean_steps']} | Progress: {res_no_sup['summary']['mean_forward_progress_m']:.2f}m")
    print("=" * 80)


if __name__ == "__main__":
    run_reproduction_suite()
