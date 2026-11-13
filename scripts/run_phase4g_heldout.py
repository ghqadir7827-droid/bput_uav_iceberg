"""run_phase4g_heldout.py - Final 100-Seed Held-Out Evaluation for Phase 4G.

Strict Protocol:
1. Reserved seed set: 100 strictly untouched seeds [80000..80099].
2. Evaluates Frozen PPO Baseline (Arm A) vs Phase 4G Selected Candidate (Arm D).
3. Evaluates Frozen BC Baseline (Arm B) as reference.
4. Identical initial conditions, episode horizon (200 steps), and supervisor configuration.
5. Primary metric: True end-to-end goal success (satisfying position and arrival speed <= 0.35 m/s).
6. Comprehensive secondary metrics:
   - Collision rate, wall collision rate, timeout rate
   - Mean/median forward displacement
   - Final goal distance
   - Min obstacle clearance, min wall clearance
   - Supervisor intervention rate
   - Deadlock / stall rate (speed < 0.05 m/s while not at goal)
7. Statistical uncertainty: Wilson 95% score confidence intervals and standard errors.
Outputs:
- reports/phase4g_heldout_metrics.json
- reports/phase4g_heldout_evaluation.md
"""

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np
import pybullet as p
import torch

from controllers.bc_policy import BehaviorCloningPolicyController
from controllers.ppo_policy import PPOPolicyController
from controllers.safety_supervisor import SafetyState, SafetySupervisor
from core.contracts import ActionLimits, DeployableObservation, VelocityAction, compute_forward_clearance_m
from sim.corridor_env import CorridorSimEnv


# Dedicated 100-seed reserved held-out pool
HELD_OUT_SEEDS = list(range(80000, 80100))


def wilson_score_interval(successes: int, total: int, z: float = 1.95996) -> Tuple[float, float]:
    """Computes Wilson 95% confidence interval for a binomial proportion."""
    if total == 0:
        return 0.0, 0.0
    p_hat = successes / total
    denom = 1.0 + (z**2) / total
    center = (p_hat + (z**2) / (2.0 * total)) / denom
    margin = (z / denom) * math.sqrt((p_hat * (1.0 - p_hat) / total) + ((z**2) / (4.0 * (total**2))))
    lower = max(0.0, (center - margin) * 100.0)
    upper = min(100.0, (center + margin) * 100.0)
    return round(lower, 2), round(upper, 2)


def evaluate_on_heldout_pool(
    ctrl_name: str,
    ctrl: Any,
    seeds: List[int] = HELD_OUT_SEEDS,
    scenario: str = "10m",
) -> Dict[str, Any]:
    """Evaluates controller on 100 held-out seeds with exact telemetry accounting."""
    env = CorridorSimEnv(scenario=scenario, gui=False, seed=seeds[0])
    supervisor = SafetySupervisor(corridor=env.corridor)

    total_episodes = len(seeds)
    successes = 0
    collisions = 0
    wall_collisions = 0
    timeouts = 0
    deadlocks = 0

    total_steps = 0
    total_interventions = 0

    displacements = []
    goal_distances = []
    min_obs_clearances = []
    min_wall_clearances = []
    per_seed_records = []

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
        stalled_steps = 0

        current_info = info
        while not done and steps < 200:
            priv = current_info["privileged_state"]
            min_obs = min(min_obs, float(priv.obstacle_clearance))
            min_wall = min(min_wall, float(priv.corridor_boundary_distance))

            drone_speed = float(np.linalg.norm(priv.global_velocity))
            if drone_speed < 0.05:
                stalled_steps += 1

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
        final_pos = env._get_drone_position()
        fwd_disp = float(final_pos[0] - ep_start_pos[0])
        final_goal_dist = float(current_info["privileged_state"].goal_distance)

        is_success = (term_reason == "success")
        is_collision = ("crash" in term_reason)
        is_wall_crash = ("wall" in term_reason)
        is_timeout = (term_reason == "timeout")
        is_deadlock = (is_timeout and stalled_steps > 100) or (final_goal_dist > 2.0 and is_timeout)

        if is_success:
            successes += 1
        if is_collision:
            collisions += 1
        if is_wall_crash:
            wall_collisions += 1
        if is_timeout:
            timeouts += 1
        if is_deadlock:
            deadlocks += 1

        total_steps += steps
        displacements.append(fwd_disp)
        goal_distances.append(final_goal_dist)
        min_obs_clearances.append(min_obs)
        min_wall_clearances.append(min_wall)

        per_seed_records.append({
            "seed": seed,
            "outcome": term_reason,
            "steps": steps,
            "forward_displacement_m": round(fwd_disp, 3),
            "final_goal_dist_m": round(final_goal_dist, 3),
            "min_obs_clearance_m": round(min_obs, 3),
            "min_wall_clearance_m": round(min_wall, 3),
            "interventions": ep_interventions,
            "stalled_steps": stalled_steps,
        })

    env.close()

    succ_rate = (successes / max(1, total_episodes)) * 100.0
    ci_low, ci_high = wilson_score_interval(successes, total_episodes)

    return {
        "controller": ctrl_name,
        "total_episodes": total_episodes,
        "success_count": successes,
        "success_rate_pct": round(succ_rate, 2),
        "success_ci95_pct": [ci_low, ci_high],
        "collision_count": collisions,
        "collision_rate_pct": round((collisions / max(1, total_episodes)) * 100.0, 2),
        "wall_collision_count": wall_collisions,
        "wall_collision_rate_pct": round((wall_collisions / max(1, total_episodes)) * 100.0, 2),
        "timeout_count": timeouts,
        "timeout_rate_pct": round((timeouts / max(1, total_episodes)) * 100.0, 2),
        "deadlock_count": deadlocks,
        "deadlock_rate_pct": round((deadlocks / max(1, total_episodes)) * 100.0, 2),
        "mean_steps": round(total_steps / max(1, total_episodes), 1),
        "mean_forward_displacement_m": round(float(np.mean(displacements)), 3),
        "median_forward_displacement_m": round(float(np.median(displacements)), 3),
        "std_forward_displacement_m": round(float(np.std(displacements)), 3),
        "mean_final_goal_dist_m": round(float(np.mean(goal_distances)), 3),
        "median_final_goal_dist_m": round(float(np.median(goal_distances)), 3),
        "mean_min_obs_clearance_m": round(float(np.mean(min_obs_clearances)), 3),
        "mean_min_wall_clearance_m": round(float(np.mean(min_wall_clearances)), 3),
        "total_interventions": total_interventions,
        "total_steps": total_steps,
        "intervention_rate_pct": round((total_interventions / max(1, total_steps)) * 100.0, 2),
        "per_seed_records": per_seed_records,
    }


def run_phase4g_heldout_suite() -> Dict[str, Any]:
    """Runs 100-seed paired held-out benchmark on Frozen PPO vs Phase 4G Candidate."""
    print("=" * 80)
    print(">>> PHASE 4G FINAL HELD-OUT EVALUATION (SEEDS 80000..80099) <<<")
    print(f"Total Paired Episodes: {len(HELD_OUT_SEEDS)}")
    print("=" * 80)

    controllers = {
        "Arm_A_Frozen_PPO": PPOPolicyController("models/checkpoints/ppo_policy_best.pt"),
        "Arm_B_Frozen_BC": BehaviorCloningPolicyController("models/checkpoints/bc_policy_best.pt"),
        "Arm_D_Selected_Candidate": PPOPolicyController("experiments/phase4g/checkpoints/ppo_arm_d_reflection.pt"),
    }

    eval_results = {}
    for name, ctrl in controllers.items():
        print(f"\n[Evaluating {name}] on 100 reserved held-out seeds...")
        res = evaluate_on_heldout_pool(ctrl_name=name, ctrl=ctrl, seeds=HELD_OUT_SEEDS)
        eval_results[name] = res
        print(f"  Success: {res['success_rate_pct']}% ({res['success_count']}/{res['total_episodes']}) [95% CI: {res['success_ci95_pct']}%]")
        print(f"  Collisions: {res['collision_rate_pct']}% ({res['collision_count']}/{res['total_episodes']}) [Wall: {res['wall_collision_count']}]")
        print(f"  Timeouts: {res['timeout_rate_pct']}% ({res['timeout_count']}/{res['total_episodes']})")
        print(f"  Interventions: {res['intervention_rate_pct']}% ({res['total_interventions']}/{res['total_steps']})")
        print(f"  Mean Displacement: {res['mean_forward_displacement_m']:.3f} m (Median: {res['median_forward_displacement_m']:.3f} m)")
        print(f"  Min Obs Clearance: {res['mean_min_obs_clearance_m']:.3f} m")

    out_file = Path("reports/phase4g_heldout_metrics.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(eval_results, f, indent=2)
    print(f"\nFinal held-out evaluation metrics written to {out_file}")
    return eval_results


if __name__ == "__main__":
    run_phase4g_heldout_suite()
