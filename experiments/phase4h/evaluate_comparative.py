"""evaluate_comparative.py - Phase 4H Final Comparative Held-Out Evaluation.

Evaluates 4 policies across the 100 reserved, pristine held-out seeds (90000..90099)
under the canonical 500-step horizon (25.0s, dt=0.05s):
1. BC Baseline:              models/checkpoints/bc_policy_best.pt
2. PPO Baseline:             models/checkpoints/ppo_policy_best.pt
3. Phase 4H BC Candidate:    experiments/phase4h/checkpoints/bc_phase4h_candidate.pt
4. Phase 4H PPO Candidate:   experiments/phase4h/checkpoints/ppo_phase4h_candidate.pt

Strict protocol requirements:
- Identical initial conditions and scene geometry per seed
- Active canonical SafetySupervisor with full state reset per episode
- Log step-level metrics and episode outcomes:
  - Goal success rate and 95% Wilson score confidence interval
  - Collision rate (obstacle and wall)
  - Timeout rate
  - Deadlock rate (drone halted for > 3.0s before goal)
  - Mean and median forward displacement
  - Mean final goal distance
  - Minimum obstacle and wall clearance
  - Supervisor intervention rate
  - Terminal speed and altitude
"""

import json
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch

from controllers.safety_supervisor import SafetyState, SafetySupervisor
from core.contracts import ActionLimits, VelocityAction, compute_forward_clearance_m
from core.reflection import reflect_observation_np, reflect_action_np
from models.actor_critic import ActorCriticPPO
from models.policy_net import NavigationPolicyMLP
from sim.corridor_env import CorridorSimEnv


HELD_OUT_SEEDS = list(range(90000, 90100)) # Exactly 100 reserved seeds


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


def wilson_score_interval(successes: int, total: int, z: float = 1.96) -> Tuple[float, float]:
    """Calculates Wilson score 95% confidence interval for a binomial proportion."""
    if total == 0:
        return 0.0, 0.0
    p = successes / total
    denom = 1.0 + (z**2) / total
    center = (p + (z**2) / (2.0 * total)) / denom
    margin = (z / denom) * np.sqrt((p * (1.0 - p) / total) + (z**2) / (4.0 * (total**2)))
    low = max(0.0, float(center - margin))
    high = min(1.0, float(center + margin))
    return round(low * 100.0, 1), round(high * 100.0, 1)


def evaluate_policy_on_heldout(
    policy_name: str,
    policy_type: str, # "bc" or "ppo"
    checkpoint_path: str,
    seeds: List[int] = HELD_OUT_SEEDS,
    max_steps: int = 500,
    device: torch.device = torch.device("cpu"),
    use_reflection: bool = False,
) -> Dict[str, Any]:
    print(f"\nEvaluating {policy_name} on {len(seeds)} held-out seeds (Horizon: {max_steps} steps)...")
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Load model
    if policy_type == "bc":
        model = NavigationPolicyMLP(obs_dim=23, act_dim=4).to(device)
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
    elif policy_type == "ppo":
        model = ActorCriticPPO(obs_dim=23, act_dim=4).to(device)
        success = model.load_ppo_checkpoint(checkpoint_path, device=device)
        assert success, f"Failed to load PPO checkpoint: {checkpoint_path}"
    else:
        raise ValueError(f"Unknown policy type: {policy_type}")

    model.eval()

    env = CorridorSimEnv(scenario="10m", gui=False, seed=seeds[0])
    supervisor = SafetySupervisor(corridor=env.corridor)

    episode_results = []
    successes = 0
    collisions = 0
    timeouts = 0
    deadlocks = 0

    forward_progresses = []
    final_goal_distances = []
    min_obs_clearances = []
    min_wall_clearances = []
    terminal_speeds = []
    terminal_altitudes = []
    intervention_rates = []

    for seed_idx, seed in enumerate(seeds):
        obs, info = env.reset(seed=seed)
        supervisor.loss_timer_s = 0.0
        supervisor.stalled_timer_s = 0.0
        supervisor.last_safe_action = VelocityAction(0.0, 0.0, 0.0, 0.0)

        ep_start_pos = env._get_drone_position()
        done = False
        steps = 0
        ep_interventions = 0
        min_obs = 10.0
        min_wall = 10.0
        stalled_steps = 0

        current_info = info
        while not done and steps < max_steps:
            priv = current_info["privileged_state"]
            min_obs = min(min_obs, float(priv.obstacle_clearance))
            min_wall = min(min_wall, float(priv.corridor_boundary_distance))

            with torch.no_grad():
                t_obs = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                if policy_type == "bc":
                    raw_act = model(t_obs, bounded=True).squeeze(0).cpu().numpy()
                else:
                    raw_act = model.actor(t_obs, bounded=True).squeeze(0).cpu().numpy()

            prop_act = VelocityAction(vx=float(raw_act[0]), vy=float(raw_act[1]), vz=float(raw_act[2]), yaw_rate=float(raw_act[3]))
            clamped_act = VelocityAction(
                vx=float(np.clip(prop_act.vx, env.action_limits.min_fwd_vel_mps, env.action_limits.max_fwd_vel_mps)),
                vy=float(np.clip(prop_act.vy, -env.action_limits.max_lat_vel_mps, env.action_limits.max_lat_vel_mps)),
                vz=float(np.clip(prop_act.vz, -env.action_limits.max_vert_vel_mps, env.action_limits.max_vert_vel_mps)),
                yaw_rate=float(np.clip(prop_act.yaw_rate, -env.action_limits.max_yaw_rate_radps, env.action_limits.max_yaw_rate_radps)),
            )

            front_dist = compute_forward_clearance_m(obs, max_range_m=env.sensor_config.max_range_m)
            safe_act, s_state, s_reason = supervisor.filter_action(
                proposed_action=clamped_act,
                current_position=priv.global_position,
                current_velocity=priv.global_velocity,
                min_front_range=front_dist,
                dt=env.dt,
                is_localization_healthy=True,
                telemetry_age_s=0.0,
                observation=obs,
            )

            is_intervened = (s_state != SafetyState.NOMINAL)
            if is_intervened:
                ep_interventions += 1

            # Check deadlock (velocity < 0.05 m/s when not at goal)
            curr_speed = float(np.linalg.norm(priv.global_velocity))
            if curr_speed < 0.05 and priv.goal_distance > 0.50:
                stalled_steps += 1
            else:
                stalled_steps = 0

            norm_act = safe_act.to_normalized(env.action_limits)
            obs, rew, term, trunc, current_info = env.step(norm_act)
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
        is_deadlock = (stalled_steps >= 60) or (steps >= max_steps and final_vel < 0.05 and final_goal_dist > 0.50)
        is_timeout = (steps >= max_steps and not is_success and not is_collision)

        if is_success:
            successes += 1
        elif is_collision:
            collisions += 1
        elif is_deadlock:
            deadlocks += 1
        elif is_timeout:
            timeouts += 1

        forward_progresses.append(fwd_prog)
        final_goal_distances.append(final_goal_dist)
        min_obs_clearances.append(min_obs)
        min_wall_clearances.append(min_wall)
        terminal_speeds.append(final_vel)
        terminal_altitudes.append(final_alt)
        intervention_rates.append(ep_interventions / max(1, steps))

        episode_results.append({
            "seed": seed,
            "status": "success" if is_success else term_reason,
            "is_success": is_success,
            "is_collision": is_collision,
            "is_deadlock": is_deadlock,
            "is_timeout": is_timeout,
            "steps": steps,
            "forward_progress_m": round(fwd_prog, 3),
            "final_goal_distance_m": round(final_goal_dist, 3),
            "min_obs_clearance_m": round(min_obs, 3),
            "min_wall_clearance_m": round(min_wall, 3),
            "terminal_speed_mps": round(final_vel, 3),
            "terminal_altitude_m": round(final_alt, 3),
            "interventions": ep_interventions,
        })

        if (seed_idx + 1) % 25 == 0 or (seed_idx + 1) == len(seeds):
            print(f"  Processed {seed_idx + 1:3d}/{len(seeds)} | Successes: {successes} | Collisions: {collisions} | Deadlocks: {deadlocks} | Timeouts: {timeouts}")

    env.close()

    total_eps = len(seeds)
    succ_rate = round((successes / total_eps) * 100.0, 1)
    ci_low, ci_high = wilson_score_interval(successes, total_eps)

    summary = {
        "policy_name": policy_name,
        "policy_type": policy_type,
        "checkpoint": checkpoint_path,
        "total_episodes": total_eps,
        "horizon_steps": max_steps,
        "success_rate_pct": succ_rate,
        "success_ci_95_pct": [ci_low, ci_high],
        "success_count": successes,
        "collision_rate_pct": round((collisions / total_eps) * 100.0, 1),
        "collision_count": collisions,
        "deadlock_rate_pct": round((deadlocks / total_eps) * 100.0, 1),
        "deadlock_count": deadlocks,
        "timeout_rate_pct": round((timeouts / total_eps) * 100.0, 1),
        "timeout_count": timeouts,
        "mean_forward_progress_m": round(float(np.mean(forward_progresses)), 3),
        "median_forward_progress_m": round(float(np.median(forward_progresses)), 3),
        "mean_final_goal_distance_m": round(float(np.mean(final_goal_distances)), 3),
        "mean_min_obs_clearance_m": round(float(np.mean(min_obs_clearances)), 3),
        "mean_min_wall_clearance_m": round(float(np.mean(min_wall_clearances)), 3),
        "mean_terminal_speed_mps": round(float(np.mean(terminal_speeds)), 3),
        "mean_terminal_altitude_m": round(float(np.mean(terminal_altitudes)), 3),
        "mean_intervention_rate": round(float(np.mean(intervention_rates)), 3),
    }

    return {"summary": summary, "episodes": episode_results}


def run_full_comparative_benchmark(
    output_report_json: str = "reports/phase4h_comparative_metrics.json",
) -> Dict[str, Any]:
    policies = [
        {
            "name": "Arm_B_BC_Baseline",
            "type": "bc",
            "checkpoint": "models/checkpoints/bc_policy_best.pt",
        },
        {
            "name": "Arm_A_PPO_Baseline",
            "type": "ppo",
            "checkpoint": "models/checkpoints/ppo_policy_best.pt",
        },
        {
            "name": "BC_Phase4H_Candidate",
            "type": "bc",
            "checkpoint": "experiments/phase4h/checkpoints/bc_phase4h_candidate.pt",
        },
        {
            "name": "PPO_Phase4H_Candidate",
            "type": "ppo",
            "checkpoint": "experiments/phase4h/checkpoints/ppo_phase4h_candidate.pt",
        },
    ]

    all_results = {}
    summaries = {}

    for pol in policies:
        res = evaluate_policy_on_heldout(
            policy_name=pol["name"],
            policy_type=pol["type"],
            checkpoint_path=pol["checkpoint"],
            seeds=HELD_OUT_SEEDS,
            max_steps=500,
        )
        all_results[pol["name"]] = res
        summaries[pol["name"]] = res["summary"]

    out_file = Path(output_report_json)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(all_results, f, indent=2, cls=NumpyEncoder)

    print("\n" + "=" * 90)
    print(">>> PHASE 4H FINAL COMPARATIVE EVALUATION (100 HELD-OUT SEEDS) <<<")
    print(f"{'Policy Name':<24} | {'Success Rate (95% CI)':<22} | {'Collision':<10} | {'Deadlock':<10} | {'Fwd Prog':<10}")
    print("-" * 90)
    for name, s in summaries.items():
        ci_str = f"{s['success_rate_pct']}% [{s['success_ci_95_pct'][0]}-{s['success_ci_95_pct'][1]}%]"
        print(f"{name:<24} | {ci_str:<22} | {s['collision_rate_pct']:<9.1f}% | {s['deadlock_rate_pct']:<9.1f}% | {s['mean_forward_progress_m']:<8.2f}m")
    print("=" * 90)

    return summaries


if __name__ == "__main__":
    run_full_comparative_benchmark()
