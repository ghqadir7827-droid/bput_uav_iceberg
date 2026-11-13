"""run_phase4c_evaluation.py - Phase 4C Held-Out Benchmark Evaluator.

Evaluates the frozen baseline PPO policy (ppo_policy_best.pt) and the fine-tuned
candidate policy (ppo_phase4b_candidate.pt) on the reserved held-out seed pool
(seeds 50000–50099).

Strictly read-only evaluation: does not train, modify checkpoints, or alter
environment dynamics. Computes all required metrics:
- Goal success rate and episode counts
- Collision rate (obstacle vs boundary)
- Timeout rate
- Net forward displacement (mean, median)
- Final distance to goal (mean, median)
- Mean episode return
- Path length and path-length-to-net-displacement ratio
- Minimum obstacle and wall clearances
- Supervisor intervention rate (denominator = total environment steps)
- Component-wise intervention magnitudes (vx, vy, vz, yaw_rate)
- Reverse steps and recovery episodes
- Proposed vs applied saturation accounting
- Per-seed breakdown
"""

import argparse
import json
from pathlib import Path
import time
from typing import Any, Dict, List, Tuple
import numpy as np
import torch

from controllers.safety_supervisor import SafetyState, SafetySupervisor
from core.contracts import VelocityAction, compute_forward_clearance_m
from models.actor_critic import ActorCriticPPO
from sim.corridor_env import CorridorSimEnv


def evaluate_model_on_seeds(
    model: ActorCriticPPO,
    seeds: List[int],
    scenario: str = "10m",
    device: torch.device = torch.device("cpu"),
    use_supervisor: bool = True,
    model_name: str = "model",
) -> Dict[str, Any]:
    """Runs repeatable evaluation across the specified seed list."""
    env = CorridorSimEnv(scenario=scenario, gui=False, seed=seeds[0])
    safety_sup = SafetySupervisor(corridor=env.corridor) if use_supervisor else None
    goal_pos = np.array(env.corridor.goal_position, dtype=np.float32)

    records = []
    total_steps = 0
    intervention_steps = 0
    reverse_steps = 0
    recovery_episodes = 0

    d_vx_sum = 0.0; d_vx_max = 0.0; d_vx_steps = 0
    d_vy_sum = 0.0; d_vy_max = 0.0; d_vy_steps = 0
    d_vz_sum = 0.0; d_vz_max = 0.0; d_vz_steps = 0
    d_yr_sum = 0.0; d_yr_max = 0.0; d_yr_steps = 0

    sat_policy = {"vx": 0, "vy": 0, "vz": 0, "yaw_rate": 0}
    sat_applied = {"vx": 0, "vy": 0, "vz": 0, "yaw_rate": 0}

    global_min_obs_clearance = 10.0
    global_min_wall_clearance = 10.0

    print(f"\n[Evaluating {model_name}] Starting benchmark across {len(seeds)} held-out seeds ({seeds[0]}–{seeds[-1]})...")
    t0 = time.time()

    model.eval()
    for idx, s in enumerate(seeds):
        obs, info = env.reset(seed=s)
        if safety_sup is not None:
            safety_sup.loss_timer_s = 0.0
            safety_sup.stalled_timer_s = 0.0
            safety_sup.last_safe_action = VelocityAction(0.0, 0.0, 0.0, 0.0)

        done = False
        step = 0
        ep_rew = 0.0
        ep_dist = 0.0
        start_pos = env._get_drone_position().copy()
        prev_pos = start_pos.copy()
        min_obs_clearance = 10.0
        min_wall_clearance = 10.0
        ep_interventions = 0
        ep_reverse_steps = 0
        ep_recovery = False

        while not done and step < 500:
            step += 1
            total_steps += 1
            with torch.no_grad():
                t_obs = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                act, _, _, _ = model.get_action_and_value(t_obs, deterministic=True)
                act_np = act.squeeze(0).cpu().numpy()

            prop_act = VelocityAction(
                vx=float(act_np[0]), vy=float(act_np[1]), vz=float(act_np[2]), yaw_rate=float(act_np[3])
            )

            # Proposed saturation
            norm_prop = prop_act.to_normalized(env.action_limits)
            if abs(norm_prop[0]) >= 0.999: sat_policy["vx"] += 1
            if abs(norm_prop[1]) >= 0.999: sat_policy["vy"] += 1
            if abs(norm_prop[2]) >= 0.999: sat_policy["vz"] += 1
            if abs(norm_prop[3]) >= 0.999: sat_policy["yaw_rate"] += 1

            clamped_vx = float(np.clip(prop_act.vx, env.action_limits.min_fwd_vel_mps, env.action_limits.max_fwd_vel_mps))
            clamped_vy = float(np.clip(prop_act.vy, -env.action_limits.max_lat_vel_mps, env.action_limits.max_lat_vel_mps))
            clamped_vz = float(np.clip(prop_act.vz, -env.action_limits.max_vert_vel_mps, env.action_limits.max_vert_vel_mps))
            clamped_yr = float(np.clip(prop_act.yaw_rate, -env.action_limits.max_yaw_rate_radps, env.action_limits.max_yaw_rate_radps))
            clamped_act = VelocityAction(vx=clamped_vx, vy=clamped_vy, vz=clamped_vz, yaw_rate=clamped_yr)

            if safety_sup is not None:
                priv = info["privileged_state"]
                front_dist = compute_forward_clearance_m(obs, max_range_m=env.sensor_config.max_range_m)
                safe_act, s_state, s_reason = safety_sup.filter_action(
                    proposed_action=clamped_act,
                    current_position=priv.global_position,
                    current_velocity=priv.global_velocity,
                    min_front_range=front_dist,
                    dt=env.dt,
                    is_localization_healthy=True,
                    telemetry_age_s=0.0,
                    observation=obs,
                )
            else:
                safe_act = clamped_act

            # Intervention deltas
            d_vx = abs(clamped_act.vx - safe_act.vx)
            d_vy = abs(clamped_act.vy - safe_act.vy)
            d_vz = abs(clamped_act.vz - safe_act.vz)
            d_yr = abs(clamped_act.yaw_rate - safe_act.yaw_rate)

            if d_vx > 1e-4:
                d_vx_sum += d_vx; d_vx_max = max(d_vx_max, d_vx); d_vx_steps += 1
            if d_vy > 1e-4:
                d_vy_sum += d_vy; d_vy_max = max(d_vy_max, d_vy); d_vy_steps += 1
            if d_vz > 1e-4:
                d_vz_sum += d_vz; d_vz_max = max(d_vz_max, d_vz); d_vz_steps += 1
            if d_yr > 1e-4:
                d_yr_sum += d_yr; d_yr_max = max(d_yr_max, d_yr); d_yr_steps += 1

            if (d_vx > 1e-4 or d_vy > 1e-4 or d_vz > 1e-4 or d_yr > 1e-4):
                intervention_steps += 1
                ep_interventions += 1

            if safe_act.vx < -1e-4:
                reverse_steps += 1
                ep_reverse_steps += 1
                ep_recovery = True

            norm_applied = safe_act.to_normalized(env.action_limits)
            if abs(norm_applied[0]) >= 0.999: sat_applied["vx"] += 1
            if abs(norm_applied[1]) >= 0.999: sat_applied["vy"] += 1
            if abs(norm_applied[2]) >= 0.999: sat_applied["vz"] += 1
            if abs(norm_applied[3]) >= 0.999: sat_applied["yaw_rate"] += 1

            obs, rew, term, trunc, info = env.step(norm_applied)
            done = term or trunc
            ep_rew += float(rew)

            curr_pos = env._get_drone_position()
            ep_dist += float(np.linalg.norm(curr_pos - prev_pos))
            prev_pos = curr_pos

            b_dist = float(info.get("boundary_distance", 10.0))
            o_dist = float(info.get("obstacle_clearance", 10.0))
            min_wall_clearance = min(min_wall_clearance, b_dist)
            min_obs_clearance = min(min_obs_clearance, o_dist)
            global_min_wall_clearance = min(global_min_wall_clearance, b_dist)
            global_min_obs_clearance = min(global_min_obs_clearance, o_dist)

        if ep_recovery:
            recovery_episodes += 1

        final_pos = env._get_drone_position()
        net_fwd = float(final_pos[0] - start_pos[0])
        final_goal_dist = float(np.linalg.norm(final_pos - goal_pos))
        path_ratio = float(ep_dist / max(1e-4, net_fwd)) if net_fwd > 0.01 else float("nan")
        status = info.get("termination_reason", "unknown")

        records.append({
            "seed": s,
            "status": status,
            "steps": step,
            "episode_return": round(ep_rew, 2),
            "distance_travelled_m": round(ep_dist, 2),
            "net_forward_displacement_m": round(net_fwd, 2),
            "final_distance_to_goal_m": round(final_goal_dist, 2),
            "path_to_displacement_ratio": round(path_ratio, 2) if not np.isnan(path_ratio) else None,
            "min_obstacle_clearance_m": round(min_obs_clearance, 3),
            "min_wall_clearance_m": round(min_wall_clearance, 3),
            "interventions": ep_interventions,
            "reverse_steps": ep_reverse_steps,
            "final_position": [round(float(x), 2) for x in final_pos],
        })

        if (idx + 1) % 25 == 0 or (idx + 1) == len(seeds):
            print(f"  [{model_name}] Completed {idx + 1}/{len(seeds)} episodes (elapsed: {time.time() - t0:.1f}s)")

    env.close()
    elapsed = time.time() - t0

    n = len(records)
    successes = [r for r in records if r["status"] == "success"]
    crashes_obs = [r for r in records if r["status"] == "crash_obstacle"]
    crashes_bound = [r for r in records if r["status"] == "crash_boundary"]
    all_crashes = crashes_obs + crashes_bound
    timeouts = [r for r in records if r["status"] == "timeout"]

    displacements = [r["net_forward_displacement_m"] for r in records]
    goal_dists = [r["final_distance_to_goal_m"] for r in records]
    returns = [r["episode_return"] for r in records]
    path_lengths = [r["distance_travelled_m"] for r in records]
    valid_path_ratios = [r["path_to_displacement_ratio"] for r in records if r["path_to_displacement_ratio"] is not None]

    return {
        "model_name": model_name,
        "total_episodes": n,
        "evaluation_time_s": round(elapsed, 2),
        "success_count": len(successes),
        "success_rate_pct": round((len(successes) / n) * 100.0, 1),
        "collision_count": len(all_crashes),
        "collision_rate_pct": round((len(all_crashes) / n) * 100.0, 1),
        "obstacle_collision_count": len(crashes_obs),
        "obstacle_collision_rate_pct": round((len(crashes_obs) / n) * 100.0, 1),
        "boundary_collision_count": len(crashes_bound),
        "boundary_collision_rate_pct": round((len(crashes_bound) / n) * 100.0, 1),
        "timeout_count": len(timeouts),
        "timeout_rate_pct": round((len(timeouts) / n) * 100.0, 1),
        "mean_return": round(float(np.mean(returns)), 2) if returns else 0.0,
        "std_return": round(float(np.std(returns)), 2) if returns else 0.0,
        "mean_net_forward_displacement_m": round(float(np.mean(displacements)), 2) if displacements else 0.0,
        "median_net_forward_displacement_m": round(float(np.median(displacements)), 2) if displacements else 0.0,
        "mean_final_distance_to_goal_m": round(float(np.mean(goal_dists)), 2) if goal_dists else 0.0,
        "median_final_distance_to_goal_m": round(float(np.median(goal_dists)), 2) if goal_dists else 0.0,
        "mean_path_length_m": round(float(np.mean(path_lengths)), 2) if path_lengths else 0.0,
        "mean_path_to_displacement_ratio": round(float(np.mean(valid_path_ratios)), 2) if valid_path_ratios else 0.0,
        "min_obstacle_clearance_m": round(global_min_obs_clearance, 3),
        "min_wall_clearance_m": round(global_min_wall_clearance, 3),
        "total_steps": total_steps,
        "intervention_steps": intervention_steps,
        "intervention_rate_pct": round((intervention_steps / max(1, total_steps)) * 100.0, 1),
        "intervention_denominator_definition": "Total environment steps executed across all evaluated episodes",
        "reverse_control_steps": reverse_steps,
        "recovery_episodes": recovery_episodes,
        "component_interventions": {
            "vx": {
                "mean_mps": round(d_vx_sum / max(1, d_vx_steps), 4),
                "max_mps": round(d_vx_max, 4),
                "intervened_steps": d_vx_steps,
                "unit": "m/s"
            },
            "vy": {
                "mean_mps": round(d_vy_sum / max(1, d_vy_steps), 4),
                "max_mps": round(d_vy_max, 4),
                "intervened_steps": d_vy_steps,
                "unit": "m/s"
            },
            "vz": {
                "mean_mps": round(d_vz_sum / max(1, d_vz_steps), 4),
                "max_mps": round(d_vz_max, 4),
                "intervened_steps": d_vz_steps,
                "unit": "m/s"
            },
            "yaw_rate": {
                "mean_radps": round(d_yr_sum / max(1, d_yr_steps), 4),
                "max_radps": round(d_yr_max, 4),
                "intervened_steps": d_yr_steps,
                "unit": "rad/s"
            },
        },
        "policy_saturation": {
            "vx": sat_policy["vx"],
            "vy": sat_policy["vy"],
            "vz": sat_policy["vz"],
            "yaw_rate": sat_policy["yaw_rate"],
            "total_steps": total_steps,
            "rate_pct": {k: round((v / max(1, total_steps)) * 100.0, 1) for k, v in sat_policy.items()}
        },
        "applied_saturation": {
            "vx": sat_applied["vx"],
            "vy": sat_applied["vy"],
            "vz": sat_applied["vz"],
            "yaw_rate": sat_applied["yaw_rate"],
            "total_steps": total_steps,
            "rate_pct": {k: round((v / max(1, total_steps)) * 100.0, 1) for k, v in sat_applied.items()}
        },
        "per_seed_records": records,
    }


def main():
    parser = argparse.ArgumentParser(description="Phase 4C Held-Out Benchmark Evaluator")
    parser.add_argument("--baseline-ckpt", type=str, default="models/checkpoints/ppo_policy_best.pt")
    parser.add_argument("--candidate-ckpt", type=str, default="models/checkpoints/ppo_phase4b_candidate.pt")
    parser.add_argument("--seed-start", type=int, default=50000)
    parser.add_argument("--seed-end", type=int, default=50099)
    parser.add_argument("--scenario", type=str, default="10m")
    args = parser.parse_args()

    held_out_seeds = list(range(args.seed_start, args.seed_end + 1))
    assert len(held_out_seeds) == 100, f"Expected 100 held-out seeds, got {len(held_out_seeds)}"
    assert held_out_seeds[0] == 50000 and held_out_seeds[-1] == 50099, "Seeds must be exactly 50000-50099"

    device = torch.device("cpu")
    print("=" * 80)
    print(">>> PHASE 4C: HELD-OUT SEED POOL EVALUATION (50000–50099) <<<")
    print(f"Scenario:             {args.scenario}")
    print(f"Held-Out Seed Range:  {args.seed_start} to {args.seed_end} (Total: {len(held_out_seeds)})")
    print(f"Baseline Checkpoint:  {args.baseline_ckpt}")
    print(f"Candidate Checkpoint: {args.candidate_ckpt}")
    print(f"Device:               {device}")
    print("=" * 80)

    # 1. Evaluate Frozen Baseline PPO
    base_model = ActorCriticPPO(obs_dim=23, act_dim=4).to(device)
    base_loaded = base_model.load_ppo_checkpoint(args.baseline_ckpt, device=device)
    if not base_loaded:
        raise RuntimeError(f"Failed to load baseline checkpoint from {args.baseline_ckpt}")

    base_results = evaluate_model_on_seeds(
        model=base_model,
        seeds=held_out_seeds,
        scenario=args.scenario,
        device=device,
        use_supervisor=True,
        model_name="Frozen Baseline PPO (ppo_policy_best.pt)",
    )

    # 2. Evaluate Candidate PPO
    cand_model = ActorCriticPPO(obs_dim=23, act_dim=4).to(device)
    cand_loaded = cand_model.load_ppo_checkpoint(args.candidate_ckpt, device=device)
    if not cand_loaded:
        raise RuntimeError(f"Failed to load candidate checkpoint from {args.candidate_ckpt}")

    cand_results = evaluate_model_on_seeds(
        model=cand_model,
        seeds=held_out_seeds,
        scenario=args.scenario,
        device=device,
        use_supervisor=True,
        model_name="Phase 4B Candidate (ppo_phase4b_candidate.pt)",
    )

    # 3. Compute Comparative Deltas (Candidate minus Baseline)
    comparison = {
        "benchmark_seeds": {"start": args.seed_start, "end": args.seed_end, "total": len(held_out_seeds)},
        "baseline_checkpoint": args.baseline_ckpt,
        "candidate_checkpoint": args.candidate_ckpt,
        "summary": {
            "success_rate_pct": {
                "baseline": base_results["success_rate_pct"],
                "candidate": cand_results["success_rate_pct"],
                "delta": round(cand_results["success_rate_pct"] - base_results["success_rate_pct"], 2),
            },
            "collision_rate_pct": {
                "baseline": base_results["collision_rate_pct"],
                "candidate": cand_results["collision_rate_pct"],
                "delta": round(cand_results["collision_rate_pct"] - base_results["collision_rate_pct"], 2),
            },
            "timeout_rate_pct": {
                "baseline": base_results["timeout_rate_pct"],
                "candidate": cand_results["timeout_rate_pct"],
                "delta": round(cand_results["timeout_rate_pct"] - base_results["timeout_rate_pct"], 2),
            },
            "mean_episode_return": {
                "baseline": base_results["mean_return"],
                "candidate": cand_results["mean_return"],
                "delta": round(cand_results["mean_return"] - base_results["mean_return"], 2),
            },
            "mean_net_forward_displacement_m": {
                "baseline": base_results["mean_net_forward_displacement_m"],
                "candidate": cand_results["mean_net_forward_displacement_m"],
                "delta": round(cand_results["mean_net_forward_displacement_m"] - base_results["mean_net_forward_displacement_m"], 2),
            },
            "median_net_forward_displacement_m": {
                "baseline": base_results["median_net_forward_displacement_m"],
                "candidate": cand_results["median_net_forward_displacement_m"],
                "delta": round(cand_results["median_net_forward_displacement_m"] - base_results["median_net_forward_displacement_m"], 2),
            },
            "mean_final_distance_to_goal_m": {
                "baseline": base_results["mean_final_distance_to_goal_m"],
                "candidate": cand_results["mean_final_distance_to_goal_m"],
                "delta": round(cand_results["mean_final_distance_to_goal_m"] - base_results["mean_final_distance_to_goal_m"], 2),
            },
            "median_final_distance_to_goal_m": {
                "baseline": base_results["median_final_distance_to_goal_m"],
                "candidate": cand_results["median_final_distance_to_goal_m"],
                "delta": round(cand_results["median_final_distance_to_goal_m"] - base_results["median_final_distance_to_goal_m"], 2),
            },
            "mean_path_length_m": {
                "baseline": base_results["mean_path_length_m"],
                "candidate": cand_results["mean_path_length_m"],
                "delta": round(cand_results["mean_path_length_m"] - base_results["mean_path_length_m"], 2),
            },
            "mean_path_to_displacement_ratio": {
                "baseline": base_results["mean_path_to_displacement_ratio"],
                "candidate": cand_results["mean_path_to_displacement_ratio"],
                "delta": round(cand_results["mean_path_to_displacement_ratio"] - base_results["mean_path_to_displacement_ratio"], 2),
            },
            "min_obstacle_clearance_m": {
                "baseline": base_results["min_obstacle_clearance_m"],
                "candidate": cand_results["min_obstacle_clearance_m"],
                "delta": round(cand_results["min_obstacle_clearance_m"] - base_results["min_obstacle_clearance_m"], 3),
            },
            "min_wall_clearance_m": {
                "baseline": base_results["min_wall_clearance_m"],
                "candidate": cand_results["min_wall_clearance_m"],
                "delta": round(cand_results["min_wall_clearance_m"] - base_results["min_wall_clearance_m"], 3),
            },
            "intervention_rate_pct": {
                "baseline": base_results["intervention_rate_pct"],
                "candidate": cand_results["intervention_rate_pct"],
                "delta": round(cand_results["intervention_rate_pct"] - base_results["intervention_rate_pct"], 2),
            },
            "emergency_reverse_steps": {
                "baseline": base_results["reverse_control_steps"],
                "candidate": cand_results["reverse_control_steps"],
                "delta": cand_results["reverse_control_steps"] - base_results["reverse_control_steps"],
            },
            "recovery_episodes": {
                "baseline": base_results["recovery_episodes"],
                "candidate": cand_results["recovery_episodes"],
                "delta": cand_results["recovery_episodes"] - base_results["recovery_episodes"],
            },
        },
        "baseline_results": base_results,
        "candidate_results": cand_results,
    }

    # 4. Save structured report artifacts
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    base_file = reports_dir / "phase4c_heldout_eval_ppo_baseline.json"
    cand_file = reports_dir / "phase4c_heldout_eval_ppo_candidate.json"
    comp_file = reports_dir / "phase4c_heldout_comparison.json"

    with open(base_file, "w") as f:
        json.dump(base_results, f, indent=2)
    with open(cand_file, "w") as f:
        json.dump(cand_results, f, indent=2)
    with open(comp_file, "w") as f:
        json.dump(comparison, f, indent=2)

    print("\n" + "=" * 80)
    print(">>> PHASE 4C HELD-OUT EVALUATION COMPLETE <<<")
    print(f"Baseline Artifact:   {base_file}")
    print(f"Candidate Artifact:  {cand_file}")
    print(f"Comparison Artifact: {comp_file}")
    print("=" * 80)


if __name__ == "__main__":
    main()
