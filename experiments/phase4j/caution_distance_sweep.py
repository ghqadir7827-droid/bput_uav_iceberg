"""caution_distance_sweep.py - Phase 4J-A4 Isolated Caution-Distance Parametric Sweep.

Evaluates 5 controllers across a controlled parametric sweep of caution distances:
d_caut in [0.80m, 0.70m, 0.60m, 0.55m, 0.50m, 0.45m]
over the 10 development seeds (1000..1009):
- Controllers:
  1. Expert_APF_Reference
  2. Arm_B_BC_Baseline (bc_policy_best.pt)
  3. BC_Phase4H_Candidate (bc_phase4h_candidate.pt)
  4. Arm_A_PPO_Baseline (ppo_policy_best.pt)
  5. PPO_Phase4H_Candidate (ppo_phase4h_candidate.pt)

Invariant Safeguards:
- critical_obstacle_dist_m (0.30m) is strictly preserved.
- emergency_reverse_dist_m (0.15m) is strictly preserved.
- boundary_buffer_m (0.25m) is strictly preserved.
- Uses IsolatedOptimizedSupervisor without modifying canonical supervisor.
- Records full per-step telemetry and time-budget metrics.
"""

from dataclasses import asdict
import json
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pybullet as p
import torch

from controllers.apf_expert import ScriptedAPFExpertController
from controllers.safety_supervisor import SafetyState
from core.contracts import ActionLimits, VelocityAction, compute_forward_clearance_m, compute_directional_clearances_m
from experiments.phase4i.time_budget import (
    classify_step_mode,
    MovementMode,
    TimeBudgetRecord,
)
from experiments.phase4j.isolated_supervisor_opt import IsolatedOptimizedSupervisor
from models.actor_critic import ActorCriticPPO
from models.policy_net import NavigationPolicyMLP
from sim.corridor_env import CorridorSimEnv


DEVELOPMENT_SEEDS = list(range(1000, 1010))
SWEEP_CAUTION_DISTANCES = [0.80, 0.70, 0.60, 0.55, 0.50, 0.45]


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


def load_model(controller_id: str, checkpoint_path: Optional[str] = None, device: torch.device = torch.device("cpu")):
    if controller_id in ("Arm_B_BC_Baseline", "BC_Phase4H_Candidate"):
        model = NavigationPolicyMLP(obs_dim=23, act_dim=4).to(device)
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        return model
    elif controller_id in ("Arm_A_PPO_Baseline", "PPO_Phase4H_Candidate"):
        model = ActorCriticPPO(obs_dim=23, act_dim=4).to(device)
        success = model.load_ppo_checkpoint(checkpoint_path, device=device)
        assert success, f"Failed to load PPO checkpoint: {checkpoint_path}"
        model.eval()
        return model
    elif controller_id == "Expert_APF_Reference":
        return None
    else:
        raise ValueError(f"Unknown controller_id: {controller_id}")


def run_sweep_episode(
    env: CorridorSimEnv,
    supervisor: IsolatedOptimizedSupervisor,
    controller_id: str,
    model: Optional[Any],
    seed: int,
    max_steps: int = 500,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, Any]:
    obs, info = env.reset(seed=seed)
    supervisor.reset_state()

    expert = None
    if controller_id == "Expert_APF_Reference":
        expert = ScriptedAPFExpertController(corridor=env.corridor)
        expert.update_environment(env.corridor, env.obstacles)

    time_budget = TimeBudgetRecord(dt_s=env.dt)
    ep_start_pos = env._get_drone_position()
    steps = 0
    done = False

    min_obs_clear = 10.0
    min_wall_clear = 10.0
    interventions = 0
    emergency_brakes = 0
    retreats = 0
    throttled_steps = 0
    cleared_pillars = set()

    current_info = info

    while not done and steps < max_steps:
        priv = current_info["privileged_state"]
        curr_pos = env._get_drone_position()
        curr_vel = priv.global_velocity
        goal_dist = float(priv.goal_distance)
        front_dist = compute_forward_clearance_m(obs, max_range_m=env.sensor_config.max_range_m)

        min_obs_clear = min(min_obs_clear, float(priv.obstacle_clearance))
        min_wall_clear = min(min_wall_clear, float(priv.corridor_boundary_distance))

        # 1. Raw action
        if controller_id == "Expert_APF_Reference":
            raw_act = expert.compute_privileged_expert_action(
                position=priv.global_position,
                velocity=priv.global_velocity,
                obstacles=env.obstacles,
            )
        elif "BC" in controller_id:
            with torch.no_grad():
                t_obs = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                act_arr = model(t_obs, bounded=True).squeeze(0).cpu().numpy()
            raw_act = VelocityAction(vx=float(act_arr[0]), vy=float(act_arr[1]), vz=float(act_arr[2]), yaw_rate=float(act_arr[3]))
        else: # PPO
            with torch.no_grad():
                t_obs = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                act_arr = model.actor(t_obs, bounded=True).squeeze(0).cpu().numpy()
            raw_act = VelocityAction(vx=float(act_arr[0]), vy=float(act_arr[1]), vz=float(act_arr[2]), yaw_rate=float(act_arr[3]))

        clamped_act = VelocityAction(
            vx=float(np.clip(raw_act.vx, env.action_limits.min_fwd_vel_mps, env.action_limits.max_fwd_vel_mps)),
            vy=float(np.clip(raw_act.vy, -env.action_limits.max_lat_vel_mps, env.action_limits.max_lat_vel_mps)),
            vz=float(np.clip(raw_act.vz, -env.action_limits.max_vert_vel_mps, env.action_limits.max_vert_vel_mps)),
            yaw_rate=float(np.clip(raw_act.yaw_rate, -env.action_limits.max_yaw_rate_radps, env.action_limits.max_yaw_rate_radps)),
        )

        # 2. Filter with Isolated Optimized Supervisor
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

        if s_state != SafetyState.NOMINAL:
            interventions += 1
        if "Forward brake" in s_reason:
            emergency_brakes += 1
        if "Reversing" in s_reason or "Reverse retreat" in s_reason:
            retreats += 1
        if "Speed throttled" in s_reason:
            throttled_steps += 1

        # Classify mode
        mode = classify_step_mode(
            pos=curr_pos,
            vel=curr_vel,
            raw_action=raw_act,
            safe_action=safe_act,
            safety_state=s_state,
            front_clearance_m=front_dist,
            min_obs_clearance_m=min_obs_clear,
            goal_dist_m=goal_dist,
            reason=s_reason,
        )
        time_budget.record_step(mode)

        # Track obstacle clearances
        for p_idx, p_x in enumerate([3.0, 5.0, 7.0], start=1):
            if curr_pos[0] > (p_x + 0.35) and p_idx not in cleared_pillars:
                cleared_pillars.add(p_idx)

        norm_applied = safe_act.to_normalized(env.action_limits)
        obs, rew, term, trunc, current_info = env.step(norm_applied)
        steps += 1
        done = term or trunc

    time_budget.finalize()

    final_pos = env._get_drone_position()
    final_vel = float(np.linalg.norm(current_info["privileged_state"].global_velocity))
    final_goal_dist = float(current_info["privileged_state"].goal_distance)
    term_reason = current_info.get("termination_reason", "unknown")
    is_success = (term_reason == "success") and (final_vel <= 0.35)
    is_collision = ("crash" in term_reason)
    is_timeout = (steps >= max_steps and not is_success and not is_collision)

    return {
        "seed": seed,
        "controller_id": controller_id,
        "caution_distance_m": supervisor.caution_distance_m,
        "steps": steps,
        "duration_s": round(steps * env.dt, 2),
        "status": term_reason,
        "is_success": is_success,
        "is_collision": is_collision,
        "is_timeout": is_timeout,
        "forward_progress_m": round(float(final_pos[0] - ep_start_pos[0]), 3),
        "final_goal_distance_m": round(final_goal_dist, 3),
        "min_obs_clearance_m": round(min_obs_clear, 3),
        "min_wall_clearance_m": round(min_wall_clear, 3),
        "terminal_speed_mps": round(final_vel, 3),
        "terminal_altitude_m": round(float(final_pos[2]), 3),
        "interventions": interventions,
        "intervention_rate": round(interventions / max(1, steps), 3),
        "emergency_brakes": emergency_brakes,
        "retreats": retreats,
        "throttled_steps": throttled_steps,
        "obstacles_cleared_count": len(cleared_pillars),
        "time_budget": time_budget.to_dict(),
    }


def execute_parametric_sweep():
    out_dir = Path("experiments/phase4j")
    out_dir.mkdir(parents=True, exist_ok=True)

    controllers = [
        ("Expert_APF_Reference", None),
        ("BC_Phase4H_Candidate", "experiments/phase4h/checkpoints/bc_phase4h_candidate.pt"),
        ("Arm_B_BC_Baseline", "models/checkpoints/bc_policy_best.pt"),
        ("PPO_Phase4H_Candidate", "experiments/phase4h/checkpoints/ppo_phase4h_candidate.pt"),
        ("Arm_A_PPO_Baseline", "models/checkpoints/ppo_policy_best.pt"),
    ]

    env = CorridorSimEnv(scenario="10m", gui=False, seed=42)

    sweep_results = {}
    aggregated_summaries = []

    print("=" * 90)
    print(">>> EXECUTING PHASE 4J-A4 CAUTION-DISTANCE PARAMETRIC SWEEP <<<")
    print(f"Caution distances: {SWEEP_CAUTION_DISTANCES}")
    print(f"Controllers: {[c[0] for c in controllers]}")
    print(f"Development Seeds: {DEVELOPMENT_SEEDS}")
    print("=" * 90)

    for d_caut in SWEEP_CAUTION_DISTANCES:
        supervisor = IsolatedOptimizedSupervisor(corridor=env.corridor, caution_distance_m=d_caut)
        sweep_results[f"d_caut_{d_caut:.2f}"] = {}

        print(f"\n========================================================")
        print(f">>> EVALUATING CAUTION DISTANCE: d_caut = {d_caut:.2f} m <<<")
        print(f"========================================================")

        for cid, ckpt_path in controllers:
            model = load_model(cid, ckpt_path)
            ep_records = []

            for seed in DEVELOPMENT_SEEDS:
                res = run_sweep_episode(
                    env=env,
                    supervisor=supervisor,
                    controller_id=cid,
                    model=model,
                    seed=seed,
                )
                ep_records.append(res)

            succ_cnt = sum(1 for r in ep_records if r["is_success"])
            coll_cnt = sum(1 for r in ep_records if r["is_collision"])
            mean_prog = float(np.mean([r["forward_progress_m"] for r in ep_records]))
            mean_steps = float(np.mean([r["steps"] for r in ep_records]))
            mean_min_obs = float(np.mean([r["min_obs_clearance_m"] for r in ep_records]))
            global_min_obs = float(np.min([r["min_obs_clearance_m"] for r in ep_records]))
            mean_interv = float(np.mean([r["intervention_rate"] for r in ep_records]))
            mean_throttled_pct = float(np.mean([r["time_budget"]["time_percentages"]["supervisor_constrained_forward"] for r in ep_records]))
            mean_brake_pct = float(np.mean([r["time_budget"]["time_percentages"]["braking_or_stopping"] for r in ep_records]))

            summary = {
                "caution_distance_m": d_caut,
                "controller_id": cid,
                "total_episodes": len(DEVELOPMENT_SEEDS),
                "success_count": succ_cnt,
                "success_rate_pct": round((succ_cnt / len(DEVELOPMENT_SEEDS)) * 100.0, 1),
                "collision_count": coll_cnt,
                "collision_rate_pct": round((coll_cnt / len(DEVELOPMENT_SEEDS)) * 100.0, 1),
                "mean_steps": round(mean_steps, 1),
                "mean_duration_s": round(mean_steps * env.dt, 2),
                "mean_forward_progress_m": round(mean_prog, 3),
                "mean_min_obs_clearance_m": round(mean_min_obs, 3),
                "global_min_obs_clearance_m": round(global_min_obs, 3),
                "mean_intervention_rate": round(mean_interv, 3),
                "mean_throttled_pct": round(mean_throttled_pct, 1),
                "mean_brake_pct": round(mean_brake_pct, 1),
            }

            sweep_results[f"d_caut_{d_caut:.2f}"][cid] = {
                "summary": summary,
                "episodes": ep_records,
            }
            aggregated_summaries.append(summary)

            print(f"  {cid:<24} | Succ: {succ_cnt:2d}/10 ({summary['success_rate_pct']:4.1f}%) | Coll: {coll_cnt:2d} | Fwd: {mean_prog:4.2f}m | Steps: {mean_steps:5.1f} | MinObs: {global_min_obs:.3f}m | Throttled: {mean_throttled_pct:4.1f}%")

    env.close()

    # Save complete machine-readable metrics
    sweep_metrics_file = out_dir / "caution_distance_sweep_metrics.json"
    with open(sweep_metrics_file, "w") as f:
        json.dump(sweep_results, f, indent=2, cls=NumpyEncoder)

    summary_table_file = out_dir / "caution_distance_sweep_summary.json"
    with open(summary_table_file, "w") as f:
        json.dump(aggregated_summaries, f, indent=2, cls=NumpyEncoder)

    print("\n" + "=" * 90)
    print(">>> CAUTION-DISTANCE PARAMETRIC SWEEP COMPLETE <<<")
    print(f"Metrics saved to {sweep_metrics_file} and {summary_table_file}")
    print("=" * 90)

    return aggregated_summaries


if __name__ == "__main__":
    execute_parametric_sweep()
