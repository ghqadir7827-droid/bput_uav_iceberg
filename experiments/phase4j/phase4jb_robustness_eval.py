"""phase4jb_robustness_eval.py - Phase 4J-B Safety-Aware Robustness & Caution-Distance Evaluation.

Evaluates 5 controllers across 6 caution distances [0.80, 0.70, 0.60, 0.55, 0.50, 0.45]
plus a supervisor-disabled diagnostic counterfactual across 50 paired development seeds (2000..2049).

Non-negotiable constraints:
- Canonical 500-step horizon preserved.
- Critical brake (0.30m), emergency reverse (0.15m), and boundary buffer (0.25m) strictly preserved.
- Reserved held-out seeds (90000..90099) strictly avoided.
- Complete metric separation between obstacle clearance and boundary clearance.
- Envelope violations (< 0.08m) tracked separately from collisions (<= 0.0m).
- Terminal step clearances properly captured even on crash termination.
"""

from dataclasses import asdict
import json
import math
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pybullet as p
import torch

from controllers.apf_expert import ScriptedAPFExpertController
from controllers.safety_supervisor import SafetyState
from core.contracts import ActionLimits, VelocityAction, compute_directional_clearances_m, compute_forward_clearance_m
from core.geometry import Corridor3D, TerminationReason
from experiments.phase4j.isolated_supervisor_opt import IsolatedOptimizedSupervisor
from models.actor_critic import ActorCriticPPO
from models.policy_net import NavigationPolicyMLP
from sim.corridor_env import CorridorSimEnv


EVAL_SEEDS = list(range(2000, 2050)) # 50 paired seeds, disjoint from 90000..90099 and 1000..1009
SWEEP_CAUTION_DISTANCES = [0.80, 0.70, 0.60, 0.55, 0.50, 0.45]
OBSTACLE_SAFETY_ENVELOPE_M: float = 0.080
BOUNDARY_SAFETY_ENVELOPE_M: float = 0.080


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


def wilson_score_interval(successes: int, trials: int, confidence: float = 0.95) -> Tuple[float, float]:
    """Computes Wilson score 95% confidence interval for a binomial proportion."""
    if trials == 0:
        return 0.0, 0.0
    z = 1.959964  # for 95% confidence
    p_hat = successes / trials
    denominator = 1 + z**2 / trials
    center = (p_hat + z**2 / (2 * trials)) / denominator
    spread = z * math.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * trials)) / trials) / denominator
    low = max(0.0, center - spread)
    high = min(1.0, center + spread)
    return round(low * 100.0, 1), round(high * 100.0, 1)


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
    elif "Expert" in controller_id:
        return None
    else:
        raise ValueError(f"Unknown controller_id: {controller_id}")


def run_evaluation_episode(
    env: CorridorSimEnv,
    supervisor: Optional[IsolatedOptimizedSupervisor],
    controller_id: str,
    model: Optional[Any],
    seed: int,
    max_steps: int = 500,
    device: torch.device = torch.device("cpu"),
    record_trace: bool = False,
) -> Dict[str, Any]:
    obs, info = env.reset(seed=seed)
    if supervisor is not None:
        supervisor.reset_state()

    expert = None
    if "Expert" in controller_id:
        expert = ScriptedAPFExpertController(corridor=env.corridor)
        expert.update_environment(env.corridor, env.obstacles)

    ep_start_pos = env._get_drone_position()
    steps = 0
    done = False

    min_obs_clear = float(info["privileged_state"].obstacle_clearance)
    min_wall_clear = float(info["privileged_state"].corridor_boundary_distance)

    interventions = 0
    emergency_brakes = 0
    retreats = 0
    throttled_steps = 0
    steps_below_obs_envelope = 0
    steps_below_wall_envelope = 0
    action_saturated_steps = 0
    cleared_pillars = set()

    step_trace = []
    current_info = info

    while not done and steps < max_steps:
        priv = current_info["privileged_state"]
        curr_pos = env._get_drone_position()
        curr_vel = priv.global_velocity
        goal_dist = float(priv.goal_distance)
        front_dist = compute_forward_clearance_m(obs, max_range_m=env.sensor_config.max_range_m)

        obs_c = float(priv.obstacle_clearance)
        wall_c = float(priv.corridor_boundary_distance)
        min_obs_clear = min(min_obs_clear, obs_c)
        min_wall_clear = min(min_wall_clear, wall_c)

        if obs_c < OBSTACLE_SAFETY_ENVELOPE_M:
            steps_below_obs_envelope += 1
        if wall_c < BOUNDARY_SAFETY_ENVELOPE_M:
            steps_below_wall_envelope += 1

        # 1. Controller raw action
        if "Expert" in controller_id:
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

        # Action saturation check
        limits = env.action_limits
        is_sat = (
            abs(raw_act.vx - limits.max_fwd_vel_mps) < 0.01 or
            abs(raw_act.vy) >= (limits.max_lat_vel_mps * 0.98) or
            abs(raw_act.vz) >= (limits.max_vert_vel_mps * 0.98)
        )
        if is_sat:
            action_saturated_steps += 1

        # Physical clamping to valid action domain
        clamped_act = VelocityAction(
            vx=float(np.clip(raw_act.vx, limits.min_fwd_vel_mps, limits.max_fwd_vel_mps)),
            vy=float(np.clip(raw_act.vy, -limits.max_lat_vel_mps, limits.max_lat_vel_mps)),
            vz=float(np.clip(raw_act.vz, -limits.max_vert_vel_mps, limits.max_vert_vel_mps)),
            yaw_rate=float(np.clip(raw_act.yaw_rate, -limits.max_yaw_rate_radps, limits.max_yaw_rate_radps)),
        )

        # 2. Supervisor filtering (or counterfactual bypass)
        if supervisor is not None:
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
        else:
            safe_act = clamped_act
            s_state = SafetyState.NOMINAL
            s_reason = "Supervisor Disabled Counterfactual"

        # Track pillars cleared
        for p_idx, p_x in enumerate(env.corridor.pillar_x_positions, start=1):
            if curr_pos[0] > (p_x + env.corridor.pillar_radius + env.corridor.drone_radius) and p_idx not in cleared_pillars:
                cleared_pillars.add(p_idx)

        if record_trace:
            step_trace.append({
                "step": steps,
                "x": round(float(curr_pos[0]), 3),
                "y": round(float(curr_pos[1]), 3),
                "z": round(float(curr_pos[2]), 3),
                "obs_clearance": round(obs_c, 4),
                "wall_clearance": round(wall_c, 4),
                "vx": round(float(curr_vel[0]), 3),
                "vy": round(float(curr_vel[1]), 3),
                "safe_vx": round(safe_act.vx, 3),
                "safe_vy": round(safe_act.vy, 3),
                "reason": s_reason,
            })

        norm_applied = safe_act.to_normalized(env.action_limits)
        obs, rew, term, trunc, current_info = env.step(norm_applied)
        steps += 1
        done = term or trunc

    # Capture terminal state clearances
    final_priv = current_info["privileged_state"]
    min_obs_clear = min(min_obs_clear, float(final_priv.obstacle_clearance))
    min_wall_clear = min(min_wall_clear, float(final_priv.corridor_boundary_distance))

    final_pos = env._get_drone_position()
    final_vel = float(np.linalg.norm(final_priv.global_velocity))
    final_goal_dist = float(final_priv.goal_distance)
    term_reason = current_info.get("termination_reason", "unknown")

    is_success = (term_reason == "success") and (final_vel <= 0.35)
    goal_entered = (final_goal_dist <= env.corridor.goal_radius)
    is_obs_collision = (term_reason == "crash_obstacle") or (min_obs_clear <= 0.0)
    is_wall_collision = (term_reason == "crash_boundary") or (min_wall_clear <= 0.0)
    is_timeout = (steps >= max_steps and not is_success and not is_obs_collision and not is_wall_collision)

    is_obs_envelope_violation = (min_obs_clear < OBSTACLE_SAFETY_ENVELOPE_M)
    is_wall_envelope_violation = (min_wall_clear < BOUNDARY_SAFETY_ENVELOPE_M)

    return {
        "seed": seed,
        "controller_id": controller_id,
        "caution_distance_m": supervisor.caution_distance_m if supervisor else None,
        "is_counterfactual": (supervisor is None),
        "steps": steps,
        "duration_s": round(steps * env.dt, 2),
        "status": term_reason,
        "is_success": is_success,
        "goal_entered": goal_entered,
        "is_timeout": is_timeout,
        "is_obstacle_collision": is_obs_collision,
        "is_boundary_collision": is_wall_collision,
        "is_obstacle_envelope_violation": is_obs_envelope_violation,
        "is_boundary_envelope_violation": is_wall_envelope_violation,
        "forward_progress_m": round(float(final_pos[0] - ep_start_pos[0]), 3),
        "final_goal_distance_m": round(final_goal_dist, 3),
        "min_obstacle_clearance_m": round(min_obs_clear, 4),
        "min_boundary_clearance_m": round(min_wall_clear, 4),
        "pct_steps_below_obs_envelope": round((steps_below_obs_envelope / max(1, steps)) * 100.0, 2),
        "terminal_speed_mps": round(final_vel, 3),
        "terminal_altitude_m": round(float(final_pos[2]), 3),
        "terminal_altitude_error_m": round(abs(float(final_pos[2]) - 1.0), 3),
        "interventions": interventions,
        "intervention_rate": round(interventions / max(1, steps), 3),
        "emergency_brakes": emergency_brakes,
        "retreats": retreats,
        "throttled_steps": throttled_steps,
        "throttled_time_pct": round((throttled_steps / max(1, steps)) * 100.0, 1),
        "action_saturation_pct": round((action_saturated_steps / max(1, steps)) * 100.0, 1),
        "obstacles_cleared_count": len(cleared_pillars),
        "trace": step_trace if (record_trace or is_obs_envelope_violation or is_obs_collision or is_wall_collision) else None,
    }


def execute_phase4jb_evaluation():
    out_dir = Path("experiments/phase4j")
    trace_dir = out_dir / "phase4jb_traces"
    out_dir.mkdir(parents=True, exist_ok=True)
    trace_dir.mkdir(parents=True, exist_ok=True)

    controllers = [
        ("Expert_APF_Reference", None),
        ("Arm_B_BC_Baseline", "models/checkpoints/bc_policy_best.pt"),
        ("BC_Phase4H_Candidate", "experiments/phase4h/checkpoints/bc_phase4h_candidate.pt"),
        ("Arm_A_PPO_Baseline", "models/checkpoints/ppo_policy_best.pt"),
        ("PPO_Phase4H_Candidate", "experiments/phase4h/checkpoints/ppo_phase4h_candidate.pt"),
    ]

    env = CorridorSimEnv(scenario="10m", gui=False, seed=42)

    all_episode_records = []
    aggregated_summaries = []

    print("=" * 100)
    print(">>> EXECUTING PHASE 4J-B SAFETY-AWARE ROBUSTNESS EVALUATION <<<")
    print(f"Paired evaluation seeds: {len(EVAL_SEEDS)} seeds ({EVAL_SEEDS[0]}..{EVAL_SEEDS[-1]})")
    print(f"Caution distances: {SWEEP_CAUTION_DISTANCES}")
    print(f"Safety Envelope: Obs >= {OBSTACLE_SAFETY_ENVELOPE_M}m, Wall >= {BOUNDARY_SAFETY_ENVELOPE_M}m")
    print("=" * 100)

    # 1. Evaluate Supervisor Caution Distances
    for d_caut in SWEEP_CAUTION_DISTANCES:
        supervisor = IsolatedOptimizedSupervisor(corridor=env.corridor, caution_distance_m=d_caut)
        print(f"\n" + "-" * 75)
        print(f">>> EVALUATING CAUTION DISTANCE: d_caut = {d_caut:.2f} m <<<")
        print("-" * 75)

        for cid, ckpt_path in controllers:
            model = load_model(cid, ckpt_path)
            ep_records = []

            for s_idx, seed in enumerate(EVAL_SEEDS):
                res = run_evaluation_episode(
                    env=env,
                    supervisor=supervisor,
                    controller_id=cid,
                    model=model,
                    seed=seed,
                )
                ep_records.append(res)
                all_episode_records.append(res)

                # Save trace if violation occurred
                if res["trace"] is not None:
                    tr_file = trace_dir / f"trace_dcaut_{d_caut:.2f}_{cid}_seed{seed}.json"
                    with open(tr_file, "w") as f:
                        json.dump(res["trace"], f, indent=2, cls=NumpyEncoder)
                    res["trace"] = str(tr_file)

            succ_cnt = sum(1 for r in ep_records if r["is_success"])
            goal_in_cnt = sum(1 for r in ep_records if r["goal_entered"])
            obs_col_cnt = sum(1 for r in ep_records if r["is_obstacle_collision"])
            wall_col_cnt = sum(1 for r in ep_records if r["is_boundary_collision"])
            obs_viols = sum(1 for r in ep_records if r["is_obstacle_envelope_violation"])
            wall_viols = sum(1 for r in ep_records if r["is_boundary_envelope_violation"])
            timeouts = sum(1 for r in ep_records if r["is_timeout"])

            fwd_prog = [r["forward_progress_m"] for r in ep_records]
            durations = [r["duration_s"] for r in ep_records]
            obs_clears = [r["min_obstacle_clearance_m"] for r in ep_records]
            wall_clears = [r["min_boundary_clearance_m"] for r in ep_records]
            speeds = [r["terminal_speed_mps"] for r in ep_records]
            intervs = [r["intervention_rate"] for r in ep_records]
            throttled = [r["throttled_time_pct"] for r in ep_records]

            succ_low, succ_high = wilson_score_interval(succ_cnt, len(EVAL_SEEDS))
            viol_low, viol_high = wilson_score_interval(obs_viols, len(EVAL_SEEDS))

            summary = {
                "caution_distance_m": d_caut,
                "is_counterfactual": False,
                "controller_id": cid,
                "total_episodes": len(EVAL_SEEDS),
                "success_count": succ_cnt,
                "success_rate_pct": round((succ_cnt / len(EVAL_SEEDS)) * 100.0, 1),
                "success_ci95": [succ_low, succ_high],
                "goal_entered_count": goal_in_cnt,
                "goal_entered_pct": round((goal_in_cnt / len(EVAL_SEEDS)) * 100.0, 1),
                "timeout_count": timeouts,
                "timeout_pct": round((timeouts / len(EVAL_SEEDS)) * 100.0, 1),
                "obstacle_collision_count": obs_col_cnt,
                "obstacle_collision_pct": round((obs_col_cnt / len(EVAL_SEEDS)) * 100.0, 1),
                "boundary_collision_count": wall_col_cnt,
                "boundary_collision_pct": round((wall_col_cnt / len(EVAL_SEEDS)) * 100.0, 1),
                "obstacle_envelope_violation_count": obs_viols,
                "obstacle_envelope_violation_pct": round((obs_viols / len(EVAL_SEEDS)) * 100.0, 1),
                "obstacle_violation_ci95": [viol_low, viol_high],
                "boundary_envelope_violation_count": wall_viols,
                "mean_forward_progress_m": round(float(np.mean(fwd_prog)), 3),
                "median_forward_progress_m": round(float(np.median(fwd_prog)), 3),
                "mean_duration_s": round(float(np.mean(durations)), 2),
                "median_duration_s": round(float(np.median(durations)), 2),
                "mean_min_obstacle_clearance_m": round(float(np.mean(obs_clears)), 4),
                "global_min_obstacle_clearance_m": round(float(np.min(obs_clears)), 4),
                "mean_min_boundary_clearance_m": round(float(np.mean(wall_clears)), 4),
                "global_min_boundary_clearance_m": round(float(np.min(wall_clears)), 4),
                "mean_terminal_speed_mps": round(float(np.mean(speeds)), 3),
                "mean_intervention_rate": round(float(np.mean(intervs)), 3),
                "mean_throttled_time_pct": round(float(np.mean(throttled)), 1),
                "mean_pillars_cleared": round(float(np.mean([r["obstacles_cleared_count"] for r in ep_records])), 2),
            }
            aggregated_summaries.append(summary)

            print(f"  {cid:<24} | Succ: {succ_cnt:2d}/50 ({summary['success_rate_pct']:4.1f}% [{succ_low}-{succ_high}%]) | ObsCol: {obs_col_cnt} | WallCol: {wall_col_cnt} | ObsViol(<8cm): {obs_viols:2d} | MinObs: {summary['global_min_obstacle_clearance_m']:.3f}m | Fwd: {summary['mean_forward_progress_m']:.2f}m")

    # 2. Phase 4J-B-F: Supervisor-Disabled Counterfactual on Expert APF
    print(f"\n" + "-" * 75)
    print(">>> EVALUATING DIAGNOSTIC COUNTERFACTUAL: EXPERT (SUPERVISOR DISABLED) <<<")
    print("-" * 75)

    ep_records = []
    for seed in EVAL_SEEDS:
        res = run_evaluation_episode(
            env=env,
            supervisor=None,  # Supervisor disabled counterfactual
            controller_id="Expert_APF_NoSupervisor_Counterfactual",
            model=None,
            seed=seed,
        )
        ep_records.append(res)
        all_episode_records.append(res)
        if res["trace"] is not None:
            tr_file = trace_dir / f"trace_counterfactual_expert_seed{seed}.json"
            with open(tr_file, "w") as f:
                json.dump(res["trace"], f, indent=2, cls=NumpyEncoder)
            res["trace"] = str(tr_file)

    succ_cnt = sum(1 for r in ep_records if r["is_success"])
    obs_col_cnt = sum(1 for r in ep_records if r["is_obstacle_collision"])
    wall_col_cnt = sum(1 for r in ep_records if r["is_boundary_collision"])
    obs_viols = sum(1 for r in ep_records if r["is_obstacle_envelope_violation"])
    wall_viols = sum(1 for r in ep_records if r["is_boundary_envelope_violation"])
    durations = [r["duration_s"] for r in ep_records]
    obs_clears = [r["min_obstacle_clearance_m"] for r in ep_records]
    wall_clears = [r["min_boundary_clearance_m"] for r in ep_records]
    speeds = [r["terminal_speed_mps"] for r in ep_records]
    succ_low, succ_high = wilson_score_interval(succ_cnt, len(EVAL_SEEDS))

    cf_summary = {
        "caution_distance_m": None,
        "is_counterfactual": True,
        "controller_id": "Expert_APF_NoSupervisor_Counterfactual",
        "total_episodes": len(EVAL_SEEDS),
        "success_count": succ_cnt,
        "success_rate_pct": round((succ_cnt / len(EVAL_SEEDS)) * 100.0, 1),
        "success_ci95": [succ_low, succ_high],
        "goal_entered_count": succ_cnt,
        "goal_entered_pct": round((succ_cnt / len(EVAL_SEEDS)) * 100.0, 1),
        "timeout_count": sum(1 for r in ep_records if r["is_timeout"]),
        "timeout_pct": round((sum(1 for r in ep_records if r["is_timeout"]) / len(EVAL_SEEDS)) * 100.0, 1),
        "obstacle_collision_count": obs_col_cnt,
        "obstacle_collision_pct": round((obs_col_cnt / len(EVAL_SEEDS)) * 100.0, 1),
        "boundary_collision_count": wall_col_cnt,
        "boundary_collision_pct": round((wall_col_cnt / len(EVAL_SEEDS)) * 100.0, 1),
        "obstacle_envelope_violation_count": obs_viols,
        "obstacle_envelope_violation_pct": round((obs_viols / len(EVAL_SEEDS)) * 100.0, 1),
        "boundary_envelope_violation_count": wall_viols,
        "mean_forward_progress_m": round(float(np.mean([r["forward_progress_m"] for r in ep_records])), 3),
        "median_forward_progress_m": round(float(np.median([r["forward_progress_m"] for r in ep_records])), 3),
        "mean_duration_s": round(float(np.mean(durations)), 2),
        "median_duration_s": round(float(np.median(durations)), 2),
        "mean_min_obstacle_clearance_m": round(float(np.mean(obs_clears)), 4),
        "global_min_obstacle_clearance_m": round(float(np.min(obs_clears)), 4),
        "mean_min_boundary_clearance_m": round(float(np.mean(wall_clears)), 4),
        "global_min_boundary_clearance_m": round(float(np.min(wall_clears)), 4),
        "mean_terminal_speed_mps": round(float(np.mean(speeds)), 3),
        "mean_intervention_rate": 0.0,
        "mean_throttled_time_pct": 0.0,
        "mean_pillars_cleared": round(float(np.mean([r["obstacles_cleared_count"] for r in ep_records])), 2),
    }
    aggregated_summaries.append(cf_summary)
    print(f"  Expert Counterfactual    | Succ: {succ_cnt:2d}/50 ({cf_summary['success_rate_pct']:4.1f}% [{succ_low}-{succ_high}%]) | ObsCol: {obs_col_cnt} | WallCol: {wall_col_cnt} | ObsViol(<8cm): {obs_viols:2d} | MinObs: {cf_summary['global_min_obstacle_clearance_m']:.3f}m | MeanDur: {cf_summary['mean_duration_s']:.2f}s")

    env.close()

    # Save complete per-episode results
    ep_file = out_dir / "phase4jb_episode_metrics.json"
    with open(ep_file, "w") as f:
        json.dump(all_episode_records, f, indent=2, cls=NumpyEncoder)

    sum_file = out_dir / "phase4jb_summary.json"
    with open(sum_file, "w") as f:
        json.dump(aggregated_summaries, f, indent=2, cls=NumpyEncoder)

    print("\n" + "=" * 100)
    print(">>> PHASE 4J-B EVALUATION COMPLETE <<<")
    print(f"Episode metrics saved to: {ep_file}")
    print(f"Summary metrics saved to: {sum_file}")
    print(f"Traces saved to: {trace_dir}")
    print("=" * 100)

    return aggregated_summaries


if __name__ == "__main__":
    execute_phase4jb_evaluation()

