"""diagnostic_harness.py - Comprehensive Phase 4I Multi-Obstacle Time-Budget Diagnostic Suite.

Evaluates 5 controllers across controlled diagnostic scenarios:
1. Arm_B_BC_Baseline:        models/checkpoints/bc_policy_best.pt
2. BC_Phase4H_Candidate:     experiments/phase4h/checkpoints/bc_phase4h_candidate.pt
3. Arm_A_PPO_Baseline:       models/checkpoints/ppo_policy_best.pt
4. PPO_Phase4H_Candidate:    experiments/phase4h/checkpoints/ppo_phase4h_candidate.pt
5. Expert_APF_Reference:     ScriptedAPFExpertController (validated reference)

Scenarios:
- Scenario 1: Clear Corridor (Seed 1)
- Scenario 2: Single Centered Obstacle (Y_obs = 0.00m, Seed 42)
- Scenario 3: Single Off-Center Obstacle Left (Y_obs = +0.20m, Seed 42)
- Scenario 4: Single Off-Center Obstacle Right (Y_obs = -0.20m, Seed 42)
- Scenario 5: Reproducible Known Deadlock (Y_obs = 0.00m, Start X = 2.0m, vx = 0.50m/s)
- Scenario 6: Goal-Approach Braking (Start X = 7.5m, vx = 0.50m/s)
- Scenario 7: Three-Pillar Multi-Obstacle Corridor (10 development seeds: 1000..1009)

Records full step-by-step telemetry, time-budget breakdowns, and progress loss diagnostics.
"""

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pybullet as p
import torch

from controllers.apf_expert import ScriptedAPFExpertController
from controllers.safety_supervisor import SafetyState, SafetySupervisor
from core.contracts import ActionLimits, VelocityAction, compute_forward_clearance_m, compute_directional_clearances_m
from core.geometry import Corridor3D, Obstacle3D
from experiments.phase4i.time_budget import (
    classify_step_mode,
    MovementMode,
    TimeBudgetRecord,
)
from models.actor_critic import ActorCriticPPO
from models.policy_net import NavigationPolicyMLP
from sim.corridor_env import CorridorSimEnv


DEVELOPMENT_SEEDS = list(range(1000, 1010)) # Exactly 10 development seeds


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


def load_policy(controller_id: str, checkpoint_path: Optional[str] = None, device: torch.device = torch.device("cpu")):
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


def run_single_episode_diagnostic(
    env: CorridorSimEnv,
    supervisor: SafetySupervisor,
    controller_id: str,
    model: Optional[Any],
    scenario_name: str,
    seed: int,
    start_override: Optional[Tuple[float, float, float, float]] = None, # (x, y, z, vx)
    obstacle_override: Optional[List[Tuple[float, float, float]]] = None, # [(x, y, z)]
    max_steps: int = 500,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, Any]:
    obs, info = env.reset(seed=seed)
    supervisor.loss_timer_s = 0.0
    supervisor.stalled_timer_s = 0.0
    supervisor.last_safe_action = VelocityAction(0.0, 0.0, 0.0, 0.0)

    # Apply overrides
    if obstacle_override is not None:
        if len(obstacle_override) == 0:
            for obs_id in env.obstacle_ids:
                p.resetBasePositionAndOrientation(obs_id, [50.0, 50.0, -10.0], [0, 0, 0, 1], physicsClientId=env.client_id)
            env.obstacles = []
        else:
            new_obs = []
            for idx, obs_id in enumerate(env.obstacle_ids):
                if idx < len(obstacle_override):
                    ox, oy, oz = obstacle_override[idx]
                    p.resetBasePositionAndOrientation(obs_id, [ox, oy, oz], [0, 0, 0, 1], physicsClientId=env.client_id)
                    new_obs.append(Obstacle3D(center=np.array([ox, oy, oz]), radius=0.20, height=2.0))
                else:
                    p.resetBasePositionAndOrientation(obs_id, [50.0, 50.0, -10.0], [0, 0, 0, 1], physicsClientId=env.client_id)
            env.obstacles = new_obs

    if start_override is not None:
        sx, sy, sz, svx = start_override
        p.resetBasePositionAndOrientation(env.drone_id, [sx, sy, sz], [0, 0, 0, 1], physicsClientId=env.client_id)
        p.resetBaseVelocity(env.drone_id, [svx, 0.0, 0.0], [0, 0, 0], physicsClientId=env.client_id)

    # Re-sample initial observation after overrides
    deploy_obs, priv_state = env._get_observations()
    obs = deploy_obs.to_vector()
    info["privileged_state"] = priv_state

    # Reference Expert instance
    expert = None
    if controller_id == "Expert_APF_Reference":
        expert = ScriptedAPFExpertController(corridor=env.corridor)
        expert.update_environment(env.corridor, env.obstacles)

    time_budget = TimeBudgetRecord(dt_s=env.dt)
    step_traces = []

    ep_start_pos = env._get_drone_position()
    steps = 0
    done = False

    # Obstacle encounter state machine
    in_caution = False
    in_stop = False
    cleared_pillars = set()

    # Progress loss diagnostic variables
    first_progress_loss_x = None
    first_progress_loss_step = None
    first_progress_loss_obstacle = None
    first_progress_loss_reason = None
    time_spent_braking_s = 0.0
    re_acceleration_events = []
    current_reaccel_start_step = None

    current_info = info

    while not done and steps < max_steps:
        priv = current_info["privileged_state"]
        curr_pos = env._get_drone_position()
        curr_vel = priv.global_velocity
        goal_dist = float(priv.goal_distance)
        front_dist = compute_forward_clearance_m(obs, max_range_m=env.sensor_config.max_range_m)
        dir_clear = compute_directional_clearances_m(obs)
        min_obs_clear = float(priv.obstacle_clearance)
        min_wall_clear = float(priv.corridor_boundary_distance)

        # 1. Compute Raw Action
        if controller_id == "Expert_APF_Reference":
            expert_act = expert.compute_privileged_expert_action(
                position=priv.global_position,
                velocity=priv.global_velocity,
                obstacles=env.obstacles,
            )
            raw_action = expert_act
        elif "BC" in controller_id:
            with torch.no_grad():
                t_obs = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                act_arr = model(t_obs, bounded=True).squeeze(0).cpu().numpy()
            raw_action = VelocityAction(vx=float(act_arr[0]), vy=float(act_arr[1]), vz=float(act_arr[2]), yaw_rate=float(act_arr[3]))
        else: # PPO
            with torch.no_grad():
                t_obs = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                act_arr = model.actor(t_obs, bounded=True).squeeze(0).cpu().numpy()
            raw_action = VelocityAction(vx=float(act_arr[0]), vy=float(act_arr[1]), vz=float(act_arr[2]), yaw_rate=float(act_arr[3]))

        # Physical clamping of proposed action
        clamped_action = VelocityAction(
            vx=float(np.clip(raw_action.vx, env.action_limits.min_fwd_vel_mps, env.action_limits.max_fwd_vel_mps)),
            vy=float(np.clip(raw_action.vy, -env.action_limits.max_lat_vel_mps, env.action_limits.max_lat_vel_mps)),
            vz=float(np.clip(raw_action.vz, -env.action_limits.max_vert_vel_mps, env.action_limits.max_vert_vel_mps)),
            yaw_rate=float(np.clip(raw_action.yaw_rate, -env.action_limits.max_yaw_rate_radps, env.action_limits.max_yaw_rate_radps)),
        )

        # 2. Safety Supervisor Filtering
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

        # 3. Classify Movement Mode
        mode = classify_step_mode(
            pos=curr_pos,
            vel=curr_vel,
            raw_action=raw_action,
            safe_action=safe_action,
            safety_state=s_state,
            front_clearance_m=front_dist,
            min_obs_clearance_m=min_obs_clear,
            goal_dist_m=goal_dist,
            reason=s_reason,
        )
        time_budget.record_step(mode)

        # Identify which obstacle is being interacted with
        active_pillar = None
        for p_idx, p_x in enumerate([3.0, 5.0, 7.0], start=1):
            if abs(curr_pos[0] - p_x) <= 1.0:
                active_pillar = f"Pillar_{p_idx}"
                break

        # Encounter tracking
        if front_dist < 0.80 and not in_caution:
            in_caution = True
            time_budget.caution_entries += 1
        elif front_dist >= 0.80:
            in_caution = False

        if abs(curr_vel[0]) <= 0.05 and not in_stop:
            in_stop = True
            time_budget.complete_stops += 1
        elif abs(curr_vel[0]) > 0.05:
            in_stop = False

        if "Reverse retreat" in s_reason or "Stalled recovery" in s_reason:
            time_budget.stalled_recoveries += 1

        # Check cleared obstacles
        for p_idx, p_x in enumerate([3.0, 5.0, 7.0], start=1):
            if curr_pos[0] > (p_x + 0.35) and p_idx not in cleared_pillars:
                cleared_pillars.add(p_idx)
                time_budget.obstacles_cleared += 1

        # Progress loss tracking
        is_stagnant = (abs(curr_vel[0]) <= 0.10 and abs(curr_vel[1]) <= 0.10) or (s_state != SafetyState.NOMINAL and safe_action.vx <= 0.05)
        if is_stagnant:
            time_spent_braking_s += env.dt
            if first_progress_loss_x is None and goal_dist > 0.50:
                first_progress_loss_x = round(float(curr_pos[0]), 3)
                first_progress_loss_step = steps
                first_progress_loss_obstacle = active_pillar or "Corridor_Free_Space"
                first_progress_loss_reason = s_reason

        # Re-acceleration measurement: from obstacle clearance until vx >= 0.45 m/s
        if mode == MovementMode.RE_ACCELERATION and current_reaccel_start_step is None:
            current_reaccel_start_step = steps
        elif current_reaccel_start_step is not None:
            if curr_vel[0] >= 0.45 or mode != MovementMode.RE_ACCELERATION:
                reaccel_duration = (steps - current_reaccel_start_step) * env.dt
                re_acceleration_events.append(round(reaccel_duration, 3))
                current_reaccel_start_step = None

        # Record step trace
        step_traces.append({
            "step": steps,
            "time_s": round(steps * env.dt, 3),
            "pos": [round(float(x), 3) for x in curr_pos],
            "vel": [round(float(v), 3) for v in curr_vel],
            "goal_dist": round(goal_dist, 3),
            "fwd_progress": round(float(curr_pos[0] - ep_start_pos[0]), 3),
            "raw_vx": round(float(raw_action.vx), 3),
            "raw_vy": round(float(raw_action.vy), 3),
            "safe_vx": round(float(safe_action.vx), 3),
            "safe_vy": round(float(safe_action.vy), 3),
            "safety_state": s_state.value,
            "safety_reason": s_reason,
            "front_clearance_m": round(front_dist, 3),
            "min_obs_clearance_m": round(min_obs_clear, 3),
            "min_wall_clearance_m": round(min_wall_clear, 3),
            "mode": mode.value,
            "active_pillar": active_pillar,
        })

        # Step simulation
        norm_applied = safe_action.to_normalized(env.action_limits)
        obs, rew, term, trunc, current_info = env.step(norm_applied)
        steps += 1
        done = term or trunc

    time_budget.finalize()

    final_pos = env._get_drone_position()
    final_vel = float(np.linalg.norm(current_info["privileged_state"].global_velocity))
    final_goal_dist = float(current_info["privileged_state"].goal_distance)
    term_reason = current_info.get("termination_reason", "unknown")
    is_success = (term_reason == "success") and (final_vel <= 0.35)

    outcome = {
        "controller_id": controller_id,
        "scenario_name": scenario_name,
        "seed": seed,
        "total_steps": steps,
        "total_time_s": round(steps * env.dt, 3),
        "termination_reason": term_reason,
        "is_success": is_success,
        "final_position": [round(float(x), 3) for x in final_pos],
        "final_speed_mps": round(final_vel, 3),
        "final_goal_distance_m": round(final_goal_dist, 3),
        "forward_progress_m": round(float(final_pos[0] - ep_start_pos[0]), 3),
        "obstacles_cleared_count": len(cleared_pillars),
        "cleared_pillars": sorted(list(cleared_pillars)),
        "time_budget": time_budget.to_dict(),
        "progress_loss_diagnostics": {
            "first_loss_x_m": first_progress_loss_x,
            "first_loss_step": first_progress_loss_step,
            "first_loss_obstacle": first_progress_loss_obstacle,
            "first_loss_reason": first_progress_loss_reason,
            "total_braking_time_s": round(time_spent_braking_s, 3),
            "mean_reaccel_time_s": round(float(np.mean(re_acceleration_events)), 3) if re_acceleration_events else None,
            "reaccel_event_count": len(re_acceleration_events),
        },
        "step_traces": step_traces,
    }

    return outcome


def execute_diagnostic_suite(output_dir: str = "experiments/phase4i") -> Dict[str, Any]:
    out_path = Path(output_dir)
    traces_dir = out_path / "traces"
    logs_dir = out_path / "logs"
    traces_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    controllers = [
        ("Expert_APF_Reference", None),
        ("Arm_B_BC_Baseline", "models/checkpoints/bc_policy_best.pt"),
        ("BC_Phase4H_Candidate", "experiments/phase4h/checkpoints/bc_phase4h_candidate.pt"),
        ("Arm_A_PPO_Baseline", "models/checkpoints/ppo_policy_best.pt"),
        ("PPO_Phase4H_Candidate", "experiments/phase4h/checkpoints/ppo_phase4h_candidate.pt"),
    ]

    scenarios = [
        # 1. Clear Corridor
        {"name": "clear_corridor", "seed": 1, "obs_override": [], "start_override": None},
        # 2. Single Centered Obstacle (Y=0.00m)
        {"name": "single_centered_obs", "seed": 42, "obs_override": [(3.0, 0.0, 1.0)], "start_override": None},
        # 3. Single Off-Center Left (Y=+0.20m)
        {"name": "single_off_center_left", "seed": 42, "obs_override": [(3.0, 0.20, 1.0)], "start_override": None},
        # 4. Single Off-Center Right (Y=-0.20m)
        {"name": "single_off_center_right", "seed": 42, "obs_override": [(3.0, -0.20, 1.0)], "start_override": None},
        # 5. Reproducible Deadlock Probe (approaching centered obstacle at 0.50 m/s from X=2.0m)
        {"name": "known_deadlock_probe", "seed": 42, "obs_override": [(3.0, 0.0, 1.0)], "start_override": (2.0, 0.0, 1.0, 0.50)},
        # 6. Goal-Approach Braking Probe (approaching goal at 0.50 m/s from X=7.5m)
        {"name": "goal_approach_probe", "seed": 1, "obs_override": [], "start_override": (7.5, 0.0, 1.0, 0.50)},
    ]

    env = CorridorSimEnv(scenario="10m", gui=False, seed=42)
    supervisor = SafetySupervisor(corridor=env.corridor)

    all_scenario_outcomes = []
    summary_by_controller = {cid: [] for cid, _ in controllers}

    print("=" * 90)
    print(">>> EXECUTING PHASE 4I TIME-BUDGET DIAGNOSTIC SUITE <<<")
    print(f"Controllers to evaluate: {[c[0] for c in controllers]}")
    print(f"Scenarios: {[s['name'] for s in scenarios]} + 10-seed Three-Pillar Corridor")
    print("=" * 90)

    for cid, ckpt_path in controllers:
        print(f"\n--- Evaluating Controller: {cid} ---")
        model = load_policy(cid, ckpt_path)

        # 1. Fixed Diagnostic Scenarios
        for sc in scenarios:
            res = run_single_episode_diagnostic(
                env=env,
                supervisor=supervisor,
                controller_id=cid,
                model=model,
                scenario_name=sc["name"],
                seed=sc["seed"],
                start_override=sc["start_override"],
                obstacle_override=sc["obs_override"],
            )
            # Save individual trace
            trace_filename = traces_dir / f"{cid}_{sc['name']}.json"
            with open(trace_filename, "w") as f:
                json.dump(res, f, indent=2, cls=NumpyEncoder)

            tb = res["time_budget"]
            print(f"  [{sc['name']:<24}] Steps: {res['total_steps']:3d} | Fwd: {res['forward_progress_m']:4.2f}m | Term: {res['termination_reason']:<8} | Success: {res['is_success']!s:<5} | Unconstr: {tb['time_percentages']['unconstrained_forward']:4.1f}% | Brake: {tb['time_percentages']['braking_or_stopping']:4.1f}%")

            all_scenario_outcomes.append(res)
            summary_by_controller[cid].append(res)

        # 2. Multi-Pillar Three-Obstacle Development Seeds (1000..1009)
        print(f"  Running 10-seed multi-pillar development batch (seeds 1000..1009)...")
        multi_pillar_results = []
        for dev_seed in DEVELOPMENT_SEEDS:
            m_res = run_single_episode_diagnostic(
                env=env,
                supervisor=supervisor,
                controller_id=cid,
                model=model,
                scenario_name="three_pillar_multi_obs",
                seed=dev_seed,
                start_override=None,
                obstacle_override=None, # uses standard procedural 3 pillars from seed
            )
            multi_pillar_results.append(m_res)
            all_scenario_outcomes.append(m_res)
            summary_by_controller[cid].append(m_res)

        succ_cnt = sum(1 for r in multi_pillar_results if r["is_success"])
        mean_prog = float(np.mean([r["forward_progress_m"] for r in multi_pillar_results]))
        mean_brake_pct = float(np.mean([r["time_budget"]["time_percentages"]["braking_or_stopping"] for r in multi_pillar_results]))
        mean_unconstr_pct = float(np.mean([r["time_budget"]["time_percentages"]["unconstrained_forward"] for r in multi_pillar_results]))
        mean_sup_fwd_pct = float(np.mean([r["time_budget"]["time_percentages"]["supervisor_constrained_forward"] for r in multi_pillar_results]))
        mean_lat_pct = float(np.mean([r["time_budget"]["time_percentages"]["lateral_avoidance"] for r in multi_pillar_results]))
        mean_cleared = float(np.mean([r["obstacles_cleared_count"] for r in multi_pillar_results]))

        print(f"  [Multi-Pillar 10-Seed Batch] Successes: {succ_cnt}/10 | Mean Fwd: {mean_prog:.2f}m | Mean Cleared Obs: {mean_cleared:.1f}/3")
        print(f"    Time Budget Breakdown: Unconstrained={mean_unconstr_pct:.1f}%, SupConstrained={mean_sup_fwd_pct:.1f}%, Lateral={mean_lat_pct:.1f}%, Brake/Stop={mean_brake_pct:.1f}%")

        # Save multi-pillar traces
        multi_trace_file = traces_dir / f"{cid}_multi_pillar_batch.json"
        with open(multi_trace_file, "w") as f:
            json.dump(multi_pillar_results, f, indent=2, cls=NumpyEncoder)

    env.close()

    # Save summary metrics
    summary_report_file = Path("reports/phase4i_time_budget_metrics.json")
    summary_report_file.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_report_file, "w") as f:
        json.dump(summary_by_controller, f, indent=2, cls=NumpyEncoder)

    print("\n" + "=" * 90)
    print(">>> PHASE 4I TIME-BUDGET DIAGNOSTIC COMPLETE <<<")
    print(f"Artifacts saved to {out_path} and {summary_report_file}")
    print("=" * 90)

    return summary_by_controller


if __name__ == "__main__":
    execute_diagnostic_suite()
