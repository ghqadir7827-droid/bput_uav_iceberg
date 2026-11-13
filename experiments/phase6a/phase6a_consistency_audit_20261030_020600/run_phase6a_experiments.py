"""run_phase6a_experiments.py - Phase 6A Consistency Audit & Sensitivity Experiments.

Executes:
1. Forensic reproduction and per-step telemetry capture for Scenario C Wide Left.
2. Controlled sensitivity sweep of wall-proximity threshold (0.350m, 0.325m, 0.300m)
   across diagnostic corner-case scenarios and the 50 canonical development seeds (2000-2049).
3. Zero modification to model checkpoints or canonical safety supervisor.
"""

import csv
import json
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pybullet as p
import torch

from controllers.guidance_controller import StatelessClearanceGuidanceController
from controllers.safety_supervisor import SafetyState, SafetySupervisor, SafetySupervisorConfig
from core.contracts import ActionLimits, DeployableObservation, VelocityAction, compute_forward_clearance_m
from core.geometry import Corridor3D, Obstacle3D
from core.guidance import GuidanceConfig, StatelessClearanceGuidanceArbitrator
from experiments.phase4j.isolated_supervisor_opt import IsolatedOptimizedSupervisor
from models.policy_net import NavigationPolicyMLP
from sim.corridor_env import CorridorSimEnv

WORKSPACE_ROOT = Path("/Users/ghulamqadir/.gemini/antigravity-ide/scratch/uav-bput-nav")
EXP_DIR = WORKSPACE_ROOT / "experiments/phase6a/phase6a_consistency_audit_20261030_020600"
TRACES_DIR = EXP_DIR / "traces"
BASELINE_CKPT = WORKSPACE_ROOT / "models/checkpoints/bc_policy_best.pt"

CAUTION_DIST_M = 0.50
CANONICAL_HORIZON_STEPS = 500
OBSTACLE_SAFETY_ENVELOPE_M = 0.080
BOUNDARY_SAFETY_ENVELOPE_M = 0.080
DEV_SEEDS = list(range(2000, 2050))


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def setup_scenario_obstacles(env: CorridorSimEnv, scenario_id: str) -> List[Tuple[float, float, float]]:
    """Configures deterministic pillar layouts for diagnostic evaluation scenarios."""
    for o_id in env.obstacle_ids:
        p.removeBody(o_id, physicsClientId=env.client_id)
    env.obstacle_ids.clear()
    env.obstacles.clear()

    pillar_configs: List[Tuple[float, float, float]] = []
    if scenario_id == "Scenario_A_Centered":
        pillar_configs = [(3.0, 0.0, 0.15)]
    elif scenario_id == "Scenario_B_Left_Mirrored":
        pillar_configs = [(3.0, 0.10, 0.15)]
    elif scenario_id == "Scenario_B_Right_Mirrored":
        pillar_configs = [(3.0, -0.10, 0.15)]
    elif scenario_id == "Scenario_C_Wide_Right":
        pillar_configs = [(3.0, 0.25, 0.15)]
    elif scenario_id == "Scenario_C_Wide_Left":
        pillar_configs = [(3.0, -0.25, 0.15)]
    elif scenario_id == "Scenario_D_Standoff":
        pillar_configs = [(3.0, 0.0, 0.15)]
    elif scenario_id == "Scenario_E_Multi_Obstacle":
        pillar_configs = [
            (3.0, 0.0, 0.15),
            (5.5, 0.25, 0.15),
            (8.0, -0.25, 0.15),
        ]

    c = env.corridor
    for x_c, y_c, rad in pillar_configs:
        height = c.height
        col_shape = p.createCollisionShape(
            p.GEOM_CYLINDER, radius=rad, height=height, physicsClientId=env.client_id
        )
        vis_shape = p.createVisualShape(
            p.GEOM_CYLINDER, radius=rad, length=height, rgbaColor=[0.85, 0.35, 0.25, 1.0],
            physicsClientId=env.client_id
        )
        body_id = p.createMultiBody(
            baseMass=0, baseCollisionShapeIndex=col_shape, baseVisualShapeIndex=vis_shape,
            basePosition=[x_c, y_c, (c.z_floor + c.z_ceiling) / 2.0],
            physicsClientId=env.client_id
        )
        env.obstacle_ids.append(body_id)
        env.obstacles.append(Obstacle3D(center=np.array([x_c, y_c, (c.z_floor + c.z_ceiling) / 2.0]), radius=rad, height=height))

    return pillar_configs


def run_episode(
    env: CorridorSimEnv,
    supervisor: IsolatedOptimizedSupervisor,
    controller: Any,
    label: str,
    seed: Optional[int] = None,
    scenario_id: Optional[str] = None,
    max_steps: int = CANONICAL_HORIZON_STEPS,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Runs a single standardized episode and records comprehensive telemetry."""
    obs, info = env.reset(seed=seed)
    supervisor.reset_state()
    if hasattr(controller, "reset"):
        controller.reset()

    pillar_configs: List[Tuple[float, float, float]] = []
    if scenario_id is not None:
        pillar_configs = setup_scenario_obstacles(env, scenario_id)
        if scenario_id == "Scenario_D_Standoff":
            start_x, start_y, start_z = 2.45, 0.0, 1.0
            start_vx, start_vy, start_vz = 0.0, 0.0, 0.0
        else:
            start_x, start_y, start_z = 1.0, 0.0, 1.0
            start_vx, start_vy, start_vz = 0.40, 0.0, 0.0

        p.resetBasePositionAndOrientation(env.drone_id, [start_x, start_y, start_z], [0, 0, 0, 1], physicsClientId=env.client_id)
        p.resetBaseVelocity(env.drone_id, [start_vx, start_vy, start_vz], [0, 0, 0], physicsClientId=env.client_id)
        p.stepSimulation(physicsClientId=env.client_id)
        d_obs, _ = env._get_observations()
        obs = d_obs.to_vector()
    else:
        start_x = float(env._get_drone_position()[0])
        pillar_configs = [(float(o.center[0]), float(o.center[1]), float(o.radius)) for o in env.obstacles]

    steps = 0
    done = False
    min_obs_overall = 10.0
    min_wall_overall = 10.0
    standoff_steps = 0
    throttled_steps = 0
    interventions = 0
    trace: List[Dict[str, Any]] = []

    while not done and steps < max_steps:
        curr_pos = env._get_drone_position()
        curr_vel = env._get_drone_velocity()
        speed = float(np.linalg.norm(curr_vel))

        obs_c = min([o.distance_to_point(curr_pos) - env.corridor.drone_radius for o in env.obstacles]) if env.obstacles else 10.0
        wall_c = min(
            env.corridor.y_max - curr_pos[1] - env.corridor.drone_radius,
            curr_pos[1] - env.corridor.y_min - env.corridor.drone_radius,
            env.corridor.z_ceiling - curr_pos[2] - env.corridor.drone_radius,
            curr_pos[2] - env.corridor.z_floor - env.corridor.drone_radius,
        )
        min_obs_overall = min(min_obs_overall, obs_c)
        min_wall_overall = min(min_wall_overall, wall_c)

        rays = obs[0:12] * 4.0
        c_front = min(rays[0], rays[1], rays[2])
        c_left = min(rays[1], rays[3], rays[5])
        c_right = min(rays[2], rays[4], rays[6])
        diff = c_left - c_right
        wall_l = float(rays[5])
        wall_r = float(rays[6])

        # Controller action prediction
        if hasattr(controller, "predict_with_diagnostics"):
            raw_act, s_dir, _, _ = controller.predict_with_diagnostics(obs)
        elif hasattr(controller, "predict"):
            pred_res = controller.predict(obs)
            if isinstance(pred_res, tuple):
                raw_act = pred_res[0]
                s_dir = pred_res[1] if len(pred_res) > 1 else "NONE"
            else:
                raw_act = pred_res
                s_dir = "NONE"
        else:
            with torch.no_grad():
                act_arr = controller.predict_action(obs)
            raw_act = VelocityAction(vx=float(act_arr[0]), vy=float(act_arr[1]), vz=float(act_arr[2]), yaw_rate=float(act_arr[3]))
            s_dir = "NONE"

        if speed <= 0.08 and obs_c < 0.50:
            standoff_steps += 1

        clamped_act = VelocityAction(
            vx=float(np.clip(raw_act.vx, env.action_limits.min_fwd_vel_mps, env.action_limits.max_fwd_vel_mps)),
            vy=float(np.clip(raw_act.vy, -env.action_limits.max_lat_vel_mps, env.action_limits.max_lat_vel_mps)),
            vz=float(np.clip(raw_act.vz, -env.action_limits.max_vert_vel_mps, env.action_limits.max_vert_vel_mps)),
            yaw_rate=float(np.clip(raw_act.yaw_rate, -env.action_limits.max_yaw_rate_radps, env.action_limits.max_yaw_rate_radps)),
        )

        front_dist = compute_forward_clearance_m(obs, max_range_m=env.sensor_config.max_range_m)
        safe_act, s_state, s_reason = supervisor.filter_action(
            proposed_action=clamped_act,
            current_position=curr_pos,
            current_velocity=curr_vel,
            min_front_range=front_dist,
            dt=env.dt,
            is_localization_healthy=True,
            telemetry_age_s=0.0,
            observation=obs,
        )

        if s_state != SafetyState.NOMINAL:
            interventions += 1
        if "Speed throttled" in s_reason:
            throttled_steps += 1

        trace.append({
            "step": steps,
            "x": round(float(curr_pos[0]), 3),
            "y": round(float(curr_pos[1]), 3),
            "z": round(float(curr_pos[2]), 3),
            "speed_mps": round(speed, 3),
            "obs_c": round(obs_c, 3),
            "wall_c": round(wall_c, 3),
            "c_front": round(c_front, 3),
            "c_left": round(c_left, 3),
            "c_right": round(c_right, 3),
            "diff": round(diff, 3),
            "wall_l": round(wall_l, 3),
            "wall_r": round(wall_r, 3),
            "raw_vx": round(raw_act.vx, 3),
            "raw_vy": round(raw_act.vy, 3),
            "safe_vx": round(safe_act.vx, 3),
            "safe_vy": round(safe_act.vy, 3),
            "guidance_dir": s_dir,
            "safety_state": s_state.value,
            "safety_reason": s_reason,
        })

        norm_applied = safe_act.to_normalized(env.action_limits)
        obs, rew, term, trunc, info = env.step(norm_applied)
        steps += 1
        done = term or trunc

    final_pos = env._get_drone_position()
    final_vel = float(np.linalg.norm(env._get_drone_velocity()))
    term_reason = info.get("termination_reason", "unknown")
    progress = float(final_pos[0] - start_x)
    goal_dist = float(np.linalg.norm(final_pos - env.corridor.goal_position))

    is_success = (term_reason == "success") and (final_vel <= 0.35)
    is_obs_collision = (term_reason == "crash_obstacle") or (min_obs_overall <= 0.0)
    is_wall_collision = (term_reason == "crash_boundary") or (min_wall_overall <= 0.0)
    is_obs_viol = (min_obs_overall < OBSTACLE_SAFETY_ENVELOPE_M)
    is_wall_viol = (min_wall_overall < BOUNDARY_SAFETY_ENVELOPE_M)

    p_cleared = [int(final_pos[0] > (px + pr)) for (px, py, pr) in pillar_configs]
    pillars_cleared_count = sum(p_cleared)

    standoff_duration_s = round(standoff_steps * env.dt, 2)
    throttled_pct = round(100.0 * throttled_steps / max(1, steps), 1)

    if is_obs_collision:
        final_outcome = "COLLISION_OBSTACLE"
    elif is_wall_collision:
        final_outcome = "COLLISION_WALL"
    elif is_success:
        final_outcome = "SUCCESS"
    elif goal_dist < 0.60 and steps >= max_steps:
        final_outcome = "GOAL_ALIGNMENT_TIMEOUT"
    elif final_pos[0] < 4.0:
        final_outcome = "PROGRESS_STALL_PILLAR_1"
    elif final_pos[0] < 6.0:
        final_outcome = "PROGRESS_STALL_PILLAR_2"
    elif final_pos[0] < 8.0:
        final_outcome = "PROGRESS_STALL_PILLAR_3"
    elif standoff_duration_s > 10.0:
        final_outcome = "TIMEOUT_EXCESSIVE_STANDOFF"
    elif steps >= max_steps:
        final_outcome = "TIMEOUT_SLOW_TRANSIT"
    else:
        final_outcome = term_reason.upper()

    summary = {
        "seed": seed if seed is not None else -1,
        "scenario_id": scenario_id if scenario_id is not None else "canonical_dev",
        "controller": label,
        "success": int(is_success),
        "steps": steps,
        "duration_s": round(steps * env.dt, 2),
        "progress_m": round(progress, 3),
        "goal_dist_m": round(goal_dist, 3),
        "final_speed_mps": round(final_vel, 3),
        "min_obs_clearance_m": round(min_obs_overall, 4),
        "min_wall_clearance_m": round(min_wall_overall, 4),
        "obs_envelope_violations": int(is_obs_viol),
        "wall_envelope_violations": int(is_wall_viol),
        "obstacle_collisions": int(is_obs_collision),
        "boundary_collisions": int(is_wall_collision),
        "interventions": interventions,
        "throttled_fraction_pct": throttled_pct,
        "standoff_duration_s": standoff_duration_s,
        "pillars_cleared": pillars_cleared_count,
        "pillar_1_cleared": p_cleared[0] if len(p_cleared) > 0 else 0,
        "pillar_2_cleared": p_cleared[1] if len(p_cleared) > 1 else 0,
        "pillar_3_cleared": p_cleared[2] if len(p_cleared) > 2 else 0,
        "final_outcome": final_outcome,
    }
    return summary, trace


class CustomThresholdGuidanceController:
    """Wrapper applying a configurable wall-tight threshold for sensitivity testing."""

    def __init__(
        self,
        base_model: NavigationPolicyMLP,
        wall_tight_threshold_m: float = 0.35,
        activation_threshold_m: float = 0.85,
        default_tie_breaker: str = "LEFT",
    ):
        self.base_model = base_model
        self.config = GuidanceConfig(
            activation_threshold_m=activation_threshold_m,
            wall_tight_threshold_m=wall_tight_threshold_m,
            default_tie_breaker=default_tie_breaker,
        )
        self.arbitrator = StatelessClearanceGuidanceArbitrator(config=self.config)
        self.last_dir: str = "NONE"

    def reset(self) -> None:
        self.arbitrator.reset()
        self.last_dir = "NONE"

    def predict_with_diagnostics(
        self, observation: Union[np.ndarray, DeployableObservation]
    ) -> Tuple[VelocityAction, str, float, float]:
        if isinstance(observation, DeployableObservation):
            obs_vec = observation.to_vector()
        else:
            obs_vec = np.asarray(observation, dtype=np.float32)

        raw_act_arr = self.base_model.predict_action(obs_vec)
        raw_act = VelocityAction(
            vx=float(raw_act_arr[0]),
            vy=float(raw_act_arr[1]),
            vz=float(raw_act_arr[2]),
            yaw_rate=float(raw_act_arr[3]),
        )
        mod_act, s_dir, c_l, c_r = self.arbitrator.arbitrate(raw_act, obs_vec)
        self.last_dir = s_dir
        return mod_act, s_dir, c_l, c_r


def main():
    print("=" * 80)
    print(">>> PHASE 6A: PRODUCTION CONSISTENCY & SENSITIVITY EXPERIMENTS <<<")
    print(f"Output Directory: {EXP_DIR}")
    print("=" * 80)

    TRACES_DIR.mkdir(parents=True, exist_ok=True)

    env = CorridorSimEnv(gui=False)
    supervisor = IsolatedOptimizedSupervisor(
        corridor=env.corridor, caution_distance_m=CAUTION_DIST_M, limits=env.action_limits
    )

    base_model = NavigationPolicyMLP()
    ckpt = torch.load(BASELINE_CKPT, map_location="cpu")
    base_model.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
    base_model.eval()

    # Candidate controllers for sensitivity experiment:
    # 1. 0.350m (Phase 6 Current Baseline)
    # 2. 0.325m (Intermediate Candidate)
    # 3. 0.300m (Candidate)
    ctrl_350 = CustomThresholdGuidanceController(base_model, wall_tight_threshold_m=0.350)
    ctrl_325 = CustomThresholdGuidanceController(base_model, wall_tight_threshold_m=0.325)
    ctrl_300 = CustomThresholdGuidanceController(base_model, wall_tight_threshold_m=0.300)

    candidates = [
        ("Candidate_Threshold_0.350m", ctrl_350),
        ("Candidate_Threshold_0.325m", ctrl_325),
        ("Candidate_Threshold_0.300m", ctrl_300),
    ]

    # --------------------------------------------------------------------------
    # Step 1: Diagnostic Scenarios Evaluation & Forensic Trace Capture
    # --------------------------------------------------------------------------
    print("\n--- 1. Evaluating Diagnostic Corner Cases ---")
    scenarios = [
        "Scenario_A_Centered",
        "Scenario_B_Left_Mirrored",
        "Scenario_B_Right_Mirrored",
        "Scenario_C_Wide_Right",
        "Scenario_C_Wide_Left",
        "Scenario_D_Standoff",
    ]

    diag_results = []
    for sc in scenarios:
        for c_label, ctrl in candidates:
            res, trace = run_episode(env, supervisor, ctrl, c_label, scenario_id=sc)
            diag_results.append(res)
            print(f"  [{sc:<24}] [{c_label:<28}] Outcome: {res['final_outcome']:<22} Prog: {res['progress_m']:>5.2f}m | MinObs: {res['min_obs_clearance_m']:>6.4f}m | MinWall: {res['min_wall_clearance_m']:>6.4f}m")
            
            # Save forensic trace for wide-left and centered cases
            if sc in ["Scenario_C_Wide_Left", "Scenario_A_Centered"]:
                t_file = TRACES_DIR / f"{sc}_{c_label}_telemetry.json"
                with open(t_file, "w") as f:
                    json.dump(trace, f, indent=2, cls=NumpyEncoder)

    diag_csv = EXP_DIR / "diagnostic_comparison.csv"
    with open(diag_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(diag_results[0].keys()))
        writer.writeheader()
        writer.writerows(diag_results)
    print(f"Saved diagnostic comparison to {diag_csv}")

    # --------------------------------------------------------------------------
    # Step 2: 50 Development Seeds (2000-2049) Comparative Evaluation
    # --------------------------------------------------------------------------
    print("\n--- 2. Evaluating 50 Canonical Development Seeds Across Candidates ---")
    dev_results = []
    
    for c_label, ctrl in candidates:
        t0 = time.time()
        c_succ = 0
        min_o = 10.0
        min_w = 10.0
        tot_prog = 0.0

        for seed in DEV_SEEDS:
            res, trace = run_episode(env, supervisor, ctrl, c_label, seed=seed)
            dev_results.append(res)
            c_succ += res["success"]
            min_o = min(min_o, res["min_obs_clearance_m"])
            min_w = min(min_w, res["min_wall_clearance_m"])
            tot_prog += res["progress_m"]

        dt_s = time.time() - t0
        rate = c_succ / len(DEV_SEEDS) * 100.0
        mean_p = tot_prog / len(DEV_SEEDS)
        print(f"  {c_label:<28}: {c_succ}/50 ({rate:.1f}%) | MinObs: {min_o:.4f}m | MinWall: {min_w:.4f}m | MeanProg: {mean_p:.3f}m | Time: {dt_s:.1f}s")

    dev_csv = EXP_DIR / "development_seeds_comparison.csv"
    with open(dev_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(dev_results[0].keys()))
        writer.writeheader()
        writer.writerows(dev_results)
    print(f"Saved 50-seed comparative results to {dev_csv}")

    # --------------------------------------------------------------------------
    # Step 3: Compute Summary Metrics
    # --------------------------------------------------------------------------
    summary: Dict[str, Any] = {}
    for c_label, _ in candidates:
        c_rows = [r for r in dev_results if r["controller"] == c_label]
        n = len(c_rows)
        succ = sum(r["success"] for r in c_rows)
        obs_col = sum(r["obstacle_collisions"] for r in c_rows)
        wall_col = sum(r["boundary_collisions"] for r in c_rows)
        obs_env = sum(r["obs_envelope_violations"] for r in c_rows)
        wall_env = sum(r["wall_envelope_violations"] for r in c_rows)
        min_obs = min(r["min_obs_clearance_m"] for r in c_rows)
        min_wall = min(r["min_wall_clearance_m"] for r in c_rows)
        mean_prog = sum(r["progress_m"] for r in c_rows) / n
        mean_dur = sum(r["duration_s"] for r in c_rows) / n
        p1 = sum(r["pillar_1_cleared"] for r in c_rows) / n * 100.0
        p2 = sum(r["pillar_2_cleared"] for r in c_rows) / n * 100.0
        p3 = sum(r["pillar_3_cleared"] for r in c_rows) / n * 100.0

        outcomes: Dict[str, int] = {}
        for r in c_rows:
            o = r["final_outcome"]
            outcomes[o] = outcomes.get(o, 0) + 1

        summary[c_label] = {
            "trials": n,
            "successes": succ,
            "success_rate_pct": round(succ / n * 100.0, 1),
            "obstacle_collisions": obs_col,
            "boundary_collisions": wall_col,
            "obs_envelope_violations": obs_env,
            "wall_envelope_violations": wall_env,
            "min_obstacle_clearance_m": round(min_obs, 4),
            "min_wall_clearance_m": round(min_wall, 4),
            "mean_progress_m": round(mean_prog, 4),
            "mean_duration_s": round(mean_dur, 2),
            "pillar_1_passage_rate_pct": p1,
            "pillar_2_passage_rate_pct": p2,
            "pillar_3_passage_rate_pct": p3,
            "outcome_breakdown": outcomes,
        }

    sum_path = EXP_DIR / "threshold_experiment_summary.json"
    with open(sum_path, "w") as f:
        json.dump(summary, f, indent=2, cls=NumpyEncoder)
    print(f"\nSaved summary JSON to {sum_path}")

    env.close()
    print("\nPhase 6A Sensitivity Benchmark Complete!")


if __name__ == "__main__":
    main()
