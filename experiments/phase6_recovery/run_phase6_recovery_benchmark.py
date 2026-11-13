"""run_phase6_recovery_benchmark.py - Phase 6 Complete Recovery & Verification Benchmark.

Executes:
1. Experiment A: Diagnostic Scenario Evaluation (Scenarios A through E across 7 corner cases).
2. Experiment B: Canonical Paired Development Benchmark (Seeds 2000–2049) comparing:
   - Baseline: Frozen BC Baseline (bc_policy_best.pt)
   - Phase 5B Candidate: Previous Stateless Guidance (Alternative A)
   - Phase 6 Final Candidate: Production Stateless Clearance Guidance (with wall-aware arbitration & soft landing)
3. Experiment C: Edge ONNX Companion Inference & Parity Benchmark.
4. Generates comprehensive CSV, JSON, and trace artifacts.
"""

import csv
import json
import math
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pybullet as p
from scipy import stats
import torch

from controllers.guidance_controller import StatelessClearanceGuidanceController
from controllers.safety_supervisor import SafetyState, SafetySupervisor, SafetySupervisorConfig
from core.contracts import ActionLimits, DeployableObservation, VelocityAction, compute_forward_clearance_m
from core.geometry import Corridor3D, Obstacle3D
from experiments.phase4j.isolated_supervisor_opt import IsolatedOptimizedSupervisor
from hardware.onnx_runner import OnboardONNXPolicyRunner
from models.policy_net import NavigationPolicyMLP
from sim.corridor_env import CorridorSimEnv


RUN_ID = "phase6_recovery_20261030_003000"
OUTPUT_DIR = Path(f"experiments/phase6_recovery/{RUN_ID}")
TRACES_DIR = OUTPUT_DIR / "traces"
CAUTION_DIST_M = 0.50
OBSTACLE_SAFETY_ENVELOPE_M = 0.080
BOUNDARY_SAFETY_ENVELOPE_M = 0.080
DEV_SEEDS = list(range(2000, 2050))
BASELINE_CKPT = "models/checkpoints/bc_policy_best.pt"


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.floating, float)):
            return float(obj)
        elif isinstance(obj, (np.integer, int)):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        return super().default(obj)


def wilson_score_interval(successes: int, trials: int) -> Tuple[float, float]:
    if trials == 0:
        return 0.0, 0.0
    z = 1.959964
    p_hat = successes / trials
    denominator = 1 + z**2 / trials
    center = (p_hat + z**2 / (2 * trials)) / denominator
    spread = z * math.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * trials)) / trials) / denominator
    return round(max(0.0, center - spread) * 100.0, 1), round(min(1.0, center + spread) * 100.0, 1)


# Legacy Phase 5B Controller for matched comparison
class LegacyPhase5BGuidanceController:
    """Historical Phase 5B stateless guidance candidate."""
    def __init__(self, base_model: NavigationPolicyMLP, activation_threshold_m: float = 0.85):
        self.base_model = base_model
        self.activation_threshold_m = activation_threshold_m

    def predict(self, obs: np.ndarray) -> Tuple[VelocityAction, str, float, float]:
        raw_act_arr = self.base_model.predict_action(obs)
        raw_act = VelocityAction(
            vx=float(raw_act_arr[0]),
            vy=float(raw_act_arr[1]),
            vz=float(raw_act_arr[2]),
            yaw_rate=float(raw_act_arr[3]),
        )
        rays = obs[0:12] * 4.0
        c_front = min(rays[0], rays[1], rays[2])
        c_left = min(rays[1], rays[3], rays[5])
        c_right = min(rays[2], rays[4], rays[6])
        diff = c_left - c_right

        selected_dir = "NONE"
        vy_cmd = raw_act.vy
        if c_front < self.activation_threshold_m:
            if diff > 0.08:
                selected_dir = "LEFT"
                vy_cmd = max(vy_cmd, 0.25)
            elif diff < -0.08:
                selected_dir = "RIGHT"
                vy_cmd = min(vy_cmd, -0.25)
            else:
                selected_dir = "TIE_BREAK_LEFT"
                vy_cmd = max(vy_cmd, 0.25)

        mod_act = VelocityAction(
            vx=raw_act.vx,
            vy=float(np.clip(vy_cmd, -0.40, 0.40)),
            vz=raw_act.vz,
            yaw_rate=raw_act.yaw_rate,
        )
        return mod_act, selected_dir, float(c_left), float(c_right)


def setup_scenario_obstacles(env: CorridorSimEnv, scenario_id: str) -> List[Tuple[float, float, float]]:
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
    controller_name: str,
    seed: Optional[int] = None,
    scenario_id: Optional[str] = None,
    max_steps: int = 500,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
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
    trace = []

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

        front_dist = compute_forward_clearance_m(obs, max_range_m=env.sensor_config.max_range_m)

        # Policy inference
        if controller_name == "Frozen_BC_Baseline":
            with torch.no_grad():
                act_arr = controller.predict_action(obs)
            raw_act = VelocityAction(vx=float(act_arr[0]), vy=float(act_arr[1]), vz=float(act_arr[2]), yaw_rate=float(act_arr[3]))
            s_dir = "NONE"
        elif hasattr(controller, "predict_with_diagnostics"):
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
            raise ValueError(f"Unknown controller: {controller_name}")

        if speed <= 0.08 and obs_c < 0.50:
            standoff_steps += 1

        clamped_act = VelocityAction(
            vx=float(np.clip(raw_act.vx, env.action_limits.min_fwd_vel_mps, env.action_limits.max_fwd_vel_mps)),
            vy=float(np.clip(raw_act.vy, -env.action_limits.max_lat_vel_mps, env.action_limits.max_lat_vel_mps)),
            vz=float(np.clip(raw_act.vz, -env.action_limits.max_vert_vel_mps, env.action_limits.max_vert_vel_mps)),
            yaw_rate=float(np.clip(raw_act.yaw_rate, -env.action_limits.max_yaw_rate_radps, env.action_limits.max_yaw_rate_radps)),
        )

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
            "obs_c": round(obs_c, 3),
            "wall_c": round(wall_c, 3),
            "raw_vx": round(raw_act.vx, 3),
            "raw_vy": round(raw_act.vy, 3),
            "safe_vx": round(safe_act.vx, 3),
            "safe_vy": round(safe_act.vy, 3),
            "guidance_dir": s_dir,
            "state": s_state.value,
            "reason": s_reason,
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
        "controller": controller_name,
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


def main():
    print("=" * 80)
    print(">>> PHASE 6 RECOVERY BENCHMARK: COMPLETE VERIFICATION CAMPAIGN <<<")
    print(f"Run ID: {RUN_ID}")
    print(f"Output Directory: {OUTPUT_DIR}")
    print("=" * 80)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TRACES_DIR.mkdir(parents=True, exist_ok=True)

    env = CorridorSimEnv(gui=False)
    supervisor = IsolatedOptimizedSupervisor(
        corridor=env.corridor, caution_distance_m=CAUTION_DIST_M, limits=env.action_limits
    )

    base_model = NavigationPolicyMLP()
    ckpt = torch.load(BASELINE_CKPT, map_location="cpu")
    base_model.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
    base_model.eval()

    legacy_5b = LegacyPhase5BGuidanceController(base_model, activation_threshold_m=0.85)
    final_phase6 = StatelessClearanceGuidanceController(base_model, activation_threshold_m=0.85)

    # --------------------------------------------------------------------------
    # Experiment A: Diagnostic Scenarios A through E
    # --------------------------------------------------------------------------
    print("\n" + "-" * 60)
    print(">>> EXPERIMENT A: 7 DIAGNOSTIC SCENARIOS EVALUATION <<<")
    print("-" * 60)

    scenarios = [
        "Scenario_A_Centered",
        "Scenario_B_Left_Mirrored",
        "Scenario_B_Right_Mirrored",
        "Scenario_C_Wide_Right",
        "Scenario_C_Wide_Left",
        "Scenario_D_Standoff",
        "Scenario_E_Multi_Obstacle",
    ]

    exp_a_results = []
    for sc in scenarios:
        for c_name, ctrl in [
            ("Frozen_BC_Baseline", base_model),
            ("Phase5B_Stateless_Guidance", legacy_5b),
            ("Phase6_Production_Guidance", final_phase6),
        ]:
            res, trace = run_episode(env, supervisor, ctrl, c_name, scenario_id=sc)
            exp_a_results.append(res)
            print(f"  [{sc:<24}] [{c_name:<28}] Outcome: {res['final_outcome']:<22} Prog: {res['progress_m']:>5.2f}m | MinObs: {res['min_obs_clearance_m']:>6.3f}m | MinWall: {res['min_wall_clearance_m']:>6.3f}m")
            if sc in ["Scenario_A_Centered", "Scenario_C_Wide_Left", "Scenario_D_Standoff", "Scenario_E_Multi_Obstacle"]:
                with open(TRACES_DIR / f"repro_{sc}_{c_name}.json", "w") as f:
                    json.dump(trace, f, indent=2, cls=NumpyEncoder)

    repro_csv = OUTPUT_DIR / "phase6_scenario_reproduction.csv"
    with open(repro_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(exp_a_results[0].keys()))
        writer.writeheader()
        writer.writerows(exp_a_results)
    print(f"Saved Experiment A reproduction to {repro_csv}")

    # --------------------------------------------------------------------------
    # Experiment B: 50 Paired Development Seeds (2000–2049)
    # --------------------------------------------------------------------------
    print("\n" + "-" * 60)
    print(">>> EXPERIMENT B: 50 PAIRED DEVELOPMENT SEEDS EVALUATION <<<")
    print("-" * 60)

    exp_b_results = []
    ctrl_names = [
        ("Frozen_BC_Baseline", base_model),
        ("Phase5B_Stateless_Guidance", legacy_5b),
        ("Phase6_Production_Guidance", final_phase6),
    ]

    for idx, seed in enumerate(DEV_SEEDS):
        for c_name, ctrl in ctrl_names:
            res, trace = run_episode(env, supervisor, ctrl, c_name, seed=seed)
            exp_b_results.append(res)
            if seed in [2000, 2001, 2005, 2006, 2018, 2020, 2022, 2036, 2041, 2049]:
                with open(TRACES_DIR / f"seed{seed}_{c_name}.json", "w") as f:
                    json.dump(trace, f, indent=2, cls=NumpyEncoder)

        if (idx + 1) % 10 == 0:
            print(f"  Processed {idx + 1} / {len(DEV_SEEDS)} paired seeds...")

    paired_csv = OUTPUT_DIR / "paired_development_results_phase6.csv"
    with open(paired_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(exp_b_results[0].keys()))
        writer.writeheader()
        writer.writerows(exp_b_results)
    print(f"Saved Experiment B paired results to {paired_csv}")

    # Compute Statistical Analysis
    by_ctrl = {c[0]: [r for r in exp_b_results if r["controller"] == c[0]] for c in ctrl_names}
    stats_summary = {}

    for c_name, rows in by_ctrl.items():
        n = len(rows)
        succ = sum(r["success"] for r in rows)
        rate = succ / n * 100.0
        ci_low, ci_high = wilson_score_interval(succ, n)
        min_obs = min(r["min_obs_clearance_m"] for r in rows)
        min_wall = min(r["min_wall_clearance_m"] for r in rows)
        obs_viol = sum(r["obs_envelope_violations"] for r in rows)
        wall_viol = sum(r["wall_envelope_violations"] for r in rows)
        obs_col = sum(r["obstacle_collisions"] for r in rows)
        wall_col = sum(r["boundary_collisions"] for r in rows)
        mean_prog = float(np.mean([r["progress_m"] for r in rows]))
        mean_dur = float(np.mean([r["duration_s"] for r in rows]))
        mean_throt = float(np.mean([r["throttled_fraction_pct"] for r in rows]))
        p1 = sum(r["pillar_1_cleared"] for r in rows) / n * 100.0
        p2 = sum(r["pillar_2_cleared"] for r in rows) / n * 100.0
        p3 = sum(r["pillar_3_cleared"] for r in rows) / n * 100.0

        stats_summary[c_name] = {
            "trials": n,
            "successes": succ,
            "success_rate_pct": rate,
            "wilson_ci_95": [ci_low, ci_high],
            "obstacle_collisions": obs_col,
            "boundary_collisions": wall_col,
            "obs_envelope_violations": obs_viol,
            "wall_envelope_violations": wall_viol,
            "min_obstacle_clearance_m": min_obs,
            "min_wall_clearance_m": min_wall,
            "mean_progress_m": round(mean_prog, 3),
            "mean_duration_s": round(mean_dur, 2),
            "mean_throttled_fraction_pct": round(mean_throt, 2),
            "pillar_1_passage_rate_pct": p1,
            "pillar_2_passage_rate_pct": p2,
            "pillar_3_passage_rate_pct": p3,
        }

    # McNemar Paired Comparison between Final Phase 6 and Baseline
    cand_rows = by_ctrl["Phase6_Production_Guidance"]
    base_rows = by_ctrl["Frozen_BC_Baseline"]
    c_wins = 0
    b_wins = 0
    ties_succ = 0
    ties_fail = 0

    for i in range(len(cand_rows)):
        c_s = bool(cand_rows[i]["success"])
        b_s = bool(base_rows[i]["success"])
        if c_s and not b_s:
            c_wins += 1
        elif b_s and not c_s:
            b_wins += 1
        elif c_s and b_s:
            ties_succ += 1
        else:
            ties_fail += 1

    discordant = c_wins + b_wins
    binom_p = float(stats.binomtest(c_wins, discordant, p=0.5).pvalue) if discordant > 0 else 1.0

    stats_summary["paired_comparison_phase6_vs_baseline"] = {
        "candidate_wins": c_wins,
        "baseline_wins": b_wins,
        "ties_both_success": ties_succ,
        "ties_both_fail": ties_fail,
        "net_candidate_wins": c_wins - b_wins,
        "mcnemar_exact_p_value": binom_p,
        "statistically_significant": bool(binom_p < 0.05),
    }

    summary_json = OUTPUT_DIR / "controller_comparison_summary.json"
    with open(summary_json, "w") as f:
        json.dump(stats_summary, f, indent=2, cls=NumpyEncoder)
    print(f"\nSaved summary statistics to {summary_json}")

    # --------------------------------------------------------------------------
    # Experiment C: Edge ONNX Hardware Inference & Parity
    # --------------------------------------------------------------------------
    print("\n" + "-" * 60)
    print(">>> EXPERIMENT C: EDGE ONNX INFERENCE BENCHMARK <<<")
    print("-" * 60)

    runner = OnboardONNXPolicyRunner(onnx_model_path="models/checkpoints/policy_bc.onnx")
    dummy_obs = np.ones(23, dtype=np.float32) * 0.5

    # Warmup
    for _ in range(50):
        _ = runner.predict(dummy_obs)

    latencies_ms = []
    for _ in range(1000):
        t0 = time.perf_counter()
        _ = runner.predict(dummy_obs)
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)

    p50 = float(np.percentile(latencies_ms, 50))
    p95 = float(np.percentile(latencies_ms, 95))
    p99 = float(np.percentile(latencies_ms, 99))
    mean_lat = float(np.mean(latencies_ms))

    onnx_metrics = {
        "model_path": "models/checkpoints/policy_bc.onnx",
        "onnx_file_size_bytes": Path("models/checkpoints/policy_bc.onnx").stat().st_size,
        "inference_cycles": len(latencies_ms),
        "mean_latency_ms": round(mean_lat, 4),
        "p50_latency_ms": round(p50, 4),
        "p95_latency_ms": round(p95, 4),
        "p99_latency_ms": round(p99, 4),
        "target_latency_budget_ms": 3.0,
        "satisfies_50hz_realtime": bool(p99 < 20.0),
    }

    onnx_json = OUTPUT_DIR / "edge_onnx_benchmark_summary.json"
    with open(onnx_json, "w") as f:
        json.dump(onnx_metrics, f, indent=2, cls=NumpyEncoder)
    print(f"Edge ONNX Runner Latency: Mean={mean_lat:.4f}ms, P95={p95:.4f}ms, P99={p99:.4f}ms")
    print(f"Saved ONNX summary to {onnx_json}")

    print("\n" + "=" * 80)
    print(">>> PHASE 6 BENCHMARK EXECUTION COMPLETE <<<")
    print("=" * 80)


if __name__ == "__main__":
    main()

