"""run_phase5c_benchmark.py - Phase 5C Three-Controller Paired Benchmark & State Analysis.

Evaluates across development seeds 2000..2049 (50 matched seeds):
  - Controller A: Frozen BC Baseline
  - Controller B: Stateless Clearance Guidance
  - Controller C: Persistent Clearance Guidance (with SETTLING re-centering)

Generates:
  - paired_three_controller_results.csv
  - state_transition_metrics.csv
  - controller_comparison_summary.json
  - high-resolution JSON traces for discordant seeds
"""

import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pybullet as p
import torch
from scipy import stats

from controllers.safety_supervisor import SafetyState
from core.contracts import ActionLimits, VelocityAction, compute_forward_clearance_m
from core.geometry import Corridor3D, Obstacle3D
from experiments.phase4j.isolated_supervisor_opt import IsolatedOptimizedSupervisor
from models.policy_net import NavigationPolicyMLP
from sim.corridor_env import CorridorSimEnv


BASELINE_CKPT = "models/checkpoints/bc_policy_best.pt"
OUTPUT_DIR = Path("experiments/phase5c/phase5c_persistent_recenter_20261029_225000")
TRACES_DIR = OUTPUT_DIR / "traces"
CAUTION_DIST_M = 0.50
OBSTACLE_SAFETY_ENVELOPE_M = 0.080
BOUNDARY_SAFETY_ENVELOPE_M = 0.080
DEV_SEEDS = list(range(2000, 2050))


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


# ==============================================================================
# Controller Implementations
# ==============================================================================

class StatelessClearanceGuidanceController:
    """Controller B: Stateless Observable Clearance Guidance."""
    def __init__(self, base_model: NavigationPolicyMLP, activation_threshold_m: float = 0.85, tie_breaker_dir: str = "LEFT"):
        self.base_model = base_model
        self.activation_threshold_m = activation_threshold_m
        self.tie_breaker_dir = tie_breaker_dir

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
                selected_dir = f"TIE_BREAK_{self.tie_breaker_dir}"
                if self.tie_breaker_dir == "LEFT":
                    vy_cmd = max(vy_cmd, 0.25)
                else:
                    vy_cmd = min(vy_cmd, -0.25)

        mod_act = VelocityAction(
            vx=raw_act.vx,
            vy=float(np.clip(vy_cmd, -0.40, 0.40)),
            vz=raw_act.vz,
            yaw_rate=raw_act.yaw_rate,
        )
        return mod_act, selected_dir, float(c_left), float(c_right)


class PersistentClearanceGuidanceController:
    """Controller C: Persistent Clearance Guidance with SETTLING Re-centering."""
    def __init__(self, base_model: NavigationPolicyMLP, activation_threshold_m: float = 0.85, tie_breaker_dir: str = "LEFT"):
        self.base_model = base_model
        self.activation_threshold_m = activation_threshold_m
        self.tie_breaker_dir = tie_breaker_dir
        self.state = "NOMINAL"
        self.active_dir = "NONE"
        self.trigger_x = 0.0
        self.state_step_count = 0
        
        # Telemetry tracking for state analysis
        self.steps_nominal = 0
        self.steps_avoid_left = 0
        self.steps_avoid_right = 0
        self.steps_settling = 0
        self.settling_preemptions = 0
        self.settling_min_obs_clearance = 10.0
        self.settling_min_wall_clearance = 10.0
        self.state_history = []

    def reset(self):
        self.state = "NOMINAL"
        self.active_dir = "NONE"
        self.trigger_x = 0.0
        self.state_step_count = 0
        self.steps_nominal = 0
        self.steps_avoid_left = 0
        self.steps_avoid_right = 0
        self.steps_settling = 0
        self.settling_preemptions = 0
        self.settling_min_obs_clearance = 10.0
        self.settling_min_wall_clearance = 10.0
        self.state_history = []

    def predict(self, obs: np.ndarray, curr_pos: np.ndarray, obs_clearance: float, wall_clearance: float) -> Tuple[VelocityAction, str, float, float]:
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

        x = float(curr_pos[0])
        y = float(curr_pos[1])
        self.state_step_count += 1

        # Track state occupancy
        if self.state == "NOMINAL":
            self.steps_nominal += 1
        elif self.state == "AVOID_LEFT":
            self.steps_avoid_left += 1
        elif self.state == "AVOID_RIGHT":
            self.steps_avoid_right += 1
        elif self.state == "SETTLING":
            self.steps_settling += 1
            self.settling_min_obs_clearance = min(self.settling_min_obs_clearance, obs_clearance)
            self.settling_min_wall_clearance = min(self.settling_min_wall_clearance, wall_clearance)

        # State Transitions
        if self.state == "NOMINAL":
            if c_front < self.activation_threshold_m:
                if diff > 0.08:
                    self.active_dir = "LEFT"
                elif diff < -0.08:
                    self.active_dir = "RIGHT"
                else:
                    self.active_dir = self.tie_breaker_dir

                self.state = f"AVOID_{self.active_dir}"
                self.trigger_x = x
                self.state_step_count = 0
                self.state_history.append((x, self.state, "approach_trigger"))

        elif self.state in ["AVOID_LEFT", "AVOID_RIGHT"]:
            dist_adv = x - self.trigger_x
            # Exit condition: front cleared AND moved forward >= 0.65m past trigger, OR timeout (50 steps = 2.5s)
            if (c_front > 1.25 and dist_adv > 0.65) or self.state_step_count > 50:
                self.state = "SETTLING"
                self.state_step_count = 0
                self.state_history.append((x, self.state, "obstacle_cleared"))

        elif self.state == "SETTLING":
            # Exit conditions
            if c_front < self.activation_threshold_m:
                # Preemption: new obstacle encountered while re-centering
                self.settling_preemptions += 1
                if diff > 0.08:
                    self.active_dir = "LEFT"
                elif diff < -0.08:
                    self.active_dir = "RIGHT"
                else:
                    self.active_dir = self.tie_breaker_dir
                self.state = f"AVOID_{self.active_dir}"
                self.trigger_x = x
                self.state_step_count = 0
                self.state_history.append((x, self.state, "settling_preempted"))
            elif abs(y) < 0.12 or self.state_step_count > 30:
                # Completed re-centering or timeout
                self.state = "NOMINAL"
                self.active_dir = "NONE"
                self.state_history.append((x, self.state, "settling_complete"))

        # Command Generation
        vy_cmd = raw_act.vy
        vx_cmd = raw_act.vx

        if self.state == "AVOID_LEFT":
            vy_cmd = max(vy_cmd, 0.25)
            vx_cmd = max(0.30, vx_cmd)
        elif self.state == "AVOID_RIGHT":
            vy_cmd = min(vy_cmd, -0.25)
            vx_cmd = max(0.30, vx_cmd)
        elif self.state == "SETTLING":
            recenter_vy = -0.20 * np.sign(y) if abs(y) > 0.08 else 0.0
            vy_cmd = float(recenter_vy)

        mod_act = VelocityAction(
            vx=float(np.clip(vx_cmd, -0.20, 0.60)),
            vy=float(np.clip(vy_cmd, -0.40, 0.40)),
            vz=raw_act.vz,
            yaw_rate=raw_act.yaw_rate,
        )
        return mod_act, self.active_dir, float(c_left), float(c_right)


# ==============================================================================
# Simulation Episode Harness
# ==============================================================================

def run_episode(
    env: CorridorSimEnv,
    supervisor: IsolatedOptimizedSupervisor,
    controller: Any,
    controller_name: str,
    seed: int,
    max_steps: int = 500,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Dict[str, Any]]:
    obs, info = env.reset(seed=seed)
    supervisor.reset_state()
    if hasattr(controller, "reset"):
        controller.reset()

    start_x = float(env._get_drone_position()[0])
    pillar_configs = [(float(o.center[0]), float(o.center[1]), float(o.radius)) for o in env.obstacles]

    steps = 0
    done = False
    min_obs_overall = 10.0
    min_wall_overall = 10.0

    interventions = 0
    throttled_steps = 0
    standoff_steps = 0
    reversals = 0
    prev_vy_sign = 0

    trace = []
    y_at_p1 = None
    y_at_p2 = None
    y_at_p3 = None

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

        # Track lateral coordinate when crossing each obstacle longitudinal station
        if y_at_p1 is None and curr_pos[0] >= pillar_configs[0][0]:
            y_at_p1 = float(curr_pos[1])
        if y_at_p2 is None and curr_pos[0] >= pillar_configs[1][0]:
            y_at_p2 = float(curr_pos[1])
        if y_at_p3 is None and curr_pos[0] >= pillar_configs[2][0]:
            y_at_p3 = float(curr_pos[1])

        # Controller action prediction
        if controller_name == "Frozen_BC_Baseline":
            with torch.no_grad():
                act_arr = controller.predict_action(obs)
            raw_act = VelocityAction(vx=float(act_arr[0]), vy=float(act_arr[1]), vz=float(act_arr[2]), yaw_rate=float(act_arr[3]))
            active_dir = "LEFT" if raw_act.vy > 0.05 else ("RIGHT" if raw_act.vy < -0.05 else "NONE")
            c_state = "NOMINAL"

        elif controller_name == "Stateless_Clearance_Guidance":
            raw_act, active_dir, _, _ = controller.predict(obs)
            c_state = "GUIDING" if active_dir != "NONE" else "NOMINAL"

        elif controller_name == "Persistent_Clearance_Guidance":
            raw_act, active_dir, _, _ = controller.predict(obs, curr_pos, obs_c, wall_c)
            c_state = controller.state

        else:
            raise ValueError(f"Unknown controller {controller_name}")

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

        curr_vy_sign = int(np.sign(raw_act.vy))
        if prev_vy_sign != 0 and curr_vy_sign != 0 and curr_vy_sign != prev_vy_sign:
            if obs_c < 0.60:
                reversals += 1
        prev_vy_sign = curr_vy_sign

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
            "c_state": c_state,
            "active_dir": active_dir,
            "s_state": s_state.value,
            "s_reason": s_reason,
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

    # Failure Classification
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
        "seed": seed,
        "controller": controller_name,
        "success": int(is_success),
        "steps": steps,
        "duration_s": round(steps * env.dt, 2),
        "progress_m": round(progress, 3),
        "goal_dist_m": round(goal_dist, 3),
        "min_obs_clearance_m": round(min_obs_overall, 4),
        "min_wall_clearance_m": round(min_wall_overall, 4),
        "obs_envelope_violations": int(is_obs_viol),
        "wall_envelope_violations": int(is_wall_viol),
        "obstacle_collisions": int(is_obs_collision),
        "boundary_collisions": int(is_wall_collision),
        "interventions": interventions,
        "throttled_fraction_pct": throttled_pct,
        "standoff_duration_s": standoff_duration_s,
        "reversals": reversals,
        "pillars_cleared": pillars_cleared_count,
        "pillar_1_cleared": p_cleared[0],
        "pillar_2_cleared": p_cleared[1],
        "pillar_3_cleared": p_cleared[2],
        "final_outcome": final_outcome,
        "y_at_pillar_1": round(y_at_p1, 3) if y_at_p1 is not None else -999.0,
        "y_at_pillar_2": round(y_at_p2, 3) if y_at_p2 is not None else -999.0,
        "y_at_pillar_3": round(y_at_p3, 3) if y_at_p3 is not None else -999.0,
    }

    state_meta = {}
    if controller_name == "Persistent_Clearance_Guidance":
        state_meta = {
            "seed": seed,
            "steps_nominal": controller.steps_nominal,
            "steps_avoid_left": controller.steps_avoid_left,
            "steps_avoid_right": controller.steps_avoid_right,
            "steps_settling": controller.steps_settling,
            "settling_preemptions": controller.settling_preemptions,
            "settling_min_obs_clearance_m": round(controller.settling_min_obs_clearance, 4) if controller.settling_min_obs_clearance < 10.0 else -1.0,
            "settling_min_wall_clearance_m": round(controller.settling_min_wall_clearance, 4) if controller.settling_min_wall_clearance < 10.0 else -1.0,
            "y_at_pillar_1": summary["y_at_pillar_1"],
            "y_at_pillar_2": summary["y_at_pillar_2"],
            "y_at_pillar_3": summary["y_at_pillar_3"],
            "final_outcome": final_outcome,
        }

    return summary, trace, state_meta


# ==============================================================================
# Main Benchmark Pipeline
# ==============================================================================

def main():
    print("=" * 80)
    print(">>> PHASE 5C: THREE-CONTROLLER PAIRED BENCHMARK (SEEDS 2000..2049) <<<")
    print("=" * 80)

    TRACES_DIR.mkdir(parents=True, exist_ok=True)
    env = CorridorSimEnv(gui=False)
    supervisor = IsolatedOptimizedSupervisor(
        corridor=env.corridor,
        caution_distance_m=CAUTION_DIST_M,
        limits=env.action_limits,
    )

    device = torch.device("cpu")
    print(f"Loading Frozen BC Baseline from {BASELINE_CKPT}...")
    base_model = NavigationPolicyMLP()
    ckpt = torch.load(BASELINE_CKPT, map_location=device)
    base_model.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
    base_model.eval()

    ctrl_stateless = StatelessClearanceGuidanceController(base_model, activation_threshold_m=0.85, tie_breaker_dir="LEFT")
    ctrl_persistent = PersistentClearanceGuidanceController(base_model, activation_threshold_m=0.85, tie_breaker_dir="LEFT")

    all_results: List[Dict[str, Any]] = []
    state_metrics: List[Dict[str, Any]] = []

    print(f"Running paired evaluation across {len(DEV_SEEDS)} development seeds...")

    for idx, seed in enumerate(DEV_SEEDS):
        # 1. Controller A: Baseline
        res_a, trace_a, _ = run_episode(env, supervisor, base_model, "Frozen_BC_Baseline", seed)
        all_results.append(res_a)

        # 2. Controller B: Stateless Guidance
        res_b, trace_b, _ = run_episode(env, supervisor, ctrl_stateless, "Stateless_Clearance_Guidance", seed)
        all_results.append(res_b)

        # 3. Controller C: Persistent Guidance
        res_c, trace_c, meta_c = run_episode(env, supervisor, ctrl_persistent, "Persistent_Clearance_Guidance", seed)
        all_results.append(res_c)
        state_metrics.append(meta_c)

        # Save trace if outcomes differ or for key comparison seeds
        if (res_b["success"] != res_c["success"]) or (seed in [2000, 2002, 2005, 2006, 2020, 2022, 2041, 2046]):
            with open(TRACES_DIR / f"seed{seed}_ctrlA_baseline.json", "w") as f:
                json.dump(trace_a, f, indent=2, cls=NumpyEncoder)
            with open(TRACES_DIR / f"seed{seed}_ctrlB_stateless.json", "w") as f:
                json.dump(trace_b, f, indent=2, cls=NumpyEncoder)
            with open(TRACES_DIR / f"seed{seed}_ctrlC_persistent.json", "w") as f:
                json.dump(trace_c, f, indent=2, cls=NumpyEncoder)

        if (idx + 1) % 10 == 0:
            print(f"Completed {idx + 1}/{len(DEV_SEEDS)} seeds...")

    env.close()

    # Save CSVs
    results_csv = OUTPUT_DIR / "paired_three_controller_results.csv"
    with open(results_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_results[0].keys()))
        writer.writeheader()
        writer.writerows(all_results)
    print(f"\nSaved paired results to {results_csv}")

    state_csv = OUTPUT_DIR / "state_transition_metrics.csv"
    with open(state_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(state_metrics[0].keys()))
        writer.writeheader()
        writer.writerows(state_metrics)
    print(f"Saved state transition metrics to {state_csv}")

    # Aggregation & Statistical Tests
    a_evals = [r for r in all_results if r["controller"] == "Frozen_BC_Baseline"]
    b_evals = [r for r in all_results if r["controller"] == "Stateless_Clearance_Guidance"]
    c_evals = [r for r in all_results if r["controller"] == "Persistent_Clearance_Guidance"]

    def compute_stats(evals: List[Dict[str, Any]], name: str) -> Dict[str, Any]:
        n = len(evals)
        succ = sum(r["success"] for r in evals)
        w_low, w_high = wilson_score_interval(succ, n)
        mean_prog = float(np.mean([r["progress_m"] for r in evals]))
        mean_dur = float(np.mean([r["duration_s"] for r in evals]))
        mean_throttled = float(np.mean([r["throttled_fraction_pct"] for r in evals]))
        mean_interventions = float(np.mean([r["interventions"] for r in evals]))
        min_obs = float(min(r["min_obs_clearance_m"] for r in evals))
        min_wall = float(min(r["min_wall_clearance_m"] for r in evals))
        obs_viols = sum(r["obs_envelope_violations"] for r in evals)
        wall_viols = sum(r["wall_envelope_violations"] for r in evals)
        obs_colls = sum(r["obstacle_collisions"] for r in evals)
        wall_colls = sum(r["boundary_collisions"] for r in evals)
        p1 = sum(r["pillar_1_cleared"] for r in evals)
        p2 = sum(r["pillar_2_cleared"] for r in evals)
        p3 = sum(r["pillar_3_cleared"] for r in evals)
        align_fails = sum(1 for r in evals if r["final_outcome"] == "GOAL_ALIGNMENT_TIMEOUT")

        fail_breakdown = {}
        for r in evals:
            cat = r["final_outcome"]
            fail_breakdown[cat] = fail_breakdown.get(cat, 0) + 1

        return {
            "controller": name,
            "episodes": n,
            "successes": succ,
            "success_rate_pct": round(succ / n * 100.0, 2),
            "wilson_ci_95": [w_low, w_high],
            "obstacle_collisions": obs_colls,
            "boundary_collisions": wall_colls,
            "obs_envelope_violations": obs_viols,
            "wall_envelope_violations": wall_viols,
            "min_obstacle_clearance_m": round(min_obs, 4),
            "min_boundary_clearance_m": round(min_wall, 4),
            "mean_progress_m": round(mean_prog, 3),
            "mean_duration_s": round(mean_dur, 2),
            "mean_throttled_pct": round(mean_throttled, 2),
            "mean_interventions": round(mean_interventions, 2),
            "pillar_1_cleared": p1,
            "pillar_1_cleared_pct": round(p1 / n * 100.0, 1),
            "pillar_2_cleared": p2,
            "pillar_2_cleared_pct": round(p2 / n * 100.0, 1),
            "pillar_3_cleared": p3,
            "pillar_3_cleared_pct": round(p3 / n * 100.0, 1),
            "terminal_alignment_failures": align_fails,
            "failure_breakdown": fail_breakdown,
        }

    stats_a = compute_stats(a_evals, "Controller_A_Frozen_BC_Baseline")
    stats_b = compute_stats(b_evals, "Controller_B_Stateless_Clearance")
    stats_c = compute_stats(c_evals, "Controller_C_Persistent_Clearance")

    # Paired McNemar Exact Tests
    def paired_compare(e1: List[Dict[str, Any]], e2: List[Dict[str, Any]], name1: str, name2: str) -> Dict[str, Any]:
        wins = 0   # e2 succeeds, e1 fails
        losses = 0 # e1 succeeds, e2 fails
        ties_succ = 0
        ties_fail = 0

        for r1, r2 in zip(e1, e2):
            s1 = r1["success"]
            s2 = r2["success"]
            if s2 == 1 and s1 == 0:
                wins += 1
            elif s1 == 1 and s2 == 0:
                losses += 1
            elif s1 == 1 and s2 == 1:
                ties_succ += 1
            else:
                ties_fail += 1

        total_discordant = wins + losses
        p_val = float(stats.binomtest(wins, total_discordant, p=0.5).pvalue) if total_discordant > 0 else 1.0

        return {
            "comparison": f"{name2} vs {name1}",
            "wins_for_second": wins,
            "wins_for_first": losses,
            "net_difference": wins - losses,
            "both_succeeded": ties_succ,
            "both_failed": ties_fail,
            "exact_binomial_pvalue": p_val,
            "significant_at_05": p_val < 0.05,
        }

    comp_b_vs_a = paired_compare(a_evals, b_evals, "Controller_A", "Controller_B")
    comp_c_vs_b = paired_compare(b_evals, c_evals, "Controller_B", "Controller_C")
    comp_c_vs_a = paired_compare(a_evals, c_evals, "Controller_A", "Controller_C")

    summary_data = {
        "benchmark_metadata": {
            "horizon_steps": 500,
            "dt_s": 0.05,
            "caution_distance_m": CAUTION_DIST_M,
            "seed_range": [2000, 2049],
            "total_seeds": len(DEV_SEEDS),
        },
        "controller_a_baseline": stats_a,
        "controller_b_stateless": stats_b,
        "controller_c_persistent": stats_c,
        "paired_comparisons": {
            "stateless_vs_baseline": comp_b_vs_a,
            "persistent_vs_stateless": comp_c_vs_b,
            "persistent_vs_baseline": comp_c_vs_a,
        }
    }

    summary_json_path = OUTPUT_DIR / "controller_comparison_summary.json"
    with open(summary_json_path, "w") as f:
        json.dump(summary_data, f, indent=2, cls=NumpyEncoder)
    print(f"Saved comparative summary to {summary_json_path}")

    # Display Printout
    print("\n" + "=" * 80)
    print(f"{'METRIC':<32} | {'CONTROLLER A (BASE)':<18} | {'CONTROLLER B (STATELESS)':<24} | {'CONTROLLER C (PERSISTENT)':<24}")
    print("-" * 106)
    print(f"{'Goal Successes':<32} | {stats_a['successes']:<18} | {stats_b['successes']:<24} | {stats_c['successes']:<24}")
    print(f"{'Success Rate (%)':<32} | {stats_a['success_rate_pct']:<18} | {stats_b['success_rate_pct']:<24} | {stats_c['success_rate_pct']:<24}")
    print(f"{'Wilson 95% CI':<32} | {str(stats_a['wilson_ci_95']):<18} | {str(stats_b['wilson_ci_95']):<24} | {str(stats_c['wilson_ci_95']):<24}")
    print(f"{'Obstacle Collisions':<32} | {stats_a['obstacle_collisions']:<18} | {stats_b['obstacle_collisions']:<24} | {stats_c['obstacle_collisions']:<24}")
    print(f"{'Boundary Collisions':<32} | {stats_a['boundary_collisions']:<18} | {stats_b['boundary_collisions']:<24} | {stats_c['boundary_collisions']:<24}")
    print(f"{'Obs Envelope Viols (<0.08m)':<32} | {stats_a['obs_envelope_violations']:<18} | {stats_b['obs_envelope_violations']:<24} | {stats_c['obs_envelope_violations']:<24}")
    print(f"{'Wall Envelope Viols (<0.08m)':<32} | {stats_a['wall_envelope_violations']:<18} | {stats_b['wall_envelope_violations']:<24} | {stats_c['wall_envelope_violations']:<24}")
    print(f"{'Min Obstacle Clearance (m)':<32} | {stats_a['min_obstacle_clearance_m']:<18} | {stats_b['min_obstacle_clearance_m']:<24} | {stats_c['min_obstacle_clearance_m']:<24}")
    print(f"{'Min Boundary Clearance (m)':<32} | {stats_a['min_boundary_clearance_m']:<18} | {stats_b['min_boundary_clearance_m']:<24} | {stats_c['min_boundary_clearance_m']:<24}")
    print(f"{'Mean Progress (m)':<32} | {stats_a['mean_progress_m']:<18} | {stats_b['mean_progress_m']:<24} | {stats_c['mean_progress_m']:<24}")
    print(f"{'Mean Duration (s)':<32} | {stats_a['mean_duration_s']:<18} | {stats_b['mean_duration_s']:<24} | {stats_c['mean_duration_s']:<24}")
    print(f"{'Mean Throttled Fraction (%)':<32} | {stats_a['mean_throttled_pct']:<18} | {stats_b['mean_throttled_pct']:<24} | {stats_c['mean_throttled_pct']:<24}")
    print(f"{'Pillar 1 Clearance Rate (%)':<32} | {stats_a['pillar_1_cleared_pct']:<18} | {stats_b['pillar_1_cleared_pct']:<24} | {stats_c['pillar_1_cleared_pct']:<24}")
    print(f"{'Pillar 2 Clearance Rate (%)':<32} | {stats_a['pillar_2_cleared_pct']:<18} | {stats_b['pillar_2_cleared_pct']:<24} | {stats_c['pillar_2_cleared_pct']:<24}")
    print(f"{'Pillar 3 Clearance Rate (%)':<32} | {stats_a['pillar_3_cleared_pct']:<18} | {stats_b['pillar_3_cleared_pct']:<24} | {stats_c['pillar_3_cleared_pct']:<24}")
    print(f"{'Terminal Alignment Failures':<32} | {stats_a['terminal_alignment_failures']:<18} | {stats_b['terminal_alignment_failures']:<24} | {stats_c['terminal_alignment_failures']:<24}")
    print("=" * 80)

    print("\n--- PAIRED COMPARISON SUMMARY ---")
    print(f"1. Controller B vs Controller A: +{comp_b_vs_a['net_difference']} net wins (Wins: {comp_b_vs_a['wins_for_second']}, Losses: {comp_b_vs_a['wins_for_first']}, p = {comp_b_vs_a['exact_binomial_pvalue']:.4f})")
    print(f"2. Controller C vs Controller B: {comp_c_vs_b['net_difference']:+d} net wins (Wins: {comp_c_vs_b['wins_for_second']}, Losses: {comp_c_vs_b['wins_for_first']}, p = {comp_c_vs_b['exact_binomial_pvalue']:.4f})")
    print(f"3. Controller C vs Controller A: +{comp_c_vs_a['net_difference']} net wins (Wins: {comp_c_vs_a['wins_for_second']}, Losses: {comp_c_vs_a['wins_for_first']}, p = {comp_c_vs_a['exact_binomial_pvalue']:.4f})")


if __name__ == "__main__":
    main()

