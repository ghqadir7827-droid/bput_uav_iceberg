"""run_phase5b_experiments.py - Phase 5B Validation & Multi-Obstacle Robustness Suite.

Executes:
1. Experiment A: Reproduction of the 7 Phase 5A diagnostic scenarios.
2. Experiment B: Paired canonical development evaluation across seeds 2000..2049.
3. Experiment C: Multi-obstacle failure diagnosis and trace logging.
4. Experiment D: Isolated State Diagnostics (Stateless Design A vs Persistent Design B).
5. Experiment E: Threshold sensitivity sweep across C_front in [0.70, 0.80, 0.85, 0.95, 1.10]m.
"""

import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pybullet as p
import torch

from controllers.safety_supervisor import SafetyState
from core.contracts import ActionLimits, VelocityAction, compute_forward_clearance_m
from core.geometry import Corridor3D, Obstacle3D
from experiments.phase4j.isolated_supervisor_opt import IsolatedOptimizedSupervisor
from models.policy_net import NavigationPolicyMLP
from sim.corridor_env import CorridorSimEnv


BASELINE_CKPT = "models/checkpoints/bc_policy_best.pt"
OUTPUT_DIR = Path("experiments/phase5b/phase5b_clearance_audit_20261029_220000")
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
# Controller Candidates
# ==============================================================================

class StatelessClearanceGuidanceController:
    """Design A: Stateless Observable Clearance Guidance (Alternative A)."""
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


class PersistentIntentionGuidanceController:
    """Design B: Persistent Avoidance State with Explicit Release & Re-centering."""
    def __init__(self, base_model: NavigationPolicyMLP, activation_threshold_m: float = 0.85, tie_breaker_dir: str = "LEFT"):
        self.base_model = base_model
        self.activation_threshold_m = activation_threshold_m
        self.tie_breaker_dir = tie_breaker_dir
        self.state = "NOMINAL"
        self.active_dir = "NONE"
        self.trigger_x = 0.0
        self.state_step_count = 0

    def reset(self):
        self.state = "NOMINAL"
        self.active_dir = "NONE"
        self.trigger_x = 0.0
        self.state_step_count = 0

    def predict(self, obs: np.ndarray, curr_pos: np.ndarray) -> Tuple[VelocityAction, str, float, float]:
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

        elif self.state in ["AVOID_LEFT", "AVOID_RIGHT"]:
            # Exit condition: front cleared AND moved forward >= 0.70m past trigger, OR timeout (50 steps = 2.5s)
            dist_adv = x - self.trigger_x
            if (c_front > 1.25 and dist_adv > 0.65) or self.state_step_count > 50:
                self.state = "SETTLING"
                self.state_step_count = 0

        elif self.state == "SETTLING":
            # Exit condition: re-centered (|y| < 0.12m) or new obstacle detected or settling timeout (30 steps = 1.5s)
            if c_front < self.activation_threshold_m:
                # Immediate transition to new avoidance
                if diff > 0.08:
                    self.active_dir = "LEFT"
                elif diff < -0.08:
                    self.active_dir = "RIGHT"
                else:
                    self.active_dir = self.tie_breaker_dir
                self.state = f"AVOID_{self.active_dir}"
                self.trigger_x = x
                self.state_step_count = 0
            elif abs(y) < 0.12 or self.state_step_count > 30:
                self.state = "NOMINAL"
                self.active_dir = "NONE"

        # Velocity Override based on State
        vy_cmd = raw_act.vy
        vx_cmd = raw_act.vx

        if self.state == "AVOID_LEFT":
            vy_cmd = max(vy_cmd, 0.25)
            # Ensure safe speed
            vx_cmd = max(0.30, vx_cmd)
        elif self.state == "AVOID_RIGHT":
            vy_cmd = min(vy_cmd, -0.25)
            vx_cmd = max(0.30, vx_cmd)
        elif self.state == "SETTLING":
            # Gentle recentering towards corridor center
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
# PyBullet Scenario Setup
# ==============================================================================

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


# ==============================================================================
# Simulation Episode Runner
# ==============================================================================

def run_simulation_episode(
    env: CorridorSimEnv,
    supervisor: IsolatedOptimizedSupervisor,
    controller: Any,
    controller_type: str,
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
        # Development seed environment: standard corridor pillars
        start_x = float(env._get_drone_position()[0])
        pillar_configs = [(float(o.center[0]), float(o.center[1]), float(o.radius)) for o in env.obstacles]

    steps = 0
    done = False
    min_obs_overall = 10.0
    min_wall_overall = 10.0

    first_caution_step = -1
    initiate_lat_step = -1
    standoff_steps = 0
    throttled_steps = 0
    reversals = 0
    prev_vy_sign = 0
    interventions = 0

    selected_dir_first = "NONE"
    c_left_first = 0.0
    c_right_first = 0.0
    raw_vx_first = 0.0
    raw_vy_first = 0.0
    safe_vx_first = 0.0
    safe_vy_first = 0.0

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
        if controller_type == "Frozen_BC_Baseline":
            with torch.no_grad():
                act_arr = controller.predict_action(obs)
            raw_act = VelocityAction(vx=float(act_arr[0]), vy=float(act_arr[1]), vz=float(act_arr[2]), yaw_rate=float(act_arr[3]))
            selected_dir = "LEFT" if raw_act.vy > 0.05 else ("RIGHT" if raw_act.vy < -0.05 else "NONE")
            rays = obs[0:12] * 4.0
            c_left = min(rays[1], rays[3], rays[5])
            c_right = min(rays[2], rays[4], rays[6])

        elif isinstance(controller, StatelessClearanceGuidanceController) or controller_type in ["Stateless_Clearance_Guidance", "Design_A_Stateless"] or controller_type.startswith("Thresh_"):
            raw_act, selected_dir, c_left, c_right = controller.predict(obs)

        elif isinstance(controller, PersistentIntentionGuidanceController) or controller_type in ["Persistent_Intention_Guidance", "Design_B_Persistent"]:
            raw_act, selected_dir, c_left, c_right = controller.predict(obs, curr_pos)

        else:
            raise ValueError(f"Unknown controller_type {controller_type}")

        if front_dist < 0.85 and first_caution_step == -1:
            first_caution_step = steps
            selected_dir_first = selected_dir
            c_left_first = c_left
            c_right_first = c_right
            raw_vx_first = raw_act.vx
            raw_vy_first = raw_act.vy

        if first_caution_step != -1 and initiate_lat_step == -1 and abs(raw_act.vy) >= 0.05:
            initiate_lat_step = steps

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

        if first_caution_step == steps:
            safe_vx_first = safe_act.vx
            safe_vy_first = safe_act.vy

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

    # Check pillar passage
    p_cleared = [int(final_pos[0] > (px + pr)) for (px, py, pr) in pillar_configs]
    pillars_cleared_count = sum(p_cleared)

    if first_caution_step != -1 and initiate_lat_step != -1:
        time_to_lat_s = round((initiate_lat_step - first_caution_step) * env.dt, 2)
    elif first_caution_step != -1:
        time_to_lat_s = 99.0
    else:
        time_to_lat_s = 0.0

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
        "seed": seed if seed is not None else -1,
        "scenario_id": scenario_id if scenario_id is not None else "canonical_dev",
        "controller": controller_type,
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
        "time_to_initiate_lat_s": time_to_lat_s,
        "pillars_cleared": pillars_cleared_count,
        "pillar_1_cleared": p_cleared[0] if len(p_cleared) > 0 else 0,
        "pillar_2_cleared": p_cleared[1] if len(p_cleared) > 1 else 0,
        "pillar_3_cleared": p_cleared[2] if len(p_cleared) > 2 else 0,
        "final_outcome": final_outcome,
        "selected_dir_first": selected_dir_first,
        "c_left_first": round(c_left_first, 3),
        "c_right_first": round(c_right_first, 3),
    }

    return summary, trace


# ==============================================================================
# Main Phase 5B Execution Pipeline
# ==============================================================================

def main():
    print("=" * 80)
    print(">>> PHASE 5B: CLEARANCE GUIDANCE VALIDATION & MULTI-OBSTACLE AUDIT <<<")
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

    stateless_guidance = StatelessClearanceGuidanceController(base_model, activation_threshold_m=0.85, tie_breaker_dir="LEFT")
    persistent_guidance = PersistentIntentionGuidanceController(base_model, activation_threshold_m=0.85, tie_breaker_dir="LEFT")

    # --------------------------------------------------------------------------
    # Experiment A: Reproduce Phase 5A Scenarios (A through E)
    # --------------------------------------------------------------------------
    print("\n" + "=" * 50)
    print(">>> EXPERIMENT A: PHASE 5A SCENARIO REPRODUCTION <<<")
    print("=" * 50)

    scenarios = [
        "Scenario_A_Centered",
        "Scenario_B_Left_Mirrored",
        "Scenario_B_Right_Mirrored",
        "Scenario_C_Wide_Right",
        "Scenario_C_Wide_Left",
        "Scenario_D_Standoff",
        "Scenario_E_Multi_Obstacle",
    ]

    repro_csv = OUTPUT_DIR / "phase5b_scenario_reproduction.csv"
    if not repro_csv.exists():
        exp_a_results = []
        for sc in scenarios:
            for c_name, ctrl in [
                ("Frozen_BC_Baseline", base_model),
                ("Stateless_Clearance_Guidance", stateless_guidance),
                ("Persistent_Intention_Guidance", persistent_guidance),
            ]:
                res, trace = run_simulation_episode(env, supervisor, ctrl, c_name, scenario_id=sc)
                exp_a_results.append(res)
                print(f"  [{sc:<24}] [{c_name:<30}] Outcome: {res['final_outcome']:<22} Prog: {res['progress_m']:>5.2f}m | MinObs: {res['min_obs_clearance_m']:>6.3f}m")
                if sc in ["Scenario_A_Centered", "Scenario_E_Multi_Obstacle"]:
                    with open(TRACES_DIR / f"repro_{sc}_{c_name}.json", "w") as f:
                        json.dump(trace, f, indent=2, cls=NumpyEncoder)

        with open(repro_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(exp_a_results[0].keys()))
            writer.writeheader()
            writer.writerows(exp_a_results)
        print(f"Saved Experiment A reproduction to {repro_csv}")
    else:
        print(f"Experiment A reproduction already exists at {repro_csv}")

    # --------------------------------------------------------------------------
    # Experiment B: Paired Canonical Development Evaluation (Seeds 2000..2049)
    # --------------------------------------------------------------------------
    print("\n" + "=" * 50)
    print(">>> EXPERIMENT B: PAIRED CANONICAL DEVELOPMENT EVALUATION (SEEDS 2000..2049) <<<")
    print("=" * 50)

    dev_csv = OUTPUT_DIR / "paired_development_results.csv"
    if not dev_csv.exists():
        exp_b_results = []
        for idx, seed in enumerate(DEV_SEEDS):
            base_res, base_trace = run_simulation_episode(env, supervisor, base_model, "Frozen_BC_Baseline", seed=seed)
            exp_b_results.append(base_res)

            cand_res, cand_trace = run_simulation_episode(env, supervisor, stateless_guidance, "Stateless_Clearance_Guidance", seed=seed)
            exp_b_results.append(cand_res)

            if seed in [2000, 2001, 2002, 2005, 2010, 2026]:
                with open(TRACES_DIR / f"dev_seed{seed}_baseline.json", "w") as f:
                    json.dump(base_trace, f, indent=2, cls=NumpyEncoder)
                with open(TRACES_DIR / f"dev_seed{seed}_candidate.json", "w") as f:
                    json.dump(cand_trace, f, indent=2, cls=NumpyEncoder)

            if (idx + 1) % 10 == 0:
                print(f"Completed {idx + 1}/{len(DEV_SEEDS)} development seeds...")

        with open(dev_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(exp_b_results[0].keys()))
            writer.writeheader()
            writer.writerows(exp_b_results)
        print(f"Saved Experiment B paired evaluation to {dev_csv}")
    else:
        print(f"Experiment B paired evaluation already exists at {dev_csv}")
        with open(dev_csv, "r") as f:
            reader = csv.DictReader(f)
            exp_b_results = [
                {k: (float(v) if k in ["duration_s", "progress_m", "min_obs_clearance_m", "min_wall_clearance_m", "throttled_fraction_pct", "standoff_duration_s"] else (int(v) if k in ["seed", "success", "steps", "obs_envelope_violations", "wall_envelope_violations", "obstacle_collisions", "boundary_collisions", "interventions", "reversals", "pillars_cleared", "pillar_1_cleared", "pillar_2_cleared", "pillar_3_cleared"] else v)) for k, v in row.items()}
                for row in reader
            ]

    # Statistical summary for Experiment B
    b_evals = [r for r in exp_b_results if r["controller"] == "Frozen_BC_Baseline"]
    c_evals = [r for r in exp_b_results if r["controller"] == "Stateless_Clearance_Guidance"]

    def print_summary(evals: List[Dict[str, Any]], name: str):
        n = len(evals)
        succ = sum(r["success"] for r in evals)
        w_low, w_high = wilson_score_interval(succ, n)
        mean_prog = np.mean([r["progress_m"] for r in evals])
        mean_dur = np.mean([r["duration_s"] for r in evals])
        mean_throttled = np.mean([r["throttled_fraction_pct"] for r in evals])
        min_obs = min(r["min_obs_clearance_m"] for r in evals)
        min_wall = min(r["min_wall_clearance_m"] for r in evals)
        obs_viols = sum(r["obs_envelope_violations"] for r in evals)
        wall_viols = sum(r["wall_envelope_violations"] for r in evals)
        obs_colls = sum(r["obstacle_collisions"] for r in evals)
        wall_colls = sum(r["boundary_collisions"] for r in evals)
        p1_cleared = sum(r["pillar_1_cleared"] for r in evals)
        p2_cleared = sum(r["pillar_2_cleared"] for r in evals)
        p3_cleared = sum(r["pillar_3_cleared"] for r in evals)

        fail_counts = {}
        for r in evals:
            cat = r["final_outcome"]
            fail_counts[cat] = fail_counts.get(cat, 0) + 1

        print(f"\n--- {name} (N={n}) ---")
        print(f"Successes: {succ}/{n} ({succ/n*100:.1f}%, 95% CI: [{w_low}%, {w_high}%])")
        print(f"Collisions: Obstacle={obs_colls}, Wall={wall_colls}")
        print(f"Envelope Violations (<0.08m): Obstacle={obs_viols}, Wall={wall_viols}")
        print(f"Min Clearances: Obstacle={min_obs:.4f}m, Wall={min_wall:.4f}m")
        print(f"Mean Progress: {mean_prog:.3f}m | Mean Duration: {mean_dur:.2f}s | Throttled: {mean_throttled:.1f}%")
        print(f"Pillars Cleared: P1={p1_cleared}/{n} ({p1_cleared/n*100:.1f}%), P2={p2_cleared}/{n} ({p2_cleared/n*100:.1f}%), P3={p3_cleared}/{n} ({p3_cleared/n*100:.1f}%)")
        print("Outcome Breakdown:")
        for cat, cnt in sorted(fail_counts.items(), key=lambda x: -x[1]):
            print(f"  - {cat}: {cnt} ({cnt/n*100:.1f}%)")

    print_summary(b_evals, "Frozen BC Baseline")
    print_summary(c_evals, "Stateless Clearance Guidance (Candidate)")

    # --------------------------------------------------------------------------
    # Experiment D: Isolated State Diagnostics (Stateless vs Persistent)
    # --------------------------------------------------------------------------
    print("\n" + "=" * 50)
    print(">>> EXPERIMENT D: STATELESS VS PERSISTENT GUIDANCE DIAGNOSTICS <<<")
    print("=" * 50)

    exp_d_results = []
    # Test on specific challenging scenarios and subset of multi-obstacle seeds
    test_suite = [
        ("Scenario_A_Centered", None),
        ("Scenario_B_Left_Mirrored", None),
        ("Scenario_B_Right_Mirrored", None),
        ("Scenario_E_Multi_Obstacle", None),
    ] + [("canonical_dev", s) for s in [2000, 2001, 2002, 2005, 2010, 2015, 2020, 2025, 2030, 2035]]

    state_csv = OUTPUT_DIR / "guidance_state_diagnostics.csv"
    if not state_csv.exists():
        for sc, seed in test_suite:
            # Design A
            res_a, _ = run_simulation_episode(env, supervisor, stateless_guidance, "Design_A_Stateless", seed=seed, scenario_id=sc if seed is None else None)
            exp_d_results.append(res_a)

            # Design B
            res_b, _ = run_simulation_episode(env, supervisor, persistent_guidance, "Design_B_Persistent", seed=seed, scenario_id=sc if seed is None else None)
            exp_d_results.append(res_b)

            label = sc if seed is None else f"seed_{seed}"
            print(f"  [{label:<25}] Design A: {res_a['final_outcome']:<20} (Prog: {res_a['progress_m']:>5.2f}m) | "
                  f"Design B: {res_b['final_outcome']:<20} (Prog: {res_b['progress_m']:>5.2f}m)")

        with open(state_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(exp_d_results[0].keys()))
            writer.writeheader()
            writer.writerows(exp_d_results)
        print(f"Saved Experiment D state diagnostics to {state_csv}")
    else:
        print(f"Experiment D state diagnostics already exists at {state_csv}")

    # --------------------------------------------------------------------------
    # Experiment E: Threshold Sensitivity
    # --------------------------------------------------------------------------
    print("\n" + "=" * 50)
    print(">>> EXPERIMENT E: THRESHOLD SENSITIVITY (C_front in [0.70, 0.80, 0.85, 0.95, 1.10]m) <<<")
    print("=" * 50)

    thresholds = [0.70, 0.80, 0.85, 0.95, 1.10]
    exp_e_results = []
    # Test across 10 paired development seeds [2000..2009]
    sample_seeds = list(range(2000, 2010))

    for thresh in thresholds:
        t_ctrl = StatelessClearanceGuidanceController(base_model, activation_threshold_m=thresh, tie_breaker_dir="LEFT")
        for s in sample_seeds:
            res, _ = run_simulation_episode(env, supervisor, t_ctrl, f"Thresh_{thresh:.2f}m", seed=s)
            res["threshold_m"] = thresh
            exp_e_results.append(res)

        succ_cnt = sum(r["success"] for r in exp_e_results if r.get("threshold_m") == thresh)
        m_prog = np.mean([r["progress_m"] for r in exp_e_results if r.get("threshold_m") == thresh])
        print(f"  Threshold C_front < {thresh:.2f}m: Successes = {succ_cnt}/{len(sample_seeds)} ({succ_cnt/len(sample_seeds)*100:.1f}%), Mean Prog = {m_prog:.3f}m")

    thresh_csv = OUTPUT_DIR / "threshold_sensitivity.csv"
    with open(thresh_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(exp_e_results[0].keys()))
        writer.writeheader()
        writer.writerows(exp_e_results)
    print(f"Saved Experiment E threshold sensitivity to {thresh_csv}")

    env.close()
    print("\nAll Phase 5B validation experiments successfully executed.")


if __name__ == "__main__":
    main()
