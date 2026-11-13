"""run_phase5a_diagnostics.py - Controlled Diagnostic Scenarios & Alternative Evaluations.

Executes Scenarios A, B, C, D, E comparing:
1. Frozen BC Baseline (models/checkpoints/bc_policy_best.pt)
2. Alternative A: Observable Clearance-Based Decision Rule
3. Alternative B: Explicit Categorical Avoidance Intention with Latching
4. Alternative C: Temporal History Probe (Stacking Analysis & Feature Asymmetry Probe)

Saves results to symmetry_scenario_results.csv and detailed JSON traces.
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
OUTPUT_DIR = Path("experiments/phase5a/phase5a_intention_audit_20261029_215000")
TRACES_DIR = OUTPUT_DIR / "traces"
CAUTION_DIST_M = 0.50

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
OBSTACLE_SAFETY_ENVELOPE_M = 0.080
BOUNDARY_SAFETY_ENVELOPE_M = 0.080


class AltAClearanceController:
    """Alternative A: Observable Clearance-Based Decision Rule.
    
    Compares observable left vs right sector rangefinder clearances.
    If front clearance is constrained, applies lateral bias towards the wider passage.
    Includes a deterministic tie-breaker for symmetric encounters.
    """
    def __init__(self, base_model: NavigationPolicyMLP, tie_breaker_dir: str = "LEFT"):
        self.base_model = base_model
        self.tie_breaker_dir = tie_breaker_dir  # "LEFT" or "RIGHT"

    def predict(self, obs: np.ndarray) -> Tuple[VelocityAction, str, float, float]:
        raw_act_arr = self.base_model.predict_action(obs)
        raw_act = VelocityAction(
            vx=float(raw_act_arr[0]),
            vy=float(raw_act_arr[1]),
            vz=float(raw_act_arr[2]),
            yaw_rate=float(raw_act_arr[3]),
        )

        rays = obs[0:12] * 4.0
        # Sector clearances
        c_front = min(rays[0], rays[1], rays[2])
        c_left = min(rays[1], rays[3], rays[5])
        c_right = min(rays[2], rays[4], rays[6])
        diff = c_left - c_right

        selected_dir = "NONE"
        vy_cmd = raw_act.vy

        # If approaching obstacle
        if c_front < 0.85:
            if diff > 0.08:
                # Left passage is measurably wider
                selected_dir = "LEFT"
                vy_cmd = max(vy_cmd, 0.25)
            elif diff < -0.08:
                # Right passage is measurably wider
                selected_dir = "RIGHT"
                vy_cmd = min(vy_cmd, -0.25)
            else:
                # Symmetric deadband: invoke deterministic tie-breaker
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


class AltBCategoricalIntentionController:
    """Alternative B: Explicit Categorical Avoidance Intention with Latching.
    
    Maintains an explicit discrete intention state machine:
      CRUISE -> BYPASS_LEFT / BYPASS_RIGHT -> RETURN_TO_CENTER -> CRUISE
    Once an intention is selected, it remains latched across timesteps until
    the obstacle is cleared, preventing oscillation and symmetry collapse.
    """
    def __init__(self, base_model: NavigationPolicyMLP, tie_breaker_dir: str = "LEFT"):
        self.base_model = base_model
        self.tie_breaker_dir = tie_breaker_dir
        self.state = "CRUISE"
        self.latched_dir = "NONE"
        self.bypass_timer = 0
        self.start_bypass_x = 0.0

    def reset(self):
        self.state = "CRUISE"
        self.latched_dir = "NONE"
        self.bypass_timer = 0
        self.start_bypass_x = 0.0

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

        # State transitions
        if self.state == "CRUISE":
            if c_front < 0.85:
                # Intention classification
                if diff > 0.08:
                    self.latched_dir = "LEFT"
                elif diff < -0.08:
                    self.latched_dir = "RIGHT"
                else:
                    self.latched_dir = self.tie_breaker_dir
                self.state = f"BYPASS_{self.latched_dir}"
                self.start_bypass_x = x
                self.bypass_timer = 0

        elif self.state in ["BYPASS_LEFT", "BYPASS_RIGHT"]:
            self.bypass_timer += 1
            # Check if obstacle cleared (front cleared and moved forward >= 0.8m)
            if (x - self.start_bypass_x > 0.80 and c_front > 1.20) or self.bypass_timer > 60:
                self.state = "RETURN_TO_CENTER"

        elif self.state == "RETURN_TO_CENTER":
            if abs(y) < 0.08:
                self.state = "CRUISE"
                self.latched_dir = "NONE"

        # Velocity conditioning based on latched intention
        vy_cmd = raw_act.vy
        vx_cmd = raw_act.vx

        if self.state == "BYPASS_LEFT":
            vy_cmd = 0.28
            vx_cmd = max(vx_cmd, 0.35)
        elif self.state == "BYPASS_RIGHT":
            vy_cmd = -0.28
            vx_cmd = max(vx_cmd, 0.35)
        elif self.state == "RETURN_TO_CENTER":
            vy_cmd = -0.20 if y > 0 else 0.20
            vx_cmd = max(vx_cmd, 0.40)

        mod_act = VelocityAction(
            vx=float(np.clip(vx_cmd, -0.20, 0.60)),
            vy=float(np.clip(vy_cmd, -0.40, 0.40)),
            vz=raw_act.vz,
            yaw_rate=raw_act.yaw_rate,
        )
        return mod_act, self.latched_dir, float(c_left), float(c_right)


class AltCTemporalHistoryProbe:
    """Alternative C: Temporal History Probe.
    
    Evaluates observation frame stacking (k=4 steps, 0.20s).
    Inspects whether the history contains distinguishing asymmetry or whether
    the entire history remains symmetric on centered approaches.
    """
    def __init__(self, base_model: NavigationPolicyMLP, stack_size: int = 4):
        self.base_model = base_model
        self.stack_size = stack_size
        self.history: List[np.ndarray] = []

    def reset(self):
        self.history = []

    def update(self, obs: np.ndarray):
        self.history.append(obs.copy())
        if len(self.history) > self.stack_size:
            self.history.pop(0)

    def analyze_symmetry(self) -> Dict[str, float]:
        if not self.history:
            return {"max_temporal_asymmetry": 0.0, "mean_temporal_vy": 0.0}
        
        asymmetries = []
        vy_hist = []
        for o in self.history:
            rays = o[0:12] * 4.0
            # Compare left vs right rays
            d_fl30 = rays[1]
            d_fr30 = rays[2]
            d_fl60 = rays[3]
            d_fr60 = rays[4]
            d_l90 = rays[5]
            d_r90 = rays[6]
            asym = abs(d_fl30 - d_fr30) + abs(d_fl60 - d_fr60) + abs(d_l90 - d_r90)
            asymmetries.append(asym)
            vy_hist.append(o[16]) # body vy
        
        return {
            "max_temporal_asymmetry": float(np.max(asymmetries)),
            "mean_temporal_asymmetry": float(np.mean(asymmetries)),
            "mean_temporal_vy": float(np.mean(vy_hist)),
        }


def setup_scenario_obstacles(env: CorridorSimEnv, scenario_id: str) -> List[Tuple[float, float, float]]:
    """Configures specific obstacles for Scenarios A-E in PyBullet."""
    # Clear existing obstacles in PyBullet
    for o_id in env.obstacle_ids:
        p.removeBody(o_id, physicsClientId=env.client_id)
    env.obstacle_ids.clear()
    env.obstacles.clear()

    pillar_configs: List[Tuple[float, float, float]] = []

    if scenario_id == "Scenario_A_Centered":
        # Single centered pillar at x=3.0, y=0.0
        pillar_configs = [(3.0, 0.0, 0.15)]
    elif scenario_id == "Scenario_B_Left_Mirrored":
        # Pillar shifted +0.10m left
        pillar_configs = [(3.0, 0.10, 0.15)]
    elif scenario_id == "Scenario_B_Right_Mirrored":
        # Pillar shifted -0.10m right
        pillar_configs = [(3.0, -0.10, 0.15)]
    elif scenario_id == "Scenario_C_Wide_Right":
        # Pillar at +0.25m left -> Right passage is wide (0.95m), Left passage narrow (0.45m)
        pillar_configs = [(3.0, 0.25, 0.15)]
    elif scenario_id == "Scenario_C_Wide_Left":
        # Pillar at -0.25m right -> Left passage is wide (0.95m), Right passage narrow (0.45m)
        pillar_configs = [(3.0, -0.25, 0.15)]
    elif scenario_id == "Scenario_D_Standoff":
        # Centered pillar at x=3.0, y=0.0
        pillar_configs = [(3.0, 0.0, 0.15)]
    elif scenario_id == "Scenario_E_Multi_Obstacle":
        # 3 sequential pillars:
        # Pillar 1 at x=3.0, y=0.0 (Centered)
        # Pillar 2 at x=5.5, y=0.25 (Left biased)
        # Pillar 3 at x=8.0, y=-0.25 (Right biased)
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


def run_diagnostic_episode(
    env: CorridorSimEnv,
    supervisor: IsolatedOptimizedSupervisor,
    controller_type: str,
    base_model: NavigationPolicyMLP,
    scenario_id: str,
    max_steps: int = 500,
) -> Dict[str, Any]:
    obs, info = env.reset(seed=42)
    supervisor.reset_state()

    # Configure scenario obstacles
    pillar_configs = setup_scenario_obstacles(env, scenario_id)

    # Initial drone placement based on scenario
    if scenario_id == "Scenario_D_Standoff":
        start_x, start_y, start_z = 2.45, 0.0, 1.0
        start_vx, start_vy, start_vz = 0.0, 0.0, 0.0
    else:
        start_x, start_y, start_z = 1.0, 0.0, 1.0
        start_vx, start_vy, start_vz = 0.40, 0.0, 0.0

    p.resetBasePositionAndOrientation(env.drone_id, [start_x, start_y, start_z], [0, 0, 0, 1], physicsClientId=env.client_id)
    p.resetBaseVelocity(env.drone_id, [start_vx, start_vy, start_vz], [0, 0, 0], physicsClientId=env.client_id)
    p.stepSimulation(physicsClientId=env.client_id)

    # Re-cast observations after positioning
    d_obs, _ = env._get_observations()
    obs = d_obs.to_vector()

    # Controller instantiation
    alt_a = AltAClearanceController(base_model, tie_breaker_dir="LEFT")
    alt_b = AltBCategoricalIntentionController(base_model, tie_breaker_dir="LEFT")
    alt_c_probe = AltCTemporalHistoryProbe(base_model, stack_size=4)

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

    named_obs_recorded = {}
    selected_dir_recorded = "NONE"
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

        # Check distances
        obs_c = min([o.distance_to_point(curr_pos) - env.corridor.drone_radius for o in env.obstacles]) if env.obstacles else 10.0
        wall_c = min(
            env.corridor.y_max - curr_pos[1] - env.corridor.drone_radius,
            curr_pos[1] - env.corridor.y_min - env.corridor.drone_radius,
            env.corridor.z_ceiling - curr_pos[2] - env.corridor.drone_radius,
            curr_pos[2] - env.corridor.z_floor - env.corridor.drone_radius,
        )
        min_obs_overall = min(min_obs_overall, obs_c)
        min_wall_overall = min(min_wall_overall, wall_c)

        # Front clearance
        front_dist = compute_forward_clearance_m(obs, max_range_m=env.sensor_config.max_range_m)

        # Controller action prediction
        if controller_type == "Frozen_BC_Baseline":
            with torch.no_grad():
                act_arr = base_model.predict_action(obs)
            raw_act = VelocityAction(vx=float(act_arr[0]), vy=float(act_arr[1]), vz=float(act_arr[2]), yaw_rate=float(act_arr[3]))
            selected_dir = "LEFT" if raw_act.vy > 0.05 else ("RIGHT" if raw_act.vy < -0.05 else "NONE")
            rays = obs[0:12] * 4.0
            c_left = min(rays[1], rays[3], rays[5])
            c_right = min(rays[2], rays[4], rays[6])

        elif controller_type == "Alt_A_Clearance_Rule":
            raw_act, selected_dir, c_left, c_right = alt_a.predict(obs)

        elif controller_type == "Alt_B_Categorical_Intention":
            raw_act, selected_dir, c_left, c_right = alt_b.predict(obs, curr_pos)

        elif controller_type == "Alt_C_Temporal_Probe":
            alt_c_probe.update(obs)
            with torch.no_grad():
                act_arr = base_model.predict_action(obs)
            raw_act = VelocityAction(vx=float(act_arr[0]), vy=float(act_arr[1]), vz=float(act_arr[2]), yaw_rate=float(act_arr[3]))
            selected_dir = "NONE"
            rays = obs[0:12] * 4.0
            c_left = min(rays[1], rays[3], rays[5])
            c_right = min(rays[2], rays[4], rays[6])

        # Track first encounter telemetry
        if front_dist < 0.80 and first_caution_step == -1:
            first_caution_step = steps
            selected_dir_recorded = selected_dir
            c_left_first = c_left
            c_right_first = c_right
            raw_vx_first = raw_act.vx
            raw_vy_first = raw_act.vy
            named_obs_recorded = {
                "ray_front": round(float(obs[0] * 4.0), 3),
                "ray_fl_30": round(float(obs[1] * 4.0), 3),
                "ray_fr_30": round(float(obs[2] * 4.0), 3),
                "ray_fl_60": round(float(obs[3] * 4.0), 3),
                "ray_fr_60": round(float(obs[4] * 4.0), 3),
                "ray_left_90": round(float(obs[5] * 4.0), 3),
                "ray_right_90": round(float(obs[6] * 4.0), 3),
                "body_vx": round(float(obs[15]), 3),
                "body_vy": round(float(obs[16]), 3),
            }

        if first_caution_step != -1 and initiate_lat_step == -1 and abs(raw_act.vy) >= 0.05:
            initiate_lat_step = steps

        # Standoff tracking (low speed near obstacle)
        if speed <= 0.08 and obs_c < 0.50:
            standoff_steps += 1

        # Clamping and supervisor
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

        if "Speed throttled" in s_reason:
            throttled_steps += 1

        # Reversals
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
        })

        norm_applied = safe_act.to_normalized(env.action_limits)
        obs, rew, term, trunc, info = env.step(norm_applied)
        steps += 1
        done = term or trunc

    final_pos = env._get_drone_position()
    final_vel = float(np.linalg.norm(env._get_drone_velocity()))
    term_reason = info.get("termination_reason", "unknown")
    progress = float(final_pos[0] - start_x)

    is_success = (term_reason == "success") and (final_vel <= 0.35)
    is_obs_collision = (term_reason == "crash_obstacle") or (min_obs_overall <= 0.0)
    is_wall_collision = (term_reason == "crash_boundary") or (min_wall_overall <= 0.0)
    is_obs_viol = (min_obs_overall < OBSTACLE_SAFETY_ENVELOPE_M)
    is_wall_viol = (min_wall_overall < BOUNDARY_SAFETY_ENVELOPE_M)

    # Count pillars cleared
    pillars_cleared = sum(1 for (px, py, pr) in pillar_configs if final_pos[0] > px + pr)

    # Time to initiate lateral velocity
    if first_caution_step != -1 and initiate_lat_step != -1:
        time_to_lat_s = round((initiate_lat_step - first_caution_step) * env.dt, 2)
    elif first_caution_step != -1:
        time_to_lat_s = 99.0  # never initiated
    else:
        time_to_lat_s = 0.0

    standoff_duration_s = round(standoff_steps * env.dt, 2)

    # Outcome label
    if is_obs_collision:
        final_outcome = "COLLISION_OBSTACLE"
    elif is_wall_collision:
        final_outcome = "COLLISION_WALL"
    elif is_success:
        final_outcome = "SUCCESS"
    elif standoff_duration_s > 10.0 or steps >= max_steps:
        final_outcome = "STANDOFF_TIMEOUT"
    else:
        final_outcome = term_reason.upper()

    temporal_stats = alt_c_probe.analyze_symmetry() if controller_type == "Alt_C_Temporal_Probe" else {}

    summary = {
        "scenario_id": scenario_id,
        "controller": controller_type,
        "init_pos": [start_x, start_y, start_z],
        "init_vel": [start_vx, start_vy, start_vz],
        "first_caution_step": first_caution_step,
        "c_left_m": round(c_left_first, 3),
        "c_right_m": round(c_right_first, 3),
        "clearance_diff_m": round(c_left_first - c_right_first, 3),
        "raw_vx": round(raw_vx_first, 3),
        "raw_vy": round(raw_vy_first, 3),
        "safe_vx": round(safe_vx_first, 3),
        "safe_vy": round(safe_vy_first, 3),
        "selected_direction": selected_dir_recorded,
        "time_to_initiate_lat_s": time_to_lat_s,
        "standoff_duration_s": standoff_duration_s,
        "min_obs_clearance_m": round(min_obs_overall, 4),
        "min_wall_clearance_m": round(min_wall_overall, 4),
        "obs_envelope_violations": int(is_obs_viol),
        "wall_envelope_violations": int(is_wall_viol),
        "progress_m": round(progress, 3),
        "pillars_cleared": pillars_cleared,
        "steps": steps,
        "final_outcome": final_outcome,
        "named_obs_at_caution": named_obs_recorded,
        "temporal_stats": temporal_stats,
    }

    return summary, trace


def main():
    print("=" * 80)
    print(">>> PHASE 5A: CONTROLLED DIAGNOSTIC SCENARIOS & ALTERNATIVE EVALUATION <<<")
    print("=" * 80)

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

    scenarios = [
        "Scenario_A_Centered",
        "Scenario_B_Left_Mirrored",
        "Scenario_B_Right_Mirrored",
        "Scenario_C_Wide_Right",
        "Scenario_C_Wide_Left",
        "Scenario_D_Standoff",
        "Scenario_E_Multi_Obstacle",
    ]

    controllers = [
        "Frozen_BC_Baseline",
        "Alt_A_Clearance_Rule",
        "Alt_B_Categorical_Intention",
        "Alt_C_Temporal_Probe",
    ]

    TRACES_DIR.mkdir(parents=True, exist_ok=True)
    all_results: List[Dict[str, Any]] = []

    for sc in scenarios:
        print(f"\n--- Running {sc} ---")
        for ctrl in controllers:
            res, trace = run_diagnostic_episode(env, supervisor, ctrl, base_model, sc)
            all_results.append(res)

            # Print concise summary
            print(f"  [{ctrl:<28}] Outcome: {res['final_outcome']:<18} Prog: {res['progress_m']:>5.2f}m | "
                  f"MinObs: {res['min_obs_clearance_m']:>6.3f}m | Dir: {res['selected_direction']:<12} | "
                  f"LatInit: {res['time_to_initiate_lat_s']:>4.1f}s | Standoff: {res['standoff_duration_s']:>4.1f}s")

            # Save sample trace
            trace_filename = TRACES_DIR / f"trace_{sc}_{ctrl}.json"
            with open(trace_filename, "w") as f:
                json.dump(trace, f, indent=2, cls=NumpyEncoder)

    env.close()

    # Save CSV
    csv_path = OUTPUT_DIR / "symmetry_scenario_results.csv"
    csv_fields = [
        "scenario_id", "controller", "init_pos", "init_vel",
        "first_caution_step", "c_left_m", "c_right_m", "clearance_diff_m",
        "raw_vx", "raw_vy", "safe_vx", "safe_vy", "selected_direction",
        "time_to_initiate_lat_s", "standoff_duration_s",
        "min_obs_clearance_m", "min_wall_clearance_m",
        "obs_envelope_violations", "wall_envelope_violations",
        "progress_m", "pillars_cleared", "steps", "final_outcome"
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_results)
    print(f"\nSaved scenario results to {csv_path}")


if __name__ == "__main__":
    main()
