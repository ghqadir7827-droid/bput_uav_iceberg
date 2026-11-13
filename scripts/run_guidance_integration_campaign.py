"""run_guidance_integration_campaign.py - Full Evaluation of Reactive Guidance with BC and PPO.

Evaluates:
Phase A - Baselines:
  1. reactive_baseline
  2. raw_bc
  3. guided_bc_phase6a
  4. raw_ppo
  5. guided_ppo_phase6a

Phase B - Failure Case Diagnostics (Seeds 99, 686, 10009, 10131, 10667)

Phase C - Independent Guidance Improvements on BC and PPO:
  - cand1_bc_early_decel: Earlier longitudinal deceleration near obstacles
  - cand2_bc_multibeam: Multi-beam sector-clearance weighting
  - cand3_bc_centering: Weak corridor-centering bias
  - cand4_bc_hysteresis: Enhanced directional hysteresis and latching
  - cand1_ppo_early_decel: Early deceleration applied to PPO
  - cand_best_bc: Best integrated guidance candidate for BC
  - cand_best_ppo: Best integrated guidance candidate for PPO

Phase D - Independent Pristine Confirmation Benchmark (Seeds 12000..12099)
"""

import argparse
import csv
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy import stats
import torch

from controllers.bc_policy import BehaviorCloningPolicyController
from controllers.guidance_controller import StatelessClearanceGuidanceController
from controllers.ppo_policy import PPOPolicyController
from controllers.registry import ReactiveBaselineController, RandomBaselineController
from controllers.safety_supervisor import SafetyState, SafetySupervisor, SafetySupervisorConfig
from core.contracts import ActionLimits, DeployableObservation, VelocityAction, compute_forward_clearance_m
from core.geometry import Corridor3D, Obstacle3D, TerminationReason
from core.guidance import GuidanceConfig, StatelessClearanceGuidanceArbitrator
from models.policy_net import NavigationPolicyMLP
from sim.corridor_env import CorridorSimEnv


# --- Custom Guidance Arbitrators for Phase C ---

class EarlyDecelArbitrator(StatelessClearanceGuidanceArbitrator):
    """Candidate 1: Earlier longitudinal deceleration when approaching obstacles."""
    def arbitrate(self, raw_action: VelocityAction, observation: Union[np.ndarray, DeployableObservation]):
        act, sdir, cl, cr = super().arbitrate(raw_action, observation)
        obs_vec = np.nan_to_num(np.asarray(observation, dtype=np.float32))
        rays = obs_vec[0:12] * 4.0
        c_front = float(min(rays[0], rays[1], rays[2]))
        if c_front < 1.30:
            scale_vx = float(np.clip(0.25 + 0.35 * (c_front - 0.40) / 0.90, 0.22, raw_action.vx))
            act = VelocityAction(vx=scale_vx, vy=act.vy, vz=act.vz, yaw_rate=act.yaw_rate)
        return act, sdir, cl, cr


class MultiBeamWeightArbitrator(StatelessClearanceGuidanceArbitrator):
    """Candidate 2: Multi-beam sector-clearance weighting for earlier lateral evasion."""
    def arbitrate(self, raw_action: VelocityAction, observation: Union[np.ndarray, DeployableObservation]):
        obs_vec = np.nan_to_num(np.asarray(observation, dtype=np.float32))
        rays = obs_vec[0:12] * 4.0
        c_front = float(min(rays[0], rays[1], rays[2]))

        c_left = float(0.50 * rays[1] + 0.30 * rays[3] + 0.20 * rays[5])
        c_right = float(0.50 * rays[2] + 0.30 * rays[4] + 0.20 * rays[6])
        diff = c_left - c_right

        wall_l = float(rays[5])
        wall_r = float(rays[6])

        goal_vec = obs_vec[12:15]
        in_terminal = bool(np.linalg.norm(goal_vec) < self.config.terminal_goal_norm_threshold)

        vy_cmd = raw_action.vy
        override_speed = self.config.min_lateral_override_mps
        selected_dir = "NONE"

        if c_front < 1.25 and not in_terminal:
            if wall_l < 0.35 and wall_r >= 0.35:
                selected_dir = "WALL_AVOID_RIGHT"
                vy_cmd = min(vy_cmd, -override_speed)
            elif wall_r < 0.35 and wall_l >= 0.35:
                selected_dir = "WALL_AVOID_LEFT"
                vy_cmd = max(vy_cmd, override_speed)
            elif diff > 0.06:
                selected_dir = "LEFT"
                vy_cmd = max(vy_cmd, override_speed)
            elif diff < -0.06:
                selected_dir = "RIGHT"
                vy_cmd = min(vy_cmd, -override_speed)
            else:
                margin = self.config.wall_advantage_margin_m
                if wall_r > wall_l + margin:
                    selected_dir = "TIE_BREAK_RIGHT"
                    vy_cmd = min(vy_cmd, -override_speed)
                else:
                    selected_dir = "TIE_BREAK_LEFT"
                    vy_cmd = max(vy_cmd, override_speed)
            self.last_dir = selected_dir
        else:
            self.last_dir = "NONE"

        guided_act = VelocityAction(
            vx=float(np.clip(raw_action.vx, self.limits.min_fwd_vel_mps, self.limits.max_fwd_vel_mps)),
            vy=float(np.clip(vy_cmd, -self.limits.max_lat_vel_mps, self.limits.max_lat_vel_mps)),
            vz=float(np.clip(raw_action.vz, -self.limits.max_vert_vel_mps, self.limits.max_vert_vel_mps)),
            yaw_rate=float(np.clip(raw_action.yaw_rate, -self.limits.max_yaw_rate_radps, self.limits.max_yaw_rate_radps)),
        )
        return guided_act, selected_dir, c_left, c_right


class CenteringBiasArbitrator(StatelessClearanceGuidanceArbitrator):
    """Candidate 3: Weak corridor-centering bias when not evading obstacles."""
    def arbitrate(self, raw_action: VelocityAction, observation: Union[np.ndarray, DeployableObservation]):
        act, sdir, cl, cr = super().arbitrate(raw_action, observation)
        obs_vec = np.nan_to_num(np.asarray(observation, dtype=np.float32))
        rays = obs_vec[0:12] * 4.0
        wall_l = float(rays[5])
        wall_r = float(rays[6])

        if sdir == "NONE" and abs(wall_l - wall_r) > 0.15:
            centering_nudge = float(np.clip(0.30 * (wall_l - wall_r), -0.15, 0.15))
            mod_vy = float(np.clip(act.vy + centering_nudge, -self.limits.max_lat_vel_mps, self.limits.max_lat_vel_mps))
            act = VelocityAction(vx=act.vx, vy=mod_vy, vz=act.vz, yaw_rate=act.yaw_rate)
        return act, sdir, cl, cr


class EnhancedHysteresisArbitrator(StatelessClearanceGuidanceArbitrator):
    """Candidate 4: Enhanced directional hysteresis and latching."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config.hysteresis_m = 0.10
        self.config.deadband_m = 0.12
        self.latch_timer = 0
        self.latched_dir = "NONE"

    def reset(self):
        super().reset()
        self.latch_timer = 0
        self.latched_dir = "NONE"

    def arbitrate(self, raw_action: VelocityAction, observation: Union[np.ndarray, DeployableObservation]):
        obs_vec = np.nan_to_num(np.asarray(observation, dtype=np.float32))
        rays = obs_vec[0:12] * 4.0
        c_front = float(min(rays[0], rays[1], rays[2]))

        if self.latch_timer > 0:
            self.latch_timer -= 1
            if c_front < 1.0:
                override_speed = self.config.min_lateral_override_mps
                if self.latched_dir in ["LEFT", "WALL_AVOID_LEFT", "TIE_BREAK_LEFT"]:
                    vy = max(raw_action.vy, override_speed)
                else:
                    vy = min(raw_action.vy, -override_speed)
                act = VelocityAction(
                    vx=float(raw_action.vx),
                    vy=float(np.clip(vy, -self.limits.max_lat_vel_mps, self.limits.max_lat_vel_mps)),
                    vz=float(raw_action.vz),
                    yaw_rate=float(raw_action.yaw_rate)
                )
                return act, self.latched_dir, float(rays[1]), float(rays[2])

        act, sdir, cl, cr = super().arbitrate(raw_action, observation)
        if sdir != "NONE":
            self.latched_dir = sdir
            self.latch_timer = 8
        return act, sdir, cl, cr


class BestIntegratedArbitrator(StatelessClearanceGuidanceArbitrator):
    """Best Candidate: Synergistic Combination of Early Decel + Multi-Beam Sector Scoring + Soft Centering."""
    def arbitrate(self, raw_action: VelocityAction, observation: Union[np.ndarray, DeployableObservation]):
        obs_vec = np.nan_to_num(np.asarray(observation, dtype=np.float32))
        rays = obs_vec[0:12] * 4.0
        c_front = float(min(rays[0], rays[1], rays[2]))

        c_left = float(0.50 * rays[1] + 0.30 * rays[3] + 0.20 * rays[5])
        c_right = float(0.50 * rays[2] + 0.30 * rays[4] + 0.20 * rays[6])
        diff = c_left - c_right

        wall_l = float(rays[5])
        wall_r = float(rays[6])

        goal_vec = obs_vec[12:15]
        in_terminal = bool(np.linalg.norm(goal_vec) < self.config.terminal_goal_norm_threshold)

        vy_cmd = raw_action.vy
        vx_cmd = raw_action.vx
        override_speed = self.config.min_lateral_override_mps
        selected_dir = "NONE"

        # Early longitudinal deceleration
        if c_front < 1.30:
            vx_cmd = float(np.clip(0.25 + 0.35 * (c_front - 0.40) / 0.90, 0.22, raw_action.vx))

        if c_front < 1.15 and not in_terminal:
            if wall_l < 0.35 and wall_r >= 0.35:
                selected_dir = "WALL_AVOID_RIGHT"
                vy_cmd = min(vy_cmd, -override_speed)
            elif wall_r < 0.35 and wall_l >= 0.35:
                selected_dir = "WALL_AVOID_LEFT"
                vy_cmd = max(vy_cmd, override_speed)
            elif diff > 0.06:
                selected_dir = "LEFT"
                vy_cmd = max(vy_cmd, override_speed)
            elif diff < -0.06:
                selected_dir = "RIGHT"
                vy_cmd = min(vy_cmd, -override_speed)
            else:
                margin = self.config.wall_advantage_margin_m
                if wall_r > wall_l + margin:
                    selected_dir = "TIE_BREAK_RIGHT"
                    vy_cmd = min(vy_cmd, -override_speed)
                else:
                    selected_dir = "TIE_BREAK_LEFT"
                    vy_cmd = max(vy_cmd, override_speed)
            self.last_dir = selected_dir
        else:
            self.last_dir = "NONE"
            if abs(wall_l - wall_r) > 0.18:
                centering_nudge = float(np.clip(0.25 * (wall_l - wall_r), -0.12, 0.12))
                vy_cmd += centering_nudge

        guided_act = VelocityAction(
            vx=float(np.clip(vx_cmd, self.limits.min_fwd_vel_mps, self.limits.max_fwd_vel_mps)),
            vy=float(np.clip(vy_cmd, -self.limits.max_lat_vel_mps, self.limits.max_lat_vel_mps)),
            vz=float(np.clip(raw_action.vz, -self.limits.max_vert_vel_mps, self.limits.max_vert_vel_mps)),
            yaw_rate=float(np.clip(raw_action.yaw_rate, -self.limits.max_yaw_rate_radps, self.limits.max_yaw_rate_radps)),
        )
        return guided_act, selected_dir, c_left, c_right


# --- Controller Wrapper Helper ---

class GenericGuidedController:
    """Wraps any base policy controller with an arbitrary guidance arbitrator."""
    def __init__(self, base_controller, arbitrator, name: str):
        self.base_controller = base_controller
        self.arbitrator = arbitrator
        self.name = name

    def reset(self):
        if hasattr(self.base_controller, "reset"):
            self.base_controller.reset()
        if hasattr(self.arbitrator, "reset"):
            self.arbitrator.reset()

    def predict(self, obs):
        raw_act = self.base_controller.predict(obs)
        guided_act, _, _, _ = self.arbitrator.arbitrate(raw_act, obs)
        return guided_act


@dataclass
class EpisodeRecord:
    seed: int
    controller_id: str
    primary_outcome: str
    total_steps: int
    flight_time_s: float
    success: bool
    obstacle_collision: bool
    wall_collision: bool
    safety_envelope_violation: bool
    min_obstacle_clearance_m: float
    min_wall_clearance_m: float
    final_x: float
    final_y: float
    final_z: float
    forward_progress_m: float
    final_goal_dist_m: float
    pillar1_cleared: bool
    pillar2_cleared: bool
    pillar3_cleared: bool
    supervisor_interventions: int
    throttled_steps: int
    mean_throttled_fraction: float
    direction_changes: int
    status_at_500: str
    progress_at_500_m: float
    success_at_500: bool


def run_episode(
    controller,
    controller_id: str,
    seed: int,
    env: CorridorSimEnv,
    max_steps: int = 1000
) -> EpisodeRecord:
    if hasattr(controller, "reset"):
        controller.reset()

    sup_config = SafetySupervisorConfig(obstacle_caution_dist_m=0.50)
    safety_sup = SafetySupervisor(corridor=env.corridor, config=sup_config)

    obs, info = env.reset(seed=seed)
    c = env.corridor

    min_obs_clear = 999.0
    min_wall_clear = 999.0
    interventions = 0
    throttled_steps = 0
    throttled_fracs = []

    direction_changes = 0
    prev_vy_sign = 0
    consecutive_stall_steps = 0

    had_obs_collision = False
    had_wall_collision = False
    had_envelope_violation = False
    reached_goal_fast = False

    snapshot_500_taken = False
    status_at_500 = "in_flight"
    progress_at_500_m = 0.0
    success_at_500 = False

    x_start = float(c.start_position[0])
    recent_x = []

    done = False
    step = 0

    while not done and step < max_steps:
        step += 1
        priv = info["privileged_state"]
        raw_act = controller.predict(obs)
        front_d = compute_forward_clearance_m(obs, max_range_m=env.sensor_config.max_range_m)

        safe_act, s_state, _ = safety_sup.filter_action(
            proposed_action=raw_act,
            current_position=priv.global_position,
            current_velocity=priv.global_velocity,
            min_front_range=front_d,
            dt=env.dt,
            observation=obs,
        )

        if s_state != SafetyState.NOMINAL:
            interventions += 1

        if raw_act.vx > 0.05 and safe_act.vx < raw_act.vx:
            throttled_steps += 1
            t_frac = float(max(0.0, 1.0 - safe_act.vx / raw_act.vx))
        else:
            t_frac = 0.0
        throttled_fracs.append(t_frac)

        curr_vy_sign = 1 if safe_act.vy > 0.05 else (-1 if safe_act.vy < -0.05 else 0)
        if curr_vy_sign != 0 and prev_vy_sign != 0 and curr_vy_sign != prev_vy_sign:
            direction_changes += 1
        if curr_vy_sign != 0:
            prev_vy_sign = curr_vy_sign

        norm_act = safe_act.to_normalized(env.action_limits)
        obs, rew, term, trunc, info = env.step(norm_act)

        next_pos = env._get_drone_position()
        next_vel = env._get_drone_velocity()
        speed = float(np.linalg.norm(next_vel))

        d_obs = float(info.get("obstacle_clearance", 999.0))
        d_wall = float(info.get("boundary_distance", 999.0))
        d_goal = float(c.distance_to_goal(next_pos))

        min_obs_clear = min(min_obs_clear, d_obs)
        min_wall_clear = min(min_wall_clear, d_wall)

        if d_obs <= 0.0:
            had_obs_collision = True
        if d_wall <= 0.0:
            had_wall_collision = True
        if (d_obs < 0.08 or d_wall < 0.08) and not (had_obs_collision or had_wall_collision):
            had_envelope_violation = True

        if d_goal <= c.goal_radius and speed > 0.35:
            reached_goal_fast = True

        if speed < 0.02 and (d_obs < 0.50 or d_wall < 0.35):
            consecutive_stall_steps += 1
        else:
            consecutive_stall_steps = 0

        recent_x.append(float(next_pos[0]))
        if len(recent_x) > 100:
            recent_x.pop(0)

        if step == 500 and not snapshot_500_taken:
            snapshot_500_taken = True
            progress_at_500_m = float(next_pos[0] - x_start)
            if info.get("termination_reason") == TerminationReason.SUCCESS.value:
                status_at_500 = "success"
                success_at_500 = True
            elif had_obs_collision:
                status_at_500 = "obstacle_collision"
            elif had_wall_collision:
                status_at_500 = "wall_collision"
            elif consecutive_stall_steps >= 20:
                status_at_500 = "standstill"
            else:
                status_at_500 = "in_flight"

        done = term or trunc

    final_pos = env._get_drone_position()
    final_speed = float(np.linalg.norm(env._get_drone_velocity()))
    prog_m = float(final_pos[0] - x_start)
    final_goal_d = float(c.distance_to_goal(final_pos))
    term_reason = str(info.get("termination_reason", ""))

    p1 = bool(final_pos[0] > 3.35)
    p2 = bool(final_pos[0] > 5.35)
    p3 = bool(final_pos[0] > 7.35)

    if not snapshot_500_taken:
        progress_at_500_m = prog_m
        if term_reason == TerminationReason.SUCCESS.value:
            status_at_500 = "success"
            success_at_500 = True
        elif had_obs_collision:
            status_at_500 = "obstacle_collision"
        elif had_wall_collision:
            status_at_500 = "wall_collision"
        else:
            status_at_500 = "terminated_early"

    delta_x_recent = float(recent_x[-1] - recent_x[0]) if len(recent_x) > 20 else 1.0
    is_oscillating = bool(direction_changes >= 25 and abs(delta_x_recent) < 0.25 and consecutive_stall_steps < 30)

    if had_obs_collision:
        outcome = "OBSTACLE_COLLISION"
    elif had_wall_collision:
        outcome = "WALL_COLLISION"
    elif had_envelope_violation:
        outcome = "SAFETY_ENVELOPE_VIOLATION"
    elif term_reason == TerminationReason.SUCCESS.value and final_goal_d <= c.goal_radius:
        outcome = "SUCCESS"
    elif reached_goal_fast and final_goal_d <= c.goal_radius and final_speed > 0.35:
        outcome = "GOAL_ALIGNMENT_FAILURE"
    elif consecutive_stall_steps >= 30:
        outcome = "SPATIAL_STANDSTILL"
    elif is_oscillating:
        outcome = "OSCILLATION"
    elif step >= max_steps:
        outcome = "TIMEOUT_1000"
    else:
        outcome = "OTHER_FAILURE"

    return EpisodeRecord(
        seed=seed,
        controller_id=controller_id,
        primary_outcome=outcome,
        total_steps=step,
        flight_time_s=round(step * env.dt, 2),
        success=(outcome == "SUCCESS"),
        obstacle_collision=had_obs_collision,
        wall_collision=had_wall_collision,
        safety_envelope_violation=had_envelope_violation,
        min_obstacle_clearance_m=round(min_obs_clear, 4),
        min_wall_clearance_m=round(min_wall_clear, 4),
        final_x=round(float(final_pos[0]), 3),
        final_y=round(float(final_pos[1]), 3),
        final_z=round(float(final_pos[2]), 3),
        forward_progress_m=round(prog_m, 3),
        final_goal_dist_m=round(final_goal_d, 3),
        pillar1_cleared=p1,
        pillar2_cleared=p2,
        pillar3_cleared=p3,
        supervisor_interventions=interventions,
        throttled_steps=throttled_steps,
        mean_throttled_fraction=round(float(np.mean(throttled_fracs)) if throttled_fracs else 0.0, 4),
        direction_changes=direction_changes,
        status_at_500=status_at_500,
        progress_at_500_m=round(progress_at_500_m, 3),
        success_at_500=success_at_500
    )


def compute_wilson_ci(k: int, n: int, confidence: float = 0.95) -> Tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    z = stats.norm.ppf(1.0 - (1.0 - confidence) / 2.0)
    p = k / n
    denom = 1.0 + (z**2) / n
    center = (p + (z**2) / (2.0 * n)) / denom
    margin = (z * math.sqrt((p * (1.0 - p) / n) + ((z**2) / (4.0 * (n**2))))) / denom
    return (max(0.0, float(center - margin)), min(1.0, float(center + margin)))


def compute_metrics(records: List[EpisodeRecord]) -> Dict[str, Any]:
    n = len(records)
    if n == 0:
        return {}
    succ = sum(1 for r in records if r.success)
    succ_500 = sum(1 for r in records if r.success_at_500)
    obs_coll = sum(1 for r in records if r.obstacle_collision)
    wall_coll = sum(1 for r in records if r.wall_collision)
    env_viol = sum(1 for r in records if r.safety_envelope_violation)
    timeouts = sum(1 for r in records if r.primary_outcome == "TIMEOUT_1000")
    oscillations = sum(1 for r in records if r.primary_outcome == "OSCILLATION")
    standstills = sum(1 for r in records if r.primary_outcome == "SPATIAL_STANDSTILL")

    succ_steps = [r.total_steps for r in records if r.success]
    progs = [r.forward_progress_m for r in records]
    obs_clears = [r.min_obstacle_clearance_m for r in records]
    wall_clears = [r.min_wall_clearance_m for r in records]
    intervs = [r.supervisor_interventions for r in records]
    throttled = [r.mean_throttled_fraction for r in records]

    ci_l, ci_h = compute_wilson_ci(succ, n)

    # Check 14 hard negative seeds
    hn_seeds = [200, 203, 204, 206, 207, 210, 213, 215, 216, 217, 218, 220, 221, 224]
    hn_records = [r for r in records if r.seed in hn_seeds]
    hn_succ = sum(1 for r in hn_records if r.success)

    return {
        "n": n,
        "success_count": succ,
        "success_rate_pct": round(succ / n * 100.0, 2),
        "wilson_ci_95": [round(ci_l * 100.0, 2), round(ci_h * 100.0, 2)],
        "success_at_500": succ_500,
        "success_at_500_pct": round(succ_500 / n * 100.0, 2),
        "gain_extended_horizon": succ - succ_500,
        "obstacle_collisions": obs_coll,
        "wall_collisions": wall_coll,
        "safety_envelope_violations": env_viol,
        "timeouts": timeouts,
        "oscillations": oscillations,
        "standstills": standstills,
        "mean_steps_to_success": round(float(np.mean(succ_steps)), 1) if succ_steps else None,
        "median_steps_to_success": round(float(np.median(succ_steps)), 1) if succ_steps else None,
        "mean_progress_m": round(float(np.mean(progs)), 3),
        "overall_min_obs_clear_m": round(float(np.min(obs_clears)), 4),
        "mean_min_obs_clear_m": round(float(np.mean(obs_clears)), 4),
        "overall_min_wall_clear_m": round(float(np.min(wall_clears)), 4),
        "mean_min_wall_clear_m": round(float(np.mean(wall_clears)), 4),
        "mean_interventions": round(float(np.mean(intervs)), 1),
        "mean_throttled_frac": round(float(np.mean(throttled)), 4),
        "hard_negatives_14_success": f"{hn_succ}/{len(hn_records)}" if hn_records else "N/A"
    }


def write_csv_file(path: Path, data: List[Dict[str, Any]]) -> None:
    if not data:
        return
    fieldnames = list(data[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)


def main():
    parser = argparse.ArgumentParser(description="Evaluate Guidance Integration with BC and PPO")
    parser.add_argument("--output-dir", type=str, default="", help="Output directory")
    args = parser.parse_args()

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir) if args.output_dir else Path(f"experiments/guidance_integration/run_{timestamp}")
    out_dir.mkdir(parents=True, exist_ok=True)
    traj_dir = out_dir / "representative_trajectories"
    traj_dir.mkdir(exist_ok=True)

    print("=" * 80)
    print("REACTIVE GUIDANCE INTEGRATION WITH BC AND PPO (PHASES A, B, C, D)")
    print(f"Output Directory: {out_dir}")
    print("=" * 80)

    # Base models
    bc_policy = BehaviorCloningPolicyController()
    ppo_policy = PPOPolicyController()

    # Controller Registry for Benchmark
    controllers: Dict[str, Any] = {
        "reactive_baseline": ReactiveBaselineController(cruise_speed=0.70),
        "raw_bc": bc_policy,
        "guided_bc_phase6a": StatelessClearanceGuidanceController(base_model=bc_policy.model),
        "raw_ppo": ppo_policy,
        "guided_ppo_phase6a": StatelessClearanceGuidanceController(base_model=ppo_policy.model),
        "cand1_bc_early_decel": GenericGuidedController(bc_policy, EarlyDecelArbitrator(), "cand1_bc_early_decel"),
        "cand2_bc_multibeam": GenericGuidedController(bc_policy, MultiBeamWeightArbitrator(), "cand2_bc_multibeam"),
        "cand3_bc_centering": GenericGuidedController(bc_policy, CenteringBiasArbitrator(), "cand3_bc_centering"),
        "cand4_bc_hysteresis": GenericGuidedController(bc_policy, EnhancedHysteresisArbitrator(), "cand4_bc_hysteresis"),
        "cand1_ppo_early_decel": GenericGuidedController(ppo_policy, EarlyDecelArbitrator(), "cand1_ppo_early_decel"),
        "cand_best_bc": GenericGuidedController(bc_policy, BestIntegratedArbitrator(), "cand_best_bc"),
        "cand_best_ppo": GenericGuidedController(ppo_policy, BestIntegratedArbitrator(), "cand_best_ppo"),
    }

    env = CorridorSimEnv(scenario="10m", gui=False)
    env.max_steps = 1000
    env.corridor.max_steps = 1000

    # -------------------------------------------------------------
    # PHASE B: 5 Specific Failure Seeds Diagnostics
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print("PHASE B: EVALUATING SPECIFIC REACTIVE FAILURE SEEDS (99, 686, 10009, 10131, 10667)")
    print("=" * 80)
    diag_seeds = [99, 686, 10009, 10131, 10667]
    diag_rows: List[Dict[str, Any]] = []

    for s in diag_seeds:
        for cname, ctrl in controllers.items():
            rec = run_episode(ctrl, cname, s, env, max_steps=1000)
            d_row = {
                "seed": s,
                "controller": cname,
                "outcome": rec.primary_outcome,
                "steps": rec.total_steps,
                "progress_m": rec.forward_progress_m,
                "min_obs_m": rec.min_obstacle_clearance_m,
                "min_wall_m": rec.min_wall_clearance_m,
                "interventions": rec.supervisor_interventions,
            }
            diag_rows.append(d_row)
            print(f"Seed {s:<6} | {cname:<22} | {rec.primary_outcome:<16} | Steps={rec.total_steps:<5} | MinObs={rec.min_obstacle_clearance_m:.4f}m")

    write_csv_file(out_dir / "failure_cases_diagnostics.csv", diag_rows)

    # -------------------------------------------------------------
    # PHASES A & C: Development Benchmark (200 Paired Seeds)
    # -------------------------------------------------------------
    # Pool 1: 100 legacy seeds (1..86 + 14 Milestone 4 hard negatives [200, 203, ... 224])
    # Pool 2: 100 campaign seeds (10000..10099)
    hn_seeds = [200, 203, 204, 206, 207, 210, 213, 215, 216, 217, 218, 220, 221, 224]
    legacy_seeds = list(range(1, 87)) + hn_seeds
    campaign_seeds = list(range(10000, 10100))
    benchmark_seeds = sorted(list(set(legacy_seeds + campaign_seeds)))
    assert len(benchmark_seeds) == 200

    print("\n" + "=" * 80)
    print(f"PHASES A & C: BENCHMARK SUITE ON {len(benchmark_seeds)} PAIRED SEEDS (1000-STEP HORIZON)")
    print("=" * 80)

    benchmark_records: Dict[str, List[EpisodeRecord]] = {cname: [] for cname in controllers.keys()}
    t_start = time.time()

    for idx, s in enumerate(benchmark_seeds):
        for cname, ctrl in controllers.items():
            rec = run_episode(ctrl, cname, s, env, max_steps=1000)
            benchmark_records[cname].append(rec)

        if (idx + 1) % 50 == 0 or (idx + 1) == len(benchmark_seeds):
            el = time.time() - t_start
            print(f"[{idx + 1:3d}/{len(benchmark_seeds)}] Processed in {el:.1f}s...")

    # Flatten and write episode CSVs
    all_bm_rows = []
    for cname, recs in benchmark_records.items():
        for r in recs:
            all_bm_rows.append(asdict(r))
    write_csv_file(out_dir / "benchmark_episodes_all.csv", all_bm_rows)

    # Compute summary metrics for each controller
    bm_summary = {}
    for cname, recs in benchmark_records.items():
        bm_summary[cname] = compute_metrics(recs)

    # -------------------------------------------------------------
    # PHASE D: Pristine Unseen Confirmation Benchmark (100 Seeds)
    # -------------------------------------------------------------
    # Seeds 12000..12099 (never used before)
    confirmation_seeds = list(range(12000, 12100))
    print("\n" + "=" * 80)
    print(f"PHASE D: PRISTINE CONFIRMATION SUITE ON {len(confirmation_seeds)} UNSEEN SEEDS (12000..12099)")
    print("=" * 80)

    conf_controllers = [
        "reactive_baseline",
        "raw_bc",
        "guided_bc_phase6a",
        "cand_best_bc",
        "raw_ppo",
        "guided_ppo_phase6a",
        "cand_best_ppo",
    ]

    conf_records: Dict[str, List[EpisodeRecord]] = {cname: [] for cname in conf_controllers}

    for idx, s in enumerate(confirmation_seeds):
        for cname in conf_controllers:
            rec = run_episode(controllers[cname], cname, s, env, max_steps=1000)
            conf_records[cname].append(rec)

    all_conf_rows = []
    conf_summary = {}
    for cname in conf_controllers:
        for r in conf_records[cname]:
            all_conf_rows.append(asdict(r))
        conf_summary[cname] = compute_metrics(conf_records[cname])
    write_csv_file(out_dir / "confirmation_episodes_all.csv", all_conf_rows)

    # Save Aggregate JSON
    final_aggregate = {
        "timestamp": timestamp,
        "horizon_override_steps": 1000,
        "phase_b_failure_cases": diag_rows,
        "benchmark_summary_200_seeds": bm_summary,
        "confirmation_summary_100_seeds": conf_summary
    }
    with open(out_dir / "aggregate_metrics.json", "w") as f:
        json.dump(final_aggregate, f, indent=2)

    env.close()
    print(f"\nAll phases completed successfully! Artifacts written to {out_dir}")


if __name__ == "__main__":
    main()

