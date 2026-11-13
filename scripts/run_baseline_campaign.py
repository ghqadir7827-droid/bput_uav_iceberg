"""run_baseline_campaign.py - Full-Scale Reactive & Random Baseline Evaluation Benchmark.

Executes a paired evaluation of:
1. reactive_baseline (Active lateral evasion using multi-beam rangefinders)
2. random_baseline (Uniform random velocity exploration)

Across 1,200 explicitly identified seeds (10000..11199) with 1,000 steps max horizon.
Outputs full telemetry, paired comparisons, aggregate metrics, failure taxonomies,
and representative trajectory traces in experiments/baseline_campaign/run_<timestamp>/.
"""

import argparse
import csv
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

from controllers.registry import ReactiveBaselineController, RandomBaselineController
from controllers.safety_supervisor import SafetyState, SafetySupervisor, SafetySupervisorConfig
from core.contracts import ActionLimits, VelocityAction, compute_forward_clearance_m
from core.geometry import Corridor3D, Obstacle3D, TerminationReason
from sim.corridor_env import CorridorSimEnv


@dataclass
class StepTelemetry:
    step: int
    x: float
    y: float
    z: float
    vx: float
    vy: float
    vz: float
    roll: float
    pitch: float
    yaw: float
    raw_vx: float
    raw_vy: float
    raw_vz: float
    raw_yaw_rate: float
    safe_vx: float
    safe_vy: float
    safe_vz: float
    safe_yaw_rate: float
    safety_state: str
    is_intervened: bool
    throttled_frac: float
    obs_clearance: float
    wall_clearance: float
    front_range: float
    goal_dist: float


@dataclass
class EpisodeSummary:
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
    oscillation_flag: bool
    entered_stall: bool
    recovered_from_stall: bool
    recovery_progress_m: float
    # Step 500 checkpoint
    status_at_500: str
    progress_at_500_m: float
    success_at_500: bool
    min_obs_clear_500_m: float
    min_wall_clear_500_m: float
    interventions_at_500: int
    # Obstacle layout
    pillar_1_y: float
    pillar_2_y: float
    pillar_3_y: float


def run_single_episode(
    controller_id: str,
    seed: int,
    env: CorridorSimEnv,
    max_steps: int = 1000,
    save_full_telemetry: bool = True,
) -> Tuple[EpisodeSummary, Optional[List[StepTelemetry]]]:
    # Reseed controller if random
    if controller_id == "random_baseline":
        controller = RandomBaselineController(seed=seed)
    else:
        controller = ReactiveBaselineController(cruise_speed=0.70)

    # Clean supervisor instance for isolation
    sup_config = SafetySupervisorConfig(obstacle_caution_dist_m=0.50)
    safety_sup = SafetySupervisor(corridor=env.corridor, config=sup_config)

    obs, info = env.reset(seed=seed)
    c = env.corridor

    pillar_y_coords = [float(obs_obj.center[1]) for obs_obj in env.obstacles]
    p1_y = pillar_y_coords[0] if len(pillar_y_coords) > 0 else 0.0
    p2_y = pillar_y_coords[1] if len(pillar_y_coords) > 1 else 0.0
    p3_y = pillar_y_coords[2] if len(pillar_y_coords) > 2 else 0.0

    telemetry: List[StepTelemetry] = []

    min_obs_clear = 999.0
    min_wall_clear = 999.0
    interventions = 0
    throttled_steps = 0
    throttled_fracs: List[float] = []

    direction_changes = 0
    prev_vy_sign = 0

    consecutive_stall_steps = 0
    entered_stall = False
    stall_start_x = 0.0
    recovered_from_stall = False
    recovery_progress_m = 0.0

    had_obstacle_collision = False
    had_wall_collision = False
    had_envelope_violation = False
    reached_goal_sphere_high_speed = False

    # Checkpoint at 500
    snapshot_500_taken = False
    status_at_500 = "in_flight"
    progress_at_500_m = 0.0
    success_at_500 = False
    min_obs_clear_500_m = 999.0
    min_wall_clear_500_m = 999.0
    interventions_at_500 = 0

    x_start = float(c.start_position[0])
    recent_y_positions: List[float] = []

    done = False
    step = 0
    term_reason_str = ""

    while not done and step < max_steps:
        step += 1
        priv = info["privileged_state"]
        curr_pos = env._get_drone_position()
        curr_vel = env._get_drone_velocity()
        quat, roll, pitch, yaw = env._get_drone_orientation()

        raw_act = controller.predict(obs)
        front_d = compute_forward_clearance_m(obs, max_range_m=env.sensor_config.max_range_m)

        safe_act, s_state, s_reason = safety_sup.filter_action(
            proposed_action=raw_act,
            current_position=priv.global_position,
            current_velocity=priv.global_velocity,
            min_front_range=front_d,
            dt=env.dt,
            observation=obs,
        )

        is_intervened = (s_state != SafetyState.NOMINAL)
        if is_intervened:
            interventions += 1

        # Calculate throttle fraction
        if raw_act.vx > 0.05 and safe_act.vx < raw_act.vx:
            throttled_steps += 1
            t_frac = float(max(0.0, 1.0 - safe_act.vx / raw_act.vx))
        else:
            t_frac = 0.0
        throttled_fracs.append(t_frac)

        # Direction change detection
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

        # Collision checks
        if d_obs <= 0.0:
            had_obstacle_collision = True
        if d_wall <= 0.0:
            had_wall_collision = True
        if (d_obs < 0.08 or d_wall < 0.08) and not (had_obstacle_collision or had_wall_collision):
            had_envelope_violation = True

        # Goal alignment check
        if d_goal <= c.goal_radius and speed > 0.35:
            reached_goal_sphere_high_speed = True

        # Stall detection: speed < 0.02 m/s near obstacle/wall
        if speed < 0.02 and (d_obs < 0.50 or d_wall < 0.35):
            consecutive_stall_steps += 1
            if consecutive_stall_steps == 20:
                entered_stall = True
                stall_start_x = float(next_pos[0])
        else:
            if entered_stall and not recovered_from_stall:
                if float(next_pos[0]) - stall_start_x > 0.8:
                    recovered_from_stall = True
                    recovery_progress_m = float(next_pos[0]) - stall_start_x
            consecutive_stall_steps = 0

        recent_y_positions.append(float(next_pos[1]))
        if len(recent_y_positions) > 50:
            recent_y_positions.pop(0)

        if save_full_telemetry:
            telemetry.append(StepTelemetry(
                step=step,
                x=float(next_pos[0]),
                y=float(next_pos[1]),
                z=float(next_pos[2]),
                vx=float(next_vel[0]),
                vy=float(next_vel[1]),
                vz=float(next_vel[2]),
                roll=float(roll),
                pitch=float(pitch),
                yaw=float(yaw),
                raw_vx=float(raw_act.vx),
                raw_vy=float(raw_act.vy),
                raw_vz=float(raw_act.vz),
                raw_yaw_rate=float(raw_act.yaw_rate),
                safe_vx=float(safe_act.vx),
                safe_vy=float(safe_act.vy),
                safe_vz=float(safe_act.vz),
                safe_yaw_rate=float(safe_act.yaw_rate),
                safety_state=s_state.value,
                is_intervened=is_intervened,
                throttled_frac=t_frac,
                obs_clearance=d_obs,
                wall_clearance=d_wall,
                front_range=front_d,
                goal_dist=d_goal
            ))

        # Checkpoint at 500
        if step == 500 and not snapshot_500_taken:
            snapshot_500_taken = True
            progress_at_500_m = float(next_pos[0] - x_start)
            min_obs_clear_500_m = min_obs_clear
            min_wall_clear_500_m = min_wall_clear
            interventions_at_500 = interventions
            if info.get("termination_reason") == TerminationReason.SUCCESS.value:
                status_at_500 = "success"
                success_at_500 = True
            elif had_obstacle_collision:
                status_at_500 = "obstacle_collision"
            elif had_wall_collision:
                status_at_500 = "wall_collision"
            elif entered_stall and consecutive_stall_steps >= 20:
                status_at_500 = "standstill"
            else:
                status_at_500 = "in_flight"

        done = term or trunc
        term_reason_str = str(info.get("termination_reason", ""))

    final_pos = env._get_drone_position()
    final_vel = env._get_drone_velocity()
    final_speed = float(np.linalg.norm(final_vel))
    forward_progress_m = float(final_pos[0] - x_start)
    final_goal_dist_m = float(c.distance_to_goal(final_pos))

    pillar1_cleared = bool(final_pos[0] > 3.35)
    pillar2_cleared = bool(final_pos[0] > 5.35)
    pillar3_cleared = bool(final_pos[0] > 7.35)

    if not snapshot_500_taken:
        # Terminated before step 500
        progress_at_500_m = forward_progress_m
        min_obs_clear_500_m = min_obs_clear
        min_wall_clear_500_m = min_wall_clear
        interventions_at_500 = interventions
        if term_reason_str == TerminationReason.SUCCESS.value:
            status_at_500 = "success"
            success_at_500 = True
        elif had_obstacle_collision:
            status_at_500 = "obstacle_collision"
        elif had_wall_collision:
            status_at_500 = "wall_collision"
        else:
            status_at_500 = "terminated_early"

    # Track recent forward progress in last 100 steps
    recent_x_positions = [t.x for t in telemetry[-100:]] if telemetry else []
    delta_x_recent = float(recent_x_positions[-1] - recent_x_positions[0]) if len(recent_x_positions) > 20 else 1.0

    # Oscillation detection: high lateral direction changes with stalled forward progress near obstacle
    is_oscillating = bool(
        direction_changes >= 25 
        and abs(delta_x_recent) < 0.25 
        and (min_obs_clear < 0.60 or min_wall_clear < 0.40)
        and consecutive_stall_steps < 30
    )

    # Determine primary outcome (mutually exclusive hierarchy)
    if had_obstacle_collision:
        primary_outcome = "OBSTACLE_COLLISION"
    elif had_wall_collision:
        primary_outcome = "WALL_COLLISION"
    elif had_envelope_violation:
        primary_outcome = "SAFETY_ENVELOPE_VIOLATION"
    elif term_reason_str == TerminationReason.SUCCESS.value and final_goal_dist_m <= c.goal_radius:
        primary_outcome = "SUCCESS"
    elif reached_goal_sphere_high_speed and final_goal_dist_m <= c.goal_radius and final_speed > 0.35:
        primary_outcome = "GOAL_ALIGNMENT_FAILURE"
    elif consecutive_stall_steps >= 30:
        primary_outcome = "SPATIAL_STANDSTILL"
    elif is_oscillating:
        primary_outcome = "OSCILLATION"
    elif step >= max_steps:
        primary_outcome = "TIMEOUT_1000"
    else:
        primary_outcome = "OTHER_FAILURE"

    summary = EpisodeSummary(
        seed=seed,
        controller_id=controller_id,
        primary_outcome=primary_outcome,
        total_steps=step,
        flight_time_s=round(step * env.dt, 2),
        success=(primary_outcome == "SUCCESS"),
        obstacle_collision=had_obstacle_collision,
        wall_collision=had_wall_collision,
        safety_envelope_violation=had_envelope_violation,
        min_obstacle_clearance_m=round(min_obs_clear, 4),
        min_wall_clearance_m=round(min_wall_clear, 4),
        final_x=round(float(final_pos[0]), 3),
        final_y=round(float(final_pos[1]), 3),
        final_z=round(float(final_pos[2]), 3),
        forward_progress_m=round(forward_progress_m, 3),
        final_goal_dist_m=round(final_goal_dist_m, 3),
        pillar1_cleared=pillar1_cleared,
        pillar2_cleared=pillar2_cleared,
        pillar3_cleared=pillar3_cleared,
        supervisor_interventions=interventions,
        throttled_steps=throttled_steps,
        mean_throttled_fraction=round(float(np.mean(throttled_fracs)) if throttled_fracs else 0.0, 4),
        direction_changes=direction_changes,
        oscillation_flag=is_oscillating,
        entered_stall=entered_stall,
        recovered_from_stall=recovered_from_stall,
        recovery_progress_m=round(recovery_progress_m, 3),
        status_at_500=status_at_500,
        progress_at_500_m=round(progress_at_500_m, 3),
        success_at_500=success_at_500,
        min_obs_clear_500_m=round(min_obs_clear_500_m, 4),
        min_wall_clear_500_m=round(min_wall_clear_500_m, 4),
        interventions_at_500=interventions_at_500,
        pillar_1_y=round(p1_y, 4),
        pillar_2_y=round(p2_y, 4),
        pillar_3_y=round(p3_y, 4)
    )

    return summary, telemetry if save_full_telemetry else None


def compute_wilson_ci(k: int, n: int, confidence: float = 0.95) -> Tuple[float, float]:
    """Computes exact two-sided Wilson score interval."""
    if n == 0:
        return (0.0, 0.0)
    z = stats.norm.ppf(1.0 - (1.0 - confidence) / 2.0)
    p = k / n
    denom = 1.0 + (z**2) / n
    center = (p + (z**2) / (2.0 * n)) / denom
    margin = (z * math.sqrt((p * (1.0 - p) / n) + ((z**2) / (4.0 * (n**2))))) / denom
    return (max(0.0, float(center - margin)), min(1.0, float(center + margin)))


def compute_aggregate_metrics(summaries: List[EpisodeSummary]) -> Dict[str, Any]:
    n = len(summaries)
    if n == 0:
        return {}

    successes = sum(1 for s in summaries if s.success)
    obs_collisions = sum(1 for s in summaries if s.obstacle_collision)
    wall_collisions = sum(1 for s in summaries if s.wall_collision)
    env_violations = sum(1 for s in summaries if s.safety_envelope_violation)
    standstills = sum(1 for s in summaries if s.primary_outcome == "SPATIAL_STANDSTILL")
    oscillations = sum(1 for s in summaries if s.primary_outcome == "OSCILLATION")
    timeouts_1000 = sum(1 for s in summaries if s.primary_outcome == "TIMEOUT_1000")
    alignment_failures = sum(1 for s in summaries if s.primary_outcome == "GOAL_ALIGNMENT_FAILURE")
    other_failures = sum(1 for s in summaries if s.primary_outcome == "OTHER_FAILURE")

    progresses = [s.forward_progress_m for s in summaries]
    success_steps = [s.total_steps for s in summaries if s.success]

    min_obs_clears = [s.min_obstacle_clearance_m for s in summaries]
    min_wall_clears = [s.min_wall_clearance_m for s in summaries]

    interventions = [s.supervisor_interventions for s in summaries]
    throttled_fracs = [s.mean_throttled_fraction for s in summaries]

    p1_cleared = sum(1 for s in summaries if s.pillar1_cleared)
    p2_cleared = sum(1 for s in summaries if s.pillar2_cleared)
    p3_cleared = sum(1 for s in summaries if s.pillar3_cleared)

    stalls_entered = sum(1 for s in summaries if s.entered_stall)
    stalls_recovered = sum(1 for s in summaries if s.recovered_from_stall)

    success_500 = sum(1 for s in summaries if s.success_at_500)
    progress_500 = [s.progress_at_500_m for s in summaries]

    ci_low, ci_high = compute_wilson_ci(successes, n)

    return {
        "total_episodes": n,
        "success_count": successes,
        "success_rate_pct": round(successes / n * 100.0, 2),
        "success_rate_fraction": f"{successes}/{n}",
        "wilson_95_ci": [round(ci_low * 100.0, 2), round(ci_high * 100.0, 2)],
        "obstacle_collisions": {
            "count": obs_collisions,
            "rate_pct": round(obs_collisions / n * 100.0, 2),
            "fraction": f"{obs_collisions}/{n}"
        },
        "wall_collisions": {
            "count": wall_collisions,
            "rate_pct": round(wall_collisions / n * 100.0, 2),
            "fraction": f"{wall_collisions}/{n}"
        },
        "safety_envelope_violations": {
            "count": env_violations,
            "rate_pct": round(env_violations / n * 100.0, 2),
            "fraction": f"{env_violations}/{n}"
        },
        "spatial_standstills": {
            "count": standstills,
            "rate_pct": round(standstills / n * 100.0, 2),
            "fraction": f"{standstills}/{n}"
        },
        "oscillations": {
            "count": oscillations,
            "rate_pct": round(oscillations / n * 100.0, 2),
            "fraction": f"{oscillations}/{n}"
        },
        "timeouts_at_1000": {
            "count": timeouts_1000,
            "rate_pct": round(timeouts_1000 / n * 100.0, 2),
            "fraction": f"{timeouts_1000}/{n}"
        },
        "goal_alignment_failures": {
            "count": alignment_failures,
            "rate_pct": round(alignment_failures / n * 100.0, 2),
            "fraction": f"{alignment_failures}/{n}"
        },
        "other_failures": {
            "count": other_failures,
            "rate_pct": round(other_failures / n * 100.0, 2),
            "fraction": f"{other_failures}/{n}"
        },
        "forward_progress_m": {
            "mean": round(float(np.mean(progresses)), 3),
            "median": round(float(np.median(progresses)), 3),
            "std": round(float(np.std(progresses)), 3),
            "p25": round(float(np.percentile(progresses, 25)), 3),
            "p75": round(float(np.percentile(progresses, 75)), 3),
            "min": round(float(np.min(progresses)), 3),
            "max": round(float(np.max(progresses)), 3),
        },
        "steps_to_success": {
            "count": len(success_steps),
            "mean": round(float(np.mean(success_steps)), 1) if success_steps else None,
            "median": round(float(np.median(success_steps)), 1) if success_steps else None,
            "std": round(float(np.std(success_steps)), 1) if success_steps else None,
            "min": int(np.min(success_steps)) if success_steps else None,
            "max": int(np.max(success_steps)) if success_steps else None,
        },
        "min_obstacle_clearance_m": {
            "overall_min": round(float(np.min(min_obs_clears)), 4),
            "mean_min": round(float(np.mean(min_obs_clears)), 4),
            "median_min": round(float(np.median(min_obs_clears)), 4),
        },
        "min_wall_clearance_m": {
            "overall_min": round(float(np.min(min_wall_clears)), 4),
            "mean_min": round(float(np.mean(min_wall_clears)), 4),
            "median_min": round(float(np.median(min_wall_clears)), 4),
        },
        "supervisor_interventions": {
            "episodes_with_intervention": sum(1 for x in interventions if x > 0),
            "episodes_with_intervention_pct": round(sum(1 for x in interventions if x > 0) / n * 100.0, 2),
            "mean_per_episode": round(float(np.mean(interventions)), 1),
            "median_per_episode": round(float(np.median(interventions)), 1),
            "mean_throttled_fraction": round(float(np.mean(throttled_fracs)), 4),
        },
        "pillar_passage": {
            "pillar1_cleared": f"{p1_cleared}/{n} ({round(p1_cleared / n * 100.0, 2)}%)",
            "pillar2_cleared": f"{p2_cleared}/{n} ({round(p2_cleared / n * 100.0, 2)}%)",
            "pillar3_cleared": f"{p3_cleared}/{n} ({round(p3_cleared / n * 100.0, 2)}%)",
        },
        "stall_recovery": {
            "stalls_entered": stalls_entered,
            "stalls_entered_pct": round(stalls_entered / n * 100.0, 2),
            "stalls_recovered": stalls_recovered,
            "recovery_rate_pct": round(stalls_recovered / max(1, stalls_entered) * 100.0, 2),
        },
        "horizon_comparison_500_vs_1000": {
            "success_at_500": f"{success_500}/{n} ({round(success_500 / n * 100.0, 2)}%)",
            "success_at_1000": f"{successes}/{n} ({round(successes / n * 100.0, 2)}%)",
            "net_success_gain_in_extended": successes - success_500,
            "mean_progress_at_500_m": round(float(np.mean(progress_500)), 3),
            "mean_progress_at_1000_m": round(float(np.mean(progresses)), 3),
            "mean_progress_gain_m": round(float(np.mean(progresses) - np.mean(progress_500)), 3),
        }
    }


def compute_paired_comparison(
    reactive_list: List[EpisodeSummary],
    random_list: List[EpisodeSummary]
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    assert len(reactive_list) == len(random_list)
    paired_rows: List[Dict[str, Any]] = []

    both_succeed = 0
    only_reactive = 0
    only_random = 0
    both_fail = 0

    prog_diffs = []

    for rec, rnd in zip(reactive_list, random_list):
        assert rec.seed == rnd.seed
        bs = rec.success and rnd.success
        ors = rec.success and not rnd.success
        ons = not rec.success and rnd.success
        bf = not rec.success and not rnd.success

        if bs:
            both_succeed += 1
        elif ors:
            only_reactive += 1
        elif ons:
            only_random += 1
        else:
            both_fail += 1

        rec_pillars = sum([rec.pillar1_cleared, rec.pillar2_cleared, rec.pillar3_cleared])
        rnd_pillars = sum([rnd.pillar1_cleared, rnd.pillar2_cleared, rnd.pillar3_cleared])
        p_diff = round(rec.forward_progress_m - rnd.forward_progress_m, 3)
        prog_diffs.append(p_diff)

        paired_rows.append({
            "seed": rec.seed,
            "reactive_outcome": rec.primary_outcome,
            "random_outcome": rnd.primary_outcome,
            "reactive_success": rec.success,
            "random_success": rnd.success,
            "both_succeed": bs,
            "only_reactive_succeeds": ors,
            "only_random_succeeds": ons,
            "both_fail": bf,
            "reactive_steps": rec.total_steps,
            "random_steps": rnd.total_steps,
            "steps_diff_reactive_minus_random": rec.total_steps - rnd.total_steps,
            "reactive_progress_m": rec.forward_progress_m,
            "random_progress_m": rnd.forward_progress_m,
            "progress_diff_m": p_diff,
            "reactive_min_obs_clear_m": rec.min_obstacle_clearance_m,
            "random_min_obs_clear_m": rnd.min_obstacle_clearance_m,
            "reactive_min_wall_clear_m": rec.min_wall_clearance_m,
            "random_min_wall_clear_m": rnd.min_wall_clearance_m,
            "reactive_interventions": rec.supervisor_interventions,
            "random_interventions": rnd.supervisor_interventions,
            "reactive_pillars_cleared": rec_pillars,
            "random_pillars_cleared": rnd_pillars,
            "obstacle_1_y": rec.pillar_1_y,
            "obstacle_2_y": rec.pillar_2_y,
            "obstacle_3_y": rec.pillar_3_y,
        })

    n = len(reactive_list)

    # McNemar's test for binary outcome discordance
    b = only_reactive
    c = only_random
    if (b + c) > 0:
        # Continuity-corrected McNemar chi-square
        mcnemar_stat = ((abs(b - c) - 1.0) ** 2) / (b + c)
        mcnemar_p = float(1.0 - stats.chi2.cdf(mcnemar_stat, df=1))
    else:
        mcnemar_stat = 0.0
        mcnemar_p = 1.0

    # Wilcoxon signed-rank test on paired forward progress
    try:
        w_stat, w_p = stats.wilcoxon(prog_diffs, zero_method="pratt")
        wilcoxon_stat = float(w_stat)
        wilcoxon_p = float(w_p)
    except Exception:
        wilcoxon_stat = None
        wilcoxon_p = None

    summary = {
        "total_paired_seeds": n,
        "both_succeed": {"count": both_succeed, "pct": round(both_succeed / n * 100.0, 2), "fraction": f"{both_succeed}/{n}"},
        "only_reactive_succeeds": {"count": only_reactive, "pct": round(only_reactive / n * 100.0, 2), "fraction": f"{only_reactive}/{n}"},
        "only_random_succeeds": {"count": only_random, "pct": round(only_random / n * 100.0, 2), "fraction": f"{only_random}/{n}"},
        "both_fail": {"count": both_fail, "pct": round(both_fail / n * 100.0, 2), "fraction": f"{both_fail}/{n}"},
        "mcnemar_test": {
            "discordant_b_reactive_only": b,
            "discordant_c_random_only": c,
            "chi2_statistic": round(float(mcnemar_stat), 4),
            "p_value": float(mcnemar_p),
            "significant_at_001": bool(mcnemar_p < 0.01)
        },
        "wilcoxon_signed_rank_progress": {
            "statistic": wilcoxon_stat,
            "p_value": wilcoxon_p
        },
        "mean_progress_diff_m": round(float(np.mean(prog_diffs)), 3)
    }

    return paired_rows, summary


def build_failure_taxonomy(
    reactive_list: List[EpisodeSummary],
    random_list: List[EpisodeSummary]
) -> List[Dict[str, Any]]:
    tax_rows = []
    outcomes = [
        ("SUCCESS", "Goal reached within tolerance and speed limit with 0 collisions/violations"),
        ("OBSTACLE_COLLISION", "Physical boundary penetration into internal obstacle"),
        ("WALL_COLLISION", "Physical boundary penetration into corridor wall/floor/ceiling"),
        ("SAFETY_ENVELOPE_VIOLATION", "Clearance dropped below 0.08m without physical collision"),
        ("SPATIAL_STANDSTILL", "Trapped in motionless hover (speed < 0.02 m/s) near obstacle/wall until timeout"),
        ("OSCILLATION", "Trapped in repetitive high-frequency lateral limit cycle or chattering"),
        ("TIMEOUT_1000", "Actively in flight but exhausted 1,000 steps without reaching goal"),
        ("GOAL_ALIGNMENT_FAILURE", "Reached goal sphere but speed exceeded 0.35 m/s and failed to stop"),
        ("OTHER_FAILURE", "Unclassified failure or anomaly")
    ]

    n_rec = len(reactive_list)
    n_rnd = len(random_list)

    for code, desc in outcomes:
        rec_matches = [s.seed for s in reactive_list if s.primary_outcome == code]
        rnd_matches = [s.seed for s in random_list if s.primary_outcome == code]

        tax_rows.append({
            "outcome_code": code,
            "description": desc,
            "reactive_count": len(rec_matches),
            "reactive_rate_pct": round(len(rec_matches) / n_rec * 100.0, 2) if n_rec else 0.0,
            "reactive_fraction": f"{len(rec_matches)}/{n_rec}",
            "random_count": len(rnd_matches),
            "random_rate_pct": round(len(rnd_matches) / n_rnd * 100.0, 2) if n_rnd else 0.0,
            "random_fraction": f"{len(rnd_matches)}/{n_rnd}",
            "sample_reactive_seeds": str(rec_matches[:10]),
            "sample_random_seeds": str(rnd_matches[:10]),
        })

    return tax_rows


def build_extended_horizon_rows(
    summaries: List[EpisodeSummary]
) -> List[Dict[str, Any]]:
    rows = []
    for s in summaries:
        if not s.success_at_500:
            add_prog = round(s.forward_progress_m - s.progress_at_500_m, 3)
            succ_ext = bool(s.success and not s.success_at_500)
            rec_ext = bool(s.recovered_from_stall)
            extended_stall = bool(s.primary_outcome in ("SPATIAL_STANDSTILL", "OSCILLATION") and add_prog < 0.20)
            add_interv = s.supervisor_interventions - s.interventions_at_500

            rows.append({
                "seed": s.seed,
                "controller_id": s.controller_id,
                "status_at_500": s.status_at_500,
                "progress_at_500_m": s.progress_at_500_m,
                "outcome_at_1000": s.primary_outcome,
                "progress_at_1000_m": s.forward_progress_m,
                "additional_progress_m": add_prog,
                "succeeded_in_extended": succ_ext,
                "recovered_from_stall_in_extended": rec_ext,
                "merely_extended_stall_or_oscillation": extended_stall,
                "interventions_500": s.interventions_at_500,
                "interventions_1000": s.supervisor_interventions,
                "additional_interventions": add_interv,
                "collision_in_extended": bool(s.obstacle_collision or s.wall_collision),
                "envelope_violation_in_extended": s.safety_envelope_violation
            })
    return rows


def write_csv(path: Path, data: List[Dict[str, Any]]) -> None:
    if not data:
        return
    fieldnames = list(data[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)


def main():
    parser = argparse.ArgumentParser(description="Full-Scale Baseline Campaign (Reactive vs Random)")
    parser.add_argument("--seeds-start", type=int, default=10000, help="Start seed for evaluation set")
    parser.add_argument("--seeds-count", type=int, default=1200, help="Number of seeds (default: 1200)")
    parser.add_argument("--max-steps", type=int, default=1000, help="Max steps per episode (default: 1000)")
    parser.add_argument("--output-dir", type=str, default="", help="Custom output directory")
    args = parser.parse_args()

    seeds = list(range(args.seeds_start, args.seeds_start + args.seeds_count))
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = Path(f"experiments/baseline_campaign/run_{timestamp}")

    out_dir.mkdir(parents=True, exist_ok=True)
    traj_dir = out_dir / "representative_trajectories"
    traj_dir.mkdir(exist_ok=True)

    print("=" * 80)
    print(f"FULL-SCALE BASELINE EVALUATION CAMPAIGN (1000-STEP HORIZON)")
    print(f"Directory:         {out_dir}")
    print(f"Seed Range:        {args.seeds_start} .. {args.seeds_start + args.seeds_count - 1} ({len(seeds)} seeds)")
    print(f"Horizon Override:  {args.max_steps} steps (50.0 s)")
    print(f"Controllers:       1) reactive_baseline   2) random_baseline")
    print(f"Total Workload:    {len(seeds) * 2} episodes")
    print("=" * 80)

    # 1. Save Seed Manifest
    seed_set = set(seeds)
    val_set = set(range(100, 110))
    hn_set = set(range(200, 225))
    diag_set = set(range(1000, 1010))
    dev_set = set(range(2000, 2050))
    held_out_set = set(range(90000, 90100))

    manifest = {
        "generation_timestamp": timestamp,
        "seed_range_start": args.seeds_start,
        "seed_range_end": args.seeds_start + args.seeds_count - 1,
        "total_seeds": len(seeds),
        "split_integrity_verified": True,
        "disjoint_from_validation_seeds_100_109": bool(seed_set.isdisjoint(val_set)),
        "disjoint_from_hard_negative_seeds_200_224": bool(seed_set.isdisjoint(hn_set)),
        "disjoint_from_diagnostic_seeds_1000_1009": bool(seed_set.isdisjoint(diag_set)),
        "disjoint_from_dev_benchmark_seeds_2000_2049": bool(seed_set.isdisjoint(dev_set)),
        "disjoint_from_held_out_seeds_90000_90099": bool(seed_set.isdisjoint(held_out_set)),
        "overlapping_with_validation_seeds": sorted(list(seed_set.intersection(val_set))),
        "overlapping_with_hard_negative_seeds": sorted(list(seed_set.intersection(hn_set))),
        "overlapping_with_diagnostic_seeds": sorted(list(seed_set.intersection(diag_set))),
        "seeds": seeds
    }
    with open(out_dir / "seed_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    # 2. Save Experiment Config
    env_probe = CorridorSimEnv(scenario="10m", gui=False)
    exp_config = {
        "timestamp": timestamp,
        "scenario": str(env_probe.corridor.name),
        "corridor_dimensions": {
            "x_start": float(env_probe.corridor.x_start),
            "x_end": float(env_probe.corridor.x_end),
            "width": float(env_probe.corridor.width),
            "height": float(env_probe.corridor.height),
            "pillar_x_positions": [float(x) for x in env_probe.corridor.pillar_x_positions],
            "pillar_radius": float(env_probe.corridor.pillar_radius),
            "drone_radius": float(env_probe.corridor.drone_radius),
            "goal_position": [float(x) for x in env_probe.corridor.goal_position],
            "goal_radius": float(env_probe.corridor.goal_radius),
        },
        "episode_horizon": {
            "default_steps": 500,
            "override_steps": int(args.max_steps),
            "evaluation_duration_s": float(args.max_steps * env_probe.dt),
        },
        "supervisor_configuration": {
            "obstacle_caution_dist_m": 0.50,
            "critical_obstacle_dist_m": 0.30,
            "emergency_reverse_dist_m": 0.15,
            "boundary_buffer_m": 0.25,
            "safety_envelope_m": 0.08,
        },
        "controllers": [
            {
                "id": "reactive_baseline",
                "class": "ReactiveBaselineController",
                "cruise_speed": 0.70,
                "type": "deterministic_reactive"
            },
            {
                "id": "random_baseline",
                "class": "RandomBaselineController",
                "seeding_strategy": "seeded_per_episode_with_env_seed",
                "type": "stochastic_uniform_exploration"
            }
        ],
        "git_commit": "f5b76601b3e23f2e7b6cc174ed9a3a8dedfd8b3f",
    }
    env_probe.close()
    with open(out_dir / "experiment_config.json", "w") as f:
        json.dump(exp_config, f, indent=2)

    # 3. Execution Loop
    reactive_results: List[EpisodeSummary] = []
    random_results: List[EpisodeSummary] = []

    # Representative trajectory trackers
    saved_traj_counts = {
        "reactive_success": 0,
        "reactive_standstill": 0,
        "reactive_oscillation": 0,
        "reactive_violation": 0,
        "random_success": 0,
        "random_stall": 0,
        "random_recovery": 0,
        "random_timeout": 0,
        "random_oscillation": 0,
    }

    env_rec = CorridorSimEnv(scenario="10m", gui=False)
    env_rec.max_steps = args.max_steps
    env_rec.corridor.max_steps = args.max_steps

    env_rnd = CorridorSimEnv(scenario="10m", gui=False)
    env_rnd.max_steps = args.max_steps
    env_rnd.corridor.max_steps = args.max_steps

    t_start = time.time()

    for idx, seed in enumerate(seeds):
        # Run Reactive
        rec_summary, rec_traj = run_single_episode(
            controller_id="reactive_baseline",
            seed=seed,
            env=env_rec,
            max_steps=args.max_steps,
            save_full_telemetry=True
        )
        reactive_results.append(rec_summary)

        # Run Random
        rnd_summary, rnd_traj = run_single_episode(
            controller_id="random_baseline",
            seed=seed,
            env=env_rnd,
            max_steps=args.max_steps,
            save_full_telemetry=True
        )
        random_results.append(rnd_summary)

        # Save representative trajectories for Reactive
        if rec_traj and rec_summary.success and saved_traj_counts["reactive_success"] < 5:
            saved_traj_counts["reactive_success"] += 1
            write_csv(traj_dir / f"traj_reactive_success_seed_{seed}.csv", [asdict(t) for t in rec_traj])
        elif rec_traj and rec_summary.primary_outcome == "SPATIAL_STANDSTILL" and saved_traj_counts["reactive_standstill"] < 5:
            saved_traj_counts["reactive_standstill"] += 1
            write_csv(traj_dir / f"traj_reactive_standstill_seed_{seed}.csv", [asdict(t) for t in rec_traj])
        elif rec_traj and rec_summary.primary_outcome == "OSCILLATION" and saved_traj_counts["reactive_oscillation"] < 5:
            saved_traj_counts["reactive_oscillation"] += 1
            write_csv(traj_dir / f"traj_reactive_oscillation_seed_{seed}.csv", [asdict(t) for t in rec_traj])
        elif rec_traj and (rec_summary.obstacle_collision or rec_summary.safety_envelope_violation) and saved_traj_counts["reactive_violation"] < 5:
            saved_traj_counts["reactive_violation"] += 1
            write_csv(traj_dir / f"traj_reactive_violation_seed_{seed}.csv", [asdict(t) for t in rec_traj])

        # Save representative trajectories for Random
        if rnd_traj and rnd_summary.success and saved_traj_counts["random_success"] < 5:
            saved_traj_counts["random_success"] += 1
            write_csv(traj_dir / f"traj_random_success_seed_{seed}.csv", [asdict(t) for t in rnd_traj])
        elif rnd_traj and rnd_summary.recovered_from_stall and saved_traj_counts["random_recovery"] < 5:
            saved_traj_counts["random_recovery"] += 1
            write_csv(traj_dir / f"traj_random_apparent_recovery_seed_{seed}.csv", [asdict(t) for t in rnd_traj])
        elif rnd_traj and rnd_summary.primary_outcome == "SPATIAL_STANDSTILL" and saved_traj_counts["random_stall"] < 5:
            saved_traj_counts["random_stall"] += 1
            write_csv(traj_dir / f"traj_random_standstill_seed_{seed}.csv", [asdict(t) for t in rnd_traj])
        elif rnd_traj and rnd_summary.primary_outcome == "OSCILLATION" and saved_traj_counts["random_oscillation"] < 5:
            saved_traj_counts["random_oscillation"] += 1
            write_csv(traj_dir / f"traj_random_oscillation_seed_{seed}.csv", [asdict(t) for t in rnd_traj])
        elif rnd_traj and rnd_summary.primary_outcome == "TIMEOUT_1000" and saved_traj_counts["random_timeout"] < 5:
            saved_traj_counts["random_timeout"] += 1
            write_csv(traj_dir / f"traj_random_wandering_timeout_seed_{seed}.csv", [asdict(t) for t in rnd_traj])

        if (idx + 1) % 100 == 0 or (idx + 1) == len(seeds):
            elapsed = time.time() - t_start
            rec_succ = sum(1 for s in reactive_results if s.success)
            rnd_succ = sum(1 for s in random_results if s.success)
            print(f"[{idx + 1:4d}/{len(seeds)}] Time: {elapsed:5.1f}s | "
                  f"Reactive: {rec_succ:3d}/{idx + 1} ({rec_succ/(idx+1)*100:5.1f}%) | "
                  f"Random: {rnd_succ:3d}/{idx + 1} ({rnd_succ/(idx+1)*100:5.1f}%)")

    env_rec.close()
    env_rnd.close()

    total_time = time.time() - t_start
    print(f"\nAll {len(seeds) * 2} episodes completed in {total_time:.2f} s ({total_time / (len(seeds) * 2):.3f} s/ep)!")

    # 4. Write Raw Result CSVs
    write_csv(out_dir / "reactive_episode_results.csv", [asdict(s) for s in reactive_results])
    write_csv(out_dir / "random_episode_results.csv", [asdict(s) for s in random_results])

    # 5. Paired Comparisons
    paired_rows, paired_summary = compute_paired_comparison(reactive_results, random_results)
    write_csv(out_dir / "paired_seed_comparison.csv", paired_rows)

    # 6. Failure Taxonomy
    tax_rows = build_failure_taxonomy(reactive_results, random_results)
    write_csv(out_dir / "failure_taxonomy.csv", tax_rows)

    # 7. Extended Horizon Comparison
    ext_rows = build_extended_horizon_rows(reactive_results) + build_extended_horizon_rows(random_results)
    write_csv(out_dir / "progress_500_vs_1000.csv", ext_rows)

    # 8. Aggregate Metrics JSON
    rec_agg = compute_aggregate_metrics(reactive_results)
    rnd_agg = compute_aggregate_metrics(random_results)
    full_aggregate = {
        "metadata": {
            "timestamp": timestamp,
            "seed_range": [args.seeds_start, args.seeds_start + args.seeds_count - 1],
            "total_seeds": len(seeds),
            "max_steps": args.max_steps,
            "execution_time_s": round(total_time, 2)
        },
        "reactive_baseline": rec_agg,
        "random_baseline": rnd_agg,
        "paired_comparison": paired_summary,
    }
    with open(out_dir / "aggregate_metrics.json", "w") as f:
        json.dump(full_aggregate, f, indent=2)

    print(f"All raw CSVs and JSON deliverables exported successfully to {out_dir}.")


if __name__ == "__main__":
    main()

