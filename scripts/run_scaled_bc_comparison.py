"""run_scaled_bc_comparison.py - Head-to-Head Comparative Benchmark of Scaled BC.

Evaluates 5 controllers across Difficulty Levels 1, 2, 3, and 4 under identical paired seeds:
1. reactive_baseline (scaled 65cm parameters)
2. original_raw_bc (models/checkpoints/bc_policy_best.pt)
3. original_guided_bc (bc_policy_best.pt + Phase 6A clearance guidance)
4. scaled_raw_bc (models/checkpoints/bc_policy_scaled_v1.pt)
5. scaled_guided_bc (bc_policy_scaled_v1.pt + Phase 6A clearance guidance)

Evaluation Conditions:
- Level 1: Basic (N=50 seeds: 20100..20149)
- Level 2: Standard (N=100 seeds: 20000..20099)
- Level 3: Challenging (N=50 seeds: 20200..20249)
- Level 4: Feasibility & Boundary (N=50 seeds: 20300..20349, 25 feasible, 25 blocked)
- Horizon: max 1000 steps (20 Hz, 50.0s)
- Safety Supervisor active with standard physical parameters
- Reports Success Rate with 95% Wilson CI, Collision Rate, Envelope Violations, Min Clearances,
  Steps, Completion Times, Velocities, and Supervisor Interventions.
"""

import argparse
import csv
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from controllers.bc_policy import BehaviorCloningPolicyController
from controllers.registry import ReactiveBaselineController
from controllers.safety_supervisor import SafetySupervisor
from core.contracts import ActionLimits, VelocityAction, compute_forward_clearance_m
from core.geometry import Corridor3D, TerminationReason
from core.guidance import GuidanceConfig, StatelessClearanceGuidanceArbitrator
from core.physical_dimensions import (
    DifficultyLevel,
    PhysicalDroneModel,
    RouteFeasibilityChecker,
    ScaledEnvironmentFactory,
)
from sim.corridor_env import CorridorSimEnv


class GenericGuidedController:
    """Wraps a base policy with a guidance arbitrator."""
    def __init__(self, base_controller, arbitrator, name: str):
        self.base_controller = base_controller
        self.arbitrator = arbitrator
        self.name = name

    def reset(self):
        if hasattr(self.base_controller, "reset"):
            self.base_controller.reset()
        if hasattr(self.arbitrator, "reset"):
            self.arbitrator.reset()

    def predict(self, obs: np.ndarray) -> VelocityAction:
        raw_act = self.base_controller.predict(obs)
        guided_act, _, _, _ = self.arbitrator.arbitrate(raw_act, obs)
        return guided_act


@dataclass
class ScaledBCEpisodeResult:
    level: int
    level_name: str
    seed: int
    controller_id: str
    is_feasible: bool
    status: str
    success: bool
    safe_rejection: bool
    obstacle_collision: bool
    wall_collision: bool
    collision_episode: bool
    safety_envelope_violation: bool
    total_steps: int
    flight_time_s: float
    flight_distance_m: float
    mean_speed_mps: float
    min_obstacle_clearance_m: float
    min_wall_clearance_m: float
    overall_min_clearance_m: float
    safety_interventions: int
    terminal_dist_to_goal_m: float
    terminal_speed_mps: float


def compute_wilson_ci(k: int, n: int, confidence: float = 0.95) -> Tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    z = 1.959964
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    margin = (z / denom) * np.sqrt(p * (1.0 - p) / n + (z * z) / (4.0 * n * n))
    return float(max(0.0, center - margin) * 100.0), float(min(1.0, center + margin) * 100.0)


def run_episode(
    env: CorridorSimEnv,
    controller: Any,
    supervisor: SafetySupervisor,
    checker: RouteFeasibilityChecker,
    level: int,
    level_name: str,
    seed: int,
    controller_id: str,
    max_steps: int = 1000,
    envelope_threshold_m: float = 0.15,
) -> ScaledBCEpisodeResult:
    if hasattr(controller, "reset"):
        controller.reset()

    obs, _ = env.reset(seed=seed)
    corridor = env.corridor

    feas_res = checker.evaluate_feasibility(corridor, env.obstacles)

    total_steps = 0
    min_obs_clearance = float("inf")
    min_wall_clearance = float("inf")
    envelope_violation_count = 0
    safety_interventions = 0
    outcome = "TIMEOUT"
    success = False
    obstacle_collision = False
    wall_collision = False
    safe_rejection = False
    flight_distance = 0.0
    prev_pos = env._get_drone_position()

    dt = env.dt

    for step in range(max_steps):
        total_steps += 1
        pos = env._get_drone_position()
        vel = env._get_drone_velocity()

        flight_distance += float(np.linalg.norm(pos - prev_pos))
        prev_pos = pos

        d_obs = corridor.obstacle_clearance(pos, env.obstacles)
        d_wall = corridor.boundary_distance(pos)
        min_obs_clearance = min(min_obs_clearance, d_obs)
        min_wall_clearance = min(min_wall_clearance, d_wall)

        if d_obs < envelope_threshold_m or d_wall < envelope_threshold_m:
            envelope_violation_count += 1

        term_reason, _ = corridor.check_termination(pos, vel, env.obstacles, step_count=step, max_steps=max_steps)
        if term_reason == TerminationReason.SUCCESS:
            success = True
            outcome = "SUCCESS"
            break
        elif term_reason == TerminationReason.CRASH_OBSTACLE:
            obstacle_collision = True
            outcome = "CRASH_OBSTACLE"
            break
        elif term_reason == TerminationReason.CRASH_BOUNDARY:
            wall_collision = True
            outcome = "CRASH_BOUNDARY"
            break

        if not feas_res.is_feasible:
            if pos[0] >= 5.0 and np.linalg.norm(vel) < 0.06 and d_obs > envelope_threshold_m:
                if step > 200:
                    safe_rejection = True
                    outcome = "SAFE_REJECT"
                    break

        raw_act = controller.predict(obs)

        min_front = compute_forward_clearance_m(obs, max_range_m=env.sensor_config.max_range_m)
        safe_act, state, _ = supervisor.filter_action(
            proposed_action=raw_act,
            current_position=pos,
            current_velocity=vel,
            min_front_range=min_front,
            dt=dt,
            is_localization_healthy=True,
            telemetry_age_s=0.0,
            observation=obs,
        )

        if state.value != "nominal":
            safety_interventions += 1

        norm_act = safe_act.to_normalized(env.action_limits)
        obs, _, terminated, truncated, info = env.step(norm_act)

        if terminated or truncated:
            term_val = info.get("termination_reason", "")
            if term_val == "success":
                success = True
                outcome = "SUCCESS"
            elif term_val == "crash_obstacle":
                obstacle_collision = True
                outcome = "CRASH_OBSTACLE"
            elif term_val == "crash_boundary":
                wall_collision = True
                outcome = "CRASH_BOUNDARY"
            elif truncated:
                outcome = "TIMEOUT"
            break

    if not feas_res.is_feasible and not obstacle_collision and not wall_collision and min_obs_clearance >= 0.10:
        safe_rejection = True
        outcome = "SAFE_REJECT"

    final_pos = env._get_drone_position()
    final_vel = env._get_drone_velocity()
    term_dist_goal = float(np.linalg.norm(final_pos - corridor.goal_position))
    term_speed = float(np.linalg.norm(final_vel))
    flight_time_s = float(total_steps * dt)
    mean_speed = float(flight_distance / max(1e-3, flight_time_s))

    has_collision = obstacle_collision or wall_collision
    overall_min = min(min_obs_clearance, min_wall_clearance)

    return ScaledBCEpisodeResult(
        level=level,
        level_name=level_name,
        seed=seed,
        controller_id=controller_id,
        is_feasible=feas_res.is_feasible,
        status=outcome,
        success=success,
        safe_rejection=safe_rejection,
        obstacle_collision=obstacle_collision,
        wall_collision=wall_collision,
        collision_episode=has_collision,
        safety_envelope_violation=(envelope_violation_count > 0),
        total_steps=total_steps,
        flight_time_s=flight_time_s,
        flight_distance_m=round(flight_distance, 3),
        mean_speed_mps=round(mean_speed, 3),
        min_obstacle_clearance_m=float(min_obs_clearance),
        min_wall_clearance_m=float(min_wall_clearance),
        overall_min_clearance_m=float(overall_min),
        safety_interventions=safety_interventions,
        terminal_dist_to_goal_m=round(term_dist_goal, 3),
        terminal_speed_mps=round(term_speed, 3),
    )


def evaluate_scaled_bc_campaign(
    output_dir: str = "experiments/scaled_campaign/bc_comparison",
) -> Dict[str, Any]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    checker = RouteFeasibilityChecker(PhysicalDroneModel())
    drone_radius = 0.325
    guidance_cfg = GuidanceConfig.for_65cm_uav(drone_radius)
    arbitrator_bc = StatelessClearanceGuidanceArbitrator(config=guidance_cfg)

    # Instantiate the 5 controllers
    reactive_ctrl = ReactiveBaselineController(cruise_speed=0.60, drone_radius=drone_radius)
    base_bc_orig = BehaviorCloningPolicyController("models/checkpoints/bc_policy_best.pt")
    guided_bc_orig = GenericGuidedController(base_bc_orig, arbitrator_bc, "original_guided_bc")
    base_bc_scaled = BehaviorCloningPolicyController("models/checkpoints/bc_policy_scaled_v1.pt")
    guided_bc_scaled = GenericGuidedController(base_bc_scaled, arbitrator_bc, "scaled_guided_bc")

    controllers = {
        "reactive_baseline": reactive_ctrl,
        "original_raw_bc": base_bc_orig,
        "original_guided_bc": guided_bc_orig,
        "scaled_raw_bc": base_bc_scaled,
        "scaled_guided_bc": guided_bc_scaled,
    }

    level_suites = [
        (1, "Level 1 (Basic)", DifficultyLevel.LEVEL_1_BASIC, list(range(20100, 20150))),
        (2, "Level 2 (Standard)", DifficultyLevel.LEVEL_2_STANDARD, list(range(20000, 20100))),
        (3, "Level 3 (Challenging)", DifficultyLevel.LEVEL_3_CHALLENGING, list(range(20200, 20250))),
        (4, "Level 4 (Boundary & Feasibility)", DifficultyLevel.LEVEL_4_FEASIBILITY_BOUNDARY, list(range(20300, 20350))),
    ]

    all_results: List[ScaledBCEpisodeResult] = []
    t_start = time.time()
    total_episodes_planned = sum(len(seeds) * len(controllers) for _, _, _, seeds in level_suites)
    print("=" * 80)
    print(">>> RUNNING SCALED BEHAVIOR CLONING COMPARATIVE BENCHMARK <<<")
    print(f"Total Runs Planned: {total_episodes_planned} episodes across 4 difficulty tiers")
    print(f"Output Directory:   {out_dir}")
    print("=" * 80)

    for lvl_idx, lvl_name, lvl_enum, seeds in level_suites:
        print(f"\n--- Initializing {lvl_name} (N={len(seeds)} seeds) ---")
        corridor_cfg = ScaledEnvironmentFactory.get_environment_config(lvl_enum)
        env = CorridorSimEnv(scenario=corridor_cfg, seed=seeds[0])
        supervisor = SafetySupervisor(env.corridor)

        for ctrl_id, ctrl in controllers.items():
            ctrl_start = time.time()
            ctrl_success = 0
            ctrl_collisions = 0
            ctrl_envelopes = 0

            for s in seeds:
                res = run_episode(
                    env=env,
                    controller=ctrl,
                    supervisor=supervisor,
                    checker=checker,
                    level=lvl_idx,
                    level_name=lvl_name,
                    seed=s,
                    controller_id=ctrl_id,
                    max_steps=1000,
                    envelope_threshold_m=0.15,
                )
                all_results.append(res)
                if res.success:
                    ctrl_success += 1
                if res.collision_episode:
                    ctrl_collisions += 1
                if res.safety_envelope_violation:
                    ctrl_envelopes += 1

            elapsed = time.time() - ctrl_start
            print(f"  {ctrl_id:<20} | Success: {ctrl_success:>3}/{len(seeds)} ({ctrl_success/len(seeds)*100:.1f}%) | Collisions: {ctrl_collisions:>2} | Envelopes: {ctrl_envelopes:>2} | {elapsed:.1f}s")

        env.close()

    total_time = time.time() - t_start

    # Export CSV of all episodes
    csv_path = out_dir / "bc_comparison_episodes_all.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(all_results[0]).keys()))
        writer.writeheader()
        for r in all_results:
            writer.writerow(asdict(r))
    print(f"\nExported detailed episode log: {csv_path.resolve()}")

    # Aggregate Metrics by Level and Controller
    summary: Dict[str, Any] = {}
    for lvl_idx, lvl_name, _, seeds in level_suites:
        summary[lvl_name] = {}
        for ctrl_id in controllers.keys():
            ep_list = [r for r in all_results if r.level == lvl_idx and r.controller_id == ctrl_id]
            feas_eps = [r for r in ep_list if r.is_feasible]
            infeas_eps = [r for r in ep_list if not r.is_feasible]

            succ_count = sum(1 for r in feas_eps if r.success)
            ci_low, ci_high = compute_wilson_ci(succ_count, len(feas_eps))
            succ_steps = [r.total_steps for r in feas_eps if r.success]
            succ_times = [r.flight_time_s for r in feas_eps if r.success]
            succ_speeds = [r.mean_speed_mps for r in feas_eps if r.success]
            succ_min_clear = [r.overall_min_clearance_m for r in feas_eps if r.success]

            obs_cols = sum(1 for r in ep_list if r.obstacle_collision)
            wall_cols = sum(1 for r in ep_list if r.wall_collision)
            tot_cols = sum(1 for r in ep_list if r.collision_episode)
            env_breaches = sum(1 for r in ep_list if r.safety_envelope_violation)
            interventions = sum(r.safety_interventions for r in ep_list)

            infeas_rejections = sum(1 for r in infeas_eps if r.safe_rejection)

            summary[lvl_name][ctrl_id] = {
                "feasible_total": len(feas_eps),
                "feasible_success": succ_count,
                "feasible_success_pct": round(succ_count / max(1, len(feas_eps)) * 100.0, 2),
                "wilson_ci_95": [round(ci_low, 2), round(ci_high, 2)],
                "infeasible_total": len(infeas_eps),
                "infeasible_safe_rejections": infeas_rejections,
                "infeasible_rejection_pct": round(infeas_rejections / max(1, len(infeas_eps)) * 100.0, 2) if infeas_eps else 100.0,
                "obstacle_collisions": obs_cols,
                "wall_collisions": wall_cols,
                "total_collision_episodes": tot_cols,
                "envelope_violation_episodes": env_breaches,
                "total_safety_interventions": interventions,
                "min_clearance_across_all_m": round(float(min(r.overall_min_clearance_m for r in ep_list)), 4) if ep_list else 0.0,
                "min_clearance_successful_m": round(float(min(succ_min_clear)), 4) if succ_min_clear else 0.0,
                "mean_steps_successful": round(float(np.mean(succ_steps)), 1) if succ_steps else 1000.0,
                "median_steps_successful": round(float(np.median(succ_steps)), 1) if succ_steps else 1000.0,
                "mean_flight_time_s": round(float(np.mean(succ_times)), 2) if succ_times else 50.0,
                "mean_velocity_mps": round(float(np.mean(succ_speeds)), 3) if succ_speeds else 0.0,
            }

    summary_path = out_dir / "bc_comparison_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Exported comparative summary JSON: {summary_path.resolve()}")

    return summary


if __name__ == "__main__":
    evaluate_scaled_bc_campaign()

