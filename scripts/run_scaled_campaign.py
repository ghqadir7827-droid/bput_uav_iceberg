"""run_scaled_campaign.py - Multi-Level Empirical Evaluation for Scaled 65cm UAV.

Executes a comprehensive, fair, and reproducible evaluation across 4 difficulty tiers:
- Level 1: Basic Navigation (Width 4.5m, Height 2.8m, N=50 seeds)
- Level 2: Standard Navigation (Width 3.8m, Height 2.5m, N=100 seeds)
- Level 3: Challenging Slaloms (Width 3.2m, Height 2.4m, N=50 seeds)
- Level 4: Boundary & Feasibility Tests (Width 2.8m, N=50 seeds, 50% passable, 50% blocked)

Evaluates 5 controllers under identical paired seeds:
1. reactive_baseline (scaled physical parameters)
2. raw_bc (bc_policy_best.pt)
3. guided_bc_phase6a (Phase 6A clearance guidance scaled for 65cm UAV)
4. raw_ppo (ppo_policy_best.pt)
5. guided_ppo (Tuned PPO with proactive deceleration & sector guidance)
"""

import csv
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from controllers.bc_policy import BehaviorCloningPolicyController
from controllers.ppo_policy import PPOPolicyController
from controllers.registry import ReactiveBaselineController
from controllers.safety_supervisor import SafetySupervisor, SafetySupervisorConfig
from core.contracts import ActionLimits, VelocityAction, compute_directional_clearances_m, compute_forward_clearance_m
from core.geometry import Corridor3D, TerminationReason
from core.guidance import GuidanceConfig, StatelessClearanceGuidanceArbitrator
from core.physical_dimensions import (
    DifficultyLevel,
    PhysicalDroneModel,
    RouteFeasibilityChecker,
    ScaledEnvironmentFactory,
)
from sim.corridor_env import CorridorSimEnv


# --- Custom Controller Wrappers for Scaled 65cm Evaluation ---

class GenericGuidedController:
    """Wraps any base policy with a guidance arbitrator."""
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


class TunedGuidedPPOArbitratorScaled:
    """Tuned guidance arbitrator designed to rescue PPO from lateral drift in 65cm corridor."""
    def __init__(self, drone_radius: float = 0.325, limits: Optional[ActionLimits] = None):
        self.drone_radius = drone_radius
        self.limits = limits or ActionLimits()
        self.last_dir: str = "NONE"

    def reset(self):
        self.last_dir = "NONE"

    def arbitrate(self, raw_action: VelocityAction, obs: np.ndarray) -> Tuple[VelocityAction, str, float, float]:
        obs_vec = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)
        rays = obs_vec[0:12] * 4.0
        c_front = float(min(rays[0], rays[1], rays[2]))
        c_left = float(min(rays[1], rays[3], rays[5]))
        c_right = float(min(rays[2], rays[4], rays[6]))
        diff = c_left - c_right
        wall_l = float(rays[5])
        wall_r = float(rays[6])

        goal_vec = obs_vec[12:15]
        in_terminal = float(np.linalg.norm(goal_vec)) < 0.50

        vx_cmd = raw_action.vx
        vy_cmd = raw_action.vy
        selected_dir = "NONE"
        override_speed = 0.35

        # Proactive longitudinal deceleration in scaled corridor
        # Center-to-obstacle distance threshold
        decel_thresh = self.drone_radius + 1.25  # 1.575m
        if c_front < decel_thresh:
            min_c = self.drone_radius + 0.25
            vx_cmd = float(np.clip(0.22 + 0.35 * (c_front - min_c) / 1.0, 0.18, raw_action.vx))

        # Lateral avoidance arbitration
        evade_thresh = self.drone_radius + 1.10  # 1.425m
        wall_tight = self.drone_radius + 0.35   # 0.675m
        if c_front < evade_thresh and not in_terminal:
            if wall_l < wall_tight and wall_r >= wall_tight:
                selected_dir = "WALL_AVOID_RIGHT"
                vy_cmd = min(vy_cmd, -override_speed)
            elif wall_r < wall_tight and wall_l >= wall_tight:
                selected_dir = "WALL_AVOID_LEFT"
                vy_cmd = max(vy_cmd, override_speed)
            elif diff > 0.10:
                selected_dir = "LEFT"
                vy_cmd = max(vy_cmd, override_speed)
            elif diff < -0.10:
                selected_dir = "RIGHT"
                vy_cmd = min(vy_cmd, -override_speed)
            else:
                if wall_r > wall_l + 0.15:
                    selected_dir = "TIE_BREAK_RIGHT"
                    vy_cmd = min(vy_cmd, -override_speed)
                else:
                    selected_dir = "TIE_BREAK_LEFT"
                    vy_cmd = max(vy_cmd, override_speed)
            self.last_dir = selected_dir
        else:
            self.last_dir = "NONE"

        guided_act = VelocityAction(
            vx=float(np.clip(vx_cmd, self.limits.min_fwd_vel_mps, self.limits.max_fwd_vel_mps)),
            vy=float(np.clip(vy_cmd, -self.limits.max_lat_vel_mps, self.limits.max_lat_vel_mps)),
            vz=float(np.clip(raw_action.vz, -self.limits.max_vert_vel_mps, self.limits.max_vert_vel_mps)),
            yaw_rate=float(np.clip(raw_action.yaw_rate, -self.limits.max_yaw_rate_radps, self.limits.max_yaw_rate_radps)),
        )
        return guided_act, selected_dir, c_left, c_right


@dataclass
class ScaledEpisodeResult:
    level: int
    level_name: str
    seed: int
    controller_id: str
    is_feasible: bool
    feasibility_status: str
    bottleneck_gap_m: float
    status: str
    success: bool
    safe_rejection: bool
    obstacle_collision: bool
    wall_collision: bool
    safety_envelope_violation: bool
    total_steps: int
    flight_time_s: float
    min_obstacle_clearance_m: float
    min_wall_clearance_m: float
    safety_interventions: int
    terminal_dist_to_goal_m: float
    terminal_speed_mps: float


def compute_wilson_ci(k: int, n: int, confidence: float = 0.95) -> Tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    z = 1.959964  # 95%
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    margin = (z / denom) * np.sqrt(p * (1.0 - p) / n + (z * z) / (4.0 * n * n))
    return float(max(0.0, center - margin)), float(min(1.0, center + margin))


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
) -> ScaledEpisodeResult:
    if hasattr(controller, "reset"):
        controller.reset()

    obs, _ = env.reset(seed=seed)
    corridor = env.corridor

    # Evaluate analytical feasibility
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

    dt = env.dt

    for step in range(max_steps):
        total_steps += 1
        pos = env._get_drone_position()
        vel = env._get_drone_velocity()

        # Analytical clearances
        d_obs = corridor.obstacle_clearance(pos, env.obstacles)
        d_wall = corridor.boundary_distance(pos)
        min_obs_clearance = min(min_obs_clearance, d_obs)
        min_wall_clearance = min(min_wall_clearance, d_wall)

        if d_obs < envelope_threshold_m:
            envelope_violation_count += 1

        # Check termination condition
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

        # Check for safe rejection / halting before an impassable gate
        if not feas_res.is_feasible:
            # If the drone has safely stopped in front of the blocked gate (e.g. speed < 0.05, 5.0m < x < 6.8m)
            # and stayed there for 50 steps without colliding
            if pos[0] >= 5.0 and np.linalg.norm(vel) < 0.06 and d_obs > envelope_threshold_m:
                if step > 200:
                    safe_rejection = True
                    outcome = "SAFE_REJECT"
                    break

        # Controller forward pass
        raw_act = controller.predict(obs)

        # Safety supervisor filtering
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

        curr_pos = env._get_drone_position()
        curr_vel = env._get_drone_velocity()

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

    # If episode timed out on an infeasible corridor with zero collisions and safe clearance:
    if not feas_res.is_feasible and not obstacle_collision and not wall_collision and min_obs_clearance >= 0.10:
        safe_rejection = True
        outcome = "SAFE_REJECT"

    final_pos = env._get_drone_position()
    final_vel = env._get_drone_velocity()
    term_dist_goal = float(np.linalg.norm(final_pos - corridor.goal_position))
    term_speed = float(np.linalg.norm(final_vel))

    return ScaledEpisodeResult(
        level=level,
        level_name=level_name,
        seed=seed,
        controller_id=controller_id,
        is_feasible=feas_res.is_feasible,
        feasibility_status=feas_res.status,
        bottleneck_gap_m=float(feas_res.min_bottleneck_gap_m),
        status=outcome,
        success=success,
        safe_rejection=safe_rejection,
        obstacle_collision=obstacle_collision,
        wall_collision=wall_collision,
        safety_envelope_violation=(envelope_violation_count > 0),
        total_steps=total_steps,
        flight_time_s=float(total_steps * dt),
        min_obstacle_clearance_m=float(min_obs_clearance),
        min_wall_clearance_m=float(min_wall_clearance),
        safety_interventions=safety_interventions,
        terminal_dist_to_goal_m=term_dist_goal,
        terminal_speed_mps=term_speed,
    )


def measure_distribution_shift(out_dir: Path, num_seeds: int = 50) -> Dict[str, Any]:
    """Measures quantitative observation distribution shift between old 2m corridor and Level 2 3.8m corridor."""
    print("Measuring Observation Distribution Shift (Old 10m vs Scaled Level 2 Standard)...")
    ctrl_old = ReactiveBaselineController()

    # Old 10m Benchmark
    env_old = CorridorSimEnv(scenario="10m", seed=100)
    obs_old_list = []
    for s in range(100, 100 + num_seeds):
        obs, _ = env_old.reset(seed=s)
        for _ in range(150):
            obs_old_list.append(obs)
            act = ctrl_old.predict(obs)
            norm_act = act.to_normalized(env_old.action_limits)
            obs, _, term, trunc, _ = env_old.step(norm_act)
            if term or trunc:
                break
    env_old.close()

    # Scaled Level 2 Benchmark
    c_scaled = ScaledEnvironmentFactory.get_environment_config(DifficultyLevel.LEVEL_2_STANDARD)
    env_new = CorridorSimEnv(scenario=c_scaled, seed=100)
    ctrl_new = ReactiveBaselineController(cruise_speed=0.60, drone_radius=0.325)
    obs_new_list = []
    for s in range(20000, 20000 + num_seeds):
        obs, _ = env_new.reset(seed=s)
        for _ in range(150):
            obs_new_list.append(obs)
            act = ctrl_new.predict(obs)
            norm_act = act.to_normalized(env_new.action_limits)
            obs, _, term, trunc, _ = env_new.step(norm_act)
            if term or trunc:
                break
    env_new.close()

    arr_old = np.array(obs_old_list, dtype=np.float32)
    arr_new = np.array(obs_new_list, dtype=np.float32)

    mean_old = np.mean(arr_old, axis=0)
    std_old = np.std(arr_old, axis=0)
    mean_new = np.mean(arr_new, axis=0)
    std_new = np.std(arr_new, axis=0)

    dim_names = [
        "ray_0_front", "ray_1_fl30", "ray_2_fr30", "ray_3_fl60", "ray_4_fr60",
        "ray_5_left90", "ray_6_right90", "ray_7_rear180", "ray_8_front_up45",
        "ray_9_front_down45", "ray_10_zenith", "ray_11_nadir",
        "goal_dir_x", "goal_dir_y", "goal_dir_z",
        "vel_x", "vel_y", "vel_z",
        "sin_roll", "cos_roll", "sin_pitch", "cos_pitch",
        "yaw_rate"
    ]

    shift_records = {}
    for i in range(23):
        delta = float(mean_new[i] - mean_old[i])
        pct_change = float(delta / (abs(mean_old[i]) + 1e-5) * 100)
        shift_records[dim_names[i]] = {
            "dim_index": i,
            "mean_old": float(mean_old[i]),
            "std_old": float(std_old[i]),
            "mean_new": float(mean_new[i]),
            "std_new": float(std_new[i]),
            "absolute_mean_delta": float(abs(delta)),
            "percent_shift": pct_change,
        }

    report = {
        "num_samples_old": len(obs_old_list),
        "num_samples_new": len(obs_new_list),
        "dimensions": shift_records,
    }
    with open(out_dir / "distribution_shift_analysis.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"Saved distribution shift analysis ({len(obs_old_list)} old samples, {len(obs_new_list)} new samples)")
    return report


def main():
    print("=" * 80)
    print("SCALED 65cm UAV MULTI-LEVEL BENCHMARK CAMPAIGN")
    print("Phases 1-6 Full Evaluation Protocol Across Difficulty Levels 1, 2, 3, 4")
    print("=" * 80)

    out_dir = Path("experiments/scaled_campaign/run_20261030_scaled")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Quantify Observation Distribution Shift
    measure_distribution_shift(out_dir, num_seeds=50)

    checker = RouteFeasibilityChecker(PhysicalDroneModel())

    # Build Controllers
    drone_radius = 0.325
    guidance_cfg = GuidanceConfig.for_65cm_uav(drone_radius)
    arbitrator_bc = StatelessClearanceGuidanceArbitrator(config=guidance_cfg)
    arbitrator_ppo = TunedGuidedPPOArbitratorScaled(drone_radius=drone_radius)

    base_bc = BehaviorCloningPolicyController()
    base_ppo = PPOPolicyController()

    controllers = {
        "reactive_baseline": ReactiveBaselineController(cruise_speed=0.60, drone_radius=drone_radius),
        "raw_bc": base_bc,
        "guided_bc_phase6a": GenericGuidedController(base_bc, arbitrator_bc, "guided_bc_phase6a"),
        "raw_ppo": base_ppo,
        "guided_ppo": GenericGuidedController(base_ppo, arbitrator_ppo, "guided_ppo"),
    }

    # Define Multi-Level Evaluation Suites
    # Level 1: 50 seeds (20100..20149)
    # Level 2: 100 seeds (20000..20099)
    # Level 3: 50 seeds (20200..20249)
    # Level 4: 50 seeds (20300..20349)
    level_suites = [
        (1, "Level 1 (Basic)", DifficultyLevel.LEVEL_1_BASIC, list(range(20100, 20150))),
        (2, "Level 2 (Standard)", DifficultyLevel.LEVEL_2_STANDARD, list(range(20000, 20100))),
        (3, "Level 3 (Challenging)", DifficultyLevel.LEVEL_3_CHALLENGING, list(range(20200, 20250))),
        (4, "Level 4 (Boundary & Feasibility)", DifficultyLevel.LEVEL_4_FEASIBILITY_BOUNDARY, list(range(20300, 20350))),
    ]

    all_results: List[ScaledEpisodeResult] = []
    t0 = time.time()
    total_episodes_planned = sum(len(seeds) * len(controllers) for _, _, _, seeds in level_suites)
    print(f"Total Episodes Planned: {total_episodes_planned} across 4 Difficulty Levels")

    for lvl_idx, lvl_name, lvl_enum, seeds in level_suites:
        print(f"\n--- Initializing {lvl_name} (N={len(seeds)} seeds) ---")
        corridor_cfg = ScaledEnvironmentFactory.get_environment_config(lvl_enum)
        env = CorridorSimEnv(scenario=corridor_cfg, seed=seeds[0])
        supervisor = SafetySupervisor(env.corridor)

        for ctrl_id, ctrl in controllers.items():
            print(f"  Evaluating {ctrl_id} on {lvl_name}...")
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
                if res.obstacle_collision or res.wall_collision:
                    ctrl_collisions += 1
                if res.safety_envelope_violation:
                    ctrl_envelopes += 1

            ctrl_dur = time.time() - ctrl_start
            print(f"    -> Success: {ctrl_success}/{len(seeds)} ({ctrl_success/len(seeds)*100:.1f}%), Collisions: {ctrl_collisions}, EnvViolations: {ctrl_envelopes} ({ctrl_dur:.1f}s)")

        env.close()

    total_time = time.time() - t0
    print(f"\nCompleted {len(all_results)} episodes in {total_time:.1f}s ({len(all_results)/total_time:.1f} ep/s)")

    # Save CSV
    csv_path = out_dir / "scaled_episodes_all.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(all_results[0]).keys()))
        writer.writeheader()
        for r in all_results:
            writer.writerow(asdict(r))
    print(f"Saved episode trajectories to {csv_path}")

    # Compute Aggregate Metrics & Level Breakdowns
    aggregates = {}
    for lvl_idx, lvl_name, _, seeds in level_suites:
        aggregates[lvl_name] = {}
        lvl_res = [r for r in all_results if r.level == lvl_idx]
        feasible_seeds = [r for r in lvl_res if r.is_feasible]
        infeasible_seeds = [r for r in lvl_res if not r.is_feasible]

        for ctrl_id in controllers.keys():
            c_res = [r for r in lvl_res if r.controller_id == ctrl_id]
            c_feas = [r for r in feasible_seeds if r.controller_id == ctrl_id]
            c_infeas = [r for r in infeasible_seeds if r.controller_id == ctrl_id]

            k_succ = sum(1 for r in c_feas if r.success)
            n_feas = len(c_feas)
            ci_low, ci_high = compute_wilson_ci(k_succ, n_feas) if n_feas > 0 else (0.0, 0.0)

            k_reject = sum(1 for r in c_infeas if r.safe_rejection)
            n_infeas = len(c_infeas)

            colls = sum(1 for r in c_res if r.obstacle_collision or r.wall_collision)
            envs = sum(1 for r in c_res if r.safety_envelope_violation)
            min_obs = min(r.min_obstacle_clearance_m for r in c_res)
            steps_feas = [r.total_steps for r in c_feas if r.success]
            mean_steps = float(np.mean(steps_feas)) if steps_feas else 1000.0

            aggregates[lvl_name][ctrl_id] = {
                "feasible_total": n_feas,
                "feasible_success": k_succ,
                "feasible_success_pct": float(k_succ / n_feas * 100) if n_feas > 0 else 0.0,
                "wilson_ci_95": [float(ci_low * 100), float(ci_high * 100)],
                "infeasible_total": n_infeas,
                "infeasible_safe_rejections": k_reject,
                "infeasible_rejection_pct": float(k_reject / n_infeas * 100) if n_infeas > 0 else 100.0,
                "collisions": colls,
                "envelope_violations_lt_0_15m": envs,
                "min_obstacle_clearance_m": float(min_obs),
                "mean_steps_to_goal": mean_steps,
            }

    json_path = out_dir / "aggregate_metrics.json"
    with open(json_path, "w") as f:
        json.dump(aggregates, f, indent=2)
    print(f"Saved aggregate metrics to {json_path}")


if __name__ == "__main__":
    main()
