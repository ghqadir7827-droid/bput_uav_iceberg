"""run_adaptive_disturbance_evaluation.py - Controlled Disturbance Evaluation of v3 vs v4.

Compares:
- scaled_guided_bc_v3 (Phase 6 persistent guidance)
- scaled_guided_bc_v4 (Corridor-width-adaptive guidance candidate)

Across the 8 standardized disturbance classes on Level 2 Standard (Width=3.8m):
1. Baseline Deterministic (Nominal)
2. Range-Sensor Noise (sigma=0.05m) & Bias (+-0.06m)
3. Localization Drift (bounded 0.10m) & Jitter (0.03m)
4. Sensor Latency (50ms / 1 step) & 5% Packet Dropout
5. Control-Command Latency (50ms) & Bounded Actuation Error (+-8%)
6. Temporary Sensor/Communication Blackout (0.25s at approach)
7. Combined Moderate Disturbances
8. Initial State Perturbations (Lateral offset +-0.35m, Cross-track vel +-0.15m/s)

Seeds: 25000..25019 (N=20 per case, 160 paired runs = 320 episodes total).
Outputs:
- experiments/scaled_campaign/adaptive_guidance/disturbance_episodes.csv
- experiments/scaled_campaign/adaptive_guidance/disturbance_summary.json
"""

from collections import deque
import csv
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pybullet as p

from controllers.registry import ControllerRegistry
from controllers.safety_supervisor import SafetySupervisor
from core.contracts import VelocityAction, compute_forward_clearance_m
from core.geometry import TerminationReason
from core.physical_dimensions import DifficultyLevel, PhysicalDroneModel, RouteFeasibilityChecker, ScaledEnvironmentFactory
from scripts.comprehensive_statistical_audit import compute_wilson_ci, mcnemar_exact
from sim.corridor_env import CorridorSimEnv


@dataclass
class DisturbanceConfig:
    case_id: int
    name: str
    description: str
    range_noise_sigma_m: float = 0.0
    range_bias_m: float = 0.0
    loc_jitter_sigma_m: float = 0.0
    loc_drift_rate_mps: float = 0.0
    obs_delay_steps: int = 0
    obs_drop_prob: float = 0.0
    act_delay_steps: int = 0
    act_scale_factor: float = 1.0
    blackout_steps: int = 0
    blackout_trigger_x_m: float = 3.0
    initial_y_offset_m: float = 0.0
    initial_vy_mps: float = 0.0


@dataclass
class DisturbanceEpisodeRecord:
    case_id: int
    case_name: str
    seed: int
    controller_id: str
    status: str
    success: bool
    safe_rejection: bool
    timeout: bool
    collision_episode: bool
    obstacle_collision: bool
    wall_collision: bool
    safety_envelope_violation: bool
    envelope_violation_steps: int
    total_steps: int
    flight_time_s: float
    flight_distance_m: float
    mean_speed_mps: float
    min_obstacle_clearance_m: float
    min_wall_clearance_m: float
    overall_min_clearance_m: float
    safety_interventions: int
    terminal_dist_to_goal_m: float


def get_disturbance_cases() -> List[DisturbanceConfig]:
    return [
        DisturbanceConfig(
            case_id=1,
            name="Baseline Deterministic",
            description="Nominal physical simulation with zero injected disturbances.",
        ),
        DisturbanceConfig(
            case_id=2,
            name="Sensor Noise & Bias",
            description="Gaussian noise sigma=0.05m + bounded bias +0.06m on all rangefinder beams.",
            range_noise_sigma_m=0.05,
            range_bias_m=0.06,
        ),
        DisturbanceConfig(
            case_id=3,
            name="Localization Drift & Jitter",
            description="Continuous lateral drift 0.04 m/s (clamped at 0.10m) + jitter sigma=0.03m.",
            loc_jitter_sigma_m=0.03,
            loc_drift_rate_mps=0.04,
        ),
        DisturbanceConfig(
            case_id=4,
            name="Sensor Latency & Dropout",
            description="Observation delay 50ms (1 step) + 5% random packet dropout holding stale reading.",
            obs_delay_steps=1,
            obs_drop_prob=0.05,
        ),
        DisturbanceConfig(
            case_id=5,
            name="Actuation Latency & Scaling Error",
            description="Control latency 50ms (1 step) + actuation scale factor 0.92 (-8% thruster under-performance).",
            act_delay_steps=1,
            act_scale_factor=0.92,
        ),
        DisturbanceConfig(
            case_id=6,
            name="Approach Sensor Blackout",
            description="0.25s (5 consecutive steps) complete loss of rangefinder telemetry during obstacle approach at x >= 3.0m.",
            blackout_steps=5,
            blackout_trigger_x_m=3.0,
        ),
        DisturbanceConfig(
            case_id=7,
            name="Combined Moderate Disturbances",
            description="Simultaneous sensor noise (0.03m), drift (0.02 m/s), 1-step latency, and 3% dropout.",
            range_noise_sigma_m=0.03,
            range_bias_m=0.03,
            loc_jitter_sigma_m=0.02,
            loc_drift_rate_mps=0.02,
            obs_delay_steps=1,
            obs_drop_prob=0.03,
            act_delay_steps=1,
            act_scale_factor=0.95,
        ),
        DisturbanceConfig(
            case_id=8,
            name="Initial State Perturbations",
            description="Initial lateral position offset +0.35m and initial cross-track velocity -0.15 m/s toward boundary wall.",
            initial_y_offset_m=0.35,
            initial_vy_mps=-0.15,
        ),
    ]


def run_disturbance_episode(
    env: CorridorSimEnv,
    controller: Any,
    supervisor: SafetySupervisor,
    seed: int,
    controller_id: str,
    dist: DisturbanceConfig,
    max_steps: int = 1000,
    envelope_threshold_m: float = 0.15,
) -> DisturbanceEpisodeRecord:
    if hasattr(controller, "reset"):
        controller.reset()
    if hasattr(supervisor, "reset"):
        supervisor.reset()

    obs, _ = env.reset(seed=seed)
    corridor = env.corridor

    rng = np.random.RandomState(seed + dist.case_id * 1000)

    # Initial state perturbations
    if abs(dist.initial_y_offset_m) > 1e-4 or abs(dist.initial_vy_mps) > 1e-4:
        p_cur = env._get_drone_position()
        v_cur = env._get_drone_velocity()
        p_new = np.array([p_cur[0], p_cur[1] + dist.initial_y_offset_m, p_cur[2]])
        v_new = np.array([v_cur[0], v_cur[1] + dist.initial_vy_mps, v_cur[2]])
        p.resetBasePositionAndOrientation(env.drone_id, p_new, [0, 0, 0, 1], physicsClientId=env.client_id)
        p.resetBaseVelocity(env.drone_id, v_new, [0, 0, 0], physicsClientId=env.client_id)
        deployable_obs, _ = env._get_observations()
        obs = deployable_obs.to_vector()

    obs_queue = deque([obs.copy()] * (dist.obs_delay_steps + 1), maxlen=dist.obs_delay_steps + 1)
    act_queue = deque([VelocityAction(0.0, 0.0, 0.0, 0.0)] * (dist.act_delay_steps + 1), maxlen=dist.act_delay_steps + 1)

    total_steps = 0
    min_obs_clearance = float("inf")
    min_wall_clearance = float("inf")
    envelope_violation_steps = 0
    safety_interventions = 0
    blackout_remaining = 0
    blackout_triggered = False

    outcome = "TIMEOUT"
    success = False
    obstacle_collision = False
    wall_collision = False
    flight_distance = 0.0
    prev_pos = env._get_drone_position()
    dt = env.dt
    cum_drift_y = 0.0

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
            envelope_violation_steps += 1

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

        # Blackout trigger
        if dist.blackout_steps > 0 and not blackout_triggered and pos[0] >= dist.blackout_trigger_x_m:
            blackout_triggered = True
            blackout_remaining = dist.blackout_steps

        # Perturbed observation
        noisy_obs = obs.copy()

        # 1. Range noise & bias
        if dist.range_noise_sigma_m > 0:
            noise = rng.normal(0, dist.range_noise_sigma_m / 4.0, size=12)
            noisy_obs[0:12] = np.clip(noisy_obs[0:12] + noise, 0.0, 1.0)
        if abs(dist.range_bias_m) > 0:
            noisy_obs[0:12] = np.clip(noisy_obs[0:12] + (dist.range_bias_m / 4.0), 0.0, 1.0)

        # 2. Localization jitter & drift
        if dist.loc_jitter_sigma_m > 0:
            noisy_obs[12:15] += rng.normal(0, dist.loc_jitter_sigma_m, size=3)
        if dist.loc_drift_rate_mps > 0:
            cum_drift_y += dist.loc_drift_rate_mps * dt * (1.0 if rng.rand() > 0.5 else -1.0)
            cum_drift_y = np.clip(cum_drift_y, -0.10, 0.10)
            noisy_obs[13] += cum_drift_y

        # 3. Packet drop
        if dist.obs_drop_prob > 0 and rng.rand() < dist.obs_drop_prob:
            noisy_obs = obs_queue[0].copy()

        # 4. Blackout
        if blackout_remaining > 0:
            blackout_remaining -= 1
            noisy_obs[0:12] = 1.0
            noisy_obs[15:18] = 0.0

        obs_queue.append(noisy_obs)
        delayed_obs = obs_queue[0]

        raw_act = controller.predict(delayed_obs)
        min_front = compute_forward_clearance_m(delayed_obs, max_range_m=env.sensor_config.max_range_m)

        safe_act, state, _ = supervisor.filter_action(
            proposed_action=raw_act,
            current_position=pos,
            current_velocity=vel,
            min_front_range=min_front,
            dt=dt,
            is_localization_healthy=(blackout_remaining == 0),
            telemetry_age_s=0.25 if blackout_remaining > 0 else (dist.obs_delay_steps * dt),
            observation=delayed_obs,
        )

        if state.value != "nominal":
            safety_interventions += 1

        if dist.act_scale_factor != 1.0:
            safe_act = VelocityAction(
                vx=safe_act.vx * dist.act_scale_factor,
                vy=safe_act.vy * dist.act_scale_factor,
                vz=safe_act.vz,
                yaw_rate=safe_act.yaw_rate,
            )

        act_queue.append(safe_act)
        delayed_act = act_queue[0]

        norm_act = delayed_act.to_normalized(env.action_limits)
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

    has_collision = obstacle_collision or wall_collision
    final_pos = env._get_drone_position()
    dist_to_goal = float(corridor.distance_to_goal(final_pos))
    flight_time = round(total_steps * dt, 2)
    mean_speed = round(flight_distance / flight_time, 3) if flight_time > 0 else 0.0
    overall_min = min(min_obs_clearance, min_wall_clearance)

    return DisturbanceEpisodeRecord(
        case_id=dist.case_id,
        case_name=dist.name,
        seed=seed,
        controller_id=controller_id,
        status=outcome,
        success=success,
        safe_rejection=False,
        timeout=(outcome == "TIMEOUT"),
        collision_episode=has_collision,
        obstacle_collision=obstacle_collision,
        wall_collision=wall_collision,
        safety_envelope_violation=(envelope_violation_steps > 0),
        envelope_violation_steps=envelope_violation_steps,
        total_steps=total_steps,
        flight_time_s=flight_time,
        flight_distance_m=round(flight_distance, 3),
        mean_speed_mps=mean_speed,
        min_obstacle_clearance_m=round(float(min_obs_clearance), 4),
        min_wall_clearance_m=round(float(min_wall_clearance), 4),
        overall_min_clearance_m=round(float(overall_min), 4),
        safety_interventions=safety_interventions,
        terminal_dist_to_goal_m=round(dist_to_goal, 3),
    )


def run_disturbance_evaluation():
    print("=" * 80)
    print("PHASE 4: DISTURBANCE & UNCERTAINTY EVALUATION (v3 vs v4)")
    print("=" * 80)

    out_dir = Path("experiments/scaled_campaign/adaptive_guidance")
    out_dir.mkdir(parents=True, exist_ok=True)

    registry = ControllerRegistry()
    ctrl_v3 = registry.get_controller("scaled_guided_bc_v3")
    ctrl_v4 = registry.get_controller("scaled_guided_bc_v4")

    controllers = {
        "scaled_guided_bc_v3": ctrl_v3,
        "scaled_guided_bc_v4": ctrl_v4,
    }

    seeds = list(range(25000, 25020))  # N=20 seeds
    cases = get_disturbance_cases()

    corridor_cfg = ScaledEnvironmentFactory.get_environment_config(DifficultyLevel.LEVEL_2_STANDARD)
    env = CorridorSimEnv(scenario=corridor_cfg, seed=seeds[0])
    supervisor = SafetySupervisor(corridor=corridor_cfg)

    records: List[DisturbanceEpisodeRecord] = []
    total_evals = len(cases) * len(seeds) * len(controllers)
    print(f"Total planned runs: {total_evals} episodes across 8 disturbance classes and 2 controllers.")

    start_time = time.time()
    count = 0

    for dist in cases:
        print(f"\nEvaluating Case {dist.case_id}: {dist.name} (N={len(seeds)} seeds)...")
        for seed in seeds:
            for cid, ctrl in controllers.items():
                rec = run_disturbance_episode(
                    env=env,
                    controller=ctrl,
                    supervisor=supervisor,
                    seed=seed,
                    controller_id=cid,
                    dist=dist,
                )
                records.append(rec)
                count += 1
                if count % 40 == 0:
                    elapsed = time.time() - start_time
                    print(f"  Completed {count}/{total_evals} episodes ({count / total_evals * 100:.1f}%) in {elapsed:.1f}s")

    env.close()
    total_time = time.time() - start_time
    print(f"\nAll {len(records)} disturbance episodes completed in {total_time:.2f}s!")

    # Write CSV
    csv_path = out_dir / "disturbance_episodes.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[f.name for f in DisturbanceEpisodeRecord.__dataclass_fields__.values()])
        writer.writeheader()
        for r in records:
            writer.writerow(asdict(r))
    print(f"Saved disturbance episode records to {csv_path}")

    # Compute Summary
    summary = compute_disturbance_summary(records, cases, controllers)
    summary_path = out_dir / "disturbance_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved disturbance summary to {summary_path}")

    return records, summary


def compute_disturbance_summary(
    records: List[DisturbanceEpisodeRecord],
    cases: List[DisturbanceConfig],
    controllers: Dict[str, Any],
) -> Dict[str, Any]:
    cids = list(controllers.keys())
    res: Dict[str, Any] = {"per_case": {}, "overall": {}}

    for dist in cases:
        cid_data = {}
        c_recs_dist = [r for r in records if r.case_id == dist.case_id]

        for cid in cids:
            recs = [r for r in c_recs_dist if r.controller_id == cid]
            n_tot = len(recs)
            succ = sum(1 for r in recs if r.success)
            timeouts = sum(1 for r in recs if r.timeout)
            collisions = sum(1 for r in recs if r.collision_episode)
            envelope_vios = sum(1 for r in recs if r.safety_envelope_violation)
            total_vio_steps = sum(r.envelope_violation_steps for r in recs)
            total_steps = sum(r.total_steps for r in recs)

            overall_min_c = min(r.overall_min_clearance_m for r in recs) if recs else 0.0
            succ_recs = [r for r in recs if r.success]
            succ_min_c = min(r.overall_min_clearance_m for r in succ_recs) if succ_recs else 0.0
            interventions = sum(r.safety_interventions for r in recs)
            mean_flight_time = float(np.mean([r.flight_time_s for r in succ_recs])) if succ_recs else 0.0

            cid_data[cid] = {
                "total_episodes": n_tot,
                "success_count": succ,
                "success_rate": round(succ / n_tot, 4),
                "success_ci95": compute_wilson_ci(succ, n_tot),
                "timeouts": timeouts,
                "timeout_rate": round(timeouts / n_tot, 4),
                "collisions": collisions,
                "collision_rate": round(collisions / n_tot, 4),
                "collision_ci95": compute_wilson_ci(collisions, n_tot),
                "envelope_violation_episodes": envelope_vios,
                "envelope_violation_rate": round(envelope_vios / n_tot, 4),
                "envelope_violation_steps": total_vio_steps,
                "envelope_violation_step_pct": round(total_vio_steps / total_steps * 100, 2) if total_steps > 0 else 0.0,
                "overall_min_clearance_m": round(overall_min_c, 4),
                "successful_min_clearance_m": round(succ_min_c, 4),
                "total_interventions": interventions,
                "mean_interventions_per_ep": round(interventions / n_tot, 1),
                "mean_flight_time_s": round(mean_flight_time, 2),
            }

        # Paired McNemar for this case
        paired_case: Dict[int, Dict[str, DisturbanceEpisodeRecord]] = {}
        for r in c_recs_dist:
            if r.seed not in paired_case:
                paired_case[r.seed] = {}
            paired_case[r.seed][r.controller_id] = r

        n_both = 0
        n_v3_only = 0
        n_v4_only = 0
        n_neither = 0
        for s, p in paired_case.items():
            s3 = p.get("scaled_guided_bc_v3", DisturbanceEpisodeRecord(0, "", 0, "", "", False, False, False, False, False, False, False, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)).success
            s4 = p.get("scaled_guided_bc_v4", DisturbanceEpisodeRecord(0, "", 0, "", "", False, False, False, False, False, False, False, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)).success
            if s3 and s4:
                n_both += 1
            elif s3 and not s4:
                n_v3_only += 1
            elif not s3 and s4:
                n_v4_only += 1
            else:
                n_neither += 1

        p_val = mcnemar_exact(n_v3_only, n_v4_only)
        cid_data["paired_mcnemar"] = {
            "both_succeed": n_both,
            "v3_only": n_v3_only,
            "v4_only": n_v4_only,
            "both_fail": n_neither,
            "p_value": p_val,
        }

        res["per_case"][f"case_{dist.case_id}_{dist.name}"] = cid_data

    # Overall Summary across all 8 cases (160 episodes per controller)
    for cid in cids:
        recs = [r for r in records if r.controller_id == cid]
        n_tot = len(recs)
        succ = sum(1 for r in recs if r.success)
        timeouts = sum(1 for r in recs if r.timeout)
        collisions = sum(1 for r in recs if r.collision_episode)
        envelope_vios = sum(1 for r in recs if r.safety_envelope_violation)
        total_vio_steps = sum(r.envelope_violation_steps for r in recs)
        total_steps = sum(r.total_steps for r in recs)

        overall_min_c = min(r.overall_min_clearance_m for r in recs) if recs else 0.0
        succ_recs = [r for r in recs if r.success]
        succ_min_c = min(r.overall_min_clearance_m for r in succ_recs) if succ_recs else 0.0
        interventions = sum(r.safety_interventions for r in recs)
        mean_flight_time = float(np.mean([r.flight_time_s for r in succ_recs])) if succ_recs else 0.0

        res["overall"][cid] = {
            "total_episodes": n_tot,
            "success_count": succ,
            "success_rate": round(succ / n_tot, 4),
            "success_ci95": compute_wilson_ci(succ, n_tot),
            "timeouts": timeouts,
            "timeout_rate": round(timeouts / n_tot, 4),
            "collisions": collisions,
            "collision_rate": round(collisions / n_tot, 4),
            "collision_ci95": compute_wilson_ci(collisions, n_tot),
            "envelope_violation_episodes": envelope_vios,
            "envelope_violation_rate": round(envelope_vios / n_tot, 4),
            "envelope_violation_steps": total_vio_steps,
            "envelope_violation_step_pct": round(total_vio_steps / total_steps * 100, 2) if total_steps > 0 else 0.0,
            "overall_min_clearance_m": round(overall_min_c, 4),
            "successful_min_clearance_m": round(succ_min_c, 4),
            "total_interventions": interventions,
            "mean_interventions_per_ep": round(interventions / n_tot, 1),
            "mean_flight_time_s": round(mean_flight_time, 2),
        }

    return res


if __name__ == "__main__":
    run_disturbance_evaluation()

