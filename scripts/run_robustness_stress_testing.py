"""run_robustness_stress_testing.py - Controlled Simulation Uncertainty & Disturbance Suite.

Progressively evaluates v2 vs v3 across 8 controlled disturbance cases:
1. Baseline Deterministic (Nominal)
2. Range-Sensor Noise (sigma=0.05m) & Bias (+-0.06m)
3. Localization Drift (bounded 0.10m) & Jitter (0.03m)
4. Sensor Latency (50ms / 1 step) & 5% Packet Dropout
5. Control-Command Latency (50ms) & Bounded Actuation Error (+-8%)
6. Temporary Sensor/Communication Blackout (0.25s at approach)
7. Combined Moderate Disturbances
8. Initial State Perturbations (Lateral offset +-0.35m, Cross-track vel +-0.15m/s)

Seeds: Dedicated clean range 25000..25019 (N=20 seeds per case across Level 2 Standard, 160 paired runs = 320 episodes).
Outputs:
- data/stress_test_config_manifest.json
- experiments/scaled_campaign/robustness_stress_test/stress_test_episodes.csv
- experiments/scaled_campaign/robustness_stress_test/stress_test_summary.json
"""

from collections import deque
import csv
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pybullet as p

from controllers.bc_policy import BehaviorCloningPolicyController
from controllers.safety_supervisor import SafetySupervisor
from core.contracts import VelocityAction, compute_forward_clearance_m
from core.geometry import TerminationReason
from core.guidance import GuidanceConfig, StatelessClearanceGuidanceArbitrator
from core.physical_dimensions import DifficultyLevel, ScaledEnvironmentFactory
from scripts.run_exceed_90_experiments import GenericGuidedController
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
class StressEpisodeRecord:
    case_id: int
    case_name: str
    seed: int
    controller_id: str
    status: str
    success: bool
    timeout: bool
    collision_episode: bool
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


def run_stress_episode(
    env: CorridorSimEnv,
    controller: Any,
    supervisor: SafetySupervisor,
    seed: int,
    controller_id: str,
    dist: DisturbanceConfig,
    max_steps: int = 1000,
    envelope_threshold_m: float = 0.15,
) -> StressEpisodeRecord:
    controller.reset()
    supervisor.reset()

    # Reset environment with reproducible seed
    obs, _ = env.reset(seed=seed)
    corridor = env.corridor

    # Seed local numpy generator for disturbances
    rng = np.random.RandomState(seed + dist.case_id * 1000)

    # Initial state perturbations
    if abs(dist.initial_y_offset_m) > 1e-4 or abs(dist.initial_vy_mps) > 1e-4:
        # Override initial drone position & velocity
        p_cur = env._get_drone_position()
        v_cur = env._get_drone_velocity()
        p_new = np.array([p_cur[0], p_cur[1] + dist.initial_y_offset_m, p_cur[2]])
        v_new = np.array([v_cur[0], v_cur[1] + dist.initial_vy_mps, v_cur[2]])
        # Set PyBullet base state
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

        # Check for blackout trigger
        if dist.blackout_steps > 0 and not blackout_triggered and pos[0] >= dist.blackout_trigger_x_m:
            blackout_triggered = True
            blackout_remaining = dist.blackout_steps

        # Construct perturbed observation
        noisy_obs = obs.copy()

        # 1. Range noise & bias (normalized range [0..1] represents 0..4.0m)
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

        # 3. Packet drop simulation (stale data holding)
        if dist.obs_drop_prob > 0 and rng.rand() < dist.obs_drop_prob:
            noisy_obs = obs_queue[0].copy()

        # 4. Blackout simulation (temporary sensor loss)
        if blackout_remaining > 0:
            blackout_remaining -= 1
            noisy_obs[0:12] = 1.0  # Max range / no returns
            noisy_obs[15:18] = 0.0  # Stale zero velocity

        # Observation delay queue
        obs_queue.append(noisy_obs)
        delayed_obs = obs_queue[0]

        # Policy prediction
        raw_act = controller.predict(delayed_obs)
        min_front = compute_forward_clearance_m(delayed_obs, max_range_m=env.sensor_config.max_range_m)

        # Safety supervisor filtering
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

        # Actuation scale error
        if dist.act_scale_factor != 1.0:
            safe_act = VelocityAction(
                vx=safe_act.vx * dist.act_scale_factor,
                vy=safe_act.vy * dist.act_scale_factor,
                vz=safe_act.vz,
                yaw_rate=safe_act.yaw_rate,
            )

        # Actuator latency queue
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

    final_pos = env._get_drone_position()
    dist_to_goal = float(corridor.distance_to_goal(final_pos))
    flight_time = round(total_steps * dt, 2)
    mean_speed = round(flight_distance / flight_time, 3) if flight_time > 0 else 0.0
    overall_min = min(min_obs_clearance, min_wall_clearance)

    return StressEpisodeRecord(
        case_id=dist.case_id,
        case_name=dist.name,
        seed=seed,
        controller_id=controller_id,
        status=outcome,
        success=success,
        timeout=(outcome == "TIMEOUT"),
        collision_episode=(obstacle_collision or wall_collision),
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


def build_stress_suite() -> List[DisturbanceConfig]:
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
            description="Position jitter sigma=0.03m + velocity jitter + systematic drift up to 0.10m.",
            loc_jitter_sigma_m=0.03,
            loc_drift_rate_mps=0.02,
        ),
        DisturbanceConfig(
            case_id=4,
            name="Sensor Latency & Packet Drops",
            description="1-step observation delay (50ms) + 5% random telemetry packet dropout.",
            obs_delay_steps=1,
            obs_drop_prob=0.05,
        ),
        DisturbanceConfig(
            case_id=5,
            name="Control Actuation Latency & Scale Error",
            description="1-step actuation delay (50ms) + -8% motor scale calibration error.",
            act_delay_steps=1,
            act_scale_factor=0.92,
        ),
        DisturbanceConfig(
            case_id=6,
            name="Temporary Telemetry Blackout",
            description="Complete sensor blackout for 0.25s (5 steps) at x=3.0m during approach.",
            blackout_steps=5,
            blackout_trigger_x_m=3.0,
        ),
        DisturbanceConfig(
            case_id=7,
            name="Combined Moderate Disturbances",
            description="Noise sigma=0.03m + 50ms latency + 2% packet drop + 5% actuator error.",
            range_noise_sigma_m=0.03,
            obs_delay_steps=1,
            obs_drop_prob=0.02,
            act_scale_factor=0.95,
        ),
        DisturbanceConfig(
            case_id=8,
            name="Initial State Perturbations",
            description="Initial lateral offset y0 = +0.35m + initial cross-track velocity vy0 = -0.15m/s.",
            initial_y_offset_m=0.35,
            initial_vy_mps=-0.15,
        ),
    ]


def run_robustness_campaign():
    print("=" * 80)
    print("PHASE 4: CONTROLLED SIMULATION STRESS TESTING CAMPAIGN")
    print("=" * 80)

    out_dir = Path("experiments/scaled_campaign/robustness_stress_test")
    out_dir.mkdir(parents=True, exist_ok=True)

    stress_cases = build_stress_suite()
    test_seeds = list(range(25000, 25020))  # N=20 seeds per case
    print(f"Executing {len(stress_cases)} disturbance classes across {len(test_seeds)} paired seeds (Total {len(stress_cases)*len(test_seeds)*2} episodes)...")

    # Save Configuration Manifest
    manifest = {
        "title": "UAV Navigation Robustness & Uncertainty Stress Test Suite",
        "timestamp_iso": "2026-10-31T01:30:00Z",
        "seeds_evaluated": test_seeds,
        "difficulty_tier": "Level 2 (Standard, Width=3.8m)",
        "cases": [asdict(c) for c in stress_cases],
    }
    with open("data/stress_test_config_manifest.json", "w") as fp:
        json.dump(manifest, fp, indent=2)
    print("Saved data/stress_test_config_manifest.json")

    cfg = ScaledEnvironmentFactory.get_environment_config(DifficultyLevel.LEVEL_2_STANDARD)
    env = CorridorSimEnv(scenario=cfg, seed=test_seeds[0])
    bc = BehaviorCloningPolicyController("models/checkpoints/bc_policy_scaled_v2.pt")

    # Controller 1: v2
    g_cfg_v2 = GuidanceConfig.for_65cm_uav_v2()
    arb_v2 = StatelessClearanceGuidanceArbitrator(config=g_cfg_v2)
    ctrl_v2 = GenericGuidedController(bc, arb_v2, "scaled_guided_bc_v2")

    # Controller 2: v3
    g_cfg_v3 = GuidanceConfig.for_65cm_uav_v3()
    arb_v3 = StatelessClearanceGuidanceArbitrator(config=g_cfg_v3)
    ctrl_v3 = GenericGuidedController(bc, arb_v3, "scaled_guided_bc_v3")

    controllers = [
        ("scaled_guided_bc_v2", ctrl_v2),
        ("scaled_guided_bc_v3", ctrl_v3),
    ]

    all_records: List[StressEpisodeRecord] = []
    case_summaries: Dict[str, Any] = {}
    t_start = time.time()

    for dist in stress_cases:
        print(f"\n[{dist.case_id}/8] Testing: {dist.name} (N={len(test_seeds)} seeds)")
        sup = SafetySupervisor(env.corridor)

        for cid, ctrl in controllers:
            t0 = time.time()
            c_recs = []
            for s in test_seeds:
                rec = run_stress_episode(
                    env=env,
                    controller=ctrl,
                    supervisor=sup,
                    seed=s,
                    controller_id=cid,
                    dist=dist,
                )
                c_recs.append(rec)
                all_records.append(rec)

            succ = sum(1 for r in c_recs if r.success)
            cols = sum(1 for r in c_recs if r.collision_episode)
            envs = sum(1 for r in c_recs if r.safety_envelope_violation)
            env_steps = sum(r.envelope_violation_steps for r in c_recs)
            intervs = sum(r.safety_interventions for r in c_recs)
            tot_steps = sum(r.total_steps for r in c_recs)
            c_vals = [r.overall_min_clearance_m for r in c_recs]
            ci = compute_wilson_ci(succ, len(test_seeds))
            elapsed = time.time() - t0

            summary_key = f"Case_{dist.case_id}_{cid}"
            case_summaries[summary_key] = {
                "case_id": dist.case_id,
                "case_name": dist.name,
                "controller_id": cid,
                "n_episodes": len(test_seeds),
                "success_count": succ,
                "success_pct": round(succ / len(test_seeds) * 100.0, 2),
                "wilson_ci_95": ci,
                "collisions": cols,
                "envelope_violations": envs,
                "envelope_violation_pct": round(envs / len(test_seeds) * 100.0, 2),
                "envelope_violation_steps": env_steps,
                "interventions": intervs,
                "intervention_rate_per_kstep": round(intervs / tot_steps * 1000.0, 2) if tot_steps > 0 else 0.0,
                "min_clearance_m": round(min(c_vals), 4) if c_vals else 0.0,
                "median_clearance_m": round(float(np.median(c_vals)), 4) if c_vals else 0.0,
                "elapsed_s": round(elapsed, 2),
            }

            print(f"  {cid:<22} | Succ: {succ:>2}/{len(test_seeds)} ({round(succ/len(test_seeds)*100, 1)}%) "
                  f"[95% CI: {ci}] | Cols: {cols} | Envs: {envs} ({env_steps} steps) | {elapsed:.1f}s")

    env.close()
    total_time = time.time() - t_start
    print(f"\nAll {len(all_records)} stress episodes completed in {total_time:.1f}s.")

    # Save CSV
    csv_file = out_dir / "stress_test_episodes.csv"
    with open(csv_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(all_records[0]).keys()))
        writer.writeheader()
        for r in all_records:
            writer.writerow(asdict(r))
    print(f"Saved episode data to {csv_file}")

    # Compute Paired McNemar for each case
    paired_summary: Dict[str, Any] = {}
    for dist in stress_cases:
        cid_v2_recs = {r.seed: r for r in all_records if r.case_id == dist.case_id and r.controller_id == "scaled_guided_bc_v2"}
        cid_v3_recs = {r.seed: r for r in all_records if r.case_id == dist.case_id and r.controller_id == "scaled_guided_bc_v3"}
        both_succ = sum(1 for s in test_seeds if cid_v2_recs[s].success and cid_v3_recs[s].success)
        v2_only = sum(1 for s in test_seeds if cid_v2_recs[s].success and not cid_v3_recs[s].success)
        v3_only = sum(1 for s in test_seeds if not cid_v2_recs[s].success and cid_v3_recs[s].success)
        both_fail = sum(1 for s in test_seeds if not cid_v2_recs[s].success and not cid_v3_recs[s].success)
        p_mcnemar = mcnemar_exact(v2_only, v3_only)

        paired_summary[dist.name] = {
            "v2_summary": case_summaries[f"Case_{dist.case_id}_scaled_guided_bc_v2"],
            "v3_summary": case_summaries[f"Case_{dist.case_id}_scaled_guided_bc_v3"],
            "both_succeed": both_succ,
            "v2_only": v2_only,
            "v3_only": v3_only,
            "both_fail": both_fail,
            "mcnemar_p_value": p_mcnemar,
        }

    # Save JSON Summary
    json_file = out_dir / "stress_test_summary.json"
    with open(json_file, "w") as fp:
        json.dump(paired_summary, fp, indent=2)
    print(f"Saved stress test summary to {json_file}")


if __name__ == "__main__":
    run_robustness_campaign()
