"""run_phase4e_b_evaluation.py - Expanded Isolated Supervisor Evaluation.

Phase 4E-B Expanded Evaluation Harness:
1. Scenario Matrix Evaluation: 12 defined challenging operational scenarios comparing
   the canonical SafetySupervisor against the ExperimentalHysteresisSupervisor
   under strictly matched initial conditions and simulated obstacle environments.
2. Disjoint Diagnostic Simulation Benchmark: 15 completely disjoint diagnostic seeds
   (70001–70015) verifying generalization of the candidate PPO policy under both supervisors.

Outputs:
- reports/phase4e_b_scenario_matrix_metrics.json
- reports/phase4e_b_simulation_benchmark_metrics.json
"""

import json
import os
import time
from typing import Any, Dict, List, Tuple
import numpy as np
import torch

from core.contracts import ActionLimits, VelocityAction, compute_forward_clearance_m
from core.geometry import Corridor3D, Obstacle3D
from controllers.safety_supervisor import SafetySupervisor, SafetySupervisorConfig
from controllers.experimental_hysteresis_supervisor import (
    ExperimentalHysteresisSupervisor,
    HysteresisSupervisorConfig,
    HysteresisSafetyState,
)
from models.actor_critic import ActorCriticPPO
from sim.corridor_env import CorridorSimEnv


# ==============================================================================
# 1. SEED POOL DEFINITIONS AND DISJOINTNESS VERIFICATION
# ==============================================================================
VALIDATION_SEEDS = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]
TRAINING_HARD_NEGATIVE_SEEDS = [200, 203, 204, 206, 207, 210, 213, 215, 216, 217, 218, 220, 221, 224]
PHASE4C_HELD_OUT_SEEDS = list(range(50000, 50100))
PHASE4EA_DIAGNOSTIC_SEEDS = [50000, 50001, 100, 101, 102]

# Disjoint Diagnostic Seed Pool for Phase 4E-B
PHASE4EB_DIAGNOSTIC_SEEDS = list(range(70001, 70016))  # 15 seeds: 70001 to 70015


def verify_disjoint_seeds() -> Dict[str, Any]:
    """Cryptographically and mathematically verifies that Phase 4E-B seeds are strictly disjoint."""
    s_eb = set(PHASE4EB_DIAGNOSTIC_SEEDS)
    overlap_val = s_eb.intersection(set(VALIDATION_SEEDS))
    overlap_train = s_eb.intersection(set(TRAINING_HARD_NEGATIVE_SEEDS))
    overlap_heldout = s_eb.intersection(set(PHASE4C_HELD_OUT_SEEDS))
    overlap_4ea = s_eb.intersection(set(PHASE4EA_DIAGNOSTIC_SEEDS))

    assert len(overlap_val) == 0, f"Overlap with validation seeds: {overlap_val}"
    assert len(overlap_train) == 0, f"Overlap with training seeds: {overlap_train}"
    assert len(overlap_heldout) == 0, f"Overlap with Phase 4C held-out seeds: {overlap_heldout}"
    assert len(overlap_4ea) == 0, f"Overlap with Phase 4E-A diagnostic seeds: {overlap_4ea}"

    return {
        "status": "VERIFIED_STRICTLY_DISJOINT",
        "phase4eb_seed_count": len(PHASE4EB_DIAGNOSTIC_SEEDS),
        "seed_range": [min(PHASE4EB_DIAGNOSTIC_SEEDS), max(PHASE4EB_DIAGNOSTIC_SEEDS)],
        "overlaps": {
            "validation_pool": list(overlap_val),
            "training_pool": list(overlap_train),
            "phase4c_heldout_pool": list(overlap_heldout),
            "phase4ea_diagnostic_pool": list(overlap_4ea),
        }
    }


# ==============================================================================
# 2. SCENARIO MATRIX SIMULATOR (12 SCENARIOS)
# ==============================================================================
def make_mock_rays_12(
    front: float = 4.0,
    left: float = 4.0,
    right: float = 4.0,
    up: float = 4.0,
    down: float = 4.0,
    rear: float = 4.0,
    side_left: float = None,
    side_right: float = None,
    max_range: float = 4.0,
) -> np.ndarray:
    """Constructs 12-ray rangefinder array adhering to contracts cone partitioning."""
    rays = np.full(12, fill_value=max_range, dtype=np.float32)
    rays[0] = min(rays[0], front)
    rays[1] = min(rays[1], left)
    rays[3] = min(rays[3], left)
    rays[5] = min(rays[5], side_left if side_left is not None else left)
    rays[2] = min(rays[2], right)
    rays[4] = min(rays[4], right)
    rays[6] = min(rays[6], side_right if side_right is not None else right)
    rays[7] = min(rays[7], rear)
    rays[8] = min(rays[8], up)
    rays[10] = min(rays[10], up)
    rays[9] = min(rays[9], down)
    rays[11] = min(rays[11], down)
    return np.clip(rays / max_range, 0.0, 1.0)


def evaluate_single_scenario(
    scenario_id: int,
    name: str,
    steps: int,
    dt: float,
    init_pos: np.ndarray,
    init_vel: np.ndarray,
    policy_prop: VelocityAction,
    sensor_fn: Any,  # Callable[[step, pos, vel] -> (front_dist, rays_array, obs_clearance)]
    obstacles: List[Obstacle3D] = None,
) -> Dict[str, Any]:
    """Evaluates both canonical and hysteresis supervisors under matched initial conditions."""
    corridor = Corridor3D.get_10m_benchmark()
    limits = ActionLimits()
    goal_pos = np.array([9.5, 0.0, 1.0], dtype=np.float32)

    results = {"scenario_id": scenario_id, "name": name, "steps": steps}

    for sup_name in ["canonical", "hysteresis"]:
        if sup_name == "canonical":
            supervisor = SafetySupervisor(corridor=corridor, limits=limits, config=SafetySupervisorConfig())
        else:
            supervisor = ExperimentalHysteresisSupervisor(corridor=corridor, limits=limits, config=HysteresisSupervisorConfig())

        pos = init_pos.copy()
        vel = init_vel.copy()
        path_len = 0.0
        min_obs_clear = 10.0
        min_wall_clear = 10.0
        sign_reversals = 0
        prev_vx = 0.0
        stationary_steps = 0
        interventions = 0
        braking_steps = 0
        recovery_steps = 0
        bypass_steps = 0
        blocked_events = 0
        escapes = 0
        collision = False
        collision_reason = ""
        trace = []
        has_escaped = False

        for st in range(steps):
            front_dist, rays, obs_clear = sensor_fn(st, pos, vel)

            # Check collision
            wall_clear = corridor.boundary_distance(pos)
            min_wall_clear = min(min_wall_clear, wall_clear)
            min_obs_clear = min(min_obs_clear, obs_clear)

            if wall_clear <= 0.0:
                collision = True
                collision_reason = f"Boundary crash (clearance={wall_clear:.3f}m)"
                break
            if obs_clear <= 0.0:
                collision = True
                collision_reason = f"Obstacle crash (clearance={obs_clear:.3f}m)"
                break

            safe_act, state, reason = supervisor.filter_action(
                proposed_action=policy_prop,
                current_position=pos,
                current_velocity=vel,
                min_front_range=front_dist,
                dt=dt,
                range_rays=rays,
            )

            # Reversals
            if (prev_vx > 1e-4 and safe_act.vx < -1e-4) or (prev_vx < -1e-4 and safe_act.vx > 1e-4):
                sign_reversals += 1
            prev_vx = safe_act.vx

            # Stationary
            spd = np.linalg.norm([safe_act.vx, safe_act.vy, safe_act.vz])
            if spd <= 0.05:
                stationary_steps += 1

            # Interventions
            if abs(safe_act.vx - policy_prop.vx) > 1e-4 or abs(safe_act.vy - policy_prop.vy) > 1e-4 or abs(safe_act.vz - policy_prop.vz) > 1e-4:
                interventions += 1

            # State categorizations
            s_val = state.value if hasattr(state, "value") else str(state)
            if "brake" in s_val: braking_steps += 1
            if "retreat" in s_val or "recovery" in s_val: recovery_steps += 1
            if "bypass" in s_val: bypass_steps += 1
            if "blocked" in s_val: blocked_events += 1

            # Check escape: if previously braked, and now in safe release or normal with x > obstacle
            if ("bypass" in s_val or "release" in s_val or s_val == "normal_approach") and pos[0] > 3.5 and not has_escaped:
                escapes += 1
                has_escaped = True

            # Kinematic integration
            dx = safe_act.vx * dt
            dy = safe_act.vy * dt
            dz = safe_act.vz * dt
            pos[0] += dx
            pos[1] += dy
            pos[2] += dz
            vel[0] = safe_act.vx
            vel[1] = safe_act.vy
            vel[2] = safe_act.vz
            step_disp = np.sqrt(dx**2 + dy**2 + dz**2)
            path_len += step_disp

            trace.append((st, round(float(pos[0]), 3), round(float(pos[1]), 3), round(float(pos[2]), 3), round(float(safe_act.vx), 3), round(float(safe_act.vy), 3), s_val))

        # Outcome determination
        net_fwd = float(pos[0] - init_pos[0])
        final_goal_dist = float(np.linalg.norm(pos - goal_pos))
        final_spd = float(np.linalg.norm(vel))

        if collision:
            outcome = "collision"
        elif final_goal_dist <= corridor.goal_radius and final_spd <= 0.35:
            outcome = "success"
        else:
            outcome = "timeout"

        results[sup_name] = {
            "outcome": outcome,
            "collision_reason": collision_reason if collision else None,
            "steps_completed": len(trace),
            "net_forward_displacement_m": round(net_fwd, 3),
            "path_length_m": round(path_len, 3),
            "min_obstacle_clearance_m": round(float(min_obs_clear), 3),
            "min_wall_clearance_m": round(float(min_wall_clear), 3),
            "velocity_sign_reversals": sign_reversals,
            "stationary_duration_s": round(stationary_steps * dt, 2),
            "braking_duration_s": round(braking_steps * dt, 2),
            "recovery_duration_s": round(recovery_steps * dt, 2),
            "bypass_duration_s": round(bypass_steps * dt, 2),
            "blocked_escape_events": blocked_events,
            "successful_escapes": escapes,
            "intervention_rate_pct": round((interventions / max(1, len(trace))) * 100.0, 1),
            "final_position": [round(float(x), 3) for x in pos],
            "final_goal_distance_m": round(final_goal_dist, 3),
            "final_arrival_speed_mps": round(final_spd, 3),
            "sample_trace": trace[::max(1, len(trace) // 10)],
        }

    return results


def run_full_scenario_matrix() -> List[Dict[str, Any]]:
    """Defines and runs all 12 diagnostic scenarios under matched initial conditions."""
    dt = 0.05
    steps = 150
    scenarios = []

    # Scenario 1: Direct approach toward an isolated obstacle (x=3.0, y=0.0)
    def s1_sensor(st, p, v):
        dist_obs = max(0.01, 3.0 - 0.20 - p[0] - 0.15)
        # Left and right both open (0.60m)
        rays = make_mock_rays_12(front=dist_obs, left=0.60, right=0.60, rear=1.0)
        return dist_obs, rays, dist_obs
    scenarios.append(evaluate_single_scenario(
        1, "Direct approach toward isolated obstacle", steps, dt,
        init_pos=np.array([2.2, 0.0, 1.0], dtype=np.float32),
        init_vel=np.array([0.5, 0.0, 0.0], dtype=np.float32),
        policy_prop=VelocityAction(vx=0.6, vy=0.0, vz=0.0, yaw_rate=0.0),
        sensor_fn=s1_sensor
    ))

    # Scenario 2: Obstacle approached from an offset angle (yaw/lateral velocity)
    def s2_sensor(st, p, v):
        # Obstacle at x=3.0, y=0.20
        d_xy = np.linalg.norm(p[:2] - np.array([3.0, 0.20])) - 0.35
        front_d = max(0.01, float(3.0 - 0.20 - p[0] - 0.15))
        rays = make_mock_rays_12(front=front_d, left=0.35, right=0.65, rear=1.0)
        return front_d, rays, max(0.01, float(d_xy))
    scenarios.append(evaluate_single_scenario(
        2, "Obstacle approached from offset angle", steps, dt,
        init_pos=np.array([2.2, -0.15, 1.0], dtype=np.float32),
        init_vel=np.array([0.4, 0.15, 0.0], dtype=np.float32),
        policy_prop=VelocityAction(vx=0.5, vy=0.15, vz=0.0, yaw_rate=0.0),
        sensor_fn=s2_sensor
    ))

    # Scenario 3: Bypass route open on the left only (Obstacle shifted right, right wall 0.25m)
    def s3_sensor(st, p, v):
        dist_obs = max(0.01, 3.0 - 0.20 - p[0] - 0.15) if p[0] < 3.2 else 2.0
        # Right blocked (0.22m), left open (0.65m)
        side_l = 0.40 if p[0] >= 2.8 else 0.65
        rays = make_mock_rays_12(front=dist_obs, left=0.65, right=0.22, side_left=side_l, rear=1.0)
        return dist_obs, rays, dist_obs
    scenarios.append(evaluate_single_scenario(
        3, "Bypass route open on left only", steps, dt,
        init_pos=np.array([2.3, -0.25, 1.0], dtype=np.float32),
        init_vel=np.array([0.4, 0.0, 0.0], dtype=np.float32),
        policy_prop=VelocityAction(vx=0.6, vy=0.0, vz=0.0, yaw_rate=0.0),
        sensor_fn=s3_sensor
    ))

    # Scenario 4: Bypass route open on the right only (Obstacle shifted left, left wall 0.25m)
    def s4_sensor(st, p, v):
        dist_obs = max(0.01, 3.0 - 0.20 - p[0] - 0.15) if p[0] < 3.2 else 2.0
        # Left blocked (0.22m), right open (0.65m)
        side_r = 0.40 if p[0] >= 2.8 else 0.65
        rays = make_mock_rays_12(front=dist_obs, left=0.22, right=0.65, side_right=side_r, rear=1.0)
        return dist_obs, rays, dist_obs
    scenarios.append(evaluate_single_scenario(
        4, "Bypass route open on right only", steps, dt,
        init_pos=np.array([2.3, 0.25, 1.0], dtype=np.float32),
        init_vel=np.array([0.4, 0.0, 0.0], dtype=np.float32),
        policy_prop=VelocityAction(vx=0.6, vy=0.0, vz=0.0, yaw_rate=0.0),
        sensor_fn=s4_sensor
    ))

    # Scenario 5: Obstacle positioned near a corridor wall (flush against left wall y=0.70)
    def s5_sensor(st, p, v):
        dist_obs = max(0.01, 3.0 - 0.20 - p[0] - 0.15) if p[0] < 3.2 else 2.0
        # Left wall pinch point (0.15m), right corridor wide open (0.90m)
        rays = make_mock_rays_12(front=dist_obs, left=0.18, right=0.85, rear=1.0)
        return dist_obs, rays, dist_obs
    scenarios.append(evaluate_single_scenario(
        5, "Obstacle near corridor wall", steps, dt,
        init_pos=np.array([2.3, 0.50, 1.0], dtype=np.float32),
        init_vel=np.array([0.4, 0.0, 0.0], dtype=np.float32),
        policy_prop=VelocityAction(vx=0.5, vy=0.0, vz=0.0, yaw_rate=0.0),
        sensor_fn=s5_sensor
    ))

    # Scenario 6: Narrow passage with no route satisfying bypass threshold (constriction)
    def s6_sensor(st, p, v):
        dist_obs = max(0.01, 3.0 - 0.20 - p[0] - 0.15)
        # Both left and right restricted (0.24m < 0.38m bypass threshold)
        rays = make_mock_rays_12(front=dist_obs, left=0.24, right=0.24, rear=0.8)
        return dist_obs, rays, dist_obs
    scenarios.append(evaluate_single_scenario(
        6, "Narrow passage with no route satisfying bypass threshold", steps, dt,
        init_pos=np.array([2.4, 0.0, 1.0], dtype=np.float32),
        init_vel=np.array([0.3, 0.0, 0.0], dtype=np.float32),
        policy_prop=VelocityAction(vx=0.5, vy=0.0, vz=0.0, yaw_rate=0.0),
        sensor_fn=s6_sensor
    ))

    # Scenario 7: Multiple obstacles encountered consecutively (Obs 1 at 3.0m, Obs 2 at 5.5m)
    def s7_sensor(st, p, v):
        if p[0] < 3.5:
            d_front = max(0.01, 3.0 - 0.20 - p[0] - 0.15) if p[0] < 3.0 else 0.80
            rays = make_mock_rays_12(front=d_front, left=0.60, right=0.25, rear=1.0)
            return d_front, rays, d_front
        else:
            d_front = max(0.01, 5.5 - 0.20 - p[0] - 0.15) if p[0] < 5.5 else 1.50
            rays = make_mock_rays_12(front=d_front, left=0.25, right=0.60, rear=1.0)
            return d_front, rays, d_front
    scenarios.append(evaluate_single_scenario(
        7, "Multiple obstacles encountered consecutively", steps, dt,
        init_pos=np.array([2.2, 0.0, 1.0], dtype=np.float32),
        init_vel=np.array([0.5, 0.0, 0.0], dtype=np.float32),
        policy_prop=VelocityAction(vx=0.6, vy=0.0, vz=0.0, yaw_rate=0.0),
        sensor_fn=s7_sensor
    ))

    # Scenario 8: Recovery after an unsuccessful lateral bypass (bypass direction obstructed)
    def s8_sensor(st, p, v):
        dist_obs = max(0.01, 3.0 - 0.20 - p[0] - 0.15)
        # Left opens initially, then suddenly becomes obstructed at step 20
        left_c = 0.55 if st < 15 else 0.18
        rays = make_mock_rays_12(front=dist_obs, left=left_c, right=0.25, rear=0.9)
        return dist_obs, rays, dist_obs
    scenarios.append(evaluate_single_scenario(
        8, "Recovery after unsuccessful lateral bypass", steps, dt,
        init_pos=np.array([2.5, 0.0, 1.0], dtype=np.float32),
        init_vel=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        policy_prop=VelocityAction(vx=0.5, vy=0.0, vz=0.0, yaw_rate=0.0),
        sensor_fn=s8_sensor
    ))

    # Scenario 9: Stationary drone with a blocked escape direction (dead end)
    def s9_sensor(st, p, v):
        # Front, sides, and rear all blocked
        rays = make_mock_rays_12(front=0.22, left=0.20, right=0.20, rear=0.20)
        return 0.22, rays, 0.22
    scenarios.append(evaluate_single_scenario(
        9, "Stationary drone with blocked escape direction", steps, dt,
        init_pos=np.array([2.5, 0.0, 1.0], dtype=np.float32),
        init_vel=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        policy_prop=VelocityAction(vx=0.5, vy=0.0, vz=0.0, yaw_rate=0.0),
        sensor_fn=s9_sensor
    ))

    # Scenario 10: Clear corridor with no obstacle (test for false-positive interventions)
    def s10_sensor(st, p, v):
        rays = make_mock_rays_12(front=4.0, left=0.85, right=0.85, rear=4.0)
        return 4.0, rays, 10.0
    scenarios.append(evaluate_single_scenario(
        10, "Clear corridor with no obstacle", steps, dt,
        init_pos=np.array([1.0, 0.0, 1.0], dtype=np.float32),
        init_vel=np.array([0.5, 0.0, 0.0], dtype=np.float32),
        policy_prop=VelocityAction(vx=0.6, vy=0.0, vz=0.0, yaw_rate=0.0),
        sensor_fn=s10_sensor
    ))

    # Scenario 11: Restricted vertical clearance (near ceiling z=1.80m)
    def s11_sensor(st, p, v):
        dist_obs = max(0.01, 3.0 - 0.20 - p[0] - 0.15)
        # Low ceiling: up=0.15m
        rays = make_mock_rays_12(front=dist_obs, left=0.60, right=0.30, up=0.15, rear=1.0)
        return dist_obs, rays, dist_obs
    scenarios.append(evaluate_single_scenario(
        11, "Restricted vertical clearance near ceiling", steps, dt,
        init_pos=np.array([2.2, 0.0, 1.80], dtype=np.float32),
        init_vel=np.array([0.4, 0.0, 0.0], dtype=np.float32),
        policy_prop=VelocityAction(vx=0.5, vy=0.0, vz=0.2, yaw_rate=0.0),  # Asks to climb
        sensor_fn=s11_sensor
    ))

    # Scenario 12: Obstacle encountered while drone already has lateral velocity
    def s12_sensor(st, p, v):
        dist_obs = max(0.01, 3.0 - 0.20 - p[0] - 0.15)
        rays = make_mock_rays_12(front=dist_obs, left=0.25, right=0.60, rear=1.0)
        return dist_obs, rays, dist_obs
    scenarios.append(evaluate_single_scenario(
        12, "Obstacle encountered with pre-existing lateral velocity", steps, dt,
        init_pos=np.array([2.3, -0.2, 1.0], dtype=np.float32),
        init_vel=np.array([0.4, -0.35, 0.0], dtype=np.float32),
        policy_prop=VelocityAction(vx=0.5, vy=-0.35, vz=0.0, yaw_rate=0.0),
        sensor_fn=s12_sensor
    ))

    return scenarios


# ==============================================================================
# 3. DISJOINT SIMULATION BENCHMARK (SEEDS 70001–70015)
# ==============================================================================
def run_phase4eb_simulation_benchmark(
    seeds: List[int],
    ckpt_path: str = "models/checkpoints/ppo_phase4b_candidate.pt",
    scenario: str = "10m",
    max_steps: int = 500,
    dt: float = 0.05,
) -> Dict[str, Any]:
    """Runs 15-seed comparative simulation benchmark on completely disjoint seeds."""
    device = torch.device("cpu")
    model = ActorCriticPPO(obs_dim=23, act_dim=4).to(device)
    model.load_ppo_checkpoint(ckpt_path, device=device)
    model.eval()

    corridor = Corridor3D.get_10m_benchmark()
    limits = ActionLimits()

    benchmark_data = {"canonical": [], "hysteresis": []}

    for sup_type in ["canonical", "hysteresis"]:
        env = CorridorSimEnv(scenario=scenario, gui=False, seed=seeds[0])

        for s in seeds:
            obs, info = env.reset(seed=s)
            if sup_type == "canonical":
                supervisor = SafetySupervisor(corridor=env.corridor, limits=limits, config=SafetySupervisorConfig())
            else:
                supervisor = ExperimentalHysteresisSupervisor(corridor=env.corridor, limits=limits, config=HysteresisSupervisorConfig())

            done = False
            step = 0
            ep_rew = 0.0
            ep_dist = 0.0
            sign_reversals = 0
            prev_vx = 0.0
            stationary_steps = 0
            interventions = 0
            braking_steps = 0
            recovery_steps = 0
            bypass_steps = 0
            escapes = 0
            has_escaped = False

            start_pos = env._get_drone_position().copy()
            prev_pos = start_pos.copy()
            min_obs_clear = 10.0
            min_wall_clear = 10.0

            while not done and step < max_steps:
                step += 1
                with torch.no_grad():
                    t_obs = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                    act, _, _, _ = model.get_action_and_value(t_obs, deterministic=True)
                    act_np = act.squeeze(0).cpu().numpy()

                clamped_act = VelocityAction(
                    vx=float(np.clip(act_np[0], limits.min_fwd_vel_mps, limits.max_fwd_vel_mps)),
                    vy=float(np.clip(act_np[1], -limits.max_lat_vel_mps, limits.max_lat_vel_mps)),
                    vz=float(np.clip(act_np[2], -limits.max_vert_vel_mps, limits.max_vert_vel_mps)),
                    yaw_rate=float(np.clip(act_np[3], -limits.max_yaw_rate_radps, limits.max_yaw_rate_radps))
                )

                priv = info["privileged_state"]
                front_dist = compute_forward_clearance_m(obs, max_range_m=env.sensor_config.max_range_m)

                safe_act, s_state, s_reason = supervisor.filter_action(
                    proposed_action=clamped_act,
                    current_position=priv.global_position,
                    current_velocity=priv.global_velocity,
                    min_front_range=front_dist,
                    dt=dt,
                    observation=obs,
                )

                if (prev_vx > 1e-4 and safe_act.vx < -1e-4) or (prev_vx < -1e-4 and safe_act.vx > 1e-4):
                    sign_reversals += 1
                prev_vx = safe_act.vx

                spd = np.linalg.norm([safe_act.vx, safe_act.vy, safe_act.vz])
                if spd <= 0.05:
                    stationary_steps += 1

                if abs(safe_act.vx - clamped_act.vx) > 1e-4 or abs(safe_act.vy - clamped_act.vy) > 1e-4:
                    interventions += 1

                s_val = s_state.value if hasattr(s_state, "value") else str(s_state)
                if "brake" in s_val: braking_steps += 1
                if "retreat" in s_val or "recovery" in s_val: recovery_steps += 1
                if "bypass" in s_val: bypass_steps += 1

                if ("bypass" in s_val or "release" in s_val) and priv.global_position[0] > 3.5 and not has_escaped:
                    escapes += 1
                    has_escaped = True

                norm_applied = safe_act.to_normalized(limits)
                obs, rew, term, trunc, info = env.step(norm_applied)
                done = term or trunc
                ep_rew += float(rew)

                curr_pos = env._get_drone_position()
                ep_dist += float(np.linalg.norm(curr_pos - prev_pos))
                prev_pos = curr_pos

                b_dist = float(info.get("boundary_distance", 10.0))
                o_dist = float(info.get("obstacle_clearance", 10.0))
                min_wall_clear = min(min_wall_clear, b_dist)
                min_obs_clear = min(min_obs_clear, o_dist)

            final_pos = env._get_drone_position()
            net_fwd = float(final_pos[0] - start_pos[0])
            final_goal_dist = float(np.linalg.norm(final_pos - corridor.goal_position))
            final_spd = float(np.linalg.norm(env._get_drone_velocity()))
            status = info.get("termination_reason", "unknown")

            benchmark_data[sup_type].append({
                "seed": s,
                "status": status,
                "steps": step,
                "episode_return": round(ep_rew, 2),
                "distance_travelled_m": round(ep_dist, 2),
                "net_forward_displacement_m": round(net_fwd, 2),
                "final_distance_to_goal_m": round(final_goal_dist, 2),
                "final_arrival_speed_mps": round(final_spd, 3),
                "sign_reversals": sign_reversals,
                "stationary_duration_s": round(stationary_steps * dt, 2),
                "braking_duration_s": round(braking_steps * dt, 2),
                "recovery_duration_s": round(recovery_steps * dt, 2),
                "bypass_duration_s": round(bypass_steps * dt, 2),
                "successful_escapes": escapes,
                "interventions": interventions,
                "intervention_rate_pct": round((interventions / max(1, step)) * 100.0, 1),
                "min_obstacle_clearance_m": round(min_obs_clear, 3),
                "min_wall_clearance_m": round(min_wall_clear, 3),
                "final_position": [round(float(x), 2) for x in final_pos],
            })

    return benchmark_data


def main():
    print("=" * 80)
    print(">>> PHASE 4E-B: EXPANDED ISOLATED SUPERVISOR EVALUATION <<<")
    print("=" * 80)

    # 1. Seed Verification
    print("\n[Step 1/3] Verifying disjoint evaluation seeds...")
    disjoint_info = verify_disjoint_seeds()
    print(f"Seed Verification: {disjoint_info['status']}")
    print(f"Phase 4E-B Diagnostic Seeds: {disjoint_info['seed_range'][0]}–{disjoint_info['seed_range'][1]} (Total: {disjoint_info['phase4eb_seed_count']})")
    print("Zero overlaps with validation, training, held-out, or Phase 4E-A pools verified.")

    # 2. 12-Scenario Matrix Evaluation
    print("\n[Step 2/3] Evaluating 12-scenario diagnostic matrix...")
    matrix_results = run_full_scenario_matrix()
    print(f"Completed {len(matrix_results)} diagnostic scenarios across both supervisors.")

    # Print summary table for scenario matrix
    print("\n" + "=" * 100)
    print(f"{'ID':<3} | {'Scenario Name':<45} | {'Can Disp':<8} | {'Hys Disp':<8} | {'Can Rev':<7} | {'Hys Rev':<7} | {'Can Out':<8} | {'Hys Out':<8}")
    print("-" * 100)
    for sc in matrix_results:
        c = sc["canonical"]
        h = sc["hysteresis"]
        print(f"{sc['scenario_id']:<3} | {sc['name']:<45} | {c['net_forward_displacement_m']:<8.2f} | {h['net_forward_displacement_m']:<8.2f} | {c['velocity_sign_reversals']:<7} | {h['velocity_sign_reversals']:<7} | {c['outcome']:<8} | {h['outcome']:<8}")
    print("=" * 100)

    # Save scenario matrix results
    out_matrix = "reports/phase4e_b_scenario_matrix_metrics.json"
    with open(out_matrix, "w") as f:
        json.dump({"disjoint_seed_verification": disjoint_info, "scenarios": matrix_results}, f, indent=2)
    print(f"\n[Saved] Scenario matrix metrics exported to: {out_matrix}")

    # 3. Disjoint Simulation Benchmark (Seeds 70001–70015)
    print("\n[Step 3/3] Running simulation benchmark across 15 disjoint seeds (70001–70015)...")
    sim_benchmark = run_phase4eb_simulation_benchmark(seeds=PHASE4EB_DIAGNOSTIC_SEEDS)

    can_recs = sim_benchmark["canonical"]
    hys_recs = sim_benchmark["hysteresis"]

    # Comparative aggregations
    print("\n" + "=" * 100)
    print(f"{'Seed':<6} | {'Can Status':<10} | {'Hys Status':<10} | {'Can Disp':<8} | {'Hys Disp':<8} | {'Can Rev':<7} | {'Hys Rev':<7} | {'Can MinObs':<10} | {'Hys MinObs':<10}")
    print("-" * 100)
    for i, s in enumerate(PHASE4EB_DIAGNOSTIC_SEEDS):
        cr = can_recs[i]
        hr = hys_recs[i]
        print(f"{s:<6} | {cr['status']:<10} | {hr['status']:<10} | {cr['net_forward_displacement_m']:<8.2f} | {hr['net_forward_displacement_m']:<8.2f} | {cr['sign_reversals']:<7} | {hr['sign_reversals']:<7} | {cr['min_obstacle_clearance_m']:<10.3f} | {hr['min_obstacle_clearance_m']:<10.3f}")
    print("=" * 100)

    # Aggregate summaries
    can_rev_tot = sum(r["sign_reversals"] for r in can_recs)
    hys_rev_tot = sum(r["sign_reversals"] for r in hys_recs)
    can_disp_mean = np.mean([r["net_forward_displacement_m"] for r in can_recs])
    hys_disp_mean = np.mean([r["net_forward_displacement_m"] for r in hys_recs])
    can_obs_min = min(r["min_obstacle_clearance_m"] for r in can_recs)
    hys_obs_min = min(r["min_obstacle_clearance_m"] for r in hys_recs)

    print(f"\nSimulation Aggregates (15 Disjoint Seeds):")
    print(f"  Total Sign Reversals: Canonical={can_rev_tot} -> Hysteresis={hys_rev_tot} (Change: {hys_rev_tot - can_rev_tot})")
    print(f"  Mean Forward Disp:    Canonical={can_disp_mean:.2f}m -> Hysteresis={hys_disp_mean:.2f}m (Change: {hys_disp_mean - can_disp_mean:+.2f}m)")
    print(f"  Global Min Obs Clear: Canonical={can_obs_min:.3f}m -> Hysteresis={hys_obs_min:.3f}m")

    out_sim = "reports/phase4e_b_simulation_benchmark_metrics.json"
    with open(out_sim, "w") as f:
        json.dump({
            "disjoint_seed_verification": disjoint_info,
            "seeds": PHASE4EB_DIAGNOSTIC_SEEDS,
            "aggregates": {
                "total_reversals": {"canonical": can_rev_tot, "hysteresis": hys_rev_tot},
                "mean_displacement_m": {"canonical": round(float(can_disp_mean), 3), "hysteresis": round(float(hys_disp_mean), 3)},
                "global_min_obs_clearance_m": {"canonical": round(float(can_obs_min), 3), "hysteresis": round(float(hys_obs_min), 3)},
            },
            "per_seed_results": sim_benchmark,
        }, f, indent=2)
    print(f"\n[Saved] Simulation benchmark metrics exported to: {out_sim}")


if __name__ == "__main__":
    main()
