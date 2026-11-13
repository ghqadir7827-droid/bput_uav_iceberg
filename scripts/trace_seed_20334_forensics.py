"""trace_seed_20334_forensics.py - Detailed Forensic Trace of Seed 20334 for v2, v3, and v4.

Audits:
1. Physical simulated position vs estimated observations.
2. Obstacle surface clearance vs wall clearance vs center distance.
3. UAV footprint (0.65m tip-to-tip diagonal, 0.325m radius) and obstacle radius (0.25m).
4. Envelope breach events (d_obs < 0.15m or d_wall < 0.15m) and supervisor interventions.
5. Exact per-step time series around the minimum clearance event.
"""

import json
from pathlib import Path
from typing import Any, Dict, List
import numpy as np

from controllers.registry import ControllerRegistry
from controllers.safety_supervisor import SafetySupervisor
from core.contracts import compute_forward_clearance_m
from core.geometry import TerminationReason
from core.physical_dimensions import DifficultyLevel, PhysicalDroneModel, RouteFeasibilityChecker, ScaledEnvironmentFactory
from sim.corridor_env import CorridorSimEnv


def trace_controller_on_seed_20334(controller_id: str) -> Dict[str, Any]:
    cfg = ScaledEnvironmentFactory.get_environment_config(DifficultyLevel.LEVEL_4_FEASIBILITY_BOUNDARY)
    env = CorridorSimEnv(scenario=cfg, seed=20334)
    obs, _ = env.reset(seed=20334)
    corridor = env.corridor

    registry = ControllerRegistry()
    controller = registry.get_controller(controller_id)
    supervisor = SafetySupervisor(corridor=cfg)

    checker = RouteFeasibilityChecker(PhysicalDroneModel(tip_to_tip_diagonal_m=0.65, collision_radius_m=0.325))
    feas = checker.evaluate_feasibility(corridor, env.obstacles)

    trace = []
    total_steps = 0
    min_obs_clearance = float("inf")
    min_wall_clearance = float("inf")
    min_center_to_obs_dist = float("inf")
    min_step_idx = -1
    envelope_violation_steps = 0
    safety_interventions = 0
    dt = env.dt

    outcome = "TIMEOUT"
    uav_radius = float(corridor.drone_radius)  # 0.325m

    for step in range(1000):
        total_steps += 1
        pos = env._get_drone_position()
        vel = env._get_drone_velocity()

        # Analytical True Physical Surface Clearances
        # 1. Obstacle clearance:
        # Distance from drone center to obstacle center minus (drone_radius + obs_radius)
        # Note: in corridor.obstacle_clearance(pos, env.obstacles):
        # clearance = min(dist_xy - (obs.radius + self.drone_radius))
        d_obs_surf = float(corridor.obstacle_clearance(pos, env.obstacles))
        
        # Also compute center-to-center distance to closest obstacle:
        obs_center_dists = [
            float(np.linalg.norm(pos[:2] - obs.center[:2]))
            for obs in env.obstacles
        ]
        closest_obs_idx = int(np.argmin(obs_center_dists))
        d_center_to_obs = obs_center_dists[closest_obs_idx]
        closest_obs = env.obstacles[closest_obs_idx]

        # 2. Wall clearance:
        d_wall_surf = float(corridor.boundary_distance(pos))
        d_wall_l, d_wall_r, d_ceil, d_floor = corridor.boundary_clearances(pos)

        if d_obs_surf < min_obs_clearance:
            min_obs_clearance = d_obs_surf
            min_step_idx = step
            min_center_to_obs_dist = d_center_to_obs

        min_wall_clearance = min(min_wall_clearance, d_wall_surf)

        is_envelope_violation = (d_obs_surf < 0.15) or (d_wall_surf < 0.15)
        if is_envelope_violation:
            envelope_violation_steps += 1

        term_reason, _ = corridor.check_termination(pos, vel, env.obstacles, step_count=step, max_steps=1000)
        if term_reason == TerminationReason.SUCCESS:
            outcome = "SUCCESS"
            break
        elif term_reason == TerminationReason.CRASH_OBSTACLE:
            outcome = "CRASH_OBSTACLE"
            break
        elif term_reason == TerminationReason.CRASH_BOUNDARY:
            outcome = "CRASH_BOUNDARY"
            break

        # Controller action
        raw_act = controller.predict(obs)
        min_front = compute_forward_clearance_m(obs, max_range_m=env.sensor_config.max_range_m)

        # Guidance diagnostics
        arb_dir = "NONE"
        if hasattr(controller, "arbitrator"):
            arb_dir = controller.arbitrator.last_dir

        # Supervisor filtering
        safe_act, sup_state, sup_reason = supervisor.filter_action(
            proposed_action=raw_act,
            current_position=pos,
            current_velocity=vel,
            min_front_range=min_front,
            dt=dt,
            is_localization_healthy=True,
            telemetry_age_s=0.0,
            observation=obs,
        )

        sup_intervened = (sup_state.value != "nominal")
        if sup_intervened:
            safety_interventions += 1

        step_record = {
            "step": step,
            "time_s": round(step * dt, 3),
            "pos": [round(float(p), 4) for p in pos],
            "vel": [round(float(v), 4) for v in vel],
            "d_obs_surf_m": round(d_obs_surf, 4),
            "d_center_to_obs_m": round(d_center_to_obs, 4),
            "closest_obs_idx": closest_obs_idx,
            "closest_obs_center": [round(float(c), 4) for c in closest_obs.center],
            "closest_obs_radius": closest_obs.radius,
            "d_wall_surf_m": round(d_wall_surf, 4),
            "d_wall_left_m": round(d_wall_l, 4),
            "d_wall_right_m": round(d_wall_r, 4),
            "raw_act": [round(raw_act.vx, 3), round(raw_act.vy, 3)],
            "safe_act": [round(safe_act.vx, 3), round(safe_act.vy, 3)],
            "arb_dir": arb_dir,
            "sup_state": sup_state.value,
            "sup_reason": sup_reason,
            "sup_intervened": sup_intervened,
            "is_envelope_violation": is_envelope_violation,
        }
        trace.append(step_record)

        norm_act = safe_act.to_normalized(env.action_limits)
        obs, _, terminated, truncated, info = env.step(norm_act)
        if terminated or truncated:
            term_val = info.get("termination_reason", "")
            if term_val == "success":
                outcome = "SUCCESS"
            elif term_val == "crash_obstacle":
                outcome = "CRASH_OBSTACLE"
            elif term_val == "crash_boundary":
                outcome = "CRASH_BOUNDARY"
            elif truncated:
                outcome = "TIMEOUT"
            break

    env.close()

    return {
        "controller_id": controller_id,
        "seed": 20334,
        "outcome": outcome,
        "total_steps": total_steps,
        "flight_time_s": round(total_steps * dt, 2),
        "min_obs_clearance_m": round(min_obs_clearance, 4),
        "min_center_to_obs_dist_m": round(min_center_to_obs_dist, 4),
        "min_wall_clearance_m": round(min_wall_clearance, 4),
        "min_step_idx": min_step_idx,
        "envelope_violation_steps": envelope_violation_steps,
        "safety_interventions": safety_interventions,
        "uav_radius_m": uav_radius,
        "trace": trace,
    }


def main():
    controllers = ["scaled_guided_bc_v2", "scaled_guided_bc_v3", "scaled_guided_bc_v4"]
    results = {}
    for cid in controllers:
        print(f"Tracing {cid} on Seed 20334...")
        results[cid] = trace_controller_on_seed_20334(cid)
        print(f"  {cid}: outcome={results[cid]['outcome']}, steps={results[cid]['total_steps']}, min_obs_clearance={results[cid]['min_obs_clearance_m']}m at step {results[cid]['min_step_idx']}, envelope_vio_steps={results[cid]['envelope_violation_steps']}")

    out_file = Path("experiments/scaled_campaign/adaptive_guidance/seed_20334_forensic_trace.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved full forensic trace to {out_file}")


if __name__ == "__main__":
    main()

