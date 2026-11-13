"""trace_episode_telemetry.py - Step-by-Step Telemetry Diagnostic for Failing Seeds.

Diagnoses exact interaction between BC policy, Guidance Arbitrator, and Safety Supervisor
on representative failing seeds (e.g. 20000 at Obstacle 3, 20016 at Obstacle 1).
"""

import argparse
from pathlib import Path
import numpy as np

from controllers.bc_policy import BehaviorCloningPolicyController
from controllers.safety_supervisor import SafetySupervisor
from core.contracts import VelocityAction, compute_forward_clearance_m
from core.guidance import GuidanceConfig, StatelessClearanceGuidanceArbitrator
from core.physical_dimensions import DifficultyLevel, ScaledEnvironmentFactory
from sim.corridor_env import CorridorSimEnv


def trace_seed(seed: int = 20000, max_steps: int = 1000):
    print("=" * 80)
    print(f">>> STEP-BY-STEP TELEMETRY TRACE FOR SEED {seed} <<<")
    print("=" * 80)

    cfg = ScaledEnvironmentFactory.get_environment_config(DifficultyLevel.LEVEL_2_STANDARD)
    env = CorridorSimEnv(scenario=cfg, seed=seed)
    drone_radius = 0.325
    guidance_cfg = GuidanceConfig.for_65cm_uav(drone_radius)
    arbitrator = StatelessClearanceGuidanceArbitrator(config=guidance_cfg)
    bc = BehaviorCloningPolicyController("models/checkpoints/bc_policy_scaled_v1.pt")
    supervisor = SafetySupervisor(env.corridor)

    obs, _ = env.reset(seed=seed)
    arbitrator.reset()
    corridor = env.corridor

    print(f"Corridor: width={corridor.width}m, height={corridor.height}m, length={corridor.x_end}m")
    print("Obstacles:")
    for idx, o in enumerate(env.obstacles):
        print(f"  Obs {idx+1}: center=({o.center[0]:.2f}, {o.center[1]:.2f}), radius={o.radius:.2f}m")

    print("\nTracing trajectory... (logging steps with significant events or every 50 steps)")
    print(f"{'Step':<5} | {'Pos (x, y)':<16} | {'Vel (vx, vy)':<16} | {'BC act':<16} | {'Guided act':<18} | {'Superv act':<18} | {'Sup State':<14} | {'Obs/Wall Clr':<14}")
    print("-" * 125)

    brake_count = 0
    guidance_overrides = 0
    stuck_steps = 0

    for step in range(1, max_steps + 1):
        pos = env._get_drone_position()
        vel = env._get_drone_velocity()

        d_obs = corridor.obstacle_clearance(pos, env.obstacles)
        d_wall = corridor.boundary_distance(pos)

        # 1. Base BC action
        raw_act = bc.predict(obs)

        # 2. Guidance arbitration
        guided_act, direction, cl, cr = arbitrator.arbitrate(raw_act, obs)
        if direction != "NONE":
            guidance_overrides += 1

        # 3. Safety Supervisor filtering
        min_front = compute_forward_clearance_m(obs, max_range_m=env.sensor_config.max_range_m)
        safe_act, state, _ = supervisor.filter_action(
            proposed_action=guided_act,
            current_position=pos,
            current_velocity=vel,
            min_front_range=min_front,
            dt=env.dt,
            is_localization_healthy=True,
            telemetry_age_s=0.0,
            observation=obs,
        )

        if "brake" in state.value.lower() or "stop" in state.value.lower():
            brake_count += 1

        speed = float(np.linalg.norm(vel[:2]))
        if speed < 0.05 and pos[0] > 1.0:
            stuck_steps += 1

        # Print periodic or when stuck begins
        should_print = (
            step <= 10 or
            step % 50 == 0 or
            (step > 200 and step % 20 == 0 and speed < 0.08) or
            d_obs < 0.20 or
            state.value != "nominal" and step % 25 == 0
        )

        if should_print:
            pos_str = f"({pos[0]:.2f}, {pos[1]:.2f})"
            vel_str = f"({vel[0]:.2f}, {vel[1]:.2f})"
            bc_str = f"({raw_act.vx:.2f}, {raw_act.vy:.2f})"
            gd_str = f"({guided_act.vx:.2f}, {guided_act.vy:.2f}) [{direction[:4]}]"
            sp_str = f"({safe_act.vx:.2f}, {safe_act.vy:.2f})"
            clr_str = f"{d_obs:.2f}m / {d_wall:.2f}m"
            print(f"{step:<5} | {pos_str:<16} | {vel_str:<16} | {bc_str:<16} | {gd_str:<18} | {sp_str:<18} | {state.value:<14} | {clr_str:<14}")

        norm_act = safe_act.to_normalized(env.action_limits)
        obs, _, term, trunc, info = env.step(norm_act)

        if term or trunc:
            print("-" * 125)
            print(f"Terminated at step {step}: reason={info.get('termination_reason', 'unknown')}")
            print(f"Final Pos: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}), Dist to Goal: {np.linalg.norm(pos - corridor.goal_position):.2f}m")
            print(f"Total brake steps: {brake_count}, Guidance overrides: {guidance_overrides}, Stuck steps (speed < 0.05): {stuck_steps}")
            break

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20000)
    args = parser.parse_args()
    trace_seed(args.seed)

