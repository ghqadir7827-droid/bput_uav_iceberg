"""smoke_test.py - Milestone 1 End-to-End Simulation Smoke Test.

Executes a complete closed-loop rollout integrating:
- Headless PyBullet 3D corridor environment
- Multi-beam raycast distance sensors
- Modular localization interface
- Deterministic Safety Supervisor filtering commands

Measures actual step times, boundary clearance, and verifies zero crashes.
"""

import sys
import time
import numpy as np

from controllers.safety_supervisor import SafetyState, SafetySupervisor
from core.contracts import VelocityAction, compute_forward_clearance_m
from localization.interface import LocalizationManager
from localization.sim_source import SimGroundTruthSource
from sim.corridor_env import CorridorSimEnv


def run_smoke_test(num_steps: int = 50) -> None:
    print("=" * 65)
    print(">>> STARTING MILESTONE 1 SMOKE TEST <<<")
    print("=" * 65)

    # 1. Initialize Simulator
    print("\n[1/4] Initializing PyBullet 3D Corridor Environment (Headless)...")
    env = CorridorSimEnv(gui=False, seed=100)
    obs, info = env.reset(seed=100)
    print(f"      Initial Observation Shape: {obs.shape}")
    print(f"      Corridor Dimensions: Length={env.corridor.x_end}m, Width={env.corridor.width}m, Height={env.corridor.height}m")
    print(f"      Obstacles Spawned: {len(env.obstacles)} pillars")

    # 2. Initialize Localization Layer
    print("\n[2/4] Initializing Modular Localization Manager...")
    sim_loc_source = SimGroundTruthSource()
    loc_mgr = LocalizationManager(primary_source=sim_loc_source, allow_dynamic_switching=False)
    print(f"      Active Source: {loc_mgr.active_source_name} (dynamic switching: DISABLED)")

    # 3. Initialize Safety Supervisor
    print("\n[3/4] Initializing Deterministic Safety Supervisor...")
    safety_sup = SafetySupervisor(corridor=env.corridor)
    print(f"      Boundary Buffer: {safety_sup.config.boundary_buffer_m}m")
    print(f"      Critical Obstacle Brake Threshold: {safety_sup.config.critical_obstacle_dist_m}m")

    # 4. Execute Closed-Loop Simulation Rollout
    print(f"\n[4/4] Executing {num_steps}-step Closed-Loop Flight Rollout...")
    step_times = []
    crashes = 0
    safety_interventions = 0

    t_start_total = time.perf_counter()

    for step in range(1, num_steps + 1):
        t_step_start = time.perf_counter()

        priv_state = info["privileged_state"]
        sim_loc_source.update_from_sim(
            priv_state.global_position,
            priv_state.global_velocity,
            priv_state.global_orientation_quat,
            sim_time_s=env.current_sim_time
        )

        pkt, health = loc_mgr.get_state(current_time_s=env.current_sim_time)
        assert health.is_healthy, f"Localization failure at step {step}: {health.error_message}"

        # Propose forward flight command
        proposed_action = VelocityAction(vx=0.45, vy=0.0, vz=0.0, yaw_rate=0.0)

        # Multi-beam front range across forward rays (indices 0..4)
        front_dist_m = compute_forward_clearance_m(obs, max_range_m=env.sensor_config.max_range_m)

        # Safety Supervisor Filter
        safe_action, safety_state, reason = safety_sup.filter_action(
            proposed_action=proposed_action,
            current_position=pkt.position,
            current_velocity=pkt.velocity,
            min_front_range=front_dist_m,
            dt=env.dt,
            is_localization_healthy=health.is_healthy,
            telemetry_age_s=health.packet_age_s,
            observation=obs,
        )

        if safety_state != SafetyState.NOMINAL:
            safety_interventions += 1

        # Step Simulation
        norm_action = safe_action.to_normalized(env.action_limits)
        obs, reward, terminated, truncated, info = env.step(norm_action)

        t_step = (time.perf_counter() - t_step_start) * 1000.0 # ms
        step_times.append(t_step)

        if terminated:
            if info.get("termination_reason") in ("crash_boundary", "crash_obstacle"):
                crashes += 1
            print(f"      Episode terminated at step {step}: {info.get('termination_detail')}")
            break

        if step % 10 == 0 or step == 1:
            print(f"      Step {step:02d} | Pos: X={pkt.position[0]:.2f}m Y={pkt.position[1]:.2f}m Z={pkt.position[2]:.2f}m | "
                  f"Clearance: {info['boundary_distance']:.2f}m | FrontRay: {front_dist_m:.2f}m | "
                  f"Safety: {safety_state.value} | StepTime: {t_step:.2f}ms")

    t_total = time.perf_counter() - t_start_total
    env.close()

    # Metrics Summary
    mean_step_time_ms = float(np.mean(step_times))
    fps = 1000.0 / mean_step_time_ms if mean_step_time_ms > 0 else 0.0

    print("\n" + "=" * 65)
    print(">>> SMOKE TEST RESULTS SUMMARY <<<")
    print("=" * 65)
    print(f"Steps Completed:          {len(step_times)} / {num_steps}")
    print(f"Crashes:                  {crashes} (Expected: 0)")
    print(f"Safety Interventions:     {safety_interventions}")
    print(f"Total Rollout Time:       {t_total:.3f} s")
    print(f"Mean Step Latency:        {mean_step_time_ms:.3f} ms")
    print(f"Measured Simulation Rate: {fps:.1f} steps/second")
    print("=" * 65)

    assert crashes == 0, f"Smoke test failed: {crashes} crashes encountered!"
    print(">>> SMOKE TEST PASSED SUCCESSFULLY (Exit Code 0) <<<\n")


if __name__ == "__main__":
    run_smoke_test()
