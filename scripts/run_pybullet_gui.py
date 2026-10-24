"""run_pybullet_gui.py - Native PyBullet 3D Graphical Viewer & Evaluator.

Opens a native PyBullet OpenGL GUI window on macOS, visualizes the 3D corridor,
staggered obstacles, destination target, drone rigid body, real-time sensor rays,
dynamic trajectory trail, and live flight telemetry HUD.

Usage:
    python scripts/run_pybullet_gui.py --controller reactive_baseline --seed 100
"""

import argparse
from pathlib import Path
import sys
import time
from typing import List, Optional
import numpy as np
import pybullet as p
from PIL import Image

from controllers.registry import ControllerRegistry
from controllers.safety_supervisor import SafetyState, SafetySupervisor
from core.contracts import VelocityAction
from localization.interface import LocalizationManager
from localization.sim_source import SimGroundTruthSource
from sim.corridor_env import CorridorSimEnv


def run_gui(
    controller_id: str = "reactive_baseline",
    scenario: str = "10m",
    seed: int = 100,
    max_steps: int = 800,
    time_scale: float = 1.0,
    save_screenshot: Optional[str] = None,
    headless_capture: bool = False,
) -> None:
    print("=" * 70)
    print(">>> LAUNCHING NATIVE PYBULLET 3D GRAPHICAL VIEWER <<<")
    print(f"    Controller:  {controller_id}")
    print(f"    Scenario:    {scenario}")
    print(f"    Seed:        {seed}")
    print(f"    Max Steps:   {max_steps}")
    print("=" * 70)

    # 1. Initialize Registry & Controller
    registry = ControllerRegistry()
    controller = registry.get_controller(controller_id)
    if controller is None:
        print(f"ERROR: Controller '{controller_id}' not found in registry.")
        print(f"Available controllers: {list(registry._controllers.keys())}")
        sys.exit(1)

    print(f"Loaded Controller: {controller.metadata.display_name} [{controller.metadata.checkpoint_id}]")

    # 2. Launch Environment (GUI mode)
    gui_mode = not headless_capture
    env = CorridorSimEnv(scenario=scenario, gui=gui_mode, seed=seed)
    env.max_steps = max_steps
    obs, info = env.reset(seed=seed)

    if hasattr(controller, "update_environment"):
        controller.update_environment(env.corridor, env.obstacles)

    # Setup PyBullet Camera & Visualizer Settings
    cam_dist = 8.0 if env.corridor.x_end <= 12.0 else 13.0
    cam_target = [env.corridor.x_end / 2.0, 0.0, 1.0]

    if gui_mode:
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=env.client_id)
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1, physicsClientId=env.client_id)
        p.resetDebugVisualizerCamera(
            cameraDistance=cam_dist,
            cameraYaw=-60,
            cameraPitch=-28,
            cameraTargetPosition=cam_target,
            physicsClientId=env.client_id
        )

    # Visual Destination Marker
    goal_pos = env.corridor.goal_position
    goal_marker_id = p.createVisualShape(
        p.GEOM_CYLINDER,
        radius=env.corridor.goal_radius,
        length=0.08,
        rgbaColor=[0.1, 0.9, 0.3, 0.6],
        physicsClientId=env.client_id
    )
    p.createMultiBody(
        baseMass=0,
        baseVisualShapeIndex=goal_marker_id,
        basePosition=[goal_pos[0], goal_pos[1], 0.04],
        physicsClientId=env.client_id
    )

    # Initialize Safety Supervisor & Localization Layer
    safety_sup = SafetySupervisor(corridor=env.corridor)
    sim_loc_source = SimGroundTruthSource()
    loc_mgr = LocalizationManager(primary_source=sim_loc_source, allow_dynamic_switching=False)

    prev_pos = env._get_drone_position()
    trajectory_lines: List[int] = []
    sensor_ray_lines: List[int] = []
    hud_text_id = -1

    step_dt = env.dt / max(0.1, time_scale)
    screenshot_captured = False

    try:
        for step in range(1, max_steps + 1):
            t_loop_start = time.perf_counter()

            # Update Localization
            priv = info["privileged_state"]
            if hasattr(controller, "set_state"):
                controller.set_state(priv.global_position, priv.global_velocity)

            sim_loc_source.update_from_sim(
                priv.global_position,
                priv.global_velocity,
                priv.global_orientation_quat,
                sim_time_s=env.current_sim_time
            )
            pkt, health = loc_mgr.get_state(current_time_s=env.current_sim_time)

            # Controller Prediction
            raw_action = controller.predict(obs)

            # Safety Supervisor Filter
            front_dist = float(obs[0] * env.sensor_config.max_range_m)
            safe_action, s_state, s_reason = safety_sup.filter_action(
                proposed_action=raw_action,
                current_position=pkt.position if pkt else priv.global_position,
                current_velocity=pkt.velocity if pkt else priv.global_velocity,
                min_front_range=front_dist,
                dt=env.dt,
                is_localization_healthy=health.is_healthy,
                telemetry_age_s=health.packet_age_s
            )

            # Step Physics
            norm_action = safe_action.to_normalized(env.action_limits)
            obs, rew, term, trunc, info = env.step(norm_action)

            curr_pos = env._get_drone_position()
            vel = env._get_drone_velocity()
            speed = float(np.linalg.norm(vel))
            d_goal = float(env.corridor.distance_to_goal(curr_pos))
            obs_clear = float(info.get("obstacle_clearance", 0.0))

            # Draw 3D Trajectory Trail (Green Line)
            if gui_mode:
                line_id = p.addUserDebugLine(
                    lineFromXYZ=prev_pos.tolist(),
                    lineToXYZ=curr_pos.tolist(),
                    lineColorRGB=[0.1, 0.95, 0.25],
                    lineWidth=2.5,
                    physicsClientId=env.client_id
                )
                trajectory_lines.append(line_id)

                # Remove previous sensor debug rays
                for rl in sensor_ray_lines:
                    p.removeUserDebugItem(rl, physicsClientId=env.client_id)
                sensor_ray_lines.clear()

                # Draw Sensor Rays (Red if close, Cyan if clear)
                vis_state = env.get_visual_state()
                for ray in vis_state["rays"]:
                    r_dist = ray["distance_m"]
                    color = [0.95, 0.2, 0.2] if r_dist < 1.0 else [0.2, 0.8, 0.9]
                    rl_id = p.addUserDebugLine(
                        lineFromXYZ=ray["start"],
                        lineToXYZ=ray["end"],
                        lineColorRGB=color,
                        lineWidth=1.5,
                        physicsClientId=env.client_id
                    )
                    sensor_ray_lines.append(rl_id)

                # Live HUD Status Overlay
                hud_msg = (
                    f"Controller: {controller.metadata.display_name}\n"
                    f"Step: {step}/{max_steps} | Time: {env.current_sim_time:.2f}s\n"
                    f"Pos: [{curr_pos[0]:.2f}, {curr_pos[1]:.2f}, {curr_pos[2]:.2f}] | Speed: {speed:.2f} m/s\n"
                    f"Dist to Goal: {d_goal:.2f} m | Obstacle Clearance: {obs_clear:.2f} m\n"
                    f"Safety: {s_state.value.upper()} ({s_reason})"
                )
                hud_pos = [1.0, -1.1, 2.2]
                if hud_text_id >= 0:
                    hud_text_id = p.addUserDebugText(
                        hud_msg,
                        textPosition=hud_pos,
                        textColorRGB=[1.0, 1.0, 1.0],
                        textSize=1.1,
                        replaceItemUniqueId=hud_text_id,
                        physicsClientId=env.client_id
                    )
                else:
                    hud_text_id = p.addUserDebugText(
                        hud_msg,
                        textPosition=hud_pos,
                        textColorRGB=[1.0, 1.0, 1.0],
                        textSize=1.1,
                        physicsClientId=env.client_id
                    )

            prev_pos = curr_pos

            # Capture mid-flight screenshot around obstacle avoidance or goal
            if save_screenshot and (step == 140 or term or trunc) and not screenshot_captured:
                _capture_render_to_file(
                    client_id=env.client_id,
                    file_path=save_screenshot,
                    drone_pos=curr_pos,
                    step=step,
                    sim_time=env.current_sim_time,
                    controller_name=controller.metadata.display_name,
                    speed=speed,
                    d_goal=d_goal,
                    obs_clear=obs_clear,
                    s_state=s_state.value.upper(),
                )
                screenshot_captured = True

            # Real-time pacing
            if gui_mode:
                elapsed = time.perf_counter() - t_loop_start
                if elapsed < step_dt:
                    time.sleep(step_dt - elapsed)

            if term or trunc:
                term_reason = info.get("termination_reason", "unknown")
                print(f"\n[Episode Finished] Step: {step} | Result: {term_reason.upper()} | Final Pos: {curr_pos}")
                if gui_mode:
                    p.addUserDebugText(
                        f"EPISODE FINISHED: {term_reason.upper()}",
                        textPosition=[curr_pos[0], curr_pos[1], curr_pos[2] + 0.4],
                        textColorRGB=[1.0, 0.8, 0.1] if term_reason == "success" else [1.0, 0.2, 0.2],
                        textSize=1.5,
                        physicsClientId=env.client_id
                    )
                    time.sleep(1.5)
                break

    finally:
        if save_screenshot and not screenshot_captured:
            _capture_render_to_file(
                client_id=env.client_id,
                file_path=save_screenshot,
                drone_pos=env._get_drone_position(),
                step=env.step_count,
                sim_time=env.current_sim_time,
                controller_name=controller.metadata.display_name,
                speed=0.0,
                d_goal=float(env.corridor.distance_to_goal(env._get_drone_position())),
                obs_clear=0.0,
                s_state="FINISHED",
            )
        env.close()
        print("Graphical Viewer closed.")


def _capture_render_to_file(
    client_id: int,
    file_path: str,
    drone_pos: np.ndarray,
    step: int = 0,
    sim_time: float = 0.0,
    controller_name: str = "Reactive Baseline",
    speed: float = 0.0,
    d_goal: float = 0.0,
    obs_clear: float = 0.0,
    s_state: str = "NOMINAL",
) -> None:
    """Captures RGB image buffer from PyBullet camera with chase perspective and HUD."""
    from PIL import ImageDraw

    width, height = 1280, 720
    pos = drone_pos
    view_mat = p.computeViewMatrix(
        cameraEyePosition=[pos[0] - 1.8, float(np.clip(pos[1] * 0.5, -0.4, 0.4)), 1.5],
        cameraTargetPosition=[pos[0] + 3.0, 0.0, 1.0],
        cameraUpVector=[0, 0, 1],
        physicsClientId=client_id
    )
    proj_mat = p.computeProjectionMatrixFOV(
        fov=75.0,
        aspect=float(width) / float(height),
        nearVal=0.05,
        farVal=35.0,
        physicsClientId=client_id
    )
    img_data = p.getCameraImage(
        width=width,
        height=height,
        viewMatrix=view_mat,
        projectionMatrix=proj_mat,
        renderer=p.ER_TINY_RENDERER,
        physicsClientId=client_id
    )
    rgba = np.array(img_data[2], dtype=np.uint8)
    if rgba.ndim == 1:
        rgba = rgba.reshape((height, width, 4))
    im = Image.fromarray(rgba[:, :, :3])

    # Draw modern HUD overlay
    card_x, card_y, card_w, card_h = 24, 24, 460, 175
    overlay = Image.new("RGBA", im.size, (0, 0, 0, 0))
    ov_draw = ImageDraw.Draw(overlay)
    ov_draw.rectangle([card_x, card_y, card_x + card_w, card_y + card_h], fill=(15, 23, 42, 215), outline=(56, 189, 248, 255), width=2)
    im = Image.alpha_composite(im.convert("RGBA"), overlay).convert("RGB")
    draw = ImageDraw.Draw(im)

    draw.text((card_x + 16, card_y + 14), "UAV CORRIDOR NAVIGATION - PYBULLET SIMULATION", fill=(56, 189, 248))
    draw.text((card_x + 16, card_y + 40), f"Controller: {controller_name}", fill=(241, 245, 249))
    draw.text((card_x + 16, card_y + 64), f"Step: {step} ({sim_time:.2f}s) | Status: IN FLIGHT", fill=(52, 211, 153))
    draw.text((card_x + 16, card_y + 88), f"Position: X={pos[0]:.2f}m, Y={pos[1]:.2f}m, Z={pos[2]:.2f}m | Speed: {speed:.2f}m/s", fill=(241, 245, 249))
    draw.text((card_x + 16, card_y + 112), f"Goal Distance: {d_goal:.2f}m | Obstacle Clearance: {obs_clear:.2f}m", fill=(251, 191, 36))
    draw.text((card_x + 16, card_y + 136), f"Safety Supervisor: {s_state} (Nominal Envelope)", fill=(148, 163, 184))

    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    im.save(file_path)
    print(f">>> Graphical Screenshot successfully saved to: {file_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyBullet 3D Corridor Graphical Viewer")
    parser.add_argument("--controller", type=str, default="reactive_baseline", help="Controller ID (e.g. reactive_baseline, scripted_expert_apf)")
    parser.add_argument("--scenario", type=str, default="10m", choices=["10m", "20m"], help="Corridor scenario (10m benchmark or 20m extended)")
    parser.add_argument("--seed", type=int, default=100, help="Random seed for environment and obstacles")
    parser.add_argument("--max-steps", type=int, default=800, help="Maximum episode steps")
    parser.add_argument("--time-scale", type=float, default=2.0, help="Simulation playback speed multiplier")
    parser.add_argument("--save-screenshot", type=str, default=None, help="Save screenshot to file path")
    parser.add_argument("--headless-capture", action="store_true", help="Capture screenshot headlessly without GUI window")
    args = parser.parse_args()

    run_gui(
        controller_id=args.controller,
        scenario=args.scenario,
        seed=args.seed,
        max_steps=args.max_steps,
        time_scale=args.time_scale,
        save_screenshot=args.save_screenshot,
        headless_capture=args.headless_capture,
    )
