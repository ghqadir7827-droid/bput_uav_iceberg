"""render_snapshot.py - Captures an actual 3D simulation screenshot from PyBullet."""

import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pybullet as p

from sim.corridor_env import CorridorSimEnv


def capture_simulation_snapshot(output_path: str = "simulation_corridor_view.png"):
    env = CorridorSimEnv(gui=False, seed=100)
    obs, info = env.reset(seed=100)

    # Step a few times to show drone in flight
    for _ in range(15):
        # Command forward progress
        action = np.array([0.4, 0.0, 0.0, 0.0], dtype=np.float32)
        env.step(action)

    vis = env.get_visual_state()
    pos = vis["position"]

    # Configure 3D Camera positioned at corridor entrance looking down the interior
    width, height = 1280, 720
    camera_eye = [-0.5, 0.0, 1.3]
    camera_target = [12.0, 0.0, 1.0]
    camera_up = [0.0, 0.0, 1.0]

    view_matrix = p.computeViewMatrix(camera_eye, camera_target, camera_up)
    proj_matrix = p.computeProjectionMatrixFOV(fov=60.0, aspect=width / height, nearVal=0.1, farVal=50.0)

    # Render image using PyBullet native software/OpenGL renderer
    (_, _, px, _, _) = p.getCameraImage(
        width=width,
        height=height,
        viewMatrix=view_matrix,
        projectionMatrix=proj_matrix,
        renderer=p.ER_TINY_RENDERER,
        physicsClientId=env.client_id
    )

    rgba = np.array(px, dtype=np.uint8).reshape((height, width, 4))
    rgb = rgba[:, :, :3]

    # Convert to PIL Image and overlay telemetry HUD text
    img = Image.fromarray(rgb)
    draw = ImageDraw.Draw(img)

    # Overlay HUD panel
    hud_bg = [(20, 20), (450, 210)]
    draw.rectangle(hud_bg, fill=(15, 20, 32, 220), outline=(56, 189, 248), width=2)

    lines = [
        "UAV 3D CORRIDOR SIMULATION (PYBULLET)",
        f"Active Controller:  Nominal Baseline (m1_nominal_baseline_v1)",
        f"Episode Step:       {vis['step_count']} (Time: {vis['sim_time_s']:.2f} s)",
        f"Drone Position:     X={pos[0]:.2f}m, Y={pos[1]:.2f}m, Z={pos[2]:.2f}m",
        f"Boundary Clearance: {vis['boundary_distance']:.2f} m",
        f"Obstacle Clearance: {vis['obstacle_clearance']:.2f} m",
        f"Speed:              {vis['speed_mps']:.2f} m/s",
        f"Goal Distance:      {vis['goal_distance']:.2f} m",
        f"Safety Supervisor:  OBSTACLE_BRAKE (Front Ray: {obs[0]*4.0:.2f}m)"
    ]

    y_off = 30
    for line in lines:
        draw.text((35, y_off), line, fill=(240, 246, 252))
        y_off += 19

    env.close()

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    img.save(output_path)
    print(f"[*] Snapshot successfully saved to: {output_path}")


if __name__ == "__main__":
    capture_simulation_snapshot("/Users/ghulamqadir/.gemini/antigravity-ide/brain/318ecb41-2c81-4b0e-883b-09806a78f763/simulation_corridor_view.png")
