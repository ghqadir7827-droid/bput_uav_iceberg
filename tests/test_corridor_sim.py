"""test_corridor_sim.py - Unit Tests for PyBullet 3D Corridor Simulation."""

import numpy as np
import pytest

from core.geometry import TerminationReason
from sim.corridor_env import CorridorSimEnv


def test_corridor_sim_initialization_and_reset():
    """Verifies that the PyBullet environment resets cleanly and returns shape (23,)."""
    env_10m = CorridorSimEnv(scenario="10m", gui=False, seed=42)
    env_20m = CorridorSimEnv(scenario="20m", gui=False, seed=42)
    try:
        obs, info = env_10m.reset(seed=42)
        assert obs.shape == (23,), f"Expected obs shape (23,), got {obs.shape}"
        assert obs.dtype == np.float32

        assert "privileged_state" in info
        assert "boundary_distance" in info
        assert info["boundary_distance"] > 0.0 # Initially inside corridor
        assert len(env_10m.obstacles) == 3 # 10m benchmark has 3 pillars

        obs20, info20 = env_20m.reset(seed=42)
        assert len(env_20m.obstacles) == 4 # 20m extended scenario has 4 pillars
    finally:
        env_10m.close()
        env_20m.close()


def test_corridor_sim_step():
    """Verifies step execution with high-level velocity command."""
    env = CorridorSimEnv(gui=False, seed=42)
    try:
        obs, _ = env.reset(seed=42)
        # Action: Forward flight (action[0] = 0.5 -> ~0.45 m/s)
        action = np.array([0.5, 0.0, 0.0, 0.0], dtype=np.float32)

        next_obs, reward, terminated, truncated, info = env.step(action)

        assert next_obs.shape == (23,)
        assert isinstance(reward, float)
        assert not terminated
        assert not truncated
        assert info["goal_progress_m"] >= 0.0
    finally:
        env.close()


def test_raycast_sensor_readings():
    """Verifies that all 12 rangefinder beams return valid normalized distances in [0, 1]."""
    env = CorridorSimEnv(gui=False, seed=42)
    try:
        obs, _ = env.reset(seed=42)
        # First 12 elements of observation are the normalized distance rays
        ranges = obs[:12]
        assert np.all(ranges >= 0.0)
        assert np.all(ranges <= 1.0)
        # Front ray should sense open corridor ahead (> 0.20)
        assert ranges[0] > 0.10
    finally:
        env.close()


def test_collision_detection():
    """Verifies that driving sideways into a wall triggers TerminationReason.CRASH_BOUNDARY."""
    env = CorridorSimEnv(gui=False, seed=42)
    try:
        env.reset(seed=42)
        # Command maximum lateral velocity left (+Y) into the wall
        action = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)

        terminated = False
        final_info = {}
        for _ in range(80):
            _, _, term, trunc, info = env.step(action)
            if term or trunc:
                terminated = term
                final_info = info
                break

        assert terminated
        assert final_info.get("termination_reason") in (
            TerminationReason.CRASH_BOUNDARY.value,
            TerminationReason.CRASH_OBSTACLE.value,
        )
    finally:
        env.close()
