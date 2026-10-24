"""test_geometry.py - Tests for 3D Corridor Spatial Math and Termination Checks."""

import numpy as np
import pytest

from core.geometry import Corridor3D, Obstacle3D, TerminationReason


def test_corridor_boundary_distance_center():
    """At the centerline, boundary distance should equal width/2 - drone_radius."""
    corridor = Corridor3D(width=2.4, height=2.0, z_floor=0.0, z_ceiling=2.0, drone_radius=0.15)
    center_pos = np.array([5.0, 0.0, 1.0]) # Exactly centered
    # Clearances: left=1.2-0.15=1.05, right=1.2-0.15=1.05, ceil=1.0-0.15=0.85, floor=1.0-0.15=0.85
    b_dist = corridor.boundary_distance(center_pos)
    assert np.isclose(b_dist, 0.85, atol=1e-4)


def test_corridor_boundary_clearances_individual():
    """Verifies directional boundary clearances."""
    corridor = Corridor3D(width=2.0, height=2.0, z_floor=0.0, z_ceiling=2.0, drone_radius=0.10)
    pos = np.array([2.0, 0.5, 1.5]) # Shifted left (+Y) and up (+Z)
    d_left, d_right, d_ceil, d_floor = corridor.boundary_clearances(pos)
    # y_max = 1.0 -> d_left = 1.0 - 0.5 - 0.1 = 0.40
    # y_min = -1.0 -> d_right = 0.5 - (-1.0) - 0.1 = 1.40
    # z_ceiling = 2.0 -> d_ceil = 2.0 - 1.5 - 0.1 = 0.40
    # z_floor = 0.0 -> d_floor = 1.5 - 0.0 - 0.1 = 1.40
    assert np.isclose(d_left, 0.40, atol=1e-4)
    assert np.isclose(d_right, 1.40, atol=1e-4)
    assert np.isclose(d_ceil, 0.40, atol=1e-4)
    assert np.isclose(d_floor, 1.40, atol=1e-4)


def test_obstacle_distance_cylinder():
    """Tests analytical distance calculation to cylindrical obstacle."""
    obs = Obstacle3D(center=np.array([5.0, 0.0, 1.0]), radius=0.20, height=2.0, obstacle_type="cylinder")
    # Point at (5.0, 0.5, 1.0) -> distance to axis is 0.5, clearance = 0.5 - 0.20 = 0.30
    dist = obs.distance_to_point(np.array([5.0, 0.5, 1.0]))
    assert np.isclose(dist, 0.30, atol=1e-4)


def test_safe_termination_conditions():
    """Tests boundary crash, obstacle crash, success, and timeout conditions."""
    corridor = Corridor3D(x_start=0.0, x_end=20.0, width=2.0, height=2.0, drone_radius=0.15)
    obs = [Obstacle3D(center=np.array([10.0, 0.0, 1.0]), radius=0.25)]

    # 1. In-flight normal condition
    state, _ = corridor.check_termination(
        position=np.array([2.0, 0.0, 1.0]),
        velocity=np.array([0.3, 0.0, 0.0]),
        obstacles=obs,
        step_count=10
    )
    assert state == TerminationReason.CONTINUE

    # 2. Boundary crash (Y exceeds boundary)
    state, _ = corridor.check_termination(
        position=np.array([2.0, 1.1, 1.0]), # y_max is 1.0, penetrated!
        velocity=np.array([0.0, 0.0, 0.0]),
        obstacles=obs,
        step_count=20
    )
    assert state == TerminationReason.CRASH_BOUNDARY

    # 3. Obstacle collision
    state, _ = corridor.check_termination(
        position=np.array([10.0, 0.1, 1.0]), # center is at (10, 0), radius 0.25+0.15=0.40 -> collision!
        velocity=np.array([0.0, 0.0, 0.0]),
        obstacles=obs,
        step_count=30
    )
    assert state == TerminationReason.CRASH_OBSTACLE

    # 4. Successful arrival
    state, _ = corridor.check_termination(
        position=corridor.goal_position,
        velocity=np.array([0.05, 0.0, 0.0]), # Low speed
        obstacles=obs,
        step_count=40
    )
    assert state == TerminationReason.SUCCESS

    # 5. Timeout
    state, _ = corridor.check_termination(
        position=np.array([2.0, 0.0, 1.0]),
        velocity=np.array([0.0, 0.0, 0.0]),
        obstacles=obs,
        step_count=500,
        max_steps=500
    )
    assert state == TerminationReason.TIMEOUT
