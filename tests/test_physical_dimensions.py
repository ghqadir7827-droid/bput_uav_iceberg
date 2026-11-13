"""test_physical_dimensions.py - Verification of 65cm UAV Model & Scaled Environments."""

import numpy as np
import pytest

from core.geometry import Corridor3D, Obstacle3D
from core.physical_dimensions import (
    DifficultyLevel,
    FeasibilityResult,
    PhysicalDroneModel,
    RouteFeasibilityChecker,
    ScaledEnvironmentFactory,
)
from controllers.safety_supervisor import SafetySupervisor, SafetySupervisorConfig
from controllers.registry import ReactiveBaselineController
from sim.corridor_env import CorridorSimEnv


def test_physical_drone_model_dimensions():
    """Verifies physical dimensions of the 65 cm quadcopter."""
    drone = PhysicalDroneModel()
    assert drone.tip_to_tip_diagonal_m == 0.65
    assert drone.collision_radius_m == 0.325
    assert drone.sensor_offset_m > drone.collision_radius_m
    assert np.isclose(drone.sensor_offset_m, 0.355, atol=1e-4)
    assert drone.mass_kg == 1.80
    assert drone.min_stopping_distance_m >= 0.20


def test_scaled_environment_factory_levels():
    """Verifies multi-level scaled environment configurations."""
    c1 = ScaledEnvironmentFactory.get_environment_config(DifficultyLevel.LEVEL_1_BASIC)
    assert c1.width == 4.50
    assert c1.height == 2.80
    assert c1.drone_radius == 0.325
    assert c1.pillar_radius == 0.25

    c2 = ScaledEnvironmentFactory.get_environment_config(DifficultyLevel.LEVEL_2_STANDARD)
    assert c2.width == 3.80
    assert c2.height == 2.50
    assert c2.drone_radius == 0.325

    c3 = ScaledEnvironmentFactory.get_environment_config(DifficultyLevel.LEVEL_3_CHALLENGING)
    assert c3.width == 3.20
    assert c3.height == 2.40
    assert c3.drone_radius == 0.325

    c4 = ScaledEnvironmentFactory.get_environment_config(DifficultyLevel.LEVEL_4_FEASIBILITY_BOUNDARY)
    assert c4.width == 2.80
    assert c4.height == 2.40
    assert c4.drone_radius == 0.325


def test_corridor_from_scenario_scaled():
    """Verifies scenario string parsing for scaled corridors."""
    c_basic = Corridor3D.from_scenario("level_1")
    assert c_basic.width == 4.50
    assert c_basic.drone_radius == 0.325

    c_std = Corridor3D.from_scenario("scaled")
    assert c_std.width == 3.80
    assert c_std.drone_radius == 0.325

    c_chal = Corridor3D.from_scenario("level_3")
    assert c_chal.width == 3.20

    c_bound = Corridor3D.from_scenario("level_4")
    assert c_bound.width == 2.80


def test_route_feasibility_checker():
    """Tests analytical geometric feasibility checking."""
    checker = RouteFeasibilityChecker()
    c2 = ScaledEnvironmentFactory.get_environment_config(DifficultyLevel.LEVEL_2_STANDARD)

    # 1. Feasible corridor: Center pillar (radius 0.25m) in 3.8m corridor
    # Left gap: 1.90 - 0.25 = 1.65m > 0.65m (FEASIBLE)
    obs_feasible = [Obstacle3D(center=np.array([5.0, 0.0, 1.25]), radius=0.25)]
    res_f = checker.evaluate_feasibility(c2, obs_feasible)
    assert res_f.is_feasible is True
    assert res_f.status == "FEASIBLE"
    assert res_f.min_bottleneck_gap_m > 0.65

    # 2. Infeasible corridor: Enormous obstacle that blocks both sides
    # Corridor width = 2.0m, obstacle radius = 0.85m at center -> gap = 1.0 - 0.85 = 0.15m < 0.65m
    c_narrow = Corridor3D(width=2.0, height=2.0, drone_radius=0.325)
    obs_blocked = [Obstacle3D(center=np.array([5.0, 0.0, 1.0]), radius=0.85)]
    res_b = checker.evaluate_feasibility(c_narrow, obs_blocked)
    assert res_b.is_feasible is False
    assert res_b.status == "BLOCKED_PASSAGE"
    assert res_b.min_bottleneck_gap_m < 0.65


def test_safety_supervisor_auto_scaling():
    """Verifies that SafetySupervisor automatically selects 65cm configuration when drone_radius > 0.25."""
    c_std = ScaledEnvironmentFactory.get_environment_config(DifficultyLevel.LEVEL_2_STANDARD)
    sup = SafetySupervisor(c_std)
    assert sup.config.critical_obstacle_dist_m >= 0.50
    assert sup.config.boundary_buffer_m >= 0.30
    assert sup.config.obstacle_caution_dist_m >= 1.00


def test_corridor_sim_env_scaled_initialization():
    """Tests that PyBullet environment initializes and runs with scaled 65cm dimensions."""
    env = CorridorSimEnv(scenario="level_2", seed=42)
    obs, info = env.reset(seed=42)
    assert obs.shape == (23,)
    assert env.corridor.drone_radius == 0.325
    assert env.corridor.width == 3.80

    # Step one action
    action = np.array([0.5, 0.0, 0.0, 0.0], dtype=np.float32)
    next_obs, reward, terminated, truncated, step_info = env.step(action)
    assert next_obs.shape == (23,)
    env.close()

