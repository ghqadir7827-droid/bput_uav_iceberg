"""test_apf_expert.py - Unit tests for Scripted APF Expert Controller."""

import numpy as np
import pytest

from controllers.apf_expert import ScriptedAPFExpertController
from controllers.registry import ControllerRegistry
from core.geometry import Corridor3D, Obstacle3D


def test_apf_metadata_and_registry():
    registry = ControllerRegistry()
    controller = registry.get_controller("scripted_expert_apf")
    assert controller is not None
    meta = controller.metadata
    assert meta.controller_id == "scripted_expert_apf"
    assert meta.is_implemented is True
    assert "Milestone 2" in meta.milestone_tag


def test_apf_attractive_goal_gradient():
    corridor = Corridor3D.get_10m_benchmark()
    expert = ScriptedAPFExpertController(corridor=corridor)
    
    # At start position (0.5, 0.0, 1.0) heading to (9.5, 0.0, 1.0) with no obstacles
    pos = np.array([0.5, 0.0, 1.0])
    vel = np.array([0.0, 0.0, 0.0])
    action = expert.compute_privileged_expert_action(pos, vel, obstacles=[])
    
    assert action.vx > 0.4  # Forward cruise speed
    assert abs(action.vy) < 0.05  # Centered laterally
    assert abs(action.vz) < 0.05  # Centered vertically


def test_apf_obstacle_evasion_direction():
    corridor = Corridor3D.get_10m_benchmark()
    expert = ScriptedAPFExpertController(corridor=corridor)
    
    # Obstacle placed slightly to the left (y = +0.15) at X = 3.0m
    obs = [Obstacle3D(center=np.array([3.0, 0.15, 1.0]), radius=0.20, height=2.0)]
    
    # Drone approaching at X = 2.4m, Y = 0.0m
    pos = np.array([2.4, 0.0, 1.0])
    vel = np.array([0.5, 0.0, 0.0])
    action = expert.compute_privileged_expert_action(pos, vel, obstacles=obs)
    
    # Should steer right (negative vy) to avoid obstacle on the left
    assert action.vy < -0.10


def test_apf_wall_boundary_repulsion():
    corridor = Corridor3D.get_10m_benchmark()  # y in [-1.0, 1.0]
    expert = ScriptedAPFExpertController(corridor=corridor)
    
    # Drone very close to right wall (y = -0.80, wall at -1.0, drone radius 0.15 -> clearance 0.05m)
    pos = np.array([2.0, -0.80, 1.0])
    vel = np.array([0.4, -0.1, 0.0])
    action = expert.compute_privileged_expert_action(pos, vel, obstacles=[])
    
    # Wall repulsion must push strongly inward (positive vy)
    assert action.vy > 0.20


def test_apf_deployable_observation_predict():
    corridor = Corridor3D.get_10m_benchmark()
    expert = ScriptedAPFExpertController(corridor=corridor)
    
    # Deployable observation: 23 floats
    # [0:12] rays (norm [0,1]), [12:15] rel_goal, [15:18] vel, [18:22] quat, [22] yaw_rate
    dummy_obs = np.ones(23, dtype=np.float32)
    dummy_obs[12:15] = [1.0, 0.0, 0.0]  # Goal straight ahead
    
    action = expert.predict(dummy_obs)
    assert action.vx > 0.0
    assert -0.8 <= action.vx <= 0.8
    assert -0.6 <= action.vy <= 0.6
    assert -0.4 <= action.vz <= 0.4
