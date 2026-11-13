"""test_guidance_controller.py - Comprehensive Unit and Regression Tests for Guidance Controller."""

import numpy as np
import pytest
import torch

from controllers.guidance_controller import StatelessClearanceGuidanceController
from controllers.registry import ControllerRegistry
from core.contracts import ActionLimits, DeployableObservation, VelocityAction
from models.policy_net import NavigationPolicyMLP


@pytest.fixture
def base_mlp():
    torch.manual_seed(42)
    mlp = NavigationPolicyMLP()
    mlp.eval()
    return mlp


def test_guidance_controller_initialization_and_metadata():
    ctrl = StatelessClearanceGuidanceController()
    meta = ctrl.metadata
    assert meta.controller_id == "bc_clearance_guidance"
    assert meta.is_implemented is True
    assert "Stateless Clearance Guidance" in meta.display_name


def test_guidance_controller_registry_lookup():
    registry = ControllerRegistry()
    ctrl = registry.get_controller("bc_clearance_guidance")
    assert ctrl is not None
    assert isinstance(ctrl, StatelessClearanceGuidanceController)
    assert ctrl.metadata.is_implemented is True


def test_guidance_bimodal_symmetry_collapse_resolution(base_mlp):
    """Verifies that centered obstacles trigger deterministic lateral avoidance rather than zero velocity."""
    ctrl = StatelessClearanceGuidanceController(base_model=base_mlp, activation_threshold_m=0.85)
    
    # Create observation with centered obstacle ahead:
    # front ray is 0.5m, left and right rays symmetric at 0.5m
    obs = np.ones(23, dtype=np.float32) * 0.8
    obs[0] = 0.50 / 4.0  # Ray 0 (front): 0.50m < 0.85m
    obs[1] = 0.50 / 4.0  # Ray 1 (front-left): 0.50m
    obs[2] = 0.50 / 4.0  # Ray 2 (front-right): 0.50m
    obs[3] = 0.60 / 4.0
    obs[4] = 0.60 / 4.0
    obs[5] = 0.80 / 4.0  # Ray 5 (left 90): 0.80m
    obs[6] = 0.80 / 4.0  # Ray 6 (right 90): 0.80m
    # Goal far away (norm = 1.0)
    obs[12:15] = [1.0, 0.0, 0.0]

    act, s_dir, c_l, c_r = ctrl.predict_with_diagnostics(obs)
    
    assert "TIE_BREAK" in s_dir
    # Must command substantial non-zero lateral velocity (|vy| >= 0.25 m/s)
    assert abs(act.vy) >= 0.25
    assert act.vx > 0.0


def test_guidance_directional_clearance_arbitration(base_mlp):
    """Verifies steering toward the side with strictly larger clearance."""
    ctrl = StatelessClearanceGuidanceController(base_model=base_mlp, activation_threshold_m=0.85)

    # Obstacle on the left -> right side is clear
    obs_left_blocked = np.ones(23, dtype=np.float32) * 0.8
    obs_left_blocked[0] = 0.50 / 4.0
    obs_left_blocked[1] = 0.40 / 4.0  # left front blocked
    obs_left_blocked[3] = 0.40 / 4.0  # left side blocked
    obs_left_blocked[5] = 0.60 / 4.0
    obs_left_blocked[2] = 1.20 / 4.0  # right front clear
    obs_left_blocked[4] = 1.20 / 4.0  # right side clear
    obs_left_blocked[6] = 0.80 / 4.0
    obs_left_blocked[12:15] = [1.0, 0.0, 0.0]

    act_r, dir_r, _, _ = ctrl.predict_with_diagnostics(obs_left_blocked)
    assert dir_r == "RIGHT"
    assert act_r.vy <= -0.25

    # Obstacle on the right -> left side is clear
    obs_right_blocked = np.ones(23, dtype=np.float32) * 0.8
    obs_right_blocked[0] = 0.50 / 4.0
    obs_right_blocked[2] = 0.40 / 4.0  # right front blocked
    obs_right_blocked[4] = 0.40 / 4.0  # right side blocked
    obs_right_blocked[6] = 0.60 / 4.0
    obs_right_blocked[1] = 1.20 / 4.0  # left front clear
    obs_right_blocked[3] = 1.20 / 4.0  # left side clear
    obs_right_blocked[5] = 0.80 / 4.0
    obs_right_blocked[12:15] = [1.0, 0.0, 0.0]

    act_l, dir_l, _, _ = ctrl.predict_with_diagnostics(obs_right_blocked)
    assert dir_l == "LEFT"
    assert act_l.vy >= 0.25


def test_guidance_wall_pinch_guardrail(base_mlp):
    """Verifies that when close to a corridor boundary, guidance does not push the UAV into the wall."""
    ctrl = StatelessClearanceGuidanceController(base_model=base_mlp, activation_threshold_m=0.85)

    # Drone is near left wall (ray 5 = 0.25m < 0.35m threshold)
    obs = np.ones(23, dtype=np.float32) * 0.8
    obs[0] = 0.60 / 4.0
    obs[1] = 0.60 / 4.0
    obs[2] = 0.60 / 4.0
    obs[5] = 0.25 / 4.0  # left wall tight
    obs[6] = 1.20 / 4.0  # right wall open
    obs[12:15] = [1.0, 0.0, 0.0]

    act, s_dir, _, _ = ctrl.predict_with_diagnostics(obs)
    assert s_dir == "WALL_AVOID_RIGHT"
    assert act.vy <= -0.25

    # Drone is near right wall (ray 6 = 0.25m < 0.35m threshold)
    obs[5] = 1.20 / 4.0  # left wall open
    obs[6] = 0.25 / 4.0  # right wall tight

    act_l, s_dir_l, _, _ = ctrl.predict_with_diagnostics(obs)
    assert s_dir_l == "WALL_AVOID_LEFT"
    assert act_l.vy >= 0.25


def test_guidance_terminal_soft_landing_release(base_mlp):
    """Verifies that guidance releases control when approaching destination goal."""
    ctrl = StatelessClearanceGuidanceController(base_model=base_mlp, activation_threshold_m=0.85)

    # Drone in terminal region: rel_goal norm < 0.50 (distance < 1.0m)
    obs = np.ones(23, dtype=np.float32) * 0.8
    obs[0] = 0.60 / 4.0  # Forward wall detected at 0.60m
    obs[1] = 0.60 / 4.0
    obs[2] = 0.60 / 4.0
    obs[12:15] = [0.30, 0.0, 0.0]  # Goal norm = 0.30 < 0.50

    act, s_dir, _, _ = ctrl.predict_with_diagnostics(obs)
    # Must yield to baseline policy nominal behavior (NONE)
    assert s_dir == "NONE"


def test_guidance_action_limits_contract(base_mlp):
    """Verifies that velocity setpoints strictly observe ActionLimits."""
    ctrl = StatelessClearanceGuidanceController(base_model=base_mlp)
    limits = ActionLimits()

    for seed in range(20):
        rng = np.random.default_rng(seed)
        dummy_obs = rng.uniform(0.0, 1.0, size=(23,)).astype(np.float32)
        act = ctrl.predict(dummy_obs)

        assert limits.min_fwd_vel_mps <= act.vx <= limits.max_fwd_vel_mps
        assert -limits.max_lat_vel_mps <= act.vy <= limits.max_lat_vel_mps
        assert -limits.max_vert_vel_mps <= act.vz <= limits.max_vert_vel_mps
        assert -limits.max_yaw_rate_radps <= act.yaw_rate <= limits.max_yaw_rate_radps

