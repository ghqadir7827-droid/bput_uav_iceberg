"""test_clearance_guidance.py - Unit tests for Continuous Clearance-Aware Guidance (v5)."""

import numpy as np
import pytest

from core.contracts import ActionLimits, DeployableObservation, VelocityAction
from core.guidance import GuidanceConfig, StatelessClearanceGuidanceArbitrator


def _make_guidance_obs(c_front: float, c_left: float, c_right: float, wall_l: float, wall_r: float) -> np.ndarray:
    """Helper generating 23-float observation vector for guidance testing."""
    rays = np.full(12, 1.0, dtype=np.float32)
    rays[0] = c_front / 4.0
    rays[1] = c_left / 4.0
    rays[3] = c_left / 4.0
    rays[5] = wall_l / 4.0

    rays[2] = c_right / 4.0
    rays[4] = c_right / 4.0
    rays[6] = wall_r / 4.0

    obs = np.zeros(23, dtype=np.float32)
    obs[:12] = rays
    # Rel goal: dist > 2.0m so not in terminal zone
    obs[12:15] = [1.0, 0.0, 0.0]
    return obs


def test_v5_continuous_lateral_scaling():
    """Verifies that v5 continuously scales lateral override between 0.15 and 0.40 m/s based on available headroom."""
    cfg = GuidanceConfig.for_65cm_uav_v5_clearance_aware()
    arb = StatelessClearanceGuidanceArbitrator(config=cfg)

    raw_act = VelocityAction(vx=0.40, vy=0.0, vz=0.0, yaw_rate=0.0)

    # Case 1: Wide open corridor (wall_l = 2.5m, wall_r = 2.0m, W = 4.5m)
    # Headroom = 2.5 - 0.35 = 2.15m >= 0.50m -> rho = 1.0 -> vy = +0.40 m/s
    obs_wide = _make_guidance_obs(c_front=0.80, c_left=2.5, c_right=2.0, wall_l=2.5, wall_r=2.0)
    act_wide, direction_wide, _, _ = arb.arbitrate(raw_act, obs_wide)
    assert "LEFT" in direction_wide
    assert np.isclose(act_wide.vy, 0.40, atol=1e-3)

    # Case 2: Tight headroom on left (wall_l = 2.0m, c_left = 0.35m, c_right = 0.10m)
    # diff = 0.35 - 0.10 = 0.25m > deadband -> selects LEFT
    # Headroom = 0.35 - 0.35 = 0.0m -> rho = 0.0 -> vy = +0.15 m/s
    arb.reset()
    obs_tight = _make_guidance_obs(c_front=0.80, c_left=0.35, c_right=0.10, wall_l=2.0, wall_r=2.0)
    act_tight, direction_tight, _, _ = arb.arbitrate(raw_act, obs_tight)
    assert "LEFT" in direction_tight
    assert np.isclose(act_tight.vy, 0.15, atol=1e-3)

    # Case 3: Narrow corridor adaptation (wall_l = 1.4m, wall_r = 1.4m, W = 2.8m < 3.2m)
    # Target headroom >= 0.50m, but corridor adaptation caps override at 0.28 m/s
    arb.reset()
    obs_narrow = _make_guidance_obs(c_front=0.80, c_left=1.4, c_right=0.80, wall_l=1.4, wall_r=1.4)
    act_narrow, direction_narrow, _, _ = arb.arbitrate(raw_act, obs_narrow)
    assert "LEFT" in direction_narrow
    assert np.isclose(act_narrow.vy, 0.28, atol=1e-3)


def test_guidance_suppression_during_safe_arrest():
    """Verifies that suppress_guidance=True immediately zeroes lateral commands."""
    cfg = GuidanceConfig.for_65cm_uav_v5_clearance_aware()
    arb = StatelessClearanceGuidanceArbitrator(config=cfg)

    raw_act = VelocityAction(vx=0.40, vy=-0.20, vz=0.0, yaw_rate=0.0)
    obs = _make_guidance_obs(c_front=0.50, c_left=1.5, c_right=1.0, wall_l=1.5, wall_r=1.0)

    # Suppressed guidance
    act, direction, _, _ = arb.arbitrate(raw_act, obs, suppress_guidance=True)
    assert act.vy == 0.0
    assert direction == "SUPPRESSED_SAFE_ARREST"


def test_persistent_tie_breaking_retains_direction():
    """Verifies that in symmetry deadband, v5 retains persistent direction without chattering."""
    cfg = GuidanceConfig.for_65cm_uav_v5_clearance_aware()
    arb = StatelessClearanceGuidanceArbitrator(config=cfg)

    raw_act = VelocityAction(vx=0.40, vy=0.0, vz=0.0, yaw_rate=0.0)

    # 1. First trigger left evasion
    obs_left = _make_guidance_obs(c_front=0.80, c_left=1.5, c_right=0.8, wall_l=1.5, wall_r=1.5)
    act1, dir1, _, _ = arb.arbitrate(raw_act, obs_left)
    assert "LEFT" in dir1
    assert act1.vy > 0.0

    # 2. Symmetrical deadband: diff = 0.0
    obs_deadband = _make_guidance_obs(c_front=0.80, c_left=1.2, c_right=1.2, wall_l=1.5, wall_r=1.5)
    act2, dir2, _, _ = arb.arbitrate(raw_act, obs_deadband)
    # Must retain LEFT direction via persistent tie-break
    assert dir2 == "TIE_BREAK_LEFT"
    assert act2.vy > 0.0

