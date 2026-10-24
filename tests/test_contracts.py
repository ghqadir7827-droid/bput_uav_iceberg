"""test_contracts.py - Verification of Observation & Action Interfaces."""

import numpy as np
import pytest

from core.contracts import (
    ActionLimits,
    DeployableObservation,
    PrivilegedState,
    SensorConfig,
    VelocityAction,
)


def test_deployable_observation_shape():
    """Verifies that the deployable student observation vector is exactly 23 floats."""
    cfg = SensorConfig()
    obs = DeployableObservation(
        distance_rays=np.ones(cfg.num_rays, dtype=np.float32) * 0.5,
        rel_goal_direction_body=np.array([1.0, 0.0, 0.0], dtype=np.float32),
        body_velocity=np.array([0.2, 0.0, 0.0], dtype=np.float32),
        attitude_trig=np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float32), # sin/cos roll & pitch
        yaw_rate=0.05
    )
    vec = obs.to_vector()
    assert vec.shape == (23,), f"Expected shape (23,), got {vec.shape}"
    assert vec.dtype == np.float32


def test_velocity_action_roundtrip():
    """Verifies conversion between physical velocity commands and normalized [-1, 1] actions."""
    limits = ActionLimits()
    act = VelocityAction(vx=0.30, vy=-0.20, vz=0.15, yaw_rate=0.25)
    norm = act.to_normalized(limits)

    assert np.all(norm >= -1.0) and np.all(norm <= 1.0)

    recovered = VelocityAction.from_normalized(norm, limits)
    assert np.isclose(act.vx, recovered.vx, atol=1e-5)
    assert np.isclose(act.vy, recovered.vy, atol=1e-5)
    assert np.isclose(act.vz, recovered.vz, atol=1e-5)
    assert np.isclose(act.yaw_rate, recovered.yaw_rate, atol=1e-5)


def test_action_clamping():
    """Verifies that out-of-bound actions are clamped properly."""
    limits = ActionLimits()
    extreme_norm = np.array([5.0, -10.0, 2.5, -4.0], dtype=np.float32)
    act = VelocityAction.from_normalized(extreme_norm, limits)

    assert np.isclose(act.vx, limits.max_fwd_vel_mps)
    assert np.isclose(act.vy, -limits.max_lat_vel_mps)
    assert np.isclose(act.vz, limits.max_vert_vel_mps)
    assert np.isclose(act.yaw_rate, -limits.max_yaw_rate_radps)
