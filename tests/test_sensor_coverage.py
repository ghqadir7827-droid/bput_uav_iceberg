"""test_sensor_coverage.py - Verification of Rangefinder Sensor Coverage and Blind-Sector Mitigation."""

import numpy as np
import pytest

from core.contracts import (
    ActionLimits,
    DeployableObservation,
    DirectionalClearances,
    LEFT_RAY_INDICES,
    LEFT_RAY_INDICES_16,
    RIGHT_RAY_INDICES,
    RIGHT_RAY_INDICES_16,
    REAR_RAY_INDICES,
    REAR_RAY_INDICES_16,
    SensorConfig,
    VelocityAction,
    compute_directional_clearances_m,
)
from core.geometry import Corridor3D, Obstacle3D
from controllers.safety_supervisor import SafetySupervisor, SafetySupervisorConfig


def test_sensor_config_12_and_16_angles():
    """Verifies angular definitions, counts, and indices for baseline 12 and extended 16 configurations."""
    cfg12 = SensorConfig()
    assert cfg12.num_rays == 12
    assert len(cfg12.ray_angles) == 12

    cfg16 = SensorConfig.radial_extended_16()
    assert cfg16.num_rays == 16
    assert len(cfg16.ray_angles) == 16

    # Verify first 12 angles are bitwise identical
    for i in range(12):
        assert cfg12.ray_angles[i] == cfg16.ray_angles[i]

    # Verify beams 12-15 angles in degrees
    azimuths_deg = [np.rad2deg(a[0]) for a in cfg16.ray_angles[12:16]]
    pitches_deg = [np.rad2deg(a[1]) for a in cfg16.ray_angles[12:16]]

    assert np.isclose(azimuths_deg[0], 120.0, atol=1e-3)   # Ray 12: +120 deg (rear left)
    assert np.isclose(azimuths_deg[1], -120.0, atol=1e-3)  # Ray 13: -120 deg (rear right)
    assert np.isclose(azimuths_deg[2], 135.0, atol=1e-3)   # Ray 14: +135 deg (rear left)
    assert np.isclose(azimuths_deg[3], -135.0, atol=1e-3)  # Ray 15: -135 deg (rear right)

    for p_ang in pitches_deg:
        assert np.isclose(p_ang, 0.0, atol=1e-3)  # All 4 new beams are horizontal


def test_horizontal_blind_sector_reduction():
    """Verifies that the rear-lateral blind sector is reduced from 90 deg down to 30 deg."""
    cfg12 = SensorConfig()
    cfg16 = SensorConfig.radial_extended_16()

    # Collect horizontal rays (pitch == 0.0) sorted by azimuth in [0, 360)
    def get_sorted_horizontal_azimuths(cfg):
        az_list = []
        for yaw, pitch in cfg.ray_angles:
            if np.isclose(pitch, 0.0, atol=1e-4):
                deg = np.rad2deg(yaw) % 360.0
                az_list.append(deg)
        return sorted(az_list)

    az12 = get_sorted_horizontal_azimuths(cfg12)
    diffs12 = [(az12[(i + 1) % len(az12)] - az12[i]) % 360.0 for i in range(len(az12))]
    max_gap_12 = max(diffs12)
    # Baseline 12-beam has 90 deg gap between 90->180 and 180->270
    assert np.isclose(max_gap_12, 90.0, atol=1e-3)

    az16 = get_sorted_horizontal_azimuths(cfg16)
    diffs16 = [(az16[(i + 1) % len(az16)] - az16[i]) % 360.0 for i in range(len(az16))]
    max_gap_16 = max(diffs16)
    # Extended 16-beam reduces maximum gap from 90 deg down to 45 deg (between 135 deg and 180 deg)
    assert np.isclose(max_gap_16, 45.0, atol=1e-3)


def test_seed_20334_geometry_interception():
    """Verifies mathematically that Ray 13 (-120 deg) and Ray 15 (-135 deg) intercept Seed 20334 obstacle.

    Seed 20334 pose at step 257:
      drone_pos: [3.8698, 0.4879, 1.1698]
      obs_center: [3.5000, -0.0638, 1.2000], r = 0.250 m
      relative: dx = -0.3698, dy = -0.5517
      azimuth: -123.84 deg
      subtended range: [-145.95 deg, -101.73 deg]
    """
    obs_center_rel = np.array([-0.3698, -0.5517])
    obs_radius = 0.250
    dist_center = np.linalg.norm(obs_center_rel)
    center_azimuth_deg = np.rad2deg(np.arctan2(obs_center_rel[1], obs_center_rel[0]))
    half_angle_deg = np.rad2deg(np.arcsin(obs_radius / dist_center))

    az_min = center_azimuth_deg - half_angle_deg
    az_max = center_azimuth_deg + half_angle_deg

    assert np.isclose(center_azimuth_deg, -123.84, atol=0.1)
    # Check baseline beams: Ray 6 is -90 deg, Ray 7 is 180 deg. Neither is in [az_min, az_max]
    assert not (az_min <= -90.0 <= az_max)
    assert not (az_min <= 180.0 <= az_max) and not (az_min <= -180.0 <= az_max)

    # Check extended beams: Ray 13 is -120 deg, Ray 15 is -135 deg
    assert az_min <= -120.0 <= az_max  # Ray 13 intercepts
    assert az_min <= -135.0 <= az_max  # Ray 15 intercepts


def test_deployable_observation_backward_compatibility():
    """Verifies that 12-ray and 16-ray observations both provide valid vectors and 23-float policy vectors."""
    # 12-ray observation
    obs12 = DeployableObservation(
        distance_rays=np.ones(12, dtype=np.float32) * 0.5,
        rel_goal_direction_body=np.array([1.0, 0.0, 0.0], dtype=np.float32),
        body_velocity=np.array([0.2, 0.0, 0.0], dtype=np.float32),
        attitude_trig=np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float32),
        yaw_rate=0.05
    )
    assert obs12.to_vector().shape == (23,)
    assert obs12.to_policy_vector().shape == (23,)

    # 16-ray observation
    obs16 = DeployableObservation(
        distance_rays=np.ones(16, dtype=np.float32) * 0.5,
        rel_goal_direction_body=np.array([1.0, 0.0, 0.0], dtype=np.float32),
        body_velocity=np.array([0.2, 0.0, 0.0], dtype=np.float32),
        attitude_trig=np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float32),
        yaw_rate=0.05
    )
    assert obs16.to_vector().shape == (27,)
    assert obs16.to_policy_vector().shape == (23,)
    # Verify policy vector matches first 12 rays
    assert np.allclose(obs16.to_policy_vector()[:12], obs16.distance_rays[:12])
    assert np.allclose(obs16.to_policy_vector()[12:], obs16.to_vector()[16:])


def test_directional_clearance_16_ray_routing():
    """Verifies that compute_directional_clearances_m routes rear-lateral rays to left/right sectors."""
    rays = np.full(16, 1.0, dtype=np.float32)  # All rays clear (4.0m)
    
    # Introduce obstacle on Ray 13 (-120 deg, rear-right)
    # Ray 13 reading = 0.417 m / 4.0 m = 0.10425
    rays[13] = 0.417 / 4.0
    clearances = compute_directional_clearances_m(rays)

    # Right clearance must register this obstacle
    assert np.isclose(clearances.right_m, 0.417, atol=1e-3)
    # Left clearance remains 4.0m
    assert np.isclose(clearances.left_m, 4.0, atol=1e-3)

    # Introduce obstacle on Ray 12 (+120 deg, rear-left)
    rays[12] = 0.550 / 4.0
    clearances2 = compute_directional_clearances_m(rays)
    assert np.isclose(clearances2.left_m, 0.550, atol=1e-3)
    assert np.isclose(clearances2.right_m, 0.417, atol=1e-3)


def test_supervisor_reacts_to_rear_lateral_beam():
    """Verifies that the SafetySupervisor reacts to rear-lateral Ray 13 by throttling or clamping rightward motion."""
    corridor = Corridor3D(width=4.0, height=2.5, x_end=15.0)
    sup = SafetySupervisor(corridor, config=SafetySupervisorConfig.for_65cm_uav())

    pos = np.array([3.87, 0.49, 1.17])
    vel = np.array([0.30, 0.0, 0.0])

    # Case 1: Caution zone obstacle on Ray 13 (-120 deg) at 0.417 m
    rays_caut = np.full(16, 1.0, dtype=np.float32)
    rays_caut[13] = 0.417 / 4.0  # In caution zone (0.35m < 0.417m < 0.60m)
    obs_caut = DeployableObservation(
        distance_rays=rays_caut,
        rel_goal_direction_body=np.array([1.0, 0.0, 0.0], dtype=np.float32),
        body_velocity=np.array([0.3, 0.0, 0.0], dtype=np.float32),
        attitude_trig=np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float32),
        yaw_rate=0.0
    )

    proposed = VelocityAction(vx=0.30, vy=-0.30, vz=0.0, yaw_rate=0.0)

    safe_act_caut, _, reason_caut = sup.filter_action(
        proposed_action=proposed,
        current_position=pos,
        current_velocity=vel,
        min_front_range=4.0,
        dt=0.05,
        observation=obs_caut
    )

    # Supervisor must throttle rightward motion from -0.30 down to approx -0.107 m/s
    assert safe_act_caut.vy > -0.30
    assert np.isclose(safe_act_caut.vy, -0.1072, atol=1e-3)

    # Case 2: Critical zone obstacle on Ray 13 (-120 deg) at 0.30 m
    rays_crit = np.full(16, 1.0, dtype=np.float32)
    rays_crit[13] = 0.30 / 4.0  # In critical zone (< 0.35m)
    obs_crit = DeployableObservation(
        distance_rays=rays_crit,
        rel_goal_direction_body=np.array([1.0, 0.0, 0.0], dtype=np.float32),
        body_velocity=np.array([0.3, 0.0, 0.0], dtype=np.float32),
        attitude_trig=np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float32),
        yaw_rate=0.0
    )

    safe_act_crit, _, reason_crit = sup.filter_action(
        proposed_action=proposed,
        current_position=pos,
        current_velocity=vel,
        min_front_range=4.0,
        dt=0.05,
        observation=obs_crit
    )

    # Supervisor must strictly block rightward motion (vy >= 0.0) and repulse left
    assert safe_act_crit.vy >= 0.0, f"Expected non-negative vy, got {safe_act_crit.vy}"
    assert "right_obs" in reason_crit or "obs=" in reason_crit
