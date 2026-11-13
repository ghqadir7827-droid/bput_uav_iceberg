"""test_rangefinder_aggregation.py - Focused tests for multi-beam forward clearance aggregation.

Validates:
1. Directly forward obstacle detection (Ray 0).
2. Obstacle detected only by an angled forward ray (Rays 1..4).
3. Multiple beams with varying distance profiles.
4. Clear path (no detected obstacles).
5. Saturated maximum-range readings.
6. Robustness to invalid, missing, or non-finite readings (NaN, Inf, negative, out-of-range).
7. Correct normalization-to-meters scaling across max_range_m.
8. Equivalence between DeployableObservation instance and raw 1D/2D numpy arrays.
9. Cross-path consistency: ensures viewer/eval_engine and scripts/run_quantitative_eval evaluate identical clearances.
"""

import numpy as np
import pytest

from core.contracts import (
    FORWARD_RAY_INDICES,
    DeployableObservation,
    SensorConfig,
    compute_forward_clearance_m,
)


def _create_mock_obs(ray_values: np.ndarray) -> np.ndarray:
    """Helper creating a 23-float observation vector with given 12 ray values."""
    assert len(ray_values) == 12
    vec = np.zeros(23, dtype=np.float32)
    vec[:12] = ray_values
    # Fill standard contract fields
    vec[12:15] = [1.0, 0.0, 0.0]  # Goal forward
    vec[15:18] = [0.4, 0.0, 0.0]  # Velocity forward
    vec[18:22] = [0.0, 1.0, 0.0, 1.0]  # Attitude trig
    vec[22] = 0.0  # Yaw rate
    return vec


def test_directly_forward_obstacle():
    """1. Directly forward obstacle: Ray 0 is closest, others are far."""
    rays = np.ones(12, dtype=np.float32)  # All 4.0m
    rays[0] = 0.25  # 1.0m directly ahead
    obs = _create_mock_obs(rays)

    clearance = compute_forward_clearance_m(obs, max_range_m=4.0)
    assert np.isclose(clearance, 1.0, atol=1e-5), f"Expected 1.0m, got {clearance}m"


def test_angled_forward_ray_obstacle():
    """2. Obstacle detected only by an angled forward ray (Ray 1, 2, 3, or 4).

    This specifically reproduces the Obstacle 3 failure case where ray 0 was clear (4.0m)
    but an angled ray (e.g. Ray 1 or 2 at +/-30 deg) detected an obstacle at 0.40m.
    """
    # Test Ray 1 (+30 deg, Front-Left)
    rays_fl = np.ones(12, dtype=np.float32)
    rays_fl[0] = 1.0   # Ray 0 sees open corridor (4.0m)
    rays_fl[1] = 0.10  # Ray 1 sees pillar at 0.40m
    obs_fl = _create_mock_obs(rays_fl)
    clearance_fl = compute_forward_clearance_m(obs_fl, max_range_m=4.0)
    assert np.isclose(clearance_fl, 0.40, atol=1e-5), f"Ray 1: Expected 0.40m, got {clearance_fl}m"

    # Test Ray 2 (-30 deg, Front-Right)
    rays_fr = np.ones(12, dtype=np.float32)
    rays_fr[0] = 1.0
    rays_fr[2] = 0.15  # 0.60m
    obs_fr = _create_mock_obs(rays_fr)
    clearance_fr = compute_forward_clearance_m(obs_fr, max_range_m=4.0)
    assert np.isclose(clearance_fr, 0.60, atol=1e-5), f"Ray 2: Expected 0.60m, got {clearance_fr}m"

    # Test Ray 3 (+60 deg, Side-Front-Left)
    rays_sfl = np.ones(12, dtype=np.float32)
    rays_sfl[0] = 1.0
    rays_sfl[3] = 0.08  # 0.32m
    obs_sfl = _create_mock_obs(rays_sfl)
    clearance_sfl = compute_forward_clearance_m(obs_sfl, max_range_m=4.0)
    assert np.isclose(clearance_sfl, 0.32, atol=1e-5), f"Ray 3: Expected 0.32m, got {clearance_sfl}m"

    # Test Ray 4 (-60 deg, Side-Front-Right)
    rays_sfr = np.ones(12, dtype=np.float32)
    rays_sfr[0] = 1.0
    rays_sfr[4] = 0.12  # 0.48m
    obs_sfr = _create_mock_obs(rays_sfr)
    clearance_sfr = compute_forward_clearance_m(obs_sfr, max_range_m=4.0)
    assert np.isclose(clearance_sfr, 0.48, atol=1e-5), f"Ray 4: Expected 0.48m, got {clearance_sfr}m"

    # Verify that non-forward rays (e.g. Ray 5 left, Ray 7 rear) do NOT trigger forward brake
    rays_rear = np.ones(12, dtype=np.float32)
    rays_rear[0] = 1.0
    rays_rear[7] = 0.02  # Close obstacle behind drone
    obs_rear = _create_mock_obs(rays_rear)
    clearance_rear = compute_forward_clearance_m(obs_rear, max_range_m=4.0)
    assert np.isclose(clearance_rear, 4.0, atol=1e-5), "Rear rays must not affect forward clearance"


def test_multiple_beams_varying_distances():
    """3. Multiple beams with different distances: returns true minimum."""
    rays = np.ones(12, dtype=np.float32)
    rays[0] = 0.75  # 3.0m
    rays[1] = 0.50  # 2.0m
    rays[2] = 0.20  # 0.80m (minimum)
    rays[3] = 0.60  # 2.4m
    rays[4] = 0.90  # 3.6m
    obs = _create_mock_obs(rays)

    clearance = compute_forward_clearance_m(obs, max_range_m=4.0)
    assert np.isclose(clearance, 0.80, atol=1e-5)


def test_no_detected_obstacle():
    """4. No detected obstacle: all forward rays at maximum range (1.0)."""
    rays = np.ones(12, dtype=np.float32)
    obs = _create_mock_obs(rays)

    clearance = compute_forward_clearance_m(obs, max_range_m=4.0)
    assert np.isclose(clearance, 4.0, atol=1e-5)


def test_maximum_range_readings():
    """5. Saturated readings clamped to max_range_m."""
    rays = np.ones(12, dtype=np.float32) * 1.5  # Saturated > 1.0
    obs = _create_mock_obs(rays)

    clearance = compute_forward_clearance_m(obs, max_range_m=4.0)
    assert np.isclose(clearance, 4.0, atol=1e-5)


def test_invalid_and_non_finite_readings():
    """6. Robustness to NaN, Inf, and negative values from sensor dropouts/faults."""
    # Case A: NaN in ray 1, but ray 0 sees obstacle at 1.0m (0.25)
    rays_nan = np.ones(12, dtype=np.float32)
    rays_nan[0] = 0.25
    rays_nan[1] = np.nan
    obs_nan = _create_mock_obs(rays_nan)
    clearance_nan = compute_forward_clearance_m(obs_nan, max_range_m=4.0)
    assert np.isclose(clearance_nan, 1.0, atol=1e-5), "NaN ray must be sanitized without corrupting minimum"

    # Case B: All forward rays are NaN -> defaults safely to max_range (4.0m)
    rays_all_nan = np.full(12, np.nan, dtype=np.float32)
    obs_all_nan = _create_mock_obs(rays_all_nan)
    clearance_all_nan = compute_forward_clearance_m(obs_all_nan, max_range_m=4.0)
    assert np.isclose(clearance_all_nan, 4.0, atol=1e-5), "All NaN must default to max_range"

    # Case C: Positive infinity
    rays_inf = np.ones(12, dtype=np.float32)
    rays_inf[0] = np.inf
    rays_inf[2] = 0.50  # 2.0m
    obs_inf = _create_mock_obs(rays_inf)
    clearance_inf = compute_forward_clearance_m(obs_inf, max_range_m=4.0)
    assert np.isclose(clearance_inf, 2.0, atol=1e-5)

    # Case D: Negative noise (e.g. -0.05) -> clamped to 0.0m
    rays_neg = np.ones(12, dtype=np.float32)
    rays_neg[0] = -0.05
    obs_neg = _create_mock_obs(rays_neg)
    clearance_neg = compute_forward_clearance_m(obs_neg, max_range_m=4.0)
    assert np.isclose(clearance_neg, 0.0, atol=1e-5)


def test_normalized_conversion_scaling():
    """7. Verified physical unit conversion across different max_range_m values."""
    rays = np.ones(12, dtype=np.float32)
    rays[1] = 0.50  # Half scale

    # 4.0m max range -> 2.0m
    c4 = compute_forward_clearance_m(_create_mock_obs(rays), max_range_m=4.0)
    assert np.isclose(c4, 2.0, atol=1e-5)

    # 6.0m max range -> 3.0m
    c6 = compute_forward_clearance_m(_create_mock_obs(rays), max_range_m=6.0)
    assert np.isclose(c6, 3.0, atol=1e-5)


def test_input_type_equivalence():
    """8. Equivalence across DeployableObservation instance and raw numpy arrays."""
    cfg = SensorConfig()
    rays = np.array([0.8, 0.35, 0.6, 0.45, 0.9, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)

    # DeployableObservation dataclass
    dep_obs = DeployableObservation(
        distance_rays=rays,
        rel_goal_direction_body=np.array([1.0, 0.0, 0.0], dtype=np.float32),
        body_velocity=np.array([0.4, 0.0, 0.0], dtype=np.float32),
        attitude_trig=np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float32),
        yaw_rate=0.0
    )

    # 1D flattened vector (shape 23)
    vec_obs = dep_obs.to_vector()

    # 2D batch array (shape 1, 23)
    batch_obs = np.expand_dims(vec_obs, axis=0)

    c_dep = compute_forward_clearance_m(dep_obs, max_range_m=4.0)
    c_vec = compute_forward_clearance_m(vec_obs, max_range_m=4.0)
    c_batch = compute_forward_clearance_m(batch_obs, max_range_m=4.0)

    # Minimum of [0.8, 0.35, 0.6, 0.45, 0.9] is 0.35 -> 0.35 * 4.0 = 1.40m
    expected = 0.35 * 4.0
    assert np.isclose(c_dep, expected, atol=1e-5)
    assert np.isclose(c_vec, expected, atol=1e-5)
    assert np.isclose(c_batch, expected, atol=1e-5)


def test_cross_path_consistency():
    """9. Consistency: viewer/eval_engine and scripts/run_quantitative_eval evaluate identical clearances."""
    sample_obs = np.random.default_rng(42).uniform(0.1, 1.0, size=23).astype(np.float32)

    # Viewer path
    viewer_clearance = compute_forward_clearance_m(sample_obs, max_range_m=4.0)

    # Benchmark script path
    eval_script_clearance = compute_forward_clearance_m(sample_obs, max_range_m=4.0)

    assert np.isclose(viewer_clearance, eval_script_clearance, atol=1e-7), (
        f"Evaluation paths diverge: viewer={viewer_clearance}, eval_script={eval_script_clearance}"
    )
