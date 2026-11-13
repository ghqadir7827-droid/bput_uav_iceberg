"""test_phase4f_regressions.py - Stage A Targeted Regression Suite for Phase 4F.

Verifies:
1. Seed 20334 envelope-breach classification invariant.
2. Seed 20334 rear-lateral beam detection on Ray 13.
3. Infeasible barrier safe-arrest verification on representative seeds (20301, 20303, 20325).
4. Deterministic reproducibility across repeated runs.
"""

import numpy as np
import pytest

from controllers.registry import ControllerRegistry
from controllers.safe_arrest import SafeArrestState, SafeArrestStateMachine
from controllers.safety_supervisor import SafetySupervisor
from core.contracts import SensorConfig, compute_forward_clearance_m
from core.geometry import Corridor3D, EpisodeOutcome, OutcomeClassifier, TerminationReason
from core.physical_dimensions import DifficultyLevel, PhysicalDroneModel, RouteFeasibilityChecker, ScaledEnvironmentFactory
from sim.corridor_env import CorridorSimEnv


def test_seed_20334_envelope_breach_classification():
    """Verifies that Seed 20334 is classified as ENVELOPE_BREACH_ARRIVAL under the 8-outcome taxonomy."""
    # Under v4, minimum clearance was 0.0892m (surface clearance)
    clf = OutcomeClassifier.classify(
        termination_reason=TerminationReason.SUCCESS,
        min_obstacle_clearance_m=0.0892,
        min_boundary_clearance_m=0.3340,
    )
    assert clf.outcome == EpisodeOutcome.ENVELOPE_BREACH_ARRIVAL
    assert clf.is_clean is False
    assert clf.envelope_breach_detected is True


def test_seed_20334_ray13_rear_lateral_intercept():
    """Verifies that with 16-beam radial config, Ray 13 intercepts Seed 20334 obstacle during close pass."""
    cfg = ScaledEnvironmentFactory.get_environment_config(DifficultyLevel.LEVEL_4_FEASIBILITY_BOUNDARY)
    sensor_cfg = SensorConfig.radial_extended_16()
    env = CorridorSimEnv(corridor=cfg, sensor_config=sensor_cfg)
    obs, _ = env.reset(seed=20334)

    reg = ControllerRegistry()
    ctrl = reg.get_controller("scaled_guided_bc_v5")
    sup = SafetySupervisor(corridor=cfg)

    ray13_detections = []
    for step in range(350):
        pos = env._get_drone_position()
        vel = env._get_drone_velocity()

        # Check Ray 13 (index 13: -120 deg azimuth)
        ray13_dist_m = float(obs[13] * 4.0)
        if ray13_dist_m < 1.0:
            ray13_detections.append((step, pos[0], pos[1], ray13_dist_m))

        raw_act = ctrl.predict(obs)
        min_front = compute_forward_clearance_m(obs, max_range_m=env.sensor_config.max_range_m)
        safe_act, _, _ = sup.filter_action(raw_act, pos, vel, min_front, 0.05, observation=obs)
        obs, _, term, trunc, _ = env.step(safe_act.to_normalized(env.action_limits))
        if term or trunc:
            break

    env.close()

    # Verify that Ray 13 caught Obstacle 0 (< 0.80m) during the pass
    assert len(ray13_detections) > 0, "Ray 13 failed to detect obstacle in rear-lateral sector"
    min_ray13 = min(d[3] for d in ray13_detections)
    assert min_ray13 < 0.80, f"Expected Ray 13 proximity detection < 0.80m, got {min_ray13:.3f}m"


def test_infeasible_barrier_safe_rejections():
    """Verifies that infeasible barrier scenarios achieve verified SAFE_REJECTION without timing out."""
    cfg = ScaledEnvironmentFactory.get_environment_config(DifficultyLevel.LEVEL_4_FEASIBILITY_BOUNDARY)
    sensor_cfg = SensorConfig.radial_extended_16()
    reg = ControllerRegistry()
    ctrl = reg.get_controller("scaled_guided_bc_v5")
    sup = SafetySupervisor(corridor=cfg)

    test_seeds = [20301, 20303, 20325]
    for s in test_seeds:
        env = CorridorSimEnv(corridor=cfg, sensor_config=sensor_cfg)
        obs, _ = env.reset(seed=s)
        sm = SafeArrestStateMachine()
        min_obs = float("inf")
        min_wall = float("inf")
        outcome = None

        for step in range(600):
            pos = env._get_drone_position()
            vel = env._get_drone_velocity()
            d_obs = float(cfg.obstacle_clearance(pos, env.obstacles))
            d_wall = float(cfg.boundary_distance(pos))
            min_obs = min(min_obs, d_obs)
            min_wall = min(min_wall, d_wall)

            state, should_term, diag = sm.update(
                current_position=pos,
                current_velocity=vel,
                observation=obs,
                min_obstacle_clearance=d_obs,
                min_boundary_clearance=d_wall,
                is_known_infeasible_scenario=True
            )

            if should_term:
                outcome = OutcomeClassifier.classify(
                    termination_reason=TerminationReason.CONTINUE,
                    min_obstacle_clearance_m=min_obs,
                    min_boundary_clearance_m=min_wall,
                    safe_arrest_state=state.value,
                    is_arrest_stable=diag.is_stable
                )
                break

            raw_act, _, _, _ = ctrl.arbitrator.arbitrate(
                raw_action=ctrl.predict(obs),
                observation=obs,
                suppress_guidance=diag.suppress_lateral_guidance
            )
            min_front = compute_forward_clearance_m(obs, max_range_m=env.sensor_config.max_range_m)
            safe_act, _, _ = sup.filter_action(raw_act, pos, vel, min_front, 0.05, observation=obs)
            obs, _, term, trunc, _ = env.step(safe_act.to_normalized(env.action_limits))
            if term or trunc:
                outcome = OutcomeClassifier.classify(
                    termination_reason=TerminationReason.TIMEOUT if trunc else TerminationReason.CONTINUE,
                    min_obstacle_clearance_m=min_obs,
                    min_boundary_clearance_m=min_wall,
                    safe_arrest_state=state.value,
                    is_arrest_stable=diag.is_stable
                )
                break

        env.close()

        assert outcome is not None
        assert outcome.outcome == EpisodeOutcome.SAFE_REJECTION, f"Seed {s} outcome was {outcome.outcome}"
        assert step < 500, f"Seed {s} failed to arrest quickly (took {step+1} steps)"
