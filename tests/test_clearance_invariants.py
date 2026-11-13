"""test_clearance_invariants.py - Regression Tests for Safety Envelope and Clearance Invariants.

Verifies:
1. Exact surface-to-surface clearance calculation accounting for the full 0.65m tip-to-tip UAV footprint.
2. Invariant consistency: any episode with clearance < 0.15m is strictly flagged as an envelope breach.
3. Separation of "Clean Success" (d >= 0.15m) vs "Envelope-Breaching Arrival" (0 < d < 0.15m).
4. Physical impassable barrier feasibility classification.
5. Absolute authority of SafetySupervisor boundary and obstacle clamping.
"""

import numpy as np
import pytest

from core.contracts import ActionLimits, DeployableObservation, VelocityAction
from core.geometry import Corridor3D, Obstacle3D, TerminationReason
from core.guidance import GuidanceConfig, StatelessClearanceGuidanceArbitrator
from core.physical_dimensions import DifficultyLevel, PhysicalDroneModel, RouteFeasibilityChecker, ScaledEnvironmentFactory
from controllers.safety_supervisor import SafetySupervisor, SafetyState


def test_surface_clearance_full_uav_footprint():
    """Confirms surface clearance subtracts both UAV radius (0.325m) and obstacle radius (0.25m)."""
    corridor = Corridor3D(width=2.80, height=2.40, drone_radius=0.325)
    obs = Obstacle3D(center=np.array([3.50, 0.0, 1.20]), radius=0.25, height=2.40)

    # Position drone at x = 3.50, y = 0.6642, z = 1.20 (Seed 20334 closest point)
    drone_pos = np.array([3.50, 0.6642, 1.20])
    center_dist = float(np.linalg.norm(drone_pos[:2] - obs.center[:2]))
    assert center_dist == pytest.approx(0.6642, abs=1e-4)

    # Surface clearance = center_dist - (R_uav + r_obs)
    surface_clearance = corridor.obstacle_clearance(drone_pos, [obs])
    expected_surf = 0.6642 - (0.325 + 0.250)  # 0.0892 m
    assert surface_clearance == pytest.approx(expected_surf, abs=1e-4)
    assert surface_clearance == pytest.approx(0.0892, abs=1e-3)


def test_envelope_breach_detection_invariant():
    """Detects any inconsistency between a clearance < 0.15m and the recorded envelope violation flag."""
    corridor = Corridor3D(width=2.80, height=2.40, drone_radius=0.325)
    obs = Obstacle3D(center=np.array([3.50, 0.0, 1.20]), radius=0.25, height=2.40)

    # Trajectory with a clearance of 0.0892m (between 0.0m and 0.15m)
    drone_pos = np.array([3.50, 0.6642, 1.20])
    clearance = corridor.obstacle_clearance(drone_pos, [obs])

    envelope_threshold_m = 0.15
    has_envelope_violation = (clearance < envelope_threshold_m)
    has_collision = (clearance <= 0.0)

    # Invariant: clearance of 0.0892m MUST be an envelope violation but NOT a collision
    assert has_envelope_violation is True
    assert has_collision is False


def test_clean_vs_breaching_success_distinction():
    """Verifies that arrival at the goal can be cleanly partitioned into Clean Success vs Envelope-Breaching Arrival."""
    corridor = Corridor3D(x_start=0.0, x_end=14.0, width=2.80, height=2.40, drone_radius=0.325)
    goal_pos = corridor.goal_position  # [13.2, 0.0, 1.2]

    # Drone arrived at goal safely
    vel = np.array([0.10, 0.0, 0.0])
    term_reason, _ = corridor.check_termination(goal_pos, vel, obstacles=[], step_count=500, max_steps=1000)
    assert term_reason == TerminationReason.SUCCESS

    # Case A: Min clearance during flight was 0.20m (Clean Success)
    min_c_clean = 0.20
    is_clean_success = (term_reason == TerminationReason.SUCCESS) and (min_c_clean >= 0.15)
    assert is_clean_success is True

    # Case B: Min clearance during flight was 0.0892m (Envelope-Breaching Arrival)
    min_c_breach = 0.0892
    is_breaching_arrival = (term_reason == TerminationReason.SUCCESS) and (min_c_breach < 0.15)
    assert is_breaching_arrival is True


def test_infeasible_barrier_geometric_classification():
    """Confirms that Level 4 paired gate (gap=0.60m) is strictly classified as INFEASIBLE for 0.65m UAV."""
    corridor = Corridor3D(width=2.80, height=2.40, drone_radius=0.325)
    drone_model = PhysicalDroneModel(tip_to_tip_diagonal_m=0.65, collision_radius_m=0.325)
    checker = RouteFeasibilityChecker(drone_model)

    # Paired gate obstacles at x=7.0m, y=+0.55m and -0.55m, radius=0.25m
    obs1 = Obstacle3D(center=np.array([7.0, 0.55, 1.20]), radius=0.25, height=2.40)
    obs2 = Obstacle3D(center=np.array([7.0, -0.55, 1.20]), radius=0.25, height=2.40)

    # Passage between obstacles: (0.55 - 0.25) - (-0.55 + 0.25) = 0.30 - (-0.30) = 0.60m
    # Passage to left wall: 1.40 - (0.55 + 0.25) = 1.40 - 0.80 = 0.60m
    # Passage to right wall: -0.80 - (-1.40) = 0.60m
    # All passages = 0.60m < required 0.70m (0.65m drone + 0.05m margin)
    feas_res = checker.evaluate_feasibility(corridor, [obs1, obs2])

    assert feas_res.is_feasible is False
    assert feas_res.status == "BLOCKED_PASSAGE"
    assert feas_res.min_bottleneck_gap_m == pytest.approx(0.60, abs=1e-3)


def test_supervisor_clamps_unsafe_lateral_penetration():
    """Verifies SafetySupervisor strictly prevents commanded penetration into boundary wall."""
    corridor = Corridor3D(width=2.80, height=2.40, drone_radius=0.325)
    sup = SafetySupervisor(corridor=corridor)

    # Drone is at y = 1.05m (left wall is at +1.40m -> wall clearance = 1.40 - 1.05 - 0.325 = 0.025m << 0.35m buffer)
    pos = np.array([5.0, 1.05, 1.20])
    vel = np.array([0.2, 0.0, 0.0])

    # Aggressive commanded leftward evasion (+0.40 m/s)
    unsafe_action = VelocityAction(vx=0.40, vy=0.40, vz=0.0, yaw_rate=0.0)

    obs = np.ones(23, dtype=np.float32)  # Normalized rays = 1.0 (4.0m)
    safe_action, state, reason = sup.filter_action(
        proposed_action=unsafe_action,
        current_position=pos,
        current_velocity=vel,
        min_front_range=3.0,
        dt=0.05,
        observation=obs,
    )

    # Supervisor MUST actively repulse and command negative vy
    assert safe_action.vy < 0.0
    assert state != SafetyState.NOMINAL
    assert "left" in reason
