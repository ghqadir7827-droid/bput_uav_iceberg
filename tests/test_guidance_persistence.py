"""test_guidance_persistence.py - Verification of Directional-Persistence Tie-Breaking.

Validates that:
1. Symmetric obstacles do not induce rapid 20 Hz chatter.
2. Small sensor fluctuations (+/- 0.03m) do not cause direction flipping.
3. Material changes in obstacle geometry (Delta > 0.15m) responsively override persistence.
4. Asymmetric corridor boundaries (wall proximity) override persistence to prevent wall contact.
5. Latch resets deterministically when front obstacle is cleared, in goal zone, or on reset().
"""

import numpy as np
import pytest

from core.contracts import VelocityAction
from core.guidance import GuidanceConfig, StatelessClearanceGuidanceArbitrator


def create_base_obs(
    front_dist: float = 1.0,
    left_dist: float = 1.8,
    right_dist: float = 1.8,
    wall_left: float = 2.0,
    wall_right: float = 2.0,
    dist_goal: float = 5.0,
) -> np.ndarray:
    """Helper to create a 23-element deployable observation with specified clearances."""
    obs = np.ones(23, dtype=np.float32)
    # Rangefinder rays (normalized by 4.0m)
    obs[0] = front_dist / 4.0   # Ray 0: front center
    obs[1] = left_dist / 4.0    # Ray 1: front left 30 deg
    obs[2] = right_dist / 4.0   # Ray 2: front right 30 deg
    obs[3] = left_dist / 4.0    # Ray 3: front left 60 deg
    obs[4] = right_dist / 4.0   # Ray 4: front right 60 deg
    obs[5] = wall_left / 4.0    # Ray 5: left 90 deg wall
    obs[6] = wall_right / 4.0   # Ray 6: right 90 deg wall
    obs[12:15] = [dist_goal / 2.0, 0.0, 0.0]  # Goal vector
    return obs


def test_symmetric_obstacle_no_chatter():
    """Verify that a centered symmetric obstacle latches a single evasion direction without chattering."""
    cfg = GuidanceConfig.for_65cm_uav_v3(drone_radius=0.325)
    arb = StatelessClearanceGuidanceArbitrator(config=cfg)

    # Initial symmetric obstacle at 1.0m (deadband range: left_dist == right_dist)
    obs = create_base_obs(front_dist=1.0, left_dist=1.8, right_dist=1.8, wall_left=2.2, wall_right=2.2)
    raw_act = VelocityAction(vx=0.60, vy=0.0, vz=0.0, yaw_rate=0.0)

    # Step 1: Initial tie-break chooses default (LEFT)
    g1, d1, _, _ = arb.arbitrate(raw_act, obs)
    assert d1 == "TIE_BREAK_LEFT"
    assert g1.vy > 0.0

    # Steps 2..20: Simulate small lateral drift as drone moves left (e.g. wall clearances shift slightly)
    # Wall right becomes slightly larger: wall_l=2.15, wall_r=2.25 (diff=0.10m, formerly caused chatter)
    directions = []
    for step in range(20):
        # Drone drifts slightly left: wall_l decreases, wall_r increases
        drift_obs = create_base_obs(
            front_dist=1.0 - step * 0.01,
            left_dist=1.78,
            right_dist=1.82,  # slight clearance diff within deadband
            wall_left=2.15,
            wall_right=2.25,
        )
        g_act, d, _, _ = arb.arbitrate(raw_act, drift_obs)
        directions.append(d)

    # All steps must persist in LEFT tie-break; ZERO flipping to RIGHT
    assert all(d == "TIE_BREAK_LEFT" for d in directions), f"Chatter observed: {set(directions)}"


def test_small_sensor_fluctuations():
    """Verify that random sensor noise (+/- 0.03m) in the deadband does not cause flipping."""
    cfg = GuidanceConfig.for_65cm_uav_v3(drone_radius=0.325)
    arb = StatelessClearanceGuidanceArbitrator(config=cfg)

    obs = create_base_obs(front_dist=1.0, left_dist=1.8, right_dist=1.8, wall_left=2.0, wall_right=2.0)
    raw_act = VelocityAction(vx=0.60, vy=0.0, vz=0.0, yaw_rate=0.0)

    # Latch initial direction
    _, d0, _, _ = arb.arbitrate(raw_act, obs)
    assert d0 == "TIE_BREAK_LEFT"

    rng = np.random.RandomState(42)
    # 50 iterations with sensor noise
    for _ in range(50):
        noise_l = rng.uniform(-0.03, 0.03)
        noise_r = rng.uniform(-0.03, 0.03)
        noisy_obs = create_base_obs(
            front_dist=1.0,
            left_dist=1.8 + noise_l,
            right_dist=1.8 + noise_r,
            wall_left=2.0,
            wall_right=2.0,
        )
        _, d, _, _ = arb.arbitrate(raw_act, noisy_obs)
        assert d in ["TIE_BREAK_LEFT", "LEFT"], f"Noise caused flip to opposite side: {d}"


def test_changing_obstacle_geometry_override():
    """Verify that a material opening on one side immediately overrides persistence."""
    cfg = GuidanceConfig.for_65cm_uav_v3(drone_radius=0.325)
    arb = StatelessClearanceGuidanceArbitrator(config=cfg)

    obs = create_base_obs(front_dist=1.0, left_dist=1.8, right_dist=1.8, wall_left=2.0, wall_right=2.0)
    raw_act = VelocityAction(vx=0.60, vy=0.0, vz=0.0, yaw_rate=0.0)

    # Latch LEFT
    _, d0, _, _ = arb.arbitrate(raw_act, obs)
    assert d0 == "TIE_BREAK_LEFT"

    # Now geometry changes materially: left side closes to 0.8m, right side opens to 2.5m (diff = -1.7m)
    new_obs = create_base_obs(front_dist=1.0, left_dist=0.8, right_dist=2.5, wall_left=2.0, wall_right=2.0)
    g_act, d_new, _, _ = arb.arbitrate(raw_act, new_obs)

    # Must override persistence and command RIGHT evasion
    assert d_new == "RIGHT"
    assert g_act.vy < 0.0


def test_asymmetric_corridor_boundary_override():
    """Verify that approaching a wall dangerously close overrides persistence to prevent collision."""
    cfg = GuidanceConfig.for_65cm_uav_v3(drone_radius=0.325)
    arb = StatelessClearanceGuidanceArbitrator(config=cfg)

    obs = create_base_obs(front_dist=1.0, left_dist=1.8, right_dist=1.8, wall_left=2.0, wall_right=2.0)
    raw_act = VelocityAction(vx=0.60, vy=0.0, vz=0.0, yaw_rate=0.0)

    # Latch LEFT
    _, d0, _, _ = arb.arbitrate(raw_act, obs)
    assert d0 == "TIE_BREAK_LEFT"

    # Left wall becomes critically tight (< 0.675m)
    tight_obs = create_base_obs(
        front_dist=1.0,
        left_dist=1.8,
        right_dist=1.8,
        wall_left=0.50,  # tight!
        wall_right=2.0,
    )
    g_act, d_tight, _, _ = arb.arbitrate(raw_act, tight_obs)

    # Must immediately enforce WALL_AVOID_RIGHT
    assert d_tight == "WALL_AVOID_RIGHT"
    assert g_act.vy < 0.0


def test_documented_reset_conditions():
    """Verify documented reset conditions for the directional persistence latch."""
    cfg = GuidanceConfig.for_65cm_uav_v3(drone_radius=0.325)
    arb = StatelessClearanceGuidanceArbitrator(config=cfg)
    raw_act = VelocityAction(vx=0.60, vy=0.0, vz=0.0, yaw_rate=0.0)

    # 1. Active evasion latches direction
    obs = create_base_obs(front_dist=1.0, left_dist=1.8, right_dist=1.8)
    arb.arbitrate(raw_act, obs)
    assert arb.last_dir == "TIE_BREAK_LEFT"

    # Reset Condition A: Obstacle cleared (c_front >= activation_threshold_m = 1.375m)
    clear_obs = create_base_obs(front_dist=2.5, left_dist=2.5, right_dist=2.5)
    _, d_clear, _, _ = arb.arbitrate(raw_act, clear_obs)
    assert d_clear == "NONE"
    assert arb.last_dir == "NONE"

    # Re-latch
    arb.arbitrate(raw_act, obs)
    assert arb.last_dir == "TIE_BREAK_LEFT"

    # Reset Condition B: In terminal goal zone (goal distance < 1.0m, norm < 0.50)
    goal_obs = create_base_obs(front_dist=1.0, left_dist=1.8, right_dist=1.8, dist_goal=0.8)
    _, d_goal, _, _ = arb.arbitrate(raw_act, goal_obs)
    assert d_goal == "NONE"
    assert arb.last_dir == "NONE"

    # Re-latch
    arb.arbitrate(raw_act, obs)
    assert arb.last_dir == "TIE_BREAK_LEFT"

    # Reset Condition C: Explicit arbitrator.reset()
    arb.reset()
    assert arb.last_dir == "NONE"
