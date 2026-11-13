"""test_adaptive_guidance.py - Unit Tests for Corridor-Width-Adaptive Lateral Guidance (v4).

Validates:
1. Adaptive guidance configuration parameters.
2. Mode switching between nominal (wide corridor) and adaptive (narrow corridor).
3. Graceful fallback on missing, non-finite, or noisy lateral rangefinder readings.
4. Retention of directional persistence for tie-breaking.
5. Absolute preservation of SafetySupervisor boundary and obstacle clamping authority.
6. Seamless registration in ControllerRegistry.
"""

import numpy as np
import pytest

from core.contracts import ActionLimits, DeployableObservation, VelocityAction
from core.geometry import Corridor3D
from core.guidance import GuidanceConfig, StatelessClearanceGuidanceArbitrator
from controllers.registry import ControllerRegistry
from controllers.safety_supervisor import SafetySupervisor, SafetyState


def test_adaptive_config_defaults():
    cfg = GuidanceConfig.for_65cm_uav_v4_adaptive(drone_radius=0.325)
    assert cfg.enable_corridor_adaptation is True
    assert cfg.persistent_tie_breaking is True
    assert cfg.min_lateral_override_mps == 0.40
    assert cfg.adaptive_lateral_override_mps == 0.30
    assert cfg.adaptive_wall_threshold_m == 0.80
    assert cfg.adaptive_corridor_width_m == 3.00
    assert cfg.activation_threshold_m == pytest.approx(1.375)
    assert cfg.forward_ray_indices == (0, 1, 2, 3, 4)


def test_adaptive_wide_corridor_nominal_speed():
    cfg = GuidanceConfig.for_65cm_uav_v4_adaptive(drone_radius=0.325)
    arb = StatelessClearanceGuidanceArbitrator(config=cfg)

    # Simulated wide corridor (Level 1 / Level 2):
    # Front obstacle at 1.0m (ray 0), left wall at 2.0m (ray 5), right wall at 2.0m (ray 6)
    # Total width = 4.0m > 3.0m, min wall = 2.0m > 0.80m
    rays = np.ones(12, dtype=np.float32) * 0.90  # 3.6m default
    rays[0] = 1.0 / 4.0  # Front = 1.0m (< 1.375m -> activates guidance)
    rays[5] = 2.0 / 4.0  # Left wall = 2.0m
    rays[6] = 2.0 / 4.0  # Right wall = 2.0m

    obs = np.zeros(23, dtype=np.float32)
    obs[0:12] = rays
    obs[12:15] = [0.8, 0.0, 0.0]  # Far from goal

    raw_act = VelocityAction(vx=0.40, vy=0.0, vz=0.0, yaw_rate=0.0)
    guided_act, sel_dir, c_left, c_right = arb.arbitrate(raw_act, obs)

    # In wide corridor, override speed should be nominal 0.40 m/s
    assert abs(guided_act.vy) == pytest.approx(0.40, abs=1e-3)
    assert "LEFT" in sel_dir or "RIGHT" in sel_dir


def test_adaptive_narrow_corridor_speed_reduction():
    cfg = GuidanceConfig.for_65cm_uav_v4_adaptive(drone_radius=0.325)
    arb = StatelessClearanceGuidanceArbitrator(config=cfg)

    # Simulated narrow corridor (Level 4: W = 2.80m):
    # Front obstacle at 1.0m, left wall at 1.40m, right wall at 1.40m
    # Total width = 2.80m < 3.00m
    rays = np.ones(12, dtype=np.float32) * 0.80
    rays[0] = 1.0 / 4.0   # Front = 1.0m
    rays[5] = 1.40 / 4.0  # Left wall = 1.40m
    rays[6] = 1.40 / 4.0  # Right wall = 1.40m

    obs = np.zeros(23, dtype=np.float32)
    obs[0:12] = rays
    obs[12:15] = [0.8, 0.0, 0.0]

    raw_act = VelocityAction(vx=0.40, vy=0.0, vz=0.0, yaw_rate=0.0)
    guided_act, sel_dir, c_left, c_right = arb.arbitrate(raw_act, obs)

    # In narrow corridor, override speed should be reduced to 0.30 m/s
    assert abs(guided_act.vy) == pytest.approx(0.30, abs=1e-3)


def test_adaptive_wall_proximity_trigger():
    cfg = GuidanceConfig.for_65cm_uav_v4_adaptive(drone_radius=0.325)
    arb = StatelessClearanceGuidanceArbitrator(config=cfg)

    # Drone is offset near a wall: total width is 3.5m, but left wall is 0.70m (< 0.80m)
    rays = np.ones(12, dtype=np.float32) * 0.80
    rays[0] = 1.0 / 4.0   # Front = 1.0m
    rays[5] = 0.70 / 4.0  # Left wall = 0.70m (< 0.80m threshold)
    rays[6] = 2.80 / 4.0  # Right wall = 2.80m

    obs = np.zeros(23, dtype=np.float32)
    obs[0:12] = rays
    obs[12:15] = [0.8, 0.0, 0.0]

    raw_act = VelocityAction(vx=0.40, vy=0.0, vz=0.0, yaw_rate=0.0)
    guided_act, sel_dir, _, _ = arb.arbitrate(raw_act, obs)

    # Close wall proximity should trigger narrow override speed of 0.30 m/s
    assert abs(guided_act.vy) == pytest.approx(0.30, abs=1e-3)


def test_adaptive_sensor_fault_fallback():
    cfg = GuidanceConfig.for_65cm_uav_v4_adaptive(drone_radius=0.325)
    arb = StatelessClearanceGuidanceArbitrator(config=cfg)

    # Non-finite or corrupt ray values
    rays = np.ones(12, dtype=np.float32) * 0.50
    rays[0] = 1.0 / 4.0
    rays[5] = np.nan       # Left wall sensor dropout
    rays[6] = np.inf       # Right wall sensor glitch

    obs = np.zeros(23, dtype=np.float32)
    obs[0:12] = rays
    obs[12:15] = [0.8, 0.0, 0.0]

    raw_act = VelocityAction(vx=0.40, vy=0.0, vz=0.0, yaw_rate=0.0)
    guided_act, sel_dir, _, _ = arb.arbitrate(raw_act, obs)

    # Should safely fallback to nominal override (0.40 m/s) without NaN
    assert np.isfinite(guided_act.vy)
    assert abs(guided_act.vy) == pytest.approx(0.40, abs=1e-3)


def test_adaptive_tie_break_persistence():
    cfg = GuidanceConfig.for_65cm_uav_v4_adaptive(drone_radius=0.325)
    arb = StatelessClearanceGuidanceArbitrator(config=cfg)

    # Step 1: Initial symmetric obstacle -> tie-break LEFT
    rays = np.ones(12, dtype=np.float32) * 0.50
    rays[0] = 1.0 / 4.0
    rays[1] = 1.0 / 4.0
    rays[2] = 1.0 / 4.0
    obs = np.zeros(23, dtype=np.float32)
    obs[0:12] = rays
    obs[12:15] = [0.8, 0.0, 0.0]

    raw_act = VelocityAction(vx=0.40, vy=0.0, vz=0.0, yaw_rate=0.0)
    _, dir1, _, _ = arb.arbitrate(raw_act, obs)
    assert "LEFT" in dir1

    # Step 2: Next step within symmetry deadband -> should maintain LEFT persistence
    _, dir2, _, _ = arb.arbitrate(raw_act, obs)
    assert dir2 == "TIE_BREAK_LEFT"


def test_adaptive_supervisor_invariance():
    # Verify SafetySupervisor completely bounds/overrides adaptive controller action
    c = Corridor3D(width=2.80, height=2.40, drone_radius=0.325)
    sup = SafetySupervisor(corridor=c)

    # UAV is near left wall (y = 1.00m, wall at 1.40m -> wall clearance = 1.40 - 1.00 - 0.325 = 0.075m < 0.35m buffer)
    pos = np.array([5.0, 1.00, 1.20])
    vel = np.array([0.2, 0.1, 0.0])

    # Candidate v4 proposes +0.30 m/s (further left towards wall)
    v4_proposed = VelocityAction(vx=0.40, vy=0.30, vz=0.0, yaw_rate=0.0)

    safe_act, state, _ = sup.filter_action(
        proposed_action=v4_proposed,
        current_position=pos,
        current_velocity=vel,
        min_front_range=2.0,
        dt=0.05,
        observation=np.zeros(23, dtype=np.float32),
    )

    # Supervisor MUST actively intervene and reject/repulse leftward motion (vy must be <= 0.0)
    assert safe_act.vy <= 0.0
    assert state != SafetyState.NOMINAL


def test_adaptive_controller_registration():
    registry = ControllerRegistry()
    ctrl = registry.get_controller("scaled_guided_bc_v4")
    assert ctrl is not None
    assert ctrl.metadata.is_implemented is True
    assert ctrl.metadata.controller_id == "scaled_guided_bc_v4"
    assert "Corridor-Adaptive" in ctrl.metadata.display_name

