"""test_mock_hil_pipeline.py - Stage 1 Hardware-in-the-Loop (HIL) Loopback Test.

Validates the full inference, guidance, safety filtering, and mock FC adapter loop:
1. Connects to MockFlightControllerAdapter.
2. Ingests mock telemetry and rangefinder inputs to construct DeployableObservation.
3. Evaluates BC v2 forward pass, guidance arbitration, and safety supervisor filtering.
4. Transmits safe velocity commands to mock FC adapter.
5. Verifies watchdog triggers: telemetry staleness (> 0.40s) and link loss (> 1.50s).
6. Verifies that RadioLinkCrossFlightAdapter strictly maintains read-only software lock.
"""

import time
import numpy as np
import pytest

from controllers.bc_policy import BehaviorCloningPolicyController
from controllers.safety_supervisor import SafetyState, SafetySupervisor, SafetySupervisorConfig
from core.contracts import ActionLimits, DeployableObservation, VelocityAction, compute_forward_clearance_m
from core.geometry import Corridor3D
from core.guidance import GuidanceConfig, StatelessClearanceGuidanceArbitrator
from hardware.fc_adapter import FlightMode, MockFlightControllerAdapter, RadioLinkCrossFlightAdapter


def test_crossflight_adapter_read_only_safety_lock():
    """Verify that RadioLinkCrossFlightAdapter strictly prevents unauthorized command transmission."""
    adapter = RadioLinkCrossFlightAdapter(read_only=True)
    assert adapter.is_read_only is True

    # Connecting should succeed in read-only mode
    assert adapter.connect("/dev/ttyAMA0", 115200) is True

    # Attempting to transmit a velocity command MUST raise PermissionError
    act = VelocityAction(vx=0.5, vy=0.0, vz=0.0, yaw_rate=0.0)
    with pytest.raises(PermissionError) as exc_info:
        adapter.send_velocity_command(act)

    assert "READ-ONLY mode" in str(exc_info.value)


def test_stage1_hil_mock_control_loop():
    """Verify Stage 1 HIL control loop at 20 Hz (50 iterations) with mock FC adapter."""
    adapter = MockFlightControllerAdapter()
    assert adapter.connect("mock", 0) is True

    # Setup controllers
    ckpt_path = "models/checkpoints/bc_policy_scaled_v2.pt"
    bc = BehaviorCloningPolicyController(ckpt_path)
    g_cfg = GuidanceConfig.for_65cm_uav_v3(drone_radius=0.325)
    arb = StatelessClearanceGuidanceArbitrator(config=g_cfg)

    corridor = Corridor3D(
        name="test_hil",
        x_start=0.0,
        x_end=14.0,
        width=3.8,
        height=2.5,
        z_floor=0.0,
        z_ceiling=2.5,
        drone_radius=0.325,
    )
    sup = SafetySupervisor(corridor=corridor)

    # Run 50 mock HIL control loop steps (simulating 2.5s of flight)
    for step in range(50):
        # 1. Fetch telemetry
        telem = adapter.get_telemetry()
        assert telem is not None
        assert telem.mode == FlightMode.GUIDED

        # 2. Construct mock 23-element DeployableObservation
        rays = np.ones(12, dtype=np.float32) * (2.5 / 4.0)  # 2.5m clearance
        rel_goal = np.array([5.0 / 2.0, 0.0, 0.0], dtype=np.float32)
        body_vel = np.array([0.5, 0.0, 0.0], dtype=np.float32)
        att_trig = np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float32)
        yaw_rate = 0.0

        obs = DeployableObservation(
            distance_rays=rays,
            rel_goal_direction_body=rel_goal,
            body_velocity=body_vel,
            attitude_trig=att_trig,
            yaw_rate=yaw_rate,
        )
        obs_vec = obs.to_vector()

        # 3. Model forward pass
        raw_act = bc.predict(obs_vec)

        # 4. Guidance arbitration
        guided_act, direction, _, _ = arb.arbitrate(raw_act, obs_vec)

        # 5. Safety Supervisor filtering
        pos = np.array([0.8 + step * 0.025, 0.0, 1.25])
        vel = np.array([0.5, 0.0, 0.0])
        min_front = compute_forward_clearance_m(obs_vec)
        safe_act, state, _ = sup.filter_action(
            proposed_action=guided_act,
            current_position=pos,
            current_velocity=vel,
            min_front_range=min_front,
            dt=0.05,
            is_localization_healthy=True,
            telemetry_age_s=0.01,
            observation=obs_vec,
        )

        assert state == SafetyState.NOMINAL

        # 6. Transmit to flight controller adapter
        ok = adapter.send_velocity_command(safe_act)
        assert ok is True
        assert adapter.last_command == safe_act


def test_stage1_hil_watchdog_and_stale_telemetry():
    """Verify watchdog response when telemetry becomes stale or lost."""
    corridor = Corridor3D(
        name="test_hil",
        x_start=0.0,
        x_end=14.0,
        width=3.8,
        height=2.5,
        drone_radius=0.325,
    )
    sup = SafetySupervisor(corridor=corridor)
    pos = np.array([1.0, 0.0, 1.25])
    vel = np.array([0.5, 0.0, 0.0])
    raw_act = VelocityAction(vx=0.6, vy=0.0, vz=0.0, yaw_rate=0.0)

    # Case A: Stale telemetry (age = 0.50s > 0.40s)
    safe_act, state, reason = sup.filter_action(
        proposed_action=raw_act,
        current_position=pos,
        current_velocity=vel,
        min_front_range=2.5,
        dt=0.05,
        is_localization_healthy=True,
        telemetry_age_s=0.50,  # STALE
    )
    assert state == SafetyState.LOCALIZATION_FAILSAFE
    assert safe_act.vx < raw_act.vx  # controlled deceleration applied

    # Case B: Prolonged telemetry loss (> 1.50s)
    sup.loss_timer_s = 1.60
    safe_act_loss, state_loss, _ = sup.filter_action(
        proposed_action=raw_act,
        current_position=pos,
        current_velocity=vel,
        min_front_range=2.5,
        dt=0.05,
        is_localization_healthy=True,
        telemetry_age_s=1.60,  # LOST
    )
    assert state_loss == SafetyState.EMERGENCY_LAND
    assert safe_act_loss.vx == 0.0
    assert safe_act_loss.vz < 0.0  # commanded downward landing velocity
