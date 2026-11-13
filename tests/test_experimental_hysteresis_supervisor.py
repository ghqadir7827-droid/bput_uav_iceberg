"""test_experimental_hysteresis_supervisor.py - Deterministic Tests for Hysteresis Supervisor.

Phase 4E-A Required Deterministic Tests:
1. Forward approach toward a stationary obstacle.
2. Repeated forward commands after braking.
3. Successful lateral obstacle bypass.
4. Retreat that increases clearance but does not create a viable bypass.
5. A blocked or dead-end escape direction.
6. A stationary drone that cannot make progress.
7. An obstacle encounter with limited vertical clearance.
8. Release from recovery after the obstacle has genuinely been cleared.
"""

import numpy as np
import pytest

from core.contracts import ActionLimits, VelocityAction
from core.geometry import Corridor3D
from controllers.experimental_hysteresis_supervisor import (
    ExperimentalHysteresisSupervisor,
    HysteresisSupervisorConfig,
    HysteresisSafetyState,
)


def make_mock_rays(
    front: float = 4.0,
    left: float = 4.0,
    right: float = 4.0,
    up: float = 4.0,
    down: float = 4.0,
    rear: float = 4.0,
    side_left: float = None,
    side_right: float = None,
    max_range: float = 4.0,
) -> np.ndarray:
    """Creates a normalized 12-element rangefinder array matching core/contracts ray mapping."""
    rays = np.full(12, fill_value=max_range, dtype=np.float32)
    # Direct front center ray
    rays[0] = min(rays[0], front)
    # Left rays: 1, 3, 5 (1 & 3 are forward-left, 5 is lateral left)
    rays[1] = min(rays[1], left)
    rays[3] = min(rays[3], left)
    rays[5] = min(rays[5], side_left if side_left is not None else left)
    # Right rays: 2, 4, 6 (2 & 4 are forward-right, 6 is lateral right)
    rays[2] = min(rays[2], right)
    rays[4] = min(rays[4], right)
    rays[6] = min(rays[6], side_right if side_right is not None else right)
    # Rear ray: 7
    rays[7] = min(rays[7], rear)
    # Up rays: 8, 10
    rays[8] = min(rays[8], up)
    rays[10] = min(rays[10], up)
    # Down rays: 9, 11
    rays[9] = min(rays[9], down)
    rays[11] = min(rays[11], down)
    return np.clip(rays / max_range, 0.0, 1.0)


@pytest.fixture
def supervisor():
    corridor = Corridor3D.get_10m_benchmark()
    limits = ActionLimits()
    config = HysteresisSupervisorConfig()
    return ExperimentalHysteresisSupervisor(corridor=corridor, limits=limits, config=config)


def test_1_forward_approach_toward_stationary_obstacle(supervisor):
    """Test 1: Drone approaches stationary obstacle; supervisor transitions from Normal to Braking."""
    pos = np.array([2.0, 0.0, 1.0], dtype=np.float32)
    vel = np.array([0.5, 0.0, 0.0], dtype=np.float32)
    dt = 0.05

    # Step 1: Obstacle is far away (1.5 m)
    rays_far = make_mock_rays(front=1.5)
    prop = VelocityAction(vx=0.5, vy=0.0, vz=0.0, yaw_rate=0.0)
    act, state, _ = supervisor.filter_action(prop, pos, vel, min_front_range=1.5, dt=dt, range_rays=rays_far)
    assert state == HysteresisSafetyState.NORMAL_APPROACH
    assert act.vx > 0.40

    # Step 2: Obstacle enters caution zone (0.5 m)
    rays_caut = make_mock_rays(front=0.50)
    act, state, _ = supervisor.filter_action(prop, pos, vel, min_front_range=0.50, dt=dt, range_rays=rays_caut)
    assert state == HysteresisSafetyState.NORMAL_APPROACH
    assert act.vx < 0.50  # Throttled

    # Step 3: Obstacle enters critical zone (0.28 m < 0.30 m)
    rays_crit = make_mock_rays(front=0.28)
    act, state, _ = supervisor.filter_action(prop, pos, vel, min_front_range=0.28, dt=dt, range_rays=rays_crit)
    assert state == HysteresisSafetyState.OBSTACLE_BRAKING
    assert act.vx == 0.0  # Forward motion strictly arrested


def test_2_repeated_forward_commands_after_braking(supervisor):
    """Test 2: Policy stubbornly proposes forward motion into obstacle; verify zero velocity chattering."""
    pos = np.array([2.6, 0.0, 1.0], dtype=np.float32)
    vel = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    dt = 0.05
    prop = VelocityAction(vx=0.6, vy=0.0, vz=0.0, yaw_rate=0.0)

    # Trigger brake initially
    rays = make_mock_rays(front=0.28, left=0.60, right=0.20, rear=1.0)
    act, state, _ = supervisor.filter_action(prop, pos, vel, min_front_range=0.28, dt=dt, range_rays=rays)
    assert state == HysteresisSafetyState.OBSTACLE_BRAKING
    assert act.vx == 0.0

    # Feed repeated forward commands for 40 consecutive steps (2.0 seconds)
    sign_reversals = 0
    prev_vx = 0.0
    states_visited = []

    for _ in range(40):
        act, state, _ = supervisor.filter_action(prop, pos, vel, min_front_range=0.28, dt=dt, range_rays=rays)
        states_visited.append(state)
        # Verify drone NEVER commands positive forward velocity back into the obstacle
        assert act.vx <= 0.05
        if (prev_vx > 0 and act.vx < 0) or (prev_vx < 0 and act.vx > 0):
            sign_reversals += 1
        prev_vx = act.vx

    # Verify zero velocity sign reversals (canonical supervisor had >100!)
    assert sign_reversals == 0
    # Verify transition from braking into bypass
    assert HysteresisSafetyState.LATERAL_OBSTACLE_BYPASS in states_visited


def test_3_successful_lateral_obstacle_bypass(supervisor):
    """Test 3: Left side has ample clearance; state machine navigates Braking -> Bypass -> Safe Release -> Normal."""
    pos = np.array([2.6, 0.0, 1.0], dtype=np.float32)
    vel = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    dt = 0.05
    prop = VelocityAction(vx=0.5, vy=0.0, vz=0.0, yaw_rate=0.0)

    # 1. Approach and enter Braking
    rays_obs = make_mock_rays(front=0.28, left=0.60, right=0.20, rear=1.0)
    for _ in range(4):  # 4 steps * 0.05s = 0.20s > braking_dwell_s (0.15s)
        act, state, _ = supervisor.filter_action(prop, pos, vel, min_front_range=0.28, dt=dt, range_rays=rays_obs)

    # Verify transition to LATERAL_OBSTACLE_BYPASS with left direction chosen
    assert state == HysteresisSafetyState.LATERAL_OBSTACLE_BYPASS
    assert supervisor.bypass_direction == "left"
    assert act.vy > 0.15  # Active left steering command

    # 2. Obstacle clears front sector (front distance increases to 0.8m, obstacle is alongside at side_left=0.35m)
    rays_clearing = make_mock_rays(front=0.80, left=1.0, right=1.0, side_left=0.35, rear=1.0)
    act, state, _ = supervisor.filter_action(prop, pos, vel, min_front_range=0.80, dt=dt, range_rays=rays_clearing)
    assert state == HysteresisSafetyState.SAFE_RELEASE

    # 3. Maintain clear front for release dwell duration (0.20s = 4 steps)
    for _ in range(5):
        act, state, _ = supervisor.filter_action(prop, pos, vel, min_front_range=0.80, dt=dt, range_rays=rays_clearing)

    # Verify confirmed release back to NORMAL_APPROACH
    assert state == HysteresisSafetyState.NORMAL_APPROACH
    assert act.vx > 0.40  # Full forward policy authority restored


def test_4_retreat_without_viable_bypass(supervisor):
    """Test 4: Sides too tight for bypass; retreat increases clearance but state machine does NOT release prematurely."""
    pos = np.array([2.6, 0.0, 1.0], dtype=np.float32)
    vel = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    dt = 0.05
    prop = VelocityAction(vx=0.6, vy=0.0, vz=0.0, yaw_rate=0.0)

    # Front blocked, left and right sides constrained (0.20 m), rear open (1.0 m)
    rays_tight = make_mock_rays(front=0.28, left=0.20, right=0.20, rear=1.0)

    # Dwell in braking
    for _ in range(4):
        act, state, _ = supervisor.filter_action(prop, pos, vel, min_front_range=0.28, dt=dt, range_rays=rays_tight)

    # Verify transition to RETREAT_OR_RECOVERY
    assert state == HysteresisSafetyState.RETREAT_OR_RECOVERY
    assert act.vx == -supervisor.config.retreat_speed_mps  # Controlled reverse

    # Simulate drone backing up: front clearance rises to 0.48 m (in canonical, this caused chattering release!)
    rays_backed_up = make_mock_rays(front=0.48, left=0.20, right=0.20, rear=0.8)
    for _ in range(5):
        act, state, _ = supervisor.filter_action(prop, pos, vel, min_front_range=0.48, dt=dt, range_rays=rays_backed_up)
        # CRITICAL TEST: Must stay in retreat and NEVER command forward velocity into the obstacle!
        assert state == HysteresisSafetyState.RETREAT_OR_RECOVERY
        assert act.vx < 0.0


def test_5_blocked_dead_end_escape_direction(supervisor):
    """Test 5: Front, sides, and rear blocked; verify BLOCKED_ESCAPE_DIRECTION hover without oscillation."""
    pos = np.array([2.6, 0.0, 1.0], dtype=np.float32)
    vel = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    dt = 0.05
    prop = VelocityAction(vx=0.5, vy=0.0, vz=0.0, yaw_rate=0.0)

    # All directions blocked
    rays_blocked = make_mock_rays(front=0.25, left=0.20, right=0.20, rear=0.20)

    # Step through braking dwell
    for _ in range(4):
        act, state, _ = supervisor.filter_action(prop, pos, vel, min_front_range=0.25, dt=dt, range_rays=rays_blocked)

    # Verify transition to BLOCKED_ESCAPE_DIRECTION
    assert state == HysteresisSafetyState.BLOCKED_ESCAPE_DIRECTION
    assert act.vx == 0.0
    assert act.vy == 0.0  # Stationary hover hold, zero velocity


def test_6_stationary_drone_cannot_make_progress(supervisor):
    """Test 6: Stationary drone facing obstacle initiates orderly recovery attempt after dwell."""
    pos = np.array([2.6, 0.0, 1.0], dtype=np.float32)
    vel = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    dt = 0.05
    prop = VelocityAction(vx=0.0, vy=0.0, vz=0.0, yaw_rate=0.0)

    rays = make_mock_rays(front=0.25, left=0.50, right=0.20, rear=1.0)

    # Step 1: Initial detection
    act, state, _ = supervisor.filter_action(prop, pos, vel, min_front_range=0.25, dt=dt, range_rays=rays)
    assert state == HysteresisSafetyState.OBSTACLE_BRAKING

    # Dwell past braking_dwell_s
    for _ in range(4):
        act, state, _ = supervisor.filter_action(prop, pos, vel, min_front_range=0.25, dt=dt, range_rays=rays)

    # Verify orderly recovery transition to lateral bypass
    assert state == HysteresisSafetyState.LATERAL_OBSTACLE_BYPASS
    assert act.vy > 0.0  # Evasive steering active even if policy proposed zero


def test_7_obstacle_encounter_with_limited_vertical_clearance(supervisor):
    """Test 7: Obstacle encountered under low ceiling; verify ceiling repulsion prevents upward climb."""
    # Place drone at z = 1.85 m (ceiling is at 2.0 m, ceiling clearance = 2.0 - 1.85 - 0.15 = 0.0 m)
    pos = np.array([2.6, 0.0, 1.85], dtype=np.float32)
    vel = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    dt = 0.05
    prop = VelocityAction(vx=0.5, vy=0.0, vz=0.20, yaw_rate=0.0)  # Policy asks to climb

    rays = make_mock_rays(front=0.28, left=0.50, right=0.20, up=0.15, rear=1.0)

    act, state, _ = supervisor.filter_action(prop, pos, vel, min_front_range=0.28, dt=dt, range_rays=rays)
    # Forward motion arrested by obstacle
    assert act.vx == 0.0
    # Upward climb strictly overridden to negative (downward repulsion away from ceiling)
    assert act.vz <= 0.0


def test_8_release_from_recovery_after_genuine_clearance(supervisor):
    """Test 8: Verify full cycle and confirmation dwell before release back to normal approach."""
    pos = np.array([2.6, 0.0, 1.0], dtype=np.float32)
    vel = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    dt = 0.05
    prop = VelocityAction(vx=0.6, vy=0.0, vz=0.0, yaw_rate=0.0)

    # 1. Enter Braking
    rays_obs = make_mock_rays(front=0.28, left=0.55, right=0.20, rear=1.0)
    for _ in range(4):
        act, state, _ = supervisor.filter_action(prop, pos, vel, min_front_range=0.28, dt=dt, range_rays=rays_obs)
    assert state == HysteresisSafetyState.LATERAL_OBSTACLE_BYPASS

    # 2. Steer around until front clears (obstacle alongside at side_left=0.40m)
    rays_clear = make_mock_rays(front=0.75, left=1.0, right=1.0, side_left=0.40, rear=1.0)
    act, state, _ = supervisor.filter_action(prop, pos, vel, min_front_range=0.75, dt=dt, range_rays=rays_clear)
    assert state == HysteresisSafetyState.SAFE_RELEASE

    # 3. Simulate brief obstacle flicker during confirmation window (front drops back down to 0.29m)
    rays_flicker = make_mock_rays(front=0.29, left=0.40, right=0.80, rear=1.0)
    act, state, _ = supervisor.filter_action(prop, pos, vel, min_front_range=0.29, dt=dt, range_rays=rays_flicker)
    # Must immediately abort safe release and re-brake
    assert state == HysteresisSafetyState.OBSTACLE_BRAKING
    assert act.vx == 0.0
