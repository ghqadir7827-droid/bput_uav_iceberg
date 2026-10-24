"""safety_supervisor.py - Deterministic Safety Barrier & Failsafe Supervisor.

Acts as a strictly verified safety layer between policy outputs and the
flight controller interface. Enforces corridor boundary constraints, obstacle
braking, and safe responses to loss of localization or telemetry dropouts.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Tuple
import numpy as np

from core.contracts import ActionLimits, VelocityAction
from core.geometry import Corridor3D


class SafetyState(Enum):
    """Operational state of the safety supervisor."""
    NOMINAL = "nominal"
    BOUNDARY_INTERVENTION = "boundary_intervention"
    OBSTACLE_BRAKE = "obstacle_brake"
    LOCALIZATION_FAILSAFE = "localization_failsafe"
    TELEMETRY_STALE = "telemetry_stale"
    EMERGENCY_LAND = "emergency_land"


@dataclass
class SafetySupervisorConfig:
    """Safety thresholds and control gains."""
    boundary_buffer_m: float = 0.25     # Distance from wall to begin active repulsion
    critical_obstacle_dist_m: float = 0.30 # Distance to trigger emergency forward brake
    emergency_reverse_dist_m: float = 0.15  # Distance to initiate slow reverse
    repulsion_gain: float = 1.5         # Proportional gain for boundary repulsion (1/s)
    max_stale_duration_s: float = 0.40  # Maximum telemetry age before triggering failsafe
    max_loss_duration_s: float = 1.50   # Duration of signal loss before emergency landing
    deceleration_limit_mps2: float = 0.8 # Controlled deceleration limit during failsafe


class SafetySupervisor:
    """Deterministic safety layer enforcing physical envelopes and failsafes."""

    def __init__(
        self,
        corridor: Corridor3D,
        limits: ActionLimits = ActionLimits(),
        config: SafetySupervisorConfig = SafetySupervisorConfig()
    ):
        self.corridor = corridor
        self.limits = limits
        self.config = config
        self.last_safe_action = VelocityAction(0.0, 0.0, 0.0, 0.0)
        self.loss_timer_s: float = 0.0

    def filter_action(
        self,
        proposed_action: VelocityAction,
        current_position: np.ndarray,
        current_velocity: np.ndarray,
        min_front_range: float,
        dt: float,
        is_localization_healthy: bool = True,
        telemetry_age_s: float = 0.0,
    ) -> Tuple[VelocityAction, SafetyState, str]:
        """Filters and overrides proposed action to guarantee boundary and collision safety.

        Parameters:
            proposed_action: Raw velocity setpoints from policy/expert.
            current_position: Estimated 3D position [x, y, z].
            current_velocity: Estimated 3D body velocity [vx, vy, vz].
            min_front_range: Smallest rangefinder reading in front hemisphere.
            dt: Time step duration (s).
            is_localization_healthy: Boolean flag from localization health check.
            telemetry_age_s: Seconds since last valid external telemetry packet.

        Returns:
            (safe_action, safety_state, reason_string)
        """
        # 1. Check for Telemetry / Localization Loss
        if not is_localization_healthy or telemetry_age_s > self.config.max_stale_duration_s:
            self.loss_timer_s += dt
            if self.loss_timer_s > self.config.max_loss_duration_s:
                # Signal emergency land: gentle controlled descent
                safe_act = VelocityAction(vx=0.0, vy=0.0, vz=-0.15, yaw_rate=0.0)
                return safe_act, SafetyState.EMERGENCY_LAND, f"Persistent localization loss ({self.loss_timer_s:.2f}s > {self.config.max_loss_duration_s}s) -> Slow descent"

            # Controlled Deceleration Failsafe (Low-drift rate-limited stop, NOT instant step to zero)
            # Ramps current forward/lateral velocity toward zero smoothly
            curr_vx = current_velocity[0]
            curr_vy = current_velocity[1]
            decel = self.config.deceleration_limit_mps2 * dt

            safe_vx = np.sign(curr_vx) * max(0.0, abs(curr_vx) - decel)
            safe_vy = np.sign(curr_vy) * max(0.0, abs(curr_vy) - decel)
            safe_vz = 0.0 # Maintain vertical height via barometer/rangefinder
            safe_yr = 0.0

            safe_act = VelocityAction(vx=float(safe_vx), vy=float(safe_vy), vz=safe_vz, yaw_rate=safe_yr)
            return safe_act, SafetyState.LOCALIZATION_FAILSAFE, f"Localization unhealthy (stale={telemetry_age_s:.2f}s) -> Controlled deceleration"

        # Reset loss timer when healthy
        self.loss_timer_s = 0.0

        # Start with bounded proposed action
        vx = np.clip(proposed_action.vx, self.limits.min_fwd_vel_mps, self.limits.max_fwd_vel_mps)
        vy = np.clip(proposed_action.vy, -self.limits.max_lat_vel_mps, self.limits.max_lat_vel_mps)
        vz = np.clip(proposed_action.vz, -self.limits.max_vert_vel_mps, self.limits.max_vert_vel_mps)
        yaw_rate = np.clip(proposed_action.yaw_rate, -self.limits.max_yaw_rate_radps, self.limits.max_yaw_rate_radps)

        # 2. Obstacle Braking (Forward Hemisphere)
        if min_front_range < self.config.critical_obstacle_dist_m:
            if min_front_range < self.config.emergency_reverse_dist_m:
                # Too close: slow backward retreat
                vx = -0.10
                state = SafetyState.OBSTACLE_BRAKE
                reason = f"Emergency obstacle proximity ({min_front_range:.2f}m) -> Reversing"
            else:
                # Within critical distance: forward motion clamped to zero
                vx = 0.0
                state = SafetyState.OBSTACLE_BRAKE
                reason = f"Front obstacle detected ({min_front_range:.2f}m < {self.config.critical_obstacle_dist_m}m) -> Forward brake"
            safe_act = VelocityAction(vx=vx, vy=vy, vz=vz, yaw_rate=yaw_rate)
            return safe_act, state, reason

        # 3. Corridor Boundary Enforcement & Active Inward Repulsion
        d_left, d_right, d_ceil, d_floor = self.corridor.boundary_clearances(current_position)
        buf = self.config.boundary_buffer_m
        intervened = False

        # Left wall intervention (+Y boundary)
        if d_left < buf:
            intervened = True
            # Disallow positive vy (movement toward left wall)
            vy = min(vy, 0.0)
            # Add inward restorative velocity proportional to penetration of buffer
            repulsion = self.config.repulsion_gain * (buf - d_left)
            vy -= repulsion

        # Right wall intervention (-Y boundary)
        if d_right < buf:
            intervened = True
            # Disallow negative vy (movement toward right wall)
            vy = max(vy, 0.0)
            # Add inward restorative velocity
            repulsion = self.config.repulsion_gain * (buf - d_right)
            vy += repulsion

        # Ceiling intervention (+Z boundary)
        if d_ceil < buf:
            intervened = True
            vz = min(vz, 0.0)
            repulsion = self.config.repulsion_gain * (buf - d_ceil)
            vz -= repulsion

        # Floor intervention (-Z boundary)
        if d_floor < buf:
            intervened = True
            vz = max(vz, 0.0)
            repulsion = self.config.repulsion_gain * (buf - d_floor)
            vz += repulsion

        # Final physical clamp
        vy = float(np.clip(vy, -self.limits.max_lat_vel_mps, self.limits.max_lat_vel_mps))
        vz = float(np.clip(vz, -self.limits.max_vert_vel_mps, self.limits.max_vert_vel_mps))

        safe_action = VelocityAction(vx=float(vx), vy=vy, vz=vz, yaw_rate=float(yaw_rate))
        self.last_safe_action = safe_action

        if intervened:
            return safe_action, SafetyState.BOUNDARY_INTERVENTION, f"Boundary buffer active (left={d_left:.2f}m, right={d_right:.2f}m) -> Inward repulsion applied"

        return safe_action, SafetyState.NOMINAL, "All safety envelopes satisfied"
