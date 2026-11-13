"""experimental_hysteresis_supervisor.py - Isolated Hysteresis Safety Supervisor.

Phase 4E-A Experimental Prototype:
Implements an explicit multi-state hysteresis barrier to eliminate the obstacle-pinning
limit cycles identified in Phase 4D, without modifying canonical supervisor code.

States:
- NORMAL_APPROACH: Nominal obstacle caution and boundary buffering.
- OBSTACLE_BRAKING: Strict forward velocity arrest upon entering critical front zone.
- RETREAT_OR_RECOVERY: Controlled backward retreat to gain standoff distance.
- LATERAL_OBSTACLE_BYPASS: Coordinated active lateral steering around obstacle.
- SAFE_RELEASE: Dwell confirmation window before returning to unconstrained flight.
- BLOCKED_ESCAPE_DIRECTION: Safe stationary hover when all escape paths are restricted.
- LOCALIZATION_FAILSAFE / EMERGENCY_LAND: Failsafes for telemetry loss.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple, Union
import numpy as np

from core.contracts import (
    ActionLimits,
    DeployableObservation,
    VelocityAction,
    compute_directional_clearances_m,
)
from core.geometry import Corridor3D


class HysteresisSafetyState(Enum):
    """Operational states of the hysteresis supervisor."""
    NORMAL_APPROACH = "normal_approach"
    OBSTACLE_BRAKING = "obstacle_braking"
    RETREAT_OR_RECOVERY = "retreat_or_recovery"
    LATERAL_OBSTACLE_BYPASS = "lateral_obstacle_bypass"
    SAFE_RELEASE = "safe_release"
    BLOCKED_ESCAPE_DIRECTION = "blocked_escape_direction"
    LOCALIZATION_FAILSAFE = "localization_failsafe"
    EMERGENCY_LAND = "emergency_land"


@dataclass
class HysteresisSupervisorConfig:
    """Configuration thresholds for hysteresis state machine."""
    # Critical and caution distances
    critical_obstacle_dist_m: float = 0.30     # Front clearance threshold for emergency braking
    emergency_reverse_dist_m: float = 0.15     # Threshold for immediate emergency reverse
    obstacle_caution_dist_m: float = 0.80      # Forward throttling caution distance
    boundary_buffer_m: float = 0.25            # Distance from wall to begin boundary repulsion
    critical_lateral_dist_m: float = 0.32      # Lateral clearance threshold for constraint
    caution_lateral_dist_m: float = 0.55       # Lateral clearance threshold for caution throttling
    
    # Hysteresis and state-machine thresholds
    standoff_release_dist_m: float = 0.55      # Front clearance required during retreat before attempting bypass
    release_front_clearance_m: float = 0.65    # Front clearance required to confirm genuine obstacle bypass
    bypass_lateral_clearance_m: float = 0.38   # Minimum lateral clearance required to initiate bypass
    braking_dwell_s: float = 0.15              # Minimum time in OBSTACLE_BRAKING before deciding next state
    release_dwell_s: float = 0.20              # Confirmation dwell time in SAFE_RELEASE
    max_retreat_duration_s: float = 2.0        # Maximum retreat duration before re-evaluating
    
    # Speeds and control gains
    repulsion_gain: float = 1.5                # Gain for boundary repulsion
    retreat_speed_mps: float = 0.10            # Fixed reverse velocity during retreat
    bypass_lateral_speed_mps: float = 0.20     # Active lateral evasion velocity
    bypass_crawl_speed_mps: float = 0.15       # Maximum forward velocity during lateral bypass
    safe_release_ramp_mps: float = 0.30        # Maximum forward velocity during SAFE_RELEASE confirmation
    
    # Telemetry failsafe thresholds
    max_stale_duration_s: float = 0.40         # Maximum telemetry age before failsafe
    max_loss_duration_s: float = 1.50          # Maximum telemetry loss duration before landing
    deceleration_limit_mps2: float = 0.8       # Deceleration limit during failsafe


class ExperimentalHysteresisSupervisor:
    """Isolated supervisor prototype implementing state-machine hysteresis."""

    def __init__(
        self,
        corridor: Corridor3D,
        limits: ActionLimits = ActionLimits(),
        config: HysteresisSupervisorConfig = HysteresisSupervisorConfig(),
    ):
        self.corridor = corridor
        self.limits = limits
        self.config = config
        
        # State machine tracking
        self.state = HysteresisSafetyState.NORMAL_APPROACH
        self.state_timer_s: float = 0.0
        self.release_confirm_timer_s: float = 0.0
        self.bypass_direction: Optional[str] = None  # "left" or "right"
        self.loss_timer_s: float = 0.0
        self.last_safe_action = VelocityAction(0.0, 0.0, 0.0, 0.0)

    def reset_state(self) -> None:
        """Resets the supervisor state machine to initial conditions."""
        self.state = HysteresisSafetyState.NORMAL_APPROACH
        self.state_timer_s = 0.0
        self.release_confirm_timer_s = 0.0
        self.bypass_direction = None
        self.loss_timer_s = 0.0
        self.last_safe_action = VelocityAction(0.0, 0.0, 0.0, 0.0)

    def filter_action(
        self,
        proposed_action: VelocityAction,
        current_position: np.ndarray,
        current_velocity: np.ndarray,
        min_front_range: float,
        dt: float,
        is_localization_healthy: bool = True,
        telemetry_age_s: float = 0.0,
        observation: Optional[Union[DeployableObservation, np.ndarray]] = None,
        range_rays: Optional[np.ndarray] = None,
    ) -> Tuple[VelocityAction, HysteresisSafetyState, str]:
        """Filters proposed action through the hysteresis state machine.
        
        Returns:
            (safe_action, hysteresis_state, reason_string)
        """
        # 1. Telemetry and Localization Health Check
        if not is_localization_healthy or telemetry_age_s > self.config.max_stale_duration_s:
            self.loss_timer_s += dt
            if self.loss_timer_s > self.config.max_loss_duration_s:
                self.state = HysteresisSafetyState.EMERGENCY_LAND
                safe_act = VelocityAction(vx=0.0, vy=0.0, vz=-0.15, yaw_rate=0.0)
                return safe_act, self.state, f"Persistent signal loss ({self.loss_timer_s:.2f}s) -> Emergency landing"

            self.state = HysteresisSafetyState.LOCALIZATION_FAILSAFE
            curr_vx = current_velocity[0]
            curr_vy = current_velocity[1]
            decel = self.config.deceleration_limit_mps2 * dt
            safe_vx = np.sign(curr_vx) * max(0.0, abs(curr_vx) - decel)
            safe_vy = np.sign(curr_vy) * max(0.0, abs(curr_vy) - decel)
            safe_act = VelocityAction(vx=float(safe_vx), vy=float(safe_vy), vz=0.0, yaw_rate=0.0)
            return safe_act, self.state, f"Telemetry stale ({telemetry_age_s:.2f}s) -> Controlled deceleration"

        self.loss_timer_s = 0.0

        # Start with bounded proposed action
        vx = float(np.clip(proposed_action.vx, self.limits.min_fwd_vel_mps, self.limits.max_fwd_vel_mps))
        vy = float(np.clip(proposed_action.vy, -self.limits.max_lat_vel_mps, self.limits.max_lat_vel_mps))
        vz = float(np.clip(proposed_action.vz, -self.limits.max_vert_vel_mps, self.limits.max_vert_vel_mps))
        yaw_rate = float(np.clip(proposed_action.yaw_rate, -self.limits.max_yaw_rate_radps, self.limits.max_yaw_rate_radps))

        # 2. Multi-Directional Sensor & Boundary Extraction
        sensor_input = observation if observation is not None else range_rays
        if sensor_input is not None:
            dir_clear = compute_directional_clearances_m(sensor_input)
            eff_front = float(min(dir_clear.front_m, min_front_range))
            d_left_obs = dir_clear.left_m
            d_right_obs = dir_clear.right_m
            d_up_obs = dir_clear.up_m
            d_down_obs = dir_clear.down_m
            d_rear = dir_clear.rear_m
        else:
            eff_front = float(min_front_range)
            d_left_obs, d_right_obs, d_up_obs, d_down_obs, d_rear = 4.0, 4.0, 4.0, 4.0, 4.0

        d_left_wall, d_right_wall, d_ceil, d_floor = self.corridor.boundary_clearances(current_position)
        buf = self.config.boundary_buffer_m
        d_crit_lat = self.config.critical_lateral_dist_m
        d_caut_lat = self.config.caution_lateral_dist_m

        C_left = min(d_left_wall, d_left_obs)
        C_right = min(d_right_wall, d_right_obs)
        C_up = min(d_ceil, d_up_obs)
        C_down = min(d_floor, d_down_obs)

        left_constrained = (C_left < d_crit_lat)
        right_constrained = (C_right < d_crit_lat)

        details = []

        # 3. State Machine Transitions
        self.state_timer_s += dt

        # Global immediate override: if dangerously close to obstacle, enforce emergency retreat
        if eff_front < self.config.emergency_reverse_dist_m and self.state != HysteresisSafetyState.RETREAT_OR_RECOVERY:
            self.state = HysteresisSafetyState.RETREAT_OR_RECOVERY
            self.state_timer_s = 0.0
            details.append(f"Emergency proximity ({eff_front:.2f}m < {self.config.emergency_reverse_dist_m:.2f}m) -> Immediate retreat")

        if self.state == HysteresisSafetyState.NORMAL_APPROACH:
            if eff_front < self.config.critical_obstacle_dist_m:
                self.state = HysteresisSafetyState.OBSTACLE_BRAKING
                self.state_timer_s = 0.0
                self.bypass_direction = None
                vx = 0.0
                details.append(f"Front obstacle detected ({eff_front:.2f}m < {self.config.critical_obstacle_dist_m:.2f}m) -> Enter OBSTACLE_BRAKING")
            elif eff_front < self.config.obstacle_caution_dist_m:
                frac = (eff_front - self.config.critical_obstacle_dist_m) / (self.config.obstacle_caution_dist_m - self.config.critical_obstacle_dist_m)
                max_caut_vx = max(0.10, frac * self.limits.max_fwd_vel_mps)
                if vx > max_caut_vx:
                    vx = max_caut_vx
                    details.append(f"Caution zone ({eff_front:.2f}m) -> Speed throttled to {vx:.2f}m/s")

        elif self.state == HysteresisSafetyState.OBSTACLE_BRAKING:
            # Enforce zero forward velocity
            vx = 0.0
            details.append(f"Holding brake (dwell={self.state_timer_s:.2f}s, front={eff_front:.2f}m)")

            # After minimum braking dwell, assess viable paths
            if self.state_timer_s >= self.config.braking_dwell_s:
                # Check for viable lateral bypass
                can_bypass_left = (C_left >= self.config.bypass_lateral_clearance_m and not left_constrained)
                can_bypass_right = (C_right >= self.config.bypass_lateral_clearance_m and not right_constrained)

                if can_bypass_left or can_bypass_right:
                    # Choose side with greater clearance
                    if can_bypass_left and can_bypass_right:
                        self.bypass_direction = "left" if C_left >= C_right else "right"
                    elif can_bypass_left:
                        self.bypass_direction = "left"
                    else:
                        self.bypass_direction = "right"
                    self.state = HysteresisSafetyState.LATERAL_OBSTACLE_BYPASS
                    self.state_timer_s = 0.0
                    vy = self.config.bypass_lateral_speed_mps if self.bypass_direction == "left" else -self.config.bypass_lateral_speed_mps
                    vx = 0.0
                    details.append(f"Viable path found -> Enter LATERAL_OBSTACLE_BYPASS ({self.bypass_direction})")
                elif d_rear >= 0.35:
                    # No immediate lateral bypass; initiate retreat to gain standoff
                    self.state = HysteresisSafetyState.RETREAT_OR_RECOVERY
                    self.state_timer_s = 0.0
                    vx = -self.config.retreat_speed_mps
                    details.append(f"Lateral paths restricted (L={C_left:.2f}m, R={C_right:.2f}m) -> Enter RETREAT_OR_RECOVERY")
                else:
                    # Blocked on front, sides, and rear
                    self.state = HysteresisSafetyState.BLOCKED_ESCAPE_DIRECTION
                    self.state_timer_s = 0.0
                    vx = 0.0
                    vy = 0.0
                    details.append("All escape directions restricted -> Enter BLOCKED_ESCAPE_DIRECTION")

        elif self.state == HysteresisSafetyState.RETREAT_OR_RECOVERY:
            # Controlled reverse retreat
            vx = -self.config.retreat_speed_mps
            details.append(f"Retreating (front={eff_front:.2f}m, rear={d_rear:.2f}m, t={self.state_timer_s:.2f}s)")

            # Check if lateral asymmetry provides a preferred lateral bias during retreat
            if C_left > C_right + 0.15 and not left_constrained:
                vy = max(vy, 0.10)
            elif C_right > C_left + 0.15 and not right_constrained:
                vy = min(vy, -0.10)

            # Check transition out of retreat:
            # 1. Gained sufficient standoff and lateral opening is viable
            can_bypass_left = (C_left >= self.config.bypass_lateral_clearance_m and not left_constrained)
            can_bypass_right = (C_right >= self.config.bypass_lateral_clearance_m and not right_constrained)

            if eff_front >= self.config.standoff_release_dist_m and (can_bypass_left or can_bypass_right):
                self.bypass_direction = "left" if C_left >= C_right else "right"
                self.state = HysteresisSafetyState.LATERAL_OBSTACLE_BYPASS
                self.state_timer_s = 0.0
                details.append(f"Standoff achieved ({eff_front:.2f}m) -> Enter LATERAL_OBSTACLE_BYPASS ({self.bypass_direction})")
            elif d_rear <= 0.25 or self.state_timer_s >= self.config.max_retreat_duration_s:
                # Rear buffer reached or max retreat duration exceeded
                vx = 0.0
                if can_bypass_left or can_bypass_right:
                    self.bypass_direction = "left" if C_left >= C_right else "right"
                    self.state = HysteresisSafetyState.LATERAL_OBSTACLE_BYPASS
                    self.state_timer_s = 0.0
                    details.append(f"Retreat limit reached -> Enter LATERAL_OBSTACLE_BYPASS ({self.bypass_direction})")
                else:
                    self.state = HysteresisSafetyState.BLOCKED_ESCAPE_DIRECTION
                    self.state_timer_s = 0.0
                    details.append("Retreat limit reached with no bypass -> Enter BLOCKED_ESCAPE_DIRECTION")

        elif self.state == HysteresisSafetyState.LATERAL_OBSTACLE_BYPASS:
            # Actively steer around obstacle while crawling forward if safe
            if self.bypass_direction == "left":
                vy = self.config.bypass_lateral_speed_mps
                if left_constrained:
                    vy = 0.0  # Clamp if left clearance degrades
            else:
                vy = -self.config.bypass_lateral_speed_mps
                if right_constrained:
                    vy = 0.0  # Clamp if right clearance degrades

            # Modest forward crawl if front has clearance
            if eff_front >= 0.35:
                vx = min(vx, self.config.bypass_crawl_speed_mps)
            else:
                vx = 0.0

            details.append(f"Bypassing {self.bypass_direction} (front={eff_front:.2f}m, vy={vy:.2f}m/s)")

            # Check if front sector has cleared
            if eff_front >= self.config.release_front_clearance_m:
                self.state = HysteresisSafetyState.SAFE_RELEASE
                self.state_timer_s = 0.0
                self.release_confirm_timer_s = 0.0
                details.append(f"Front sector clear ({eff_front:.2f}m >= {self.config.release_front_clearance_m:.2f}m) -> Enter SAFE_RELEASE")
            elif (self.bypass_direction == "left" and left_constrained) or (self.bypass_direction == "right" and right_constrained):
                # Chosen direction became blocked
                vx = 0.0
                self.state = HysteresisSafetyState.OBSTACLE_BRAKING
                self.state_timer_s = 0.0
                details.append("Bypass direction obstructed -> Re-enter OBSTACLE_BRAKING")

        elif self.state == HysteresisSafetyState.SAFE_RELEASE:
            # Confirmation dwell window
            self.release_confirm_timer_s += dt
            vx = min(vx, self.config.safe_release_ramp_mps)
            details.append(f"Confirming safe release (front={eff_front:.2f}m, dwell={self.release_confirm_timer_s:.2f}s)")

            # Check for premature clearance drop
            if eff_front < self.config.critical_obstacle_dist_m:
                self.state = HysteresisSafetyState.OBSTACLE_BRAKING
                self.state_timer_s = 0.0
                vx = 0.0
                details.append(f"Obstacle re-detected in front ({eff_front:.2f}m) -> Re-enter OBSTACLE_BRAKING")
            elif self.release_confirm_timer_s >= self.config.release_dwell_s:
                # Confirmed clearance: release to normal navigation
                self.state = HysteresisSafetyState.NORMAL_APPROACH
                self.state_timer_s = 0.0
                self.bypass_direction = None
                details.append(f"Clearance confirmed ({eff_front:.2f}m) -> Enter NORMAL_APPROACH")

        elif self.state == HysteresisSafetyState.BLOCKED_ESCAPE_DIRECTION:
            # Safe stationary hover
            vx = 0.0
            vy = 0.0
            details.append(f"Holding stationary position (front={eff_front:.2f}m, rear={d_rear:.2f}m)")

            # Check if escape window has opened
            if self.state_timer_s >= 0.20:
                can_bypass_left = (C_left >= self.config.bypass_lateral_clearance_m and not left_constrained)
                can_bypass_right = (C_right >= self.config.bypass_lateral_clearance_m and not right_constrained)
                if can_bypass_left or can_bypass_right:
                    self.bypass_direction = "left" if C_left >= C_right else "right"
                    self.state = HysteresisSafetyState.LATERAL_OBSTACLE_BYPASS
                    self.state_timer_s = 0.0
                    details.append(f"Path opened -> Enter LATERAL_OBSTACLE_BYPASS ({self.bypass_direction})")
                elif d_rear >= 0.35:
                    self.state = HysteresisSafetyState.RETREAT_OR_RECOVERY
                    self.state_timer_s = 0.0
                    details.append("Rear opened -> Enter RETREAT_OR_RECOVERY")

        # 4. Enforce Multi-Directional Lateral and Boundary Protection
        # Unsafe lateral motion prevention (Obstacles and Walls)
        if left_constrained:
            if vy > 0.0:
                vy = 0.0
                details.append(f"left_constrained (C_left={C_left:.2f}m)")
        elif C_left < d_caut_lat:
            frac_l = max(0.0, (C_left - d_crit_lat) / (d_caut_lat - d_crit_lat))
            max_left_vy = max(0.05, frac_l * self.limits.max_lat_vel_mps)
            if vy > max_left_vy:
                vy = max_left_vy

        if right_constrained:
            if vy < 0.0:
                vy = 0.0
                details.append(f"right_constrained (C_right={C_right:.2f}m)")
        elif C_right < d_caut_lat:
            frac_r = max(0.0, (C_right - d_crit_lat) / (d_caut_lat - d_crit_lat))
            min_right_vy = -max(0.05, frac_r * self.limits.max_lat_vel_mps)
            if vy < min_right_vy:
                vy = min_right_vy

        # Lateral boundary repulsion if entering wall buffer
        if d_left_wall < buf:
            vy = min(vy, 0.0)
            repulsion = self.config.repulsion_gain * (buf - d_left_wall)
            vy -= repulsion
            details.append(f"wall_left={d_left_wall:.2f}m")
        if d_right_wall < buf:
            vy = max(vy, 0.0)
            repulsion = self.config.repulsion_gain * (buf - d_right_wall)
            vy += repulsion
            details.append(f"wall_right={d_right_wall:.2f}m")

        # Vertical boundary protection
        if d_ceil < buf or d_up_obs < 0.25:
            vz = min(vz, 0.0)
            if d_ceil < buf and d_floor > 0.35 and d_down_obs > 0.35:
                repulsion = self.config.repulsion_gain * (buf - d_ceil)
                vz -= repulsion
                details.append(f"ceiling={d_ceil:.2f}m")
        if d_floor < buf or d_down_obs < 0.25:
            vz = max(vz, 0.0)
            if d_floor < buf and d_ceil > 0.35 and d_up_obs > 0.35:
                repulsion = self.config.repulsion_gain * (buf - d_floor)
                vz += repulsion
                details.append(f"floor={d_floor:.2f}m")
        if (d_ceil < buf or d_up_obs < 0.25) and (d_floor < buf or d_down_obs < 0.25):
            vz = 0.0

        # Coupled boundary brake: if penetrating boundary, clamp forward velocity
        min_b = min(d_left_wall, d_right_wall, d_ceil, d_floor)
        if min_b <= 0.0:
            vx = min(vx, 0.0)

        # 5. Final Action Clamping to Strict Contract ActionLimits
        vx = float(np.clip(vx, self.limits.min_fwd_vel_mps, self.limits.max_fwd_vel_mps))
        vy = float(np.clip(vy, -self.limits.max_lat_vel_mps, self.limits.max_lat_vel_mps))
        vz = float(np.clip(vz, -self.limits.max_vert_vel_mps, self.limits.max_vert_vel_mps))
        yaw_rate = float(np.clip(yaw_rate, -self.limits.max_yaw_rate_radps, self.limits.max_yaw_rate_radps))

        safe_action = VelocityAction(vx=vx, vy=vy, vz=vz, yaw_rate=yaw_rate)
        self.last_safe_action = safe_action

        reason_str = " | ".join(details) if details else "Nominal flight"
        return safe_action, self.state, reason_str
