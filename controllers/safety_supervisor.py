"""safety_supervisor.py - Deterministic Safety Barrier & Failsafe Supervisor.

Acts as a strictly verified safety layer between policy outputs and the
flight controller interface. Enforces corridor boundary constraints, obstacle
braking, and safe responses to loss of localization or telemetry dropouts.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple, Union
import numpy as np

from core.contracts import (
    ActionLimits,
    DeployableObservation,
    DirectionalClearances,
    VelocityAction,
    compute_directional_clearances_m,
)
from core.geometry import Corridor3D


class SafetyState(Enum):
    """Operational state of the safety supervisor."""
    NOMINAL = "nominal"
    BOUNDARY_INTERVENTION = "boundary_intervention"
    OBSTACLE_BRAKE = "obstacle_brake"
    BOUNDARY_AND_OBSTACLE_BRAKE = "boundary_and_obstacle_brake"
    LOCALIZATION_FAILSAFE = "localization_failsafe"
    TELEMETRY_STALE = "telemetry_stale"
    EMERGENCY_LAND = "emergency_land"


@dataclass
class SafetySupervisorConfig:
    """Safety thresholds and control gains."""
    boundary_buffer_m: float = 0.25        # Distance from wall to begin active repulsion
    critical_obstacle_dist_m: float = 0.30 # Distance to trigger emergency forward brake
    emergency_reverse_dist_m: float = 0.15 # Distance to initiate slow reverse
    repulsion_gain: float = 1.5            # Proportional gain for boundary repulsion (1/s)
    max_stale_duration_s: float = 0.40     # Maximum telemetry age before triggering failsafe
    max_loss_duration_s: float = 1.50      # Duration of signal loss before emergency landing
    deceleration_limit_mps2: float = 0.8   # Controlled deceleration limit during failsafe
    obstacle_caution_dist_m: float = 0.80  # Distance below which forward velocity is throttled near obstacles
    critical_lateral_dist_m: float = 0.32  # Lateral distance to trigger lateral constraint / clamp
    caution_lateral_dist_m: float = 0.55   # Lateral distance below which lateral velocity is throttled
    stalled_duration_s: float = 0.35       # Time trapped near obstacle before stalled recovery retreat

    @classmethod
    def for_65cm_uav(cls, drone_radius: float = 0.325) -> "SafetySupervisorConfig":
        """Safety thresholds scaled for 65 cm physical quadcopter (R=0.325m)."""
        return cls(
            boundary_buffer_m=0.35,
            critical_obstacle_dist_m=float(drone_radius + 0.20),  # 0.525 m from center
            emergency_reverse_dist_m=float(drone_radius + 0.10),  # 0.425 m from center
            repulsion_gain=1.5,
            deceleration_limit_mps2=0.8,
            obstacle_caution_dist_m=float(drone_radius + 0.85),   # 1.175 m from center
            critical_lateral_dist_m=0.35,                         # 0.35 m surface clearance
            caution_lateral_dist_m=0.60,
            stalled_duration_s=0.35,
        )


class SafetySupervisor:
    """Deterministic safety layer enforcing physical envelopes and multi-directional failsafes."""

    def __init__(
        self,
        corridor: Corridor3D,
        limits: ActionLimits = ActionLimits(),
        config: Optional[SafetySupervisorConfig] = None
    ):
        self.corridor = corridor
        self.limits = limits
        if config is None:
            if hasattr(corridor, "drone_radius") and corridor.drone_radius > 0.25:
                self.config = SafetySupervisorConfig.for_65cm_uav(corridor.drone_radius)
            else:
                self.config = SafetySupervisorConfig()
        else:
            self.config = config
        self.last_safe_action = VelocityAction(0.0, 0.0, 0.0, 0.0)
        self.loss_timer_s: float = 0.0
        self.stalled_timer_s: float = 0.0

    def reset(self) -> None:
        """Reset internal timers and cached actions to guarantee no inter-episode state leakage."""
        self.last_safe_action = VelocityAction(0.0, 0.0, 0.0, 0.0)
        self.loss_timer_s = 0.0
        self.stalled_timer_s = 0.0

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
    ) -> Tuple[VelocityAction, SafetyState, str]:
        """Filters and overrides proposed action to guarantee boundary and multi-directional obstacle safety.

        Parameters:
            proposed_action: Raw velocity setpoints from policy/expert.
            current_position: Estimated 3D position [x, y, z].
            current_velocity: Estimated 3D body velocity [vx, vy, vz].
            min_front_range: Smallest rangefinder reading in front hemisphere.
            dt: Time step duration (s).
            is_localization_healthy: Boolean flag from localization health check.
            telemetry_age_s: Seconds since last valid external telemetry packet.
            observation: Optional DeployableObservation or raw 23-float vector for multi-directional sensing.
            range_rays: Optional 12-element rangefinder array in meters or normalized.

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

            # Controlled Deceleration Failsafe
            curr_vx = current_velocity[0]
            curr_vy = current_velocity[1]
            decel = self.config.deceleration_limit_mps2 * dt

            safe_vx = np.sign(curr_vx) * max(0.0, abs(curr_vx) - decel)
            safe_vy = np.sign(curr_vy) * max(0.0, abs(curr_vy) - decel)
            safe_vz = 0.0
            safe_yr = 0.0

            safe_act = VelocityAction(vx=float(safe_vx), vy=float(safe_vy), vz=safe_vz, yaw_rate=safe_yr)
            return safe_act, SafetyState.LOCALIZATION_FAILSAFE, f"Localization unhealthy (stale={telemetry_age_s:.2f}s) -> Controlled deceleration"

        # Reset loss timer when healthy
        self.loss_timer_s = 0.0

        # Start with bounded proposed action
        vx = float(np.clip(proposed_action.vx, self.limits.min_fwd_vel_mps, self.limits.max_fwd_vel_mps))
        vy = float(np.clip(proposed_action.vy, -self.limits.max_lat_vel_mps, self.limits.max_lat_vel_mps))
        vz = float(np.clip(proposed_action.vz, -self.limits.max_vert_vel_mps, self.limits.max_vert_vel_mps))
        yaw_rate = float(np.clip(proposed_action.yaw_rate, -self.limits.max_yaw_rate_radps, self.limits.max_yaw_rate_radps))

        # 2. Extract Multi-Directional Sensor Evidence
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
            # Fallback for calls providing only scalar min_front_range
            eff_front = float(min_front_range)
            d_left_obs = 4.0
            d_right_obs = 4.0
            d_up_obs = 4.0
            d_down_obs = 4.0
            d_rear = 4.0

        # 3. Corridor Boundary Clearances
        d_left_wall, d_right_wall, d_ceil, d_floor = self.corridor.boundary_clearances(current_position)
        buf = self.config.boundary_buffer_m
        d_crit_lat = self.config.critical_lateral_dist_m
        d_caut_lat = self.config.caution_lateral_dist_m

        # Net directional safety envelopes
        C_left = min(d_left_wall, d_left_obs)
        C_right = min(d_right_wall, d_right_obs)
        C_up = min(d_ceil, d_up_obs)
        C_down = min(d_floor, d_down_obs)

        obstacle_intervened = False
        obstacle_details = []
        boundary_intervened = False
        boundary_details = []

        # 4. Forward Obstacle Braking & Emergency Reverse
        caution_dist = self.config.obstacle_caution_dist_m
        if eff_front < self.config.critical_obstacle_dist_m:
            obstacle_intervened = True
            if eff_front < self.config.emergency_reverse_dist_m:
                vx = -0.10
                obstacle_details.append(f"Emergency obstacle proximity ({eff_front:.2f}m < {self.config.emergency_reverse_dist_m:.2f}m) -> Reversing")
            else:
                vx = 0.0
                obstacle_details.append(f"Front obstacle detected ({eff_front:.2f}m < {self.config.critical_obstacle_dist_m:.2f}m) -> Forward brake")
        elif eff_front < caution_dist:
            frac = (eff_front - self.config.critical_obstacle_dist_m) / (caution_dist - self.config.critical_obstacle_dist_m)
            max_caut_vx = max(0.10, frac * self.limits.max_fwd_vel_mps)
            if vx > max_caut_vx:
                vx = max_caut_vx
                obstacle_intervened = True
                obstacle_details.append(f"Obstacle caution zone ({eff_front:.2f}m < {caution_dist:.2f}m) -> Speed throttled to {vx:.2f}m/s")

        # 5. Multi-Directional Lateral Obstacle and Boundary Arbitration
        left_constrained = (C_left < d_crit_lat)
        right_constrained = (C_right < d_crit_lat)

        # Unsafe lateral motion prevention
        if left_constrained:
            if vy > 0.0:
                vy = 0.0
                if d_left_obs < d_crit_lat:
                    obstacle_intervened = True
                    obstacle_details.append(f"left_obs={d_left_obs:.2f}m")
                if d_left_wall < buf:
                    boundary_intervened = True
                    boundary_details.append(f"left={d_left_wall:.2f}m")
        elif C_left < d_caut_lat:
            frac_l = max(0.0, (C_left - d_crit_lat) / (d_caut_lat - d_crit_lat))
            max_left_vy = max(0.05, frac_l * self.limits.max_lat_vel_mps)
            if vy > max_left_vy:
                vy = max_left_vy

        if right_constrained:
            if vy < 0.0:
                vy = 0.0
                if d_right_obs < d_crit_lat:
                    obstacle_intervened = True
                    obstacle_details.append(f"right_obs={d_right_obs:.2f}m")
                if d_right_wall < buf:
                    boundary_intervened = True
                    boundary_details.append(f"right={d_right_wall:.2f}m")
        elif C_right < d_caut_lat:
            frac_r = max(0.0, (C_right - d_crit_lat) / (d_caut_lat - d_crit_lat))
            min_right_vy = -max(0.05, frac_r * self.limits.max_lat_vel_mps)
            if vy < min_right_vy:
                vy = min_right_vy

        # Coordinated escape and repulsion
        if left_constrained and right_constrained:
            # Both sides constrained: strictly arrest all lateral motion, no opposing repulsions
            vy = 0.0
            if eff_front < caution_dist:
                vx = min(vx, 0.05)
        elif left_constrained:
            # Left constrained, right has headroom
            if d_left_wall < buf or d_left_obs < 0.25:
                pen = max(0.0, buf - d_left_wall if d_left_wall < buf else 0.0, 0.25 - d_left_obs if d_left_obs < 0.25 else 0.0)
                headroom = min(1.0, max(0.0, (C_right - d_crit_lat) / (d_caut_lat - d_crit_lat)))
                rep = self.config.repulsion_gain * pen * headroom
                vy -= rep
                if d_left_wall < buf:
                    boundary_intervened = True
                    if f"left={d_left_wall:.2f}m" not in boundary_details:
                        boundary_details.append(f"left={d_left_wall:.2f}m")
        elif right_constrained:
            # Right constrained, left has headroom
            if d_right_wall < buf or d_right_obs < 0.25:
                pen = max(0.0, buf - d_right_wall if d_right_wall < buf else 0.0, 0.25 - d_right_obs if d_right_obs < 0.25 else 0.0)
                headroom = min(1.0, max(0.0, (C_left - d_crit_lat) / (d_caut_lat - d_crit_lat)))
                rep = self.config.repulsion_gain * pen * headroom
                vy += rep
                if d_right_wall < buf:
                    boundary_intervened = True
                    if f"right={d_right_wall:.2f}m" not in boundary_details:
                        boundary_details.append(f"right={d_right_wall:.2f}m")
        else:
            # Neither side constrained by obstacles; apply normal boundary buffer repulsion if near wall
            if d_left_wall < buf:
                boundary_intervened = True
                vy = min(vy, 0.0)
                repulsion = self.config.repulsion_gain * (buf - d_left_wall)
                vy -= repulsion
                boundary_details.append(f"left={d_left_wall:.2f}m")
            if d_right_wall < buf:
                boundary_intervened = True
                vy = max(vy, 0.0)
                repulsion = self.config.repulsion_gain * (buf - d_right_wall)
                vy += repulsion
                boundary_details.append(f"right={d_right_wall:.2f}m")

        # 6. Vertical Directional Arbitration
        if d_ceil < buf or d_up_obs < 0.25:
            vz = min(vz, 0.0)
            if d_ceil < buf and d_floor > 0.35 and d_down_obs > 0.35:
                boundary_intervened = True
                repulsion = self.config.repulsion_gain * (buf - d_ceil)
                vz -= repulsion
                boundary_details.append(f"ceil={d_ceil:.2f}m")
        if d_floor < buf or d_down_obs < 0.25:
            vz = max(vz, 0.0)
            if d_floor < buf and d_ceil > 0.35 and d_up_obs > 0.35:
                boundary_intervened = True
                repulsion = self.config.repulsion_gain * (buf - d_floor)
                vz += repulsion
                boundary_details.append(f"floor={d_floor:.2f}m")
        if (d_ceil < buf or d_up_obs < 0.25) and (d_floor < buf or d_down_obs < 0.25):
            vz = 0.0

        # 7. Stalled-State Recovery Handling
        if vx <= 0.05 and eff_front < 0.38:
            self.stalled_timer_s += dt
            if self.stalled_timer_s >= self.config.stalled_duration_s:
                # Check rear clearance before initiating backward retreat
                if d_rear > 0.45:
                    vx = -0.10
                    obstacle_intervened = True
                    obstacle_details.append(f"Stalled recovery active ({self.stalled_timer_s:.2f}s) -> Reverse retreat")
                    # Assist lateral steering toward wider side if clearance asymmetry is significant
                    if C_left > C_right + 0.15 and not left_constrained:
                        vy = max(vy, 0.15)
                    elif C_right > C_left + 0.15 and not right_constrained:
                        vy = min(vy, -0.15)
                else:
                    vx = 0.0
        elif eff_front >= 0.45 or vx > 0.10:
            self.stalled_timer_s = 0.0

        # 8. Coupled Boundary-Forward Velocity Arbitration
        min_b = min(d_left_wall, d_right_wall, d_ceil, d_floor)
        if boundary_intervened:
            if min_b <= 0.0:
                vx = min(vx, 0.0)
            elif eff_front < caution_dist:
                b_ratio = max(0.0, min_b / buf)
                fwd_cap = max(0.05, b_ratio * self.limits.max_fwd_vel_mps * 0.5)
                if vx > fwd_cap:
                    vx = fwd_cap

        # Final physical clamp strictly adhering to ActionLimits
        vx = float(np.clip(vx, self.limits.min_fwd_vel_mps, self.limits.max_fwd_vel_mps))
        vy = float(np.clip(vy, -self.limits.max_lat_vel_mps, self.limits.max_lat_vel_mps))
        vz = float(np.clip(vz, -self.limits.max_vert_vel_mps, self.limits.max_vert_vel_mps))
        yaw_rate = float(np.clip(yaw_rate, -self.limits.max_yaw_rate_radps, self.limits.max_yaw_rate_radps))

        safe_action = VelocityAction(vx=vx, vy=vy, vz=vz, yaw_rate=yaw_rate)
        self.last_safe_action = safe_action

        # 9. Unified State and Diagnostic Reason Reporting
        obs_desc = "; ".join(obstacle_details) if obstacle_details else ""
        b_desc = ", ".join(boundary_details) if boundary_details else ""

        if obstacle_intervened and boundary_intervened:
            reason = f"Combined intervention: {obs_desc} | Boundaries ({b_desc})"
            return safe_action, SafetyState.BOUNDARY_AND_OBSTACLE_BRAKE, reason
        elif obstacle_intervened:
            return safe_action, SafetyState.OBSTACLE_BRAKE, obs_desc
        elif boundary_intervened:
            return safe_action, SafetyState.BOUNDARY_INTERVENTION, f"Boundary buffer active ({b_desc}) -> Inward repulsion applied"

        return safe_action, SafetyState.NOMINAL, "All safety envelopes satisfied"

