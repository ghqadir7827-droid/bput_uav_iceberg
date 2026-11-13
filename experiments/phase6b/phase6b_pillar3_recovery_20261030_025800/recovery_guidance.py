"""recovery_guidance.py - Experimental Post-Evasion Centerline Recovery Guidance Arbitrator.

Phase 6B Isolated Candidate:
Extends Stateless Clearance Guidance with an observable post-evasion centerline restorative state:
- Activates ONLY AFTER an obstacle has been encountered and passed (c_front opened back up >= 0.90m).
- Observable lateral displacement estimated from lateral 90-degree rangefinder beams.
- Strong hysteresis band (activation at |y_obs| >= 0.22m, release at |y_obs| <= 0.08m) to prevent chattering.
- Clearance validation in intended recovery direction (>= 0.55m).
- Bounded, gradual lateral setpoints (|vy| <= 0.15 m/s, gain = 0.50).
- Pure-NumPy, deployment-ready (zero PyTorch, zero privileged state).
- Resets cleanly between episodes.
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple, Union
import numpy as np

from core.contracts import ActionLimits, DeployableObservation, VelocityAction
from core.guidance import GuidanceConfig, StatelessClearanceGuidanceArbitrator


@dataclass
class RecoveryConfig:
    """Configuration parameters for post-evasion centerline recovery."""
    guidance_config: GuidanceConfig = field(default_factory=GuidanceConfig)
    recovery_activate_y_m: float = 0.22        # Lateral displacement threshold to activate recovery
    recovery_release_y_m: float = 0.08         # Lateral displacement threshold to deactivate (hysteresis)
    recovery_min_front_c_m: float = 0.90       # Minimum front clearance to permit recovery
    recovery_min_side_c_m: float = 0.55        # Minimum clearance in intended recovery direction
    recovery_corridor_width_min_m: float = 1.60# Corridor width validation lower bound
    recovery_corridor_width_max_m: float = 2.40# Corridor width validation upper bound
    recovery_max_speed_mps: float = 0.15       # Bounded gradual lateral recovery speed
    recovery_min_fwd_speed_mps: float = 0.35   # Maintain healthy forward cruise during recovery
    recovery_gain: float = 0.50                # Proportional recovery gain


class PostEvasionRecoveryArbitrator:
    """Arbitrates nominal policy actions with clearance guidance and post-evasion centerline recovery."""

    def __init__(
        self,
        config: Optional[RecoveryConfig] = None,
        limits: Optional[ActionLimits] = None,
    ):
        self.config = config or RecoveryConfig()
        self.limits = limits or ActionLimits()
        self.base_arbitrator = StatelessClearanceGuidanceArbitrator(
            config=self.config.guidance_config,
            limits=self.limits,
        )
        self.obstacle_encountered: bool = False
        self.recovery_active: bool = False
        self.last_dir: str = "NONE"

    def reset(self) -> None:
        """Resets persistent direction latch, obstacle history, and recovery states."""
        self.base_arbitrator.reset()
        self.obstacle_encountered = False
        self.recovery_active = False
        self.last_dir = "NONE"

    def arbitrate(
        self,
        raw_action: VelocityAction,
        observation: Union[np.ndarray, DeployableObservation],
    ) -> Tuple[VelocityAction, str, float, float]:
        """Arbitrates velocity commands with obstacle evasion and centerline recovery.
        
        Args:
            raw_action: Nominal policy VelocityAction [vx, vy, vz, yaw_rate].
            observation: 23-float observation vector or DeployableObservation instance.
            
        Returns:
            Tuple of (guided_action, selected_dir, c_left, c_right)
        """
        if isinstance(observation, DeployableObservation):
            obs_vec = observation.to_vector()
        else:
            obs_vec = np.asarray(observation, dtype=np.float32)

        # Sanitize non-finite values (fail-closed guard)
        obs_vec = np.nan_to_num(obs_vec, nan=0.0, posinf=1.0, neginf=-1.0)

        # 1. Extract Sector Clearances
        rays = obs_vec[0:12] * 4.0
        c_front = float(min(rays[0], rays[1], rays[2]))
        c_left = float(min(rays[1], rays[3], rays[5]))
        c_right = float(min(rays[2], rays[4], rays[6]))

        wall_l = float(rays[5])  # +90 deg azimuth (left wall)
        wall_r = float(rays[6])  # -90 deg azimuth (right wall)

        # 2. Terminal approach detection
        goal_vec = obs_vec[12:15]
        goal_norm = float(np.linalg.norm(goal_vec))
        in_terminal_zone = (goal_norm < self.config.guidance_config.terminal_goal_norm_threshold)

        # 3. PRIORITY 1: Obstacle Clearance Guidance
        # If obstacle is detected ahead within activation threshold, clearance guidance takes strict precedence.
        if c_front < self.config.guidance_config.activation_threshold_m and not in_terminal_zone:
            self.obstacle_encountered = True
            self.recovery_active = False
            guided_act, selected_dir, cl, cr = self.base_arbitrator.arbitrate(raw_action, obs_vec)
            self.last_dir = selected_dir
            return guided_act, selected_dir, cl, cr

        # Obstacle guidance is inactive
        self.base_arbitrator.reset()

        selected_dir = "NONE"
        vx_cmd = raw_action.vx
        vy_cmd = raw_action.vy

        # In terminal zone: soft release, no recovery intervention
        if in_terminal_zone:
            self.recovery_active = False
            self.last_dir = "NONE"
            clamped_act = VelocityAction(
                vx=float(np.clip(raw_action.vx, self.limits.min_fwd_vel_mps, self.limits.max_fwd_vel_mps)),
                vy=float(np.clip(raw_action.vy, -self.limits.max_lat_vel_mps, self.limits.max_lat_vel_mps)),
                vz=float(np.clip(raw_action.vz, -self.limits.max_vert_vel_mps, self.limits.max_vert_vel_mps)),
                yaw_rate=float(np.clip(raw_action.yaw_rate, -self.limits.max_yaw_rate_radps, self.limits.max_yaw_rate_radps)),
            )
            return clamped_act, selected_dir, c_left, c_right

        # 4. PRIORITY 2: Post-Evasion Centerline Recovery
        # Requirement 1: Obstacle not yet passed -> recovery remains inactive.
        if not self.obstacle_encountered:
            self.recovery_active = False
            self.last_dir = "NONE"
            clamped_act = VelocityAction(
                vx=float(np.clip(raw_action.vx, self.limits.min_fwd_vel_mps, self.limits.max_fwd_vel_mps)),
                vy=float(np.clip(raw_action.vy, -self.limits.max_lat_vel_mps, self.limits.max_lat_vel_mps)),
                vz=float(np.clip(raw_action.vz, -self.limits.max_vert_vel_mps, self.limits.max_vert_vel_mps)),
                yaw_rate=float(np.clip(raw_action.yaw_rate, -self.limits.max_yaw_rate_radps, self.limits.max_yaw_rate_radps)),
            )
            return clamped_act, selected_dir, c_left, c_right

        # Validate observable corridor boundaries (fail closed if sensor readings invalid or obscured)
        obs_corridor_width = wall_l + wall_r
        is_width_valid = (
            self.config.recovery_corridor_width_min_m <= obs_corridor_width <= self.config.recovery_corridor_width_max_m
        )

        if is_width_valid and c_front >= self.config.recovery_min_front_c_m:
            # Observable lateral displacement: positive = displaced left, negative = displaced right
            y_obs = float((wall_r - wall_l) / 2.0)

            # State machine with hysteresis
            if self.recovery_active:
                if abs(y_obs) <= self.config.recovery_release_y_m:
                    self.recovery_active = False
            else:
                if abs(y_obs) >= self.config.recovery_activate_y_m:
                    self.recovery_active = True

            if self.recovery_active:
                max_rec_spd = self.config.recovery_max_speed_mps
                gain = self.config.recovery_gain

                if y_obs > 0:
                    # Displaced LEFT -> Recover toward RIGHT (negative vy)
                    if c_right >= self.config.recovery_min_side_c_m:
                        selected_dir = "RECOVERY_RIGHT"
                        rec_vy = -float(np.clip(gain * y_obs, 0.0, max_rec_spd))
                        vy_cmd = min(raw_action.vy, rec_vy)
                        vx_cmd = max(raw_action.vx, self.config.recovery_min_fwd_speed_mps)
                    else:
                        # Inadequate clearance to the right: fail closed
                        self.recovery_active = False
                else:
                    # Displaced RIGHT -> Recover toward LEFT (positive vy)
                    if c_left >= self.config.recovery_min_side_c_m:
                        selected_dir = "RECOVERY_LEFT"
                        rec_vy = float(np.clip(gain * abs(y_obs), 0.0, max_rec_spd))
                        vy_cmd = max(raw_action.vy, rec_vy)
                        vx_cmd = max(raw_action.vx, self.config.recovery_min_fwd_speed_mps)
                    else:
                        # Inadequate clearance to the left: fail closed
                        self.recovery_active = False
            else:
                # Centerline restored (|y_obs| <= release threshold):
                # Damp lateral drift away from centerline to prevent BC velocity-copy overshoot
                if abs(y_obs) <= 0.12:
                    if y_obs >= 0:
                        vy_cmd = min(raw_action.vy, 0.08)
                    else:
                        vy_cmd = max(raw_action.vy, -0.08)
        else:
            self.recovery_active = False

        self.last_dir = selected_dir

        # 5. Enforce Action Limits
        guided_action = VelocityAction(
            vx=float(np.clip(vx_cmd, self.limits.min_fwd_vel_mps, self.limits.max_fwd_vel_mps)),
            vy=float(np.clip(vy_cmd, -self.limits.max_lat_vel_mps, self.limits.max_lat_vel_mps)),
            vz=float(np.clip(raw_action.vz, -self.limits.max_vert_vel_mps, self.limits.max_vert_vel_mps)),
            yaw_rate=float(np.clip(raw_action.yaw_rate, -self.limits.max_yaw_rate_radps, self.limits.max_yaw_rate_radps)),
        )

        return guided_action, selected_dir, c_left, c_right
