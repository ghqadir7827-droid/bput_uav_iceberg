"""guidance.py - Pure-NumPy Stateless Clearance Guidance Engine.

Zero-dependency (PyTorch-free) shared implementation of the observable clearance
guidance layer for autonomous UAV corridor navigation:
- Evaluates observable rangefinder sectors (front, left, right) at 20 Hz.
- Resolves centered bimodal symmetry collapse deterministically.
- Features wall-proximity margin arbitration to eliminate sequential corner-pinches.
- Integrates directional hysteresis to suppress high-frequency chattering.
- Implements soft terminal landing release near the destination goal sphere.
- Enforces strict velocity action limits.
- Shared directly by both simulator controllers and companion hardware runners.
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Union
import numpy as np

from core.contracts import ActionLimits, DeployableObservation, VelocityAction


@dataclass
class GuidanceConfig:
    """Configurable thresholds for clearance guidance arbitration."""
    activation_threshold_m: float = 0.85
    deadband_m: float = 0.08
    hysteresis_m: float = 0.04
    wall_tight_threshold_m: float = 0.35
    wall_advantage_margin_m: float = 0.10
    min_lateral_override_mps: float = 0.25
    terminal_goal_norm_threshold: float = 0.50  # Corresponds to 3D distance < 1.0m
    default_tie_breaker: str = "LEFT"
    forward_ray_indices: Tuple[int, ...] = (0, 1, 2)
    persistent_tie_breaking: bool = False
    enable_corridor_adaptation: bool = False
    adaptive_wall_threshold_m: float = 0.80
    adaptive_corridor_width_m: float = 3.00
    adaptive_lateral_override_mps: float = 0.30
    enable_continuous_clearance_scaling: bool = False
    max_lateral_override_mps: float = 0.40
    continuous_headroom_ref_m: float = 1.00
    critical_lateral_safety_m: float = 0.475

    @classmethod
    def for_65cm_uav(cls, drone_radius: float = 0.325) -> "GuidanceConfig":
        """Clearance guidance configuration scaled for 65 cm physical quadcopter."""
        return cls(
            activation_threshold_m=float(drone_radius + 0.90),  # 1.225 m from center
            deadband_m=0.12,
            hysteresis_m=0.06,
            wall_tight_threshold_m=float(drone_radius + 0.30),  # 0.625 m from center
            wall_advantage_margin_m=0.15,
            min_lateral_override_mps=0.30,
            terminal_goal_norm_threshold=0.50,
            default_tie_breaker="LEFT",
            forward_ray_indices=(0, 1, 2),
            persistent_tie_breaking=False,
        )

    @classmethod
    def for_65cm_uav_v2(cls, drone_radius: float = 0.325) -> "GuidanceConfig":
        """Harmonized clearance guidance configuration matching safety supervisor sector coverage."""
        return cls(
            activation_threshold_m=float(drone_radius + 0.95),  # 1.275 m from center
            deadband_m=0.08,
            hysteresis_m=0.06,
            wall_tight_threshold_m=float(drone_radius + 0.35),  # 0.675 m from center
            wall_advantage_margin_m=0.10,
            min_lateral_override_mps=0.35,
            terminal_goal_norm_threshold=0.50,
            default_tie_breaker="LEFT",
            forward_ray_indices=(0, 1, 2, 3, 4),
            persistent_tie_breaking=False,
        )

    @classmethod
    def for_65cm_uav_v3(cls, drone_radius: float = 0.325) -> "GuidanceConfig":
        """Guidance v3 with directional persistence in deadband to eliminate chatter."""
        return cls(
            activation_threshold_m=float(drone_radius + 1.05),  # 1.375 m from center
            deadband_m=0.08,
            hysteresis_m=0.06,
            wall_tight_threshold_m=float(drone_radius + 0.35),  # 0.675 m from center
            wall_advantage_margin_m=0.10,
            min_lateral_override_mps=0.40,
            terminal_goal_norm_threshold=0.50,
            default_tie_breaker="LEFT",
            forward_ray_indices=(0, 1, 2, 3, 4),
            persistent_tie_breaking=True,
        )

    @classmethod
    def for_65cm_uav_v4_adaptive(cls, drone_radius: float = 0.325) -> "GuidanceConfig":
        """Guidance v4 candidate: corridor-width-adaptive lateral guidance with persistent tie-breaking."""
        return cls(
            activation_threshold_m=float(drone_radius + 1.05),  # 1.375 m from center
            deadband_m=0.08,
            hysteresis_m=0.06,
            wall_tight_threshold_m=float(drone_radius + 0.35),  # 0.675 m from center
            wall_advantage_margin_m=0.10,
            min_lateral_override_mps=0.40,                      # Nominal override in standard/wide corridors
            terminal_goal_norm_threshold=0.50,
            default_tie_breaker="LEFT",
            forward_ray_indices=(0, 1, 2, 3, 4),
            persistent_tie_breaking=True,
            enable_corridor_adaptation=True,
            adaptive_wall_threshold_m=0.80,                     # Sensed lateral wall clearance < 0.80m triggers narrow mode
            adaptive_corridor_width_m=3.00,                     # Sensed corridor width < 3.00m triggers narrow mode
            adaptive_lateral_override_mps=0.30,                 # Reduced override to prevent buffer chattering in narrow corridors
        )

    @classmethod
    def for_65cm_uav_v5_clearance_aware(cls, drone_radius: float = 0.325) -> "GuidanceConfig":
        """Guidance v5 candidate: continuous clearance-aware lateral guidance scaling (0.15 - 0.40 m/s)."""
        return cls(
            activation_threshold_m=float(drone_radius + 1.05),  # 1.375 m from center
            deadband_m=0.08,
            hysteresis_m=0.08,
            wall_tight_threshold_m=float(drone_radius + 0.35),  # 0.675 m from center
            wall_advantage_margin_m=0.10,
            min_lateral_override_mps=0.15,                      # Minimum continuous override (0.15 m/s)
            max_lateral_override_mps=0.40,                      # Maximum continuous override (0.40 m/s)
            terminal_goal_norm_threshold=0.50,
            default_tie_breaker="LEFT",
            forward_ray_indices=(0, 1, 2, 3, 4),
            persistent_tie_breaking=True,
            enable_corridor_adaptation=True,
            adaptive_corridor_width_m=3.20,
            adaptive_lateral_override_mps=0.28,
            enable_continuous_clearance_scaling=True,
            continuous_headroom_ref_m=0.50,
            critical_lateral_safety_m=0.35,
        )


class StatelessClearanceGuidanceArbitrator:
    """Pure-NumPy Stateless Clearance Guidance Arbitrator.
    
    Provides exact mathematical parity across simulator and embedded edge environments
    with zero PyTorch or runtime-heavy dependencies.
    """

    def __init__(
        self,
        config: Optional[GuidanceConfig] = None,
        limits: Optional[ActionLimits] = None,
    ):
        self.config = config or GuidanceConfig()
        self.limits = limits or ActionLimits()
        self.last_dir: str = "NONE"

    def reset(self) -> None:
        """Resets persistent direction latch and filter states."""
        self.last_dir = "NONE"

    def arbitrate(
        self,
        raw_action: VelocityAction,
        observation: Union[np.ndarray, DeployableObservation],
        suppress_guidance: bool = False,
    ) -> Tuple[VelocityAction, str, float, float]:
        """Arbitrates velocity commands based on instantaneous spatial clearance.
        
        Args:
            raw_action: Nominal policy VelocityAction [vx, vy, vz, yaw_rate].
            observation: 23-float or 27-float observation vector or DeployableObservation instance.
            suppress_guidance: When True (e.g. during SAFE_ARREST), zeroes lateral override.
            
        Returns:
            Tuple of (guided_action, selected_dir, c_left, c_right)
        """
        if isinstance(observation, DeployableObservation):
            obs_vec = observation.to_vector()
        else:
            obs_vec = np.asarray(observation, dtype=np.float32)

        # Sanitize observation vector against non-finite values from sensor noise / dropouts
        obs_vec = np.nan_to_num(obs_vec, nan=0.0, posinf=1.0, neginf=-1.0)

        # 1. Extract Sector Clearances (physical meters from normalized rays, max_range=4.0m)
        rays = obs_vec[0:12] * 4.0
        fwd_indices = list(self.config.forward_ray_indices)
        c_front = float(min(rays[i] for i in fwd_indices))
        c_left = float(min(rays[1], rays[3], rays[5]))
        c_right = float(min(rays[2], rays[4], rays[6]))
        diff = c_left - c_right

        # Physical wall clearances from 90-degree lateral beams
        wall_l = float(rays[5])  # +90 deg azimuth (left wall)
        wall_r = float(rays[6])  # -90 deg azimuth (right wall)

        # 2. Terminal approach detection from obs[12:15] (scaled goal vector in body FLU)
        # When dist_goal < 2.0m, norm is dist_goal / 2.0. So norm < 0.50 means 3D dist_goal < 1.0m.
        # If obs_vec has 16 rays (length >= 27), goal is at index 16:19
        if len(obs_vec) >= 27:
            goal_vec = obs_vec[16:19]
        else:
            goal_vec = obs_vec[12:15]
        goal_norm = float(np.linalg.norm(goal_vec))
        in_terminal_zone = (goal_norm < self.config.terminal_goal_norm_threshold)

        # If safe-arrest state machine demands guidance suppression, zero lateral overrides immediately
        if suppress_guidance:
            self.last_dir = "SUPPRESSED_SAFE_ARREST"
            guided_action = VelocityAction(
                vx=float(np.clip(raw_action.vx, self.limits.min_fwd_vel_mps, self.limits.max_fwd_vel_mps)),
                vy=0.0,
                vz=float(np.clip(raw_action.vz, -self.limits.max_vert_vel_mps, self.limits.max_vert_vel_mps)),
                yaw_rate=float(np.clip(raw_action.yaw_rate, -self.limits.max_yaw_rate_radps, self.limits.max_yaw_rate_radps)),
            )
            return guided_action, self.last_dir, c_left, c_right

        selected_dir = "NONE"
        vy_cmd = raw_action.vy
        override_speed = self.config.max_lateral_override_mps if self.config.enable_continuous_clearance_scaling else self.config.min_lateral_override_mps

        # Dynamic corridor-width adaptation for lateral guidance
        if self.config.enable_corridor_adaptation:
            wall_l_valid = bool(np.isfinite(wall_l) and (wall_l > 0.05))
            wall_r_valid = bool(np.isfinite(wall_r) and (wall_r > 0.05))
            if wall_l_valid and wall_r_valid:
                min_wall = min(wall_l, wall_r)
                corridor_width_est = wall_l + wall_r
                if (
                    min_wall < self.config.adaptive_wall_threshold_m
                    or corridor_width_est < self.config.adaptive_corridor_width_m
                ):
                    override_speed = self.config.adaptive_lateral_override_mps

        def get_override(direction: str) -> float:
            """Computes continuous clearance-scaled override or constant override speed."""
            if not self.config.enable_continuous_clearance_scaling:
                return override_speed
            target_c = min(wall_l, c_left) if "LEFT" in direction else min(wall_r, c_right)
            headroom = max(0.0, target_c - self.config.critical_lateral_safety_m)
            rho = min(1.0, headroom / max(0.01, self.config.continuous_headroom_ref_m))
            max_v = override_speed if self.config.enable_corridor_adaptation else self.config.max_lateral_override_mps
            min_v = min(self.config.min_lateral_override_mps, max_v)
            return float(min_v + rho * (max_v - min_v))

        # 3. Clearance Guidance Arbitration
        if c_front < self.config.activation_threshold_m and not in_terminal_zone:
            left_wall_tight = (wall_l < self.config.wall_tight_threshold_m)
            right_wall_tight = (wall_r < self.config.wall_tight_threshold_m)

            if left_wall_tight and not right_wall_tight:
                # Left wall is dangerously close: enforce right evasion
                selected_dir = "WALL_AVOID_RIGHT"
                vy_cmd = min(vy_cmd, -get_override("RIGHT"))
            elif right_wall_tight and not left_wall_tight:
                # Right wall is dangerously close: enforce left evasion
                selected_dir = "WALL_AVOID_LEFT"
                vy_cmd = max(vy_cmd, get_override("LEFT"))
            else:
                # Both walls have adequate clearance: use clearance differential with directional hysteresis
                h = self.config.hysteresis_m
                deadband = self.config.deadband_m
                l_th = deadband - (h if self.last_dir in ["LEFT", "TIE_BREAK_LEFT"] else 0.0) + (h if self.last_dir in ["RIGHT", "TIE_BREAK_RIGHT"] else 0.0)
                r_th = -deadband - (h if self.last_dir in ["LEFT", "TIE_BREAK_LEFT"] else 0.0) + (h if self.last_dir in ["RIGHT", "TIE_BREAK_RIGHT"] else 0.0)

                if diff > l_th:
                    selected_dir = "LEFT"
                    vy_cmd = max(vy_cmd, get_override("LEFT"))
                elif diff < r_th:
                    selected_dir = "RIGHT"
                    vy_cmd = min(vy_cmd, -get_override("RIGHT"))
                else:
                    # In symmetry deadband:
                    if (
                        self.config.persistent_tie_breaking
                        and self.last_dir in ["LEFT", "TIE_BREAK_LEFT"]
                        and not left_wall_tight
                    ):
                        selected_dir = "TIE_BREAK_LEFT"
                        vy_cmd = max(vy_cmd, get_override("LEFT"))
                    elif (
                        self.config.persistent_tie_breaking
                        and self.last_dir in ["RIGHT", "TIE_BREAK_RIGHT"]
                        and not right_wall_tight
                    ):
                        selected_dir = "TIE_BREAK_RIGHT"
                        vy_cmd = min(vy_cmd, -get_override("RIGHT"))
                    else:
                        # Initial tie-break or wall advantage margin
                        margin = self.config.wall_advantage_margin_m
                        if wall_r > wall_l + margin:
                            selected_dir = "TIE_BREAK_RIGHT"
                            vy_cmd = min(vy_cmd, -get_override("RIGHT"))
                        elif wall_l > wall_r + margin:
                            selected_dir = "TIE_BREAK_LEFT"
                            vy_cmd = max(vy_cmd, get_override("LEFT"))
                        else:
                            selected_dir = f"TIE_BREAK_{self.config.default_tie_breaker}"
                            if self.config.default_tie_breaker == "RIGHT":
                                vy_cmd = min(vy_cmd, -get_override("RIGHT"))
                            else:
                                vy_cmd = max(vy_cmd, get_override("LEFT"))

            self.last_dir = selected_dir
        else:
            self.last_dir = "NONE"

        # 4. Enforce Action Limits
        guided_action = VelocityAction(
            vx=float(np.clip(raw_action.vx, self.limits.min_fwd_vel_mps, self.limits.max_fwd_vel_mps)),
            vy=float(np.clip(vy_cmd, -self.limits.max_lat_vel_mps, self.limits.max_lat_vel_mps)),
            vz=float(np.clip(raw_action.vz, -self.limits.max_vert_vel_mps, self.limits.max_vert_vel_mps)),
            yaw_rate=float(np.clip(raw_action.yaw_rate, -self.limits.max_yaw_rate_radps, self.limits.max_yaw_rate_radps)),
        )

        return guided_action, selected_dir, c_left, c_right

