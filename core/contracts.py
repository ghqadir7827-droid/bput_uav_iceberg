"""contracts.py - Frozen Observation and Action Contracts.

This module formalizes the interface contract between simulation, teacher
(privileged expert), student (deployable policy), and physical hardware.

CRITICAL CONTRACT RULES:
1. Teacher may access privileged ground-truth state (global coordinates, full
   obstacle coordinates, analytical wall clearances) for demonstration generation,
   reward computation, and evaluation.
2. Student (deployable policy) must ONLY observe quantities that can be computed
   onboard the physical UAV from real sensors (rangefinder readings, IMU attitude/rates,
   relative goal vector in body frame, estimated body velocity).
3. Actions are high-level bounded velocity setpoints, NOT motor commands.
4. All angles are represented via continuous trigonometric functions to eliminate
   wrapping ambiguities at +/- pi.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple
import numpy as np


class CoordinateFrame(Enum):
    """Supported coordinate frame conventions."""
    NED = "North-East-Down"        # ArduPilot / MAVLink standard
    ENU = "East-North-Up"          # ROS / PyBullet default
    BODY_FLU = "Forward-Left-Up"   # UAV body-fixed (X=Fwd, Y=Left, Z=Up)
    BODY_FRD = "Forward-Right-Down"# UAV body-fixed (X=Fwd, Y=Right, Z=Down)


@dataclass(frozen=True)
class SensorConfig:
    """Rangefinder / LiDAR array configuration."""
    num_rays: int = 12
    max_range_m: float = 4.0
    min_range_m: float = 0.05
    noise_std_m: float = 0.02
    dropout_probability: float = 0.01

    # Angular distribution of rays relative to body Forward axis (in radians)
    # 8 horizontal rays (azimuth: 0, +/-30, +/-60, +/-90, 180)
    # 2 vertical rays (pitch: +45, -45)
    # 2 vertical nadir/zenith (pitch: +90, -90)
    ray_angles: Tuple[Tuple[float, float], ...] = (
        (0.0, 0.0),            # Front center (yaw=0, pitch=0)
        (np.deg2rad(30), 0.0),  # Front left 30 deg
        (-np.deg2rad(30), 0.0), # Front right 30 deg
        (np.deg2rad(60), 0.0),  # Side front left 60 deg
        (-np.deg2rad(60), 0.0), # Side front right 60 deg
        (np.deg2rad(90), 0.0),  # Left 90 deg
        (-np.deg2rad(90), 0.0), # Right 90 deg
        (np.pi, 0.0),           # Rear 180 deg
        (0.0, np.deg2rad(45)),  # Front pitch up 45 deg
        (0.0, -np.deg2rad(45)), # Front pitch down 45 deg
        (0.0, np.deg2rad(90)),  # Ceiling zenith (straight up)
        (0.0, -np.deg2rad(90)), # Floor nadir (straight down)
    )


@dataclass(frozen=True)
class ActionLimits:
    """Physical limits for bounded velocity setpoints."""
    max_fwd_vel_mps: float = 0.60     # Maximum forward speed
    min_fwd_vel_mps: float = 0.00     # No reverse thrust during normal transit
    max_lat_vel_mps: float = 0.40     # Lateral velocity (+Left, -Right)
    max_vert_vel_mps: float = 0.30    # Vertical velocity (+Up, -Down)
    max_yaw_rate_radps: float = 0.50  # Angular yaw rate (+CCW, -CW)
    max_accel_mps2: float = 1.00      # Max linear acceleration for rate limiting


@dataclass
class DeployableObservation:
    """Deployment-available observation vector (Student Observation).

    Length breakdown:
    - distance_rays: 12 floats in [0.0, 1.0] (normalized by max_range_m)
    - rel_goal_direction_body: 3 floats (unit vector in body FLU frame)
    - body_velocity: 3 floats (estimated [vx, vy, vz] in body FLU frame, m/s)
    - attitude_trig: 4 floats [sin(roll), cos(roll), sin(pitch), cos(pitch)]
    - yaw_rate: 1 float (normalized yaw angular rate, rad/s)
    Total dimension = 12 + 3 + 3 + 4 + 1 = 23 floats.
    """
    distance_rays: np.ndarray             # shape (12,), dtype float32
    rel_goal_direction_body: np.ndarray   # shape (3,), unit vector
    body_velocity: np.ndarray             # shape (3,), m/s
    attitude_trig: np.ndarray             # shape (4,), [s_r, c_r, s_p, c_p]
    yaw_rate: float                       # rad/s

    def to_vector(self) -> np.ndarray:
        """Flattens deployable observation into 1D float32 array of shape (23,)."""
        vec = np.concatenate([
            self.distance_rays.astype(np.float32),
            self.rel_goal_direction_body.astype(np.float32),
            self.body_velocity.astype(np.float32),
            self.attitude_trig.astype(np.float32),
            np.array([self.yaw_rate], dtype=np.float32)
        ])
        assert vec.shape == (23,), f"Expected shape (23,), got {vec.shape}"
        return vec


@dataclass
class PrivilegedState:
    """Privileged state available only to Teacher/Expert and Simulation Evaluation."""
    global_position: np.ndarray       # shape (3,), [x, y, z] in world frame (m)
    global_velocity: np.ndarray       # shape (3,), [vx, vy, vz] in world frame (m/s)
    global_orientation_quat: np.ndarray # shape (4,), [x, y, z, w]
    corridor_boundary_distance: float # distance to nearest wall/ceiling/floor (m)
    obstacle_clearance: float         # distance to nearest internal obstacle (m)
    goal_distance: float              # Euclidean distance to target (m)
    timestamp_s: float                # Simulation timestamp (s)


@dataclass
class VelocityAction:
    """High-level commanded velocity setpoint."""
    vx: float  # Forward velocity in body frame (m/s)
    vy: float  # Lateral velocity in body frame (m/s)
    vz: float  # Vertical velocity in body frame (m/s)
    yaw_rate: float # Yaw angular rate (rad/s)

    def to_normalized(self, limits: ActionLimits = ActionLimits()) -> np.ndarray:
        """Converts physical velocity setpoint to normalized [-1, 1] range."""
        norm_vx = (2.0 * (self.vx - limits.min_fwd_vel_mps) / 
                   (limits.max_fwd_vel_mps - limits.min_fwd_vel_mps)) - 1.0
        norm_vy = self.vy / limits.max_lat_vel_mps
        norm_vz = self.vz / limits.max_vert_vel_mps
        norm_yr = self.yaw_rate / limits.max_yaw_rate_radps
        return np.clip(np.array([norm_vx, norm_vy, norm_vz, norm_yr], dtype=np.float32), -1.0, 1.0)

    @classmethod
    def from_normalized(cls, norm_action: np.ndarray, limits: ActionLimits = ActionLimits()) -> "VelocityAction":
        """Converts normalized [-1, 1] array back to physical velocity setpoint."""
        a = np.clip(norm_action, -1.0, 1.0)
        vx = limits.min_fwd_vel_mps + 0.5 * (a[0] + 1.0) * (limits.max_fwd_vel_mps - limits.min_fwd_vel_mps)
        vy = float(a[1] * limits.max_lat_vel_mps)
        vz = float(a[2] * limits.max_vert_vel_mps)
        yaw_rate = float(a[3] * limits.max_yaw_rate_radps)
        return cls(vx=float(vx), vy=vy, vz=vz, yaw_rate=yaw_rate)
