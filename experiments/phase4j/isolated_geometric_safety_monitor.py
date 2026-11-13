"""isolated_geometric_safety_monitor.py - Isolated Oracle-Assisted Geometric Safety Monitor.

SIMULATION-ONLY DIAGNOSTIC / ORACLE-ASSISTED SAFETY EXPERIMENT.
WARNING: Uses simulator ground-truth obstacle geometry. This is an analytical research
tool to investigate short-horizon clearance prediction and is NOT directly deployable on
physical hardware without an onboard mapping / state estimation system (e.g. 3D occupancy grid).

Design:
- Operates downstream of the supervisor.
- Projects vehicle position forward over a short horizon (tau = 0.15s, 3 steps).
- Checks ground-truth 3D clearance against obstacles (including rear-lateral quadrants).
- If predicted clearance violates the safety envelope (< 0.080m), intervenes by arresting
  the dangerous velocity component toward the obstacle.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np

from core.contracts import VelocityAction
from core.geometry import Corridor3D, Obstacle3D


GUARD_ENVELOPE_M: float = 0.080 # Minimum required physical air gap
PREDICTION_HORIZON_S: float = 0.15 # Lookahead horizon
DT_PREDICT_S: float = 0.05


class OracleGeometricSafetyMonitor:
    """Simulation-only oracle safety monitor evaluating short-horizon geometric clearance."""

    def __init__(
        self,
        corridor: Corridor3D,
        guard_envelope_m: float = GUARD_ENVELOPE_M,
        prediction_horizon_s: float = PREDICTION_HORIZON_S,
    ):
        self.corridor = corridor
        self.guard_envelope_m = guard_envelope_m
        self.prediction_horizon_s = prediction_horizon_s
        self.total_interventions: int = 0
        self.blocked_envelope_violations: int = 0

    def reset_state(self):
        self.total_interventions = 0
        self.blocked_envelope_violations = 0

    def evaluate_and_filter(
        self,
        proposed_action: VelocityAction,
        current_position: np.ndarray,
        current_velocity: np.ndarray,
        current_yaw: float,
        obstacles: List[Obstacle3D],
    ) -> Tuple[VelocityAction, bool, str]:
        """Evaluates short-horizon projected clearance. Intervenes if guard envelope is breached.

        Returns:
            (filtered_action, intervened_flag, reason_string)
        """
        pos = np.asarray(current_position, dtype=np.float64)
        r_drone = self.corridor.drone_radius

        # Convert proposed body velocity to world velocity using current yaw
        cos_y = np.cos(current_yaw)
        sin_y = np.sin(current_yaw)
        world_vx = cos_y * proposed_action.vx - sin_y * proposed_action.vy
        world_vy = sin_y * proposed_action.vx + cos_y * proposed_action.vy
        world_vz = proposed_action.vz

        cmd_world_vel = np.array([world_vx, world_vy, world_vz], dtype=np.float64)

        # Check current ground-truth clearance
        curr_min_clear = 10.0
        nearest_obs: Optional[Obstacle3D] = None
        for obs in obstacles:
            c = obs.distance_to_point(pos) - r_drone
            if c < curr_min_clear:
                curr_min_clear = c
                nearest_obs = obs

        # Short-horizon lookahead steps
        num_steps = max(1, int(round(self.prediction_horizon_s / DT_PREDICT_S)))
        min_pred_clear = curr_min_clear
        critical_t = 0.0

        for step in range(1, num_steps + 1):
            t = step * DT_PREDICT_S
            p_pred = pos + cmd_world_vel * t
            for obs in obstacles:
                c_pred = obs.distance_to_point(p_pred) - r_drone
                if c_pred < min_pred_clear:
                    min_pred_clear = c_pred
                    critical_t = t
                    nearest_obs = obs

        # If predicted clearance breaches guard envelope, intervene
        if min_pred_clear < self.guard_envelope_m and nearest_obs is not None:
            self.total_interventions += 1
            if min_pred_clear < 0.05:
                self.blocked_envelope_violations += 1

            # Relative vector from drone to nearest obstacle in world frame
            rel_vec = nearest_obs.center[:2] - pos[:2]
            dist_to_center = np.linalg.norm(rel_vec)
            if dist_to_center > 1e-5:
                unit_to_obs = rel_vec / dist_to_center
            else:
                unit_to_obs = np.array([1.0, 0.0])

            # Transform obstacle unit direction into body frame
            body_obs_x = cos_y * unit_to_obs[0] + sin_y * unit_to_obs[1]
            body_obs_y = -sin_y * unit_to_obs[0] + cos_y * unit_to_obs[1]

            filt_vx = proposed_action.vx
            filt_vy = proposed_action.vy

            # 1. Lateral arbitration: if obstacle is to the side and drone is moving laterally toward it
            if body_obs_y > 0.05 and filt_vy > 0.0:
                filt_vy = -0.10 # Push away from obstacle on the left
            elif body_obs_y < -0.05 and filt_vy < 0.0:
                filt_vy = 0.10  # Push away from obstacle on the right

            # 2. Longitudinal arbitration: if obstacle is ahead and drone is moving forward toward it
            if body_obs_x > 0.10 and filt_vx > 0.10:
                filt_vx = 0.05 # Clamp forward velocity when approaching obstacle ahead

            safe_act = VelocityAction(
                vx=float(np.clip(filt_vx, self.corridor.drone_radius, 0.60)),
                vy=float(np.clip(filt_vy, -0.40, 0.40)),
                vz=proposed_action.vz,
                yaw_rate=proposed_action.yaw_rate,
            )
            reason = f"Oracle Geometric Monitor Active: predicted clearance {min_pred_clear:.3f}m < {self.guard_envelope_m:.3f}m at t={critical_t:.2f}s"
            return safe_act, True, reason

        return proposed_action, False, "Geometric clearance envelope satisfied"
