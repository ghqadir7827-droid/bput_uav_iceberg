"""apf_expert.py - Scripted Artificial Potential Field (APF) Expert Controller.

Implements an analytical gradient-based navigation expert using attractive goal potential,
repulsive wall potential, repulsive obstacle potential, and tangential vortex circulation
to guarantee smooth, collision-free obstacle evasion and terminal landing.

PRIVILEGED INFORMATION USAGE:
- The APF expert uses privileged geometric obstacle centers [x_i, y_i, z_i] and analytical
  corridor boundaries to compute optimal potential gradients.
- STUDENT CONTRACT PRESERVATION:
  During demonstration recording, ONLY the 23-float deployable observation vector
  (multi-beam rangefinder rays, relative goal vector in FLU, body velocity, attitude trig,
  yaw rate) is logged as the training input. Privileged obstacle centers are strictly excluded
  from student demonstration inputs.
"""

from typing import List, Optional, Tuple, Union
import numpy as np

from controllers.registry import ControllerMetadata, IController
from core.contracts import ActionLimits, VelocityAction
from core.geometry import Corridor3D, Obstacle3D


class ScriptedAPFExpertController(IController):
    """Scripted Artificial Potential Field expert for corridor following and obstacle evasion."""

    def __init__(
        self,
        corridor: Optional[Corridor3D] = None,
        cruise_speed: float = 0.65,
        k_att: float = 1.0,
        k_rep_obs: float = 0.45,
        d_rep_obs: float = 1.30,
        k_circ: float = 0.55,
        k_wall: float = 1.8,
        d_rep_wall: float = 0.50,
    ):
        self.corridor = corridor or Corridor3D()
        self.cruise_speed = cruise_speed
        self.k_att = k_att
        self.k_rep_obs = k_rep_obs
        self.d_rep_obs = d_rep_obs
        self.k_circ = k_circ
        self.k_wall_inv = 0.35
        self.d_rep_wall = 0.45

        self._meta = ControllerMetadata(
            controller_id="scripted_expert_apf",
            checkpoint_id="m2_apf_expert_v1",
            display_name="Scripted APF Expert",
            is_implemented=True,
            milestone_tag="Milestone 2",
            description="Analytical Artificial Potential Field with tangential circulation and inverse boundary repulsion."
        )

        self._cached_obstacles: List[Obstacle3D] = []
        self._last_position: Optional[np.ndarray] = None
        self._last_velocity: Optional[np.ndarray] = None

    @property
    def metadata(self) -> ControllerMetadata:
        return self._meta

    def update_environment(self, corridor: Corridor3D, obstacles: List[Obstacle3D]) -> None:
        """Updates internal geometric model with active episode scene."""
        self.corridor = corridor
        self._cached_obstacles = list(obstacles)

    def set_state(self, position: np.ndarray, velocity: Optional[np.ndarray] = None) -> None:
        """Sets internal privileged position and velocity for step computation."""
        self._last_position = np.array(position, dtype=np.float64)
        if velocity is not None:
            self._last_velocity = np.array(velocity, dtype=np.float64)

    def predict(self, observation: np.ndarray) -> VelocityAction:
        """Predicts action. Uses privileged scene if set_state was called, else deployable APF."""
        if self._last_position is not None and self._cached_obstacles:
            vel = self._last_velocity if self._last_velocity is not None else np.zeros(3)
            return self.compute_privileged_expert_action(self._last_position, vel, self._cached_obstacles)

        # Fallback to rangefinder-based reactive APF
        rays = observation[0:12] * 4.0
        rel_goal = observation[12:15]
        gx, gy, gz = float(rel_goal[0]), float(rel_goal[1]), float(rel_goal[2])

        d_front = float(rays[0])
        d_fl = float(min(rays[1], rays[3]))
        d_fr = float(min(rays[2], rays[4]))
        d_left = float(rays[5])
        d_right = float(rays[6])
        d_up = float(rays[10])
        d_down = float(rays[11])

        front_min = min(d_front, d_fl, d_fr)

        if gx < 0:
            vx = -0.15
        elif front_min < 1.4:
            vx = float(np.clip(0.25 + 0.35 * (front_min - 0.4) / 1.0, 0.18, self.cruise_speed))
        else:
            vx = float(self.cruise_speed * np.clip(gx, 0.25, 1.0))

        vy = float(0.5 * gy)
        vz = float(0.5 * gz)

        if front_min < 1.5:
            left_score = d_fl + 0.65 * d_left
            right_score = d_fr + 0.65 * d_right
            repel_strength = float(np.clip((1.5 - front_min) * 0.70, 0.0, 0.65))
            if left_score > right_score:
                vy += repel_strength
            else:
                vy -= repel_strength

        if d_left < 0.65:
            vy -= float((0.65 - d_left) * 0.6)
        if d_right < 0.65:
            vy += float((0.65 - d_right) * 0.6)

        if d_down < 0.50:
            vz += float((0.50 - d_down) * 0.4)
        if d_up < 0.50:
            vz -= float((0.50 - d_up) * 0.4)

        yaw_rate = float(np.clip(1.2 * vy, -0.4, 0.4))

        return VelocityAction(
            vx=float(np.clip(vx, -0.2, 0.8)),
            vy=float(np.clip(vy, -0.6, 0.6)),
            vz=float(np.clip(vz, -0.3, 0.3)),
            yaw_rate=yaw_rate
        )

    def compute_privileged_expert_action(
        self,
        position: np.ndarray,
        velocity: np.ndarray,
        obstacles: List[Obstacle3D],
    ) -> VelocityAction:
        """Computes analytical potential field action using privileged scene geometry.
        
        This generates the optimal expert trajectory demonstrations.
        """
        p_curr = np.asarray(position, dtype=np.float64)
        c = self.corridor
        goal = np.asarray(c.goal_position, dtype=np.float64)

        # 1. Attractive Potential towards Goal
        d_goal_vec = goal - p_curr
        d_goal = float(np.linalg.norm(d_goal_vec))
        if d_goal > 1e-4:
            u_att = d_goal_vec / d_goal
        else:
            u_att = np.zeros(3)

        # Scale forward velocity: cruise speed when far, gentle proportional landing when near
        if d_goal < 1.5:
            vx_des = float(self.cruise_speed * (d_goal / 1.5))
            vx_des = min(vx_des, 0.28)
        else:
            vx_des = self.cruise_speed

        vy_des = float(0.70 * u_att[1])
        vz_des = float(0.80 * u_att[2])

        # 2. Analytical Obstacle Repulsion & Tangential Circulation
        for obs in obstacles:
            obs_c = np.asarray(obs.center, dtype=np.float64)
            delta_xy = p_curr[:2] - obs_c[:2]
            dist_center = float(np.linalg.norm(delta_xy))
            clearance = dist_center - obs.radius - c.drone_radius

            dx = obs_c[0] - p_curr[0]
            if dx > -0.60 and clearance < self.d_rep_obs:
                if dist_center > 1e-4:
                    n_rep = delta_xy / dist_center
                else:
                    n_rep = np.array([0.0, 1.0])

                rep_mag = float(np.clip(
                    self.k_rep_obs * ((1.0 / max(0.04, clearance)) - (1.0 / self.d_rep_obs)),
                    0.0, 1.2
                ))

                # Tangential Circulation Force (avoid saddle points)
                evade_dir = -1.0 if obs_c[1] >= 0 else 1.0
                circ_mag = float(self.k_circ * np.exp(-max(0.0, clearance) / 0.35))

                vy_des += float(rep_mag * n_rep[1] + circ_mag * evade_dir)

                if abs(delta_xy[1]) < (obs.radius + c.drone_radius + 0.15) and dx > 0:
                    slow_factor = np.clip(clearance / self.d_rep_obs, 0.40, 1.0)
                    vx_des *= float(slow_factor)

        # 3. Inverse Corridor Wall Repulsion (barrier potential)
        d_left = float(c.y_max - p_curr[1] - c.drone_radius)
        d_right = float(p_curr[1] - c.y_min - c.drone_radius)
        if d_left < self.d_rep_wall:
            rep_left = self.k_wall_inv * ((1.0 / max(0.02, d_left)) - (1.0 / self.d_rep_wall))
            vy_des -= float(rep_left)
        if d_right < self.d_rep_wall:
            rep_right = self.k_wall_inv * ((1.0 / max(0.02, d_right)) - (1.0 / self.d_rep_wall))
            vy_des += float(rep_right)

        # 4. Vertical Buffer (Floor & Ceiling)
        d_floor = float(p_curr[2] - c.z_floor - c.drone_radius)
        d_ceil = float(c.z_ceiling - p_curr[2] - c.drone_radius)
        if d_floor < 0.35:
            vz_des += float(0.80 * (0.35 - d_floor))
        if d_ceil < 0.35:
            vz_des -= float(0.80 * (0.35 - d_ceil))

        # 5. Yaw Rate Alignment
        yaw_rate = float(np.clip(1.2 * vy_des, -0.5, 0.5))

        return VelocityAction(
            vx=float(np.clip(vx_des, -0.15, 0.70)),
            vy=float(np.clip(vy_des, -0.60, 0.60)),
            vz=float(np.clip(vz_des, -0.30, 0.30)),
            yaw_rate=yaw_rate
        )
