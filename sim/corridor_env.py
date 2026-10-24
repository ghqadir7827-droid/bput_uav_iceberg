"""corridor_env.py - Minimal Native PyBullet 3D Corridor Environment.

Implements a standard Gymnasium environment modeling a 3D obstacle corridor.
Uses PyBullet headless physics (p.DIRECT) for fast, zero-dependency execution.
Provides multi-beam raycast distance sensing, high-level velocity control,
and explicit 3D boundary distance metrics.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p

from core.contracts import ActionLimits, DeployableObservation, PrivilegedState, SensorConfig, VelocityAction
from core.geometry import Corridor3D, Obstacle3D, TerminationReason


class CorridorSimEnv(gym.Env):
    """Gymnasium environment for 3D UAV corridor navigation."""

    metadata = {"render_modes": ["direct", "human"]}

    def __init__(
        self,
        scenario: Union[str, Corridor3D] = "10m",
        corridor: Optional[Corridor3D] = None,
        sensor_config: SensorConfig = SensorConfig(),
        action_limits: ActionLimits = ActionLimits(),
        dt: float = 0.05,
        gui: bool = False,
        seed: Optional[int] = None,
    ):
        super().__init__()
        if corridor is not None:
            self.corridor = corridor
        else:
            self.corridor = Corridor3D.from_scenario(scenario)

        self.sensor_config = sensor_config
        self.action_limits = action_limits
        self.dt = dt
        self.gui = gui

        self._rng = np.random.default_rng(seed)
        self.client_id = -1
        self.drone_id = -1
        self.wall_ids: List[int] = []
        self.obstacle_ids: List[int] = []
        self.obstacles: List[Obstacle3D] = []

        self.step_count = 0
        self.max_steps = int(self.corridor.max_steps)
        self.current_sim_time = 0.0
        self.prev_position = np.zeros(3, dtype=np.float32)

        # Define Observation Space: 23 continuous floats
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(23,), dtype=np.float32
        )

        # Define Action Space: 4 continuous bounded velocity setpoints in [-1, 1]
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(4,), dtype=np.float32
        )

        self._init_pybullet()

    def _init_pybullet(self) -> None:
        """Connects to PyBullet in headless or GUI mode."""
        if self.client_id >= 0:
            p.disconnect(self.client_id)

        mode = p.GUI if self.gui else p.DIRECT
        self.client_id = p.connect(mode)
        p.setTimeStep(self.dt, physicsClientId=self.client_id)
        p.setGravity(0, 0, 0, physicsClientId=self.client_id) # Zero gravity since flight controller handles hover equilibrium

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self.step_count = 0
        self.current_sim_time = 0.0

        p.resetSimulation(physicsClientId=self.client_id)
        p.setGravity(0, 0, 0, physicsClientId=self.client_id)

        self._build_corridor()
        self._spawn_obstacles()
        self._spawn_drone()

        self.prev_position = self._get_drone_position()

        deployable_obs, priv_state = self._get_observations()
        info = {
            "privileged_state": priv_state,
            "boundary_distance": priv_state.corridor_boundary_distance,
            "obstacle_clearance": priv_state.obstacle_clearance,
            "termination_reason": TerminationReason.CONTINUE.value,
        }
        return deployable_obs.to_vector(), info

    def _build_corridor(self) -> None:
        """Constructs boundary visual and collision boxes for walls, floor, and ceiling."""
        self.wall_ids = []
        c = self.corridor
        length = c.x_end - c.x_start
        center_x = (c.x_start + c.x_end) / 2.0
        wall_thick = 0.1

        # 1. Floor (Dark Slate)
        floor_shape = p.createCollisionShape(
            p.GEOM_BOX, halfExtents=[length / 2.0, c.width / 2.0, wall_thick / 2.0],
            physicsClientId=self.client_id
        )
        floor_vis = p.createVisualShape(
            p.GEOM_BOX, halfExtents=[length / 2.0, c.width / 2.0, wall_thick / 2.0],
            rgbaColor=[0.20, 0.23, 0.28, 1.0],
            physicsClientId=self.client_id
        )
        floor_id = p.createMultiBody(
            baseMass=0, baseCollisionShapeIndex=floor_shape, baseVisualShapeIndex=floor_vis,
            basePosition=[center_x, c.y_center, c.z_floor - wall_thick / 2.0],
            physicsClientId=self.client_id
        )
        self.wall_ids.append(floor_id)

        # 2. Ceiling (Collision only, no visual box so interior is visible from above)
        ceil_shape = p.createCollisionShape(
            p.GEOM_BOX, halfExtents=[length / 2.0, c.width / 2.0, wall_thick / 2.0],
            physicsClientId=self.client_id
        )
        ceil_id = p.createMultiBody(
            baseMass=0, baseCollisionShapeIndex=ceil_shape, baseVisualShapeIndex=-1,
            basePosition=[center_x, c.y_center, c.z_ceiling + wall_thick / 2.0],
            physicsClientId=self.client_id
        )
        self.wall_ids.append(ceil_id)

        # 3. Left Wall (+Y, Translucent Cyan-Gray)
        left_shape = p.createCollisionShape(
            p.GEOM_BOX, halfExtents=[length / 2.0, wall_thick / 2.0, c.height / 2.0],
            physicsClientId=self.client_id
        )
        left_vis = p.createVisualShape(
            p.GEOM_BOX, halfExtents=[length / 2.0, wall_thick / 2.0, c.height / 2.0],
            rgbaColor=[0.30, 0.45, 0.62, 0.35],
            physicsClientId=self.client_id
        )
        left_id = p.createMultiBody(
            baseMass=0, baseCollisionShapeIndex=left_shape, baseVisualShapeIndex=left_vis,
            basePosition=[center_x, c.y_max + wall_thick / 2.0, (c.z_floor + c.z_ceiling) / 2.0],
            physicsClientId=self.client_id
        )
        self.wall_ids.append(left_id)

        # 4. Right Wall (-Y, Translucent Cyan-Gray)
        right_shape = p.createCollisionShape(
            p.GEOM_BOX, halfExtents=[length / 2.0, wall_thick / 2.0, c.height / 2.0],
            physicsClientId=self.client_id
        )
        right_vis = p.createVisualShape(
            p.GEOM_BOX, halfExtents=[length / 2.0, wall_thick / 2.0, c.height / 2.0],
            rgbaColor=[0.30, 0.45, 0.62, 0.35],
            physicsClientId=self.client_id
        )
        right_id = p.createMultiBody(
            baseMass=0, baseCollisionShapeIndex=right_shape, baseVisualShapeIndex=right_vis,
            basePosition=[center_x, c.y_min - wall_thick / 2.0, (c.z_floor + c.z_ceiling) / 2.0],
            physicsClientId=self.client_id
        )
        self.wall_ids.append(right_id)

    def _spawn_obstacles(self) -> None:
        """Places standard internal obstacles (pillars with randomized positions)."""
        self.obstacle_ids = []
        self.obstacles = []
        c = self.corridor

        for x_pos in c.pillar_x_positions:
            y_offset = float(self._rng.uniform(-c.pillar_y_range, c.pillar_y_range))
            radius = c.pillar_radius
            height = c.height

            col_shape = p.createCollisionShape(
                p.GEOM_CYLINDER, radius=radius, height=height,
                physicsClientId=self.client_id
            )
            vis_shape = p.createVisualShape(
                p.GEOM_CYLINDER, radius=radius, length=height,
                rgbaColor=[0.95, 0.35, 0.10, 0.95],
                physicsClientId=self.client_id
            )
            obs_id = p.createMultiBody(
                baseMass=0, baseCollisionShapeIndex=col_shape, baseVisualShapeIndex=vis_shape,
                basePosition=[x_pos, y_offset, (c.z_floor + c.z_ceiling) / 2.0],
                physicsClientId=self.client_id
            )
            self.obstacle_ids.append(obs_id)
            self.obstacles.append(Obstacle3D(
                center=np.array([x_pos, y_offset, (c.z_floor + c.z_ceiling) / 2.0]),
                radius=radius,
                height=height,
                obstacle_type="cylinder"
            ))

    def _spawn_drone(self) -> None:
        """Creates drone rigid body at the entrance (Bright Cyan Drone)."""
        c = self.corridor
        start_x = float(c.start_position[0])
        # Slight initial lateral perturbation
        start_y = float(c.start_position[1] + self._rng.uniform(-0.1, 0.1))
        start_z = float(c.start_position[2])

        drone_shape = p.createCollisionShape(
            p.GEOM_SPHERE, radius=c.drone_radius,
            physicsClientId=self.client_id
        )
        drone_vis = p.createVisualShape(
            p.GEOM_SPHERE, radius=c.drone_radius,
            rgbaColor=[0.10, 0.82, 0.95, 1.0],
            physicsClientId=self.client_id
        )
        self.drone_id = p.createMultiBody(
            baseMass=1.2, baseCollisionShapeIndex=drone_shape, baseVisualShapeIndex=drone_vis,
            basePosition=[start_x, start_y, start_z],
            baseOrientation=[0, 0, 0, 1],
            physicsClientId=self.client_id
        )

    def _get_drone_position(self) -> np.ndarray:
        pos, _ = p.getBasePositionAndOrientation(self.drone_id, physicsClientId=self.client_id)
        return np.array(pos, dtype=np.float32)

    def _get_drone_velocity(self) -> np.ndarray:
        lin_vel, _ = p.getBaseVelocity(self.drone_id, physicsClientId=self.client_id)
        return np.array(lin_vel, dtype=np.float32)

    def _get_drone_orientation(self) -> Tuple[np.ndarray, float, float, float]:
        _, quat = p.getBasePositionAndOrientation(self.drone_id, physicsClientId=self.client_id)
        euler = p.getEulerFromQuaternion(quat) # roll, pitch, yaw
        return np.array(quat, dtype=np.float32), euler[0], euler[1], euler[2]

    def _cast_ray_sensors(self, pos: np.ndarray, yaw: float) -> np.ndarray:
        """Casts multi-beam distance rays using PyBullet rayTestBatch."""
        cfg = self.sensor_config
        ray_from = []
        ray_to = []

        # Ray origin offset: start slightly outside drone collision sphere (radius ~0.15m)
        offset = float(self.corridor.drone_radius + 0.03)  # 0.18 m
        eff_span = cfg.max_range_m - offset

        for azimuth, pitch in cfg.ray_angles:
            # Transform azimuth by drone current yaw
            eff_azimuth = yaw + azimuth
            ux = np.cos(pitch) * np.cos(eff_azimuth)
            uy = np.cos(pitch) * np.sin(eff_azimuth)
            uz = np.sin(pitch)

            start = [pos[0] + offset * ux, pos[1] + offset * uy, pos[2] + offset * uz]
            end = [pos[0] + cfg.max_range_m * ux, pos[1] + cfg.max_range_m * uy, pos[2] + cfg.max_range_m * uz]
            ray_from.append(start)
            ray_to.append(end)

        results = p.rayTestBatch(ray_from, ray_to, physicsClientId=self.client_id)
        ranges = []
        for res in results:
            hit_obj_id = res[0]
            hit_fraction = res[2]

            # If no hit or hit the drone body itself, report maximum range
            if hit_obj_id == -1 or hit_obj_id == self.drone_id:
                dist = cfg.max_range_m
            else:
                dist = offset + hit_fraction * eff_span

            # Add simulated sensor noise
            dist += float(self._rng.normal(0, cfg.noise_std_m))
            dist = np.clip(dist, cfg.min_range_m, cfg.max_range_m)

            # Simulated dropout
            if self._rng.random() < cfg.dropout_probability:
                dist = cfg.max_range_m

            # Normalize to [0.0, 1.0]
            ranges.append(dist / cfg.max_range_m)

        return np.array(ranges, dtype=np.float32)

    def _get_observations(self) -> Tuple[DeployableObservation, PrivilegedState]:
        """Constructs both the deployable student observation and privileged teacher state."""
        pos = self._get_drone_position()
        vel = self._get_drone_velocity()
        quat, roll, pitch, yaw = self._get_drone_orientation()

        # Rangefinder readings
        ranges = self._cast_ray_sensors(pos, yaw)

        # Relative goal vector in body FLU frame: unit direction, scaled proportionally when within 2.0m of target
        world_rel_goal = self.corridor.goal_position - pos
        dist_goal = np.linalg.norm(world_rel_goal)
        scale = float(min(1.0, dist_goal / 2.0))
        if dist_goal > 1e-4:
            unit_world_goal = (world_rel_goal / dist_goal) * scale
        else:
            unit_world_goal = np.zeros(3, dtype=np.float32)

        # Rotate goal vector from World ENU to Body FLU (using yaw rotation)
        cos_y = np.cos(yaw)
        sin_y = np.sin(yaw)
        body_goal_x = cos_y * unit_world_goal[0] + sin_y * unit_world_goal[1]
        body_goal_y = -sin_y * unit_world_goal[0] + cos_y * unit_world_goal[1]
        body_goal_z = unit_world_goal[2]
        body_goal = np.array([body_goal_x, body_goal_y, body_goal_z], dtype=np.float32)

        # Body velocities
        body_vx = cos_y * vel[0] + sin_y * vel[1]
        body_vy = -sin_y * vel[0] + cos_y * vel[1]
        body_vz = vel[2]
        body_vel = np.array([body_vx, body_vy, body_vz], dtype=np.float32)

        # Trigonometric attitude
        att_trig = np.array([np.sin(roll), np.cos(roll), np.sin(pitch), np.cos(pitch)], dtype=np.float32)

        # Yaw rate (approx from pybullet angular velocity)
        _, ang_vel = p.getBaseVelocity(self.drone_id, physicsClientId=self.client_id)
        yaw_rate = float(ang_vel[2])

        deployable_obs = DeployableObservation(
            distance_rays=ranges,
            rel_goal_direction_body=body_goal,
            body_velocity=body_vel,
            attitude_trig=att_trig,
            yaw_rate=yaw_rate
        )

        b_dist = self.corridor.boundary_distance(pos)
        obs_clear = self.corridor.obstacle_clearance(pos, self.obstacles)

        privileged_state = PrivilegedState(
            global_position=pos,
            global_velocity=vel,
            global_orientation_quat=quat,
            corridor_boundary_distance=b_dist,
            obstacle_clearance=obs_clear,
            goal_distance=float(dist_goal),
            timestamp_s=self.current_sim_time
        )

        return deployable_obs, privileged_state

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        self.step_count += 1
        self.current_sim_time += self.dt

        # Unpack high-level normalized action
        vel_cmd = VelocityAction.from_normalized(action, self.action_limits)

        # Transform body velocity command to world velocity using current yaw
        _, _, _, yaw = self._get_drone_orientation()
        cos_y = np.cos(yaw)
        sin_y = np.sin(yaw)
        world_vx = cos_y * vel_cmd.vx - sin_y * vel_cmd.vy
        world_vy = sin_y * vel_cmd.vx + cos_y * vel_cmd.vy
        world_vz = vel_cmd.vz

        # Apply velocity control to drone rigid body
        p.resetBaseVelocity(
            self.drone_id,
            linearVelocity=[world_vx, world_vy, world_vz],
            angularVelocity=[0, 0, vel_cmd.yaw_rate],
            physicsClientId=self.client_id
        )

        p.stepSimulation(physicsClientId=self.client_id)

        curr_pos = self._get_drone_position()
        curr_vel = self._get_drone_velocity()

        # Compute observations
        deployable_obs, priv_state = self._get_observations()

        # Check termination conditions
        term_reason, reason_str = self.corridor.check_termination(
            curr_pos, curr_vel, self.obstacles, self.step_count, self.max_steps
        )

        terminated = (term_reason in (TerminationReason.SUCCESS, TerminationReason.CRASH_BOUNDARY, TerminationReason.CRASH_OBSTACLE))
        truncated = (term_reason == TerminationReason.TIMEOUT)

        # Calculate Reward
        progress = self.corridor.goal_progress(self.prev_position, curr_pos)
        reward = progress * 10.0 # Forward progress reward

        # Penalty for low clearance
        if priv_state.corridor_boundary_distance < 0.20:
            reward -= 0.5 * (0.20 - priv_state.corridor_boundary_distance)
        if priv_state.obstacle_clearance < 0.20:
            reward -= 1.0 * (0.20 - priv_state.obstacle_clearance)

        # Terminal rewards / penalties
        if term_reason == TerminationReason.SUCCESS:
            reward += 100.0
        elif term_reason in (TerminationReason.CRASH_BOUNDARY, TerminationReason.CRASH_OBSTACLE):
            reward -= 50.0

        self.prev_position = curr_pos

        info = {
            "privileged_state": priv_state,
            "termination_reason": term_reason.value,
            "termination_detail": reason_str,
            "goal_progress_m": progress,
            "boundary_distance": priv_state.corridor_boundary_distance,
            "obstacle_clearance": priv_state.obstacle_clearance,
        }

        return deployable_obs.to_vector(), float(reward), terminated, truncated, info

    def get_scene_spec(self) -> Dict[str, Any]:
        """Returns analytical corridor bounds, obstacle geometry, and sensor setup."""
        c = self.corridor
        return {
            "corridor": {
                "x_start": float(c.x_start),
                "x_end": float(c.x_end),
                "width": float(c.width),
                "height": float(c.height),
                "z_floor": float(c.z_floor),
                "z_ceiling": float(c.z_ceiling),
                "y_min": float(c.y_min),
                "y_max": float(c.y_max),
                "y_center": float(c.y_center),
                "goal_position": [float(x) for x in c.goal_position],
                "goal_radius": float(c.goal_radius),
                "drone_radius": float(c.drone_radius),
            },
            "obstacles": [
                {
                    "center": [float(x) for x in obs.center],
                    "radius": float(obs.radius),
                    "height": float(obs.height),
                    "obstacle_type": str(obs.obstacle_type),
                }
                for obs in self.obstacles
            ],
            "sensor_config": {
                "num_rays": int(self.sensor_config.num_rays),
                "max_range_m": float(self.sensor_config.max_range_m),
                "min_range_m": float(self.sensor_config.min_range_m),
                "ray_angles": [[float(a), float(p_ang)] for a, p_ang in self.sensor_config.ray_angles],
            }
        }

    def get_visual_state(self) -> Dict[str, Any]:
        """Returns synchronized real-time drone state, raycast endpoints, and spatial clearances."""
        pos = self._get_drone_position()
        vel = self._get_drone_velocity()
        quat, roll, pitch, yaw = self._get_drone_orientation()

        cfg = self.sensor_config
        ray_from = []
        ray_to = []
        offset = float(self.corridor.drone_radius + 0.03)
        eff_span = cfg.max_range_m - offset

        for azimuth, p_angle in cfg.ray_angles:
            eff_azimuth = yaw + azimuth
            ux = np.cos(p_angle) * np.cos(eff_azimuth)
            uy = np.cos(p_angle) * np.sin(eff_azimuth)
            uz = np.sin(p_angle)
            ray_from.append([float(pos[0] + offset * ux), float(pos[1] + offset * uy), float(pos[2] + offset * uz)])
            ray_to.append([float(pos[0] + cfg.max_range_m * ux), float(pos[1] + cfg.max_range_m * uy), float(pos[2] + cfg.max_range_m * uz)])

        results = p.rayTestBatch(ray_from, ray_to, physicsClientId=self.client_id)
        rays_data = []
        for i, res in enumerate(results):
            hit_obj_id = res[0]
            hit_frac = float(res[2])
            if hit_obj_id == -1 or hit_obj_id == self.drone_id:
                dist = cfg.max_range_m
                hit_pos = ray_to[i]
                hit_frac = 1.0
            else:
                dist = offset + hit_frac * eff_span
                hit_pos = [
                    float(ray_from[i][0] + hit_frac * (ray_to[i][0] - ray_from[i][0])),
                    float(ray_from[i][1] + hit_frac * (ray_to[i][1] - ray_from[i][1])),
                    float(ray_from[i][2] + hit_frac * (ray_to[i][2] - ray_from[i][2])),
                ]
            rays_data.append({
                "start": ray_from[i],
                "end": hit_pos,
                "hit_fraction": hit_frac,
                "distance_m": float(dist),
            })

        b_dist = self.corridor.boundary_distance(pos)
        obs_clear = self.corridor.obstacle_clearance(pos, self.obstacles)

        return {
            "position": [float(x) for x in pos],
            "velocity": [float(x) for x in vel],
            "orientation_quat": [float(x) for x in quat],
            "orientation_euler": [float(roll), float(pitch), float(yaw)],
            "speed_mps": float(np.linalg.norm(vel)),
            "sim_time_s": float(self.current_sim_time),
            "step_count": int(self.step_count),
            "boundary_distance": float(b_dist),
            "obstacle_clearance": float(obs_clear),
            "goal_distance": float(self.corridor.distance_to_goal(pos)),
            "rays": rays_data,
        }

    def close(self) -> None:
        if self.client_id >= 0:
            p.disconnect(self.client_id)
            self.client_id = -1
