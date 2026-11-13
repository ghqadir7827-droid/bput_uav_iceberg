"""physical_dimensions.py - Physically Realistic 65cm UAV Model & Scaled Environments.

Defines the physical characteristics, geometry, collision envelopes, and multi-level
environment specifications for a 65 cm (propeller-tip-to-propeller-tip) autonomous UAV.
Includes analytical route feasibility validation to distinguish impossible geometry
from controller navigation failures.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from core.geometry import Corridor3D, Obstacle3D


class DifficultyLevel(Enum):
    """Corridor navigation difficulty tiers."""
    LEVEL_1_BASIC = 1
    LEVEL_2_STANDARD = 2
    LEVEL_3_CHALLENGING = 3
    LEVEL_4_FEASIBILITY_BOUNDARY = 4


@dataclass(frozen=True)
class PhysicalDroneModel:
    """Exact physical dimensions and kinematic envelopes of the 65 cm quadcopter."""
    # Tip-to-tip diagonal span
    tip_to_tip_diagonal_m: float = 0.65
    # Bounding collision radius in horizontal XY plane
    collision_radius_m: float = 0.325  # 0.65 m / 2
    # Vertical dimension from bottom of landing gear to propeller plane
    body_height_m: float = 0.22
    # All-up mass with battery, avionics, sensors, and Raspberry Pi 5 (kg)
    mass_kg: float = 1.80
    # Motor-to-motor diagonal frame dimension (e.g. F450 frame)
    motor_diagonal_m: float = 0.46
    # Propeller diameter (9.5" propellers)
    propeller_diameter_m: float = 0.24
    # Sensor mount offset (origin just outside propeller disk)
    sensor_offset_m: float = 0.355  # collision_radius + 0.03m

    # Kinematic Braking Limits (at 0.60 m/s cruise speed, 1.0 m/s^2 max deceleration)
    # s = v^2 / (2 * a) = 0.36 / 2.0 = 0.18 m + 0.03 m latency margin = 0.21 m
    min_stopping_distance_m: float = 0.21

    # Safety Envelope (clearance beyond the physical collision radius)
    safety_envelope_clearance_m: float = 0.15
    # Center-to-obstacle critical distance that triggers emergency intervention
    critical_stop_center_dist_m: float = 0.475  # 0.325 + 0.15


@dataclass
class FeasibilityResult:
    """Analytical evaluation of route geometric traversability."""
    is_feasible: bool
    status: str
    min_bottleneck_gap_m: float
    drone_diameter_m: float
    critical_obstacle_idx: int
    clearance_margin_m: float
    diagnostic_details: str


class RouteFeasibilityChecker:
    """Evaluates whether an obstacle corridor can be physically traversed by a vehicle.
    
    A route is deemed INFEASIBLE if at any point along the corridor, the maximum available
    free passage width between an obstacle and walls (or between adjacent obstacles) is
    strictly smaller than the vehicle's physical diameter (2 * R_drone).
    """

    def __init__(self, drone_model: Optional[PhysicalDroneModel] = None):
        self.drone = drone_model or PhysicalDroneModel()

    def evaluate_feasibility(
        self,
        corridor: Corridor3D,
        obstacles: List[Obstacle3D],
        min_safety_margin_m: float = 0.05,
    ) -> FeasibilityResult:
        """Determines if the route has a continuous collision-free channel for the drone."""
        drone_diam = 2.0 * self.drone.collision_radius_m  # 0.65 m
        required_passage = drone_diam + min_safety_margin_m

        if not obstacles:
            w = float(corridor.width)
            return FeasibilityResult(
                is_feasible=w >= required_passage,
                status="FEASIBLE" if w >= required_passage else "INFEASIBLE_WALLS_TOO_NARROW",
                min_bottleneck_gap_m=w,
                drone_diameter_m=drone_diam,
                critical_obstacle_idx=-1,
                clearance_margin_m=w - drone_diam,
                diagnostic_details=f"No obstacles present. Corridor width = {w:.2f}m.",
            )

        # 1. Check vertical clearance
        vert_gap = float(corridor.height)
        if vert_gap < self.drone.body_height_m + min_safety_margin_m:
            return FeasibilityResult(
                is_feasible=False,
                status="BLOCKED_VERTICAL",
                min_bottleneck_gap_m=vert_gap,
                drone_diameter_m=drone_diam,
                critical_obstacle_idx=-1,
                clearance_margin_m=vert_gap - self.drone.body_height_m,
                diagnostic_details=f"Corridor height {vert_gap:.2f}m is smaller than drone height {self.drone.body_height_m:.2f}m.",
            )

        # 2. Group obstacles into longitudinal cross-sections (within +/- 0.8m of each other)
        # and compute the true free intervals in Y at each cross-section
        min_gap_found = float("inf")
        crit_idx = -1
        crit_status = "FEASIBLE"
        crit_details = []

        y_wall_min = float(corridor.y_min)
        y_wall_max = float(corridor.y_max)

        # Group obstacles by X coordinate clusters
        x_clusters: Dict[float, List[Tuple[int, Obstacle3D]]] = {}
        for idx, obs in enumerate(obstacles):
            ox = float(obs.center[0])
            matched_key = None
            for key in x_clusters.keys():
                if abs(ox - key) <= 0.80:
                    matched_key = key
                    break
            if matched_key is None:
                x_clusters[ox] = [(idx, obs)]
            else:
                x_clusters[matched_key].append((idx, obs))

        for cluster_x, cluster_obs in x_clusters.items():
            # Build list of blocked Y segments [y1, y2]
            blocked_segments: List[Tuple[float, float]] = []
            for _, obs in cluster_obs:
                oy = float(obs.center[1])
                orad = float(obs.radius)
                blocked_segments.append((max(y_wall_min, oy - orad), min(y_wall_max, oy + orad)))

            # Merge overlapping blocked segments
            blocked_segments.sort(key=lambda s: s[0])
            merged_blocked: List[Tuple[float, float]] = []
            for seg in blocked_segments:
                if not merged_blocked:
                    merged_blocked.append(seg)
                else:
                    last_start, last_end = merged_blocked[-1]
                    if seg[0] <= last_end:
                        merged_blocked[-1] = (last_start, max(last_end, seg[1]))
                    else:
                        merged_blocked.append(seg)

            # Compute free passage gaps between walls and merged blocked segments
            free_gaps: List[float] = []
            current_y = y_wall_min
            for b_start, b_end in merged_blocked:
                if b_start > current_y:
                    free_gaps.append(b_start - current_y)
                current_y = max(current_y, b_end)
            if current_y < y_wall_max:
                free_gaps.append(y_wall_max - current_y)

            max_slice_gap = max(free_gaps) if free_gaps else 0.0

            if max_slice_gap < min_gap_found:
                min_gap_found = max_slice_gap
                crit_idx = cluster_obs[0][0]

            if max_slice_gap < required_passage:
                crit_status = "BLOCKED_PASSAGE"
                obs_ids_str = ", ".join(str(idx) for idx, _ in cluster_obs)
                crit_details.append(
                    f"Slice at x~{cluster_x:.2f}m (obstacles [{obs_ids_str}]) has max passage {max_slice_gap:.3f}m "
                    f"< required {required_passage:.3f}m."
                )

        is_feasible = (crit_status == "FEASIBLE") and (min_gap_found >= required_passage)
        margin = min_gap_found - drone_diam

        diag = "; ".join(crit_details) if crit_details else (
            f"Passable: bottleneck gap is {min_gap_found:.3f}m (drone={drone_diam:.2f}m, margin={margin:.3f}m)."
        )

        return FeasibilityResult(
            is_feasible=is_feasible,
            status="FEASIBLE" if is_feasible else crit_status,
            min_bottleneck_gap_m=min_gap_found,
            drone_diameter_m=drone_diam,
            critical_obstacle_idx=crit_idx,
            clearance_margin_m=margin,
            diagnostic_details=diag,
        )


class ScaledEnvironmentFactory:
    """Generates standardized, physically consistent environments scaled for the 65 cm UAV."""

    DRONE_MODEL = PhysicalDroneModel()

    @classmethod
    def get_environment_config(
        cls,
        level: Union[int, DifficultyLevel],
        corridor_length_m: float = 14.0,
    ) -> Corridor3D:
        """Returns a standardized Corridor3D scaled for the 65 cm UAV for the given level."""
        if isinstance(level, int):
            level = DifficultyLevel(level)

        drone_r = cls.DRONE_MODEL.collision_radius_m  # 0.325 m

        if level == DifficultyLevel.LEVEL_1_BASIC:
            # Level 1: Generous clearance (4.50 m width, 2.80 m height)
            # Pillars well-spaced with small lateral perturbation, generous passages (>1.5 m)
            return Corridor3D(
                name="level_1_basic_scaled_65cm",
                x_start=0.0,
                x_end=corridor_length_m,
                width=4.50,
                height=2.80,
                z_floor=0.0,
                z_ceiling=2.80,
                y_center=0.0,
                start_position=np.array([0.8, 0.0, 1.40], dtype=np.float32),
                goal_position=np.array([corridor_length_m - 0.8, 0.0, 1.40], dtype=np.float32),
                goal_radius=0.55,
                drone_radius=drone_r,
                pillar_x_positions=[3.5, 7.0, 10.5],
                pillar_y_range=0.35,
                pillar_radius=0.25,
                max_steps=1000,
            )

        elif level == DifficultyLevel.LEVEL_2_STANDARD:
            # Level 2: Standard realistic corridor (3.80 m width, 2.50 m height)
            # Balanced challenge with 0.85m - 1.20m free clearance channels
            return Corridor3D(
                name="level_2_standard_scaled_65cm",
                x_start=0.0,
                x_end=corridor_length_m,
                width=3.80,
                height=2.50,
                z_floor=0.0,
                z_ceiling=2.50,
                y_center=0.0,
                start_position=np.array([0.8, 0.0, 1.25], dtype=np.float32),
                goal_position=np.array([corridor_length_m - 0.8, 0.0, 1.25], dtype=np.float32),
                goal_radius=0.50,
                drone_radius=drone_r,
                pillar_x_positions=[3.5, 7.0, 10.5],
                pillar_y_range=0.60,
                pillar_radius=0.25,
                max_steps=1000,
            )

        elif level == DifficultyLevel.LEVEL_3_CHALLENGING:
            # Level 3: Challenging tight passages (3.20 m width, 2.40 m height)
            # Large lateral offsets (+/-0.75m), tight turning slaloms, clearance down to 0.45m
            return Corridor3D(
                name="level_3_challenging_scaled_65cm",
                x_start=0.0,
                x_end=corridor_length_m,
                width=3.20,
                height=2.40,
                z_floor=0.0,
                z_ceiling=2.40,
                y_center=0.0,
                start_position=np.array([0.8, 0.0, 1.20], dtype=np.float32),
                goal_position=np.array([corridor_length_m - 0.8, 0.0, 1.20], dtype=np.float32),
                goal_radius=0.50,
                drone_radius=drone_r,
                pillar_x_positions=[3.5, 7.0, 10.5],
                pillar_y_range=0.75,
                pillar_radius=0.25,
                max_steps=1000,
            )

        elif level == DifficultyLevel.LEVEL_4_FEASIBILITY_BOUNDARY:
            # Level 4: Feasibility boundary tests (2.80 m width, 2.40 m height)
            # Wide pillar offsets up to +/-0.85m where some seeds are intentionally near or below
            # the 0.65m passage limit to test safe stopping and feasibility rejection
            return Corridor3D(
                name="level_4_boundary_scaled_65cm",
                x_start=0.0,
                x_end=corridor_length_m,
                width=2.80,
                height=2.40,
                z_floor=0.0,
                z_ceiling=2.40,
                y_center=0.0,
                start_position=np.array([0.8, 0.0, 1.20], dtype=np.float32),
                goal_position=np.array([corridor_length_m - 0.8, 0.0, 1.20], dtype=np.float32),
                goal_radius=0.50,
                drone_radius=drone_r,
                pillar_x_positions=[3.5, 7.0, 10.5],
                pillar_y_range=0.88,
                pillar_radius=0.25,
                max_steps=1000,
            )

        raise ValueError(f"Unknown difficulty level: {level}")
