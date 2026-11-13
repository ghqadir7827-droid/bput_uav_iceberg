"""geometry.py - Explicit 3D Corridor Spatial Representation & Measurable Metrics.

Defines the mathematical geometry of the 3D flight tunnel, analytical boundary
clearance calculations, goal progress tracking, obstacle clearance, and safe
episode termination conditions.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np


class TerminationReason(Enum):
    """Episode termination status."""
    CONTINUE = "continue"
    SUCCESS = "success"
    CRASH_OBSTACLE = "crash_obstacle"
    CRASH_BOUNDARY = "crash_boundary"
    TIMEOUT = "timeout"


class EpisodeOutcome(Enum):
    """Authoritative 8-category terminal outcome taxonomy for evaluation (Phase 4F Priority C)."""
    CLEAN_SUCCESS = "clean_success"
    ENVELOPE_BREACH_ARRIVAL = "envelope_breach_arrival"
    COLLISION = "collision"
    TIMEOUT = "timeout"
    SAFE_REJECTION = "safe_rejection"
    UNRESOLVED_OBSTACLE = "unresolved_obstacle"
    UNSAFE_OR_UNSTABLE_ARREST = "unsafe_or_unstable_arrest"
    OTHER_FAILURE = "other_failure"


@dataclass(frozen=True)
class OutcomeClassificationResult:
    """Rigorous classification result with ground-truth supporting evidence."""
    outcome: EpisodeOutcome
    reason: str
    min_obstacle_clearance_m: float
    min_boundary_clearance_m: float
    envelope_breach_detected: bool
    is_clean: bool
    is_safe_rejection: bool
    is_collision: bool
    is_timeout: bool


class OutcomeClassifier:
    """Authoritative evaluator classifying terminal episodes into mutually exclusive categories."""

    SAFETY_ENVELOPE_THRESHOLD_M: float = 0.150

    @classmethod
    def classify(
        cls,
        termination_reason: Union[TerminationReason, str],
        min_obstacle_clearance_m: float,
        min_boundary_clearance_m: float,
        safe_arrest_state: Optional[str] = None,
        is_arrest_stable: bool = False,
        other_failure_reason: Optional[str] = None,
    ) -> OutcomeClassificationResult:
        """Assigns exactly one authoritative terminal classification per episode."""
        term_str = termination_reason.value if isinstance(termination_reason, TerminationReason) else str(termination_reason).lower()
        min_obs = float(min_obstacle_clearance_m)
        min_bnd = float(min_boundary_clearance_m)
        overall_min = min(min_obs, min_bnd)

        # 1. Authoritative Collision Check (collision cannot be overridden by later arrival)
        if term_str in ("crash_obstacle", "crash_boundary", "collision") or overall_min <= 0.0:
            return OutcomeClassificationResult(
                outcome=EpisodeOutcome.COLLISION,
                reason=f"Authoritative collision detected (min_obs={min_obs:.4f}m, min_bnd={min_bnd:.4f}m)",
                min_obstacle_clearance_m=min_obs,
                min_boundary_clearance_m=min_bnd,
                envelope_breach_detected=True,
                is_clean=False,
                is_safe_rejection=False,
                is_collision=True,
                is_timeout=False,
            )

        # 2. Safe Arrest / Rejection Check
        if safe_arrest_state in ("SAFE_ARREST", "VERIFIED_INFEASIBLE"):
            if is_arrest_stable and overall_min > 0.05:
                return OutcomeClassificationResult(
                    outcome=EpisodeOutcome.SAFE_REJECTION,
                    reason=f"Safely arrested in front of verified infeasible passage (dwell verified, min_c={overall_min:.4f}m)",
                    min_obstacle_clearance_m=min_obs,
                    min_boundary_clearance_m=min_bnd,
                    envelope_breach_detected=(min_obs < cls.SAFETY_ENVELOPE_THRESHOLD_M),
                    is_clean=False,
                    is_safe_rejection=True,
                    is_collision=False,
                    is_timeout=False,
                )
            else:
                return OutcomeClassificationResult(
                    outcome=EpisodeOutcome.UNSAFE_OR_UNSTABLE_ARREST,
                    reason=f"Safe arrest failed stability or clearance criteria (stable={is_arrest_stable}, min_c={overall_min:.4f}m)",
                    min_obstacle_clearance_m=min_obs,
                    min_boundary_clearance_m=min_bnd,
                    envelope_breach_detected=(min_obs < cls.SAFETY_ENVELOPE_THRESHOLD_M),
                    is_clean=False,
                    is_safe_rejection=False,
                    is_collision=False,
                    is_timeout=False,
                )

        if safe_arrest_state == "UNRESOLVED":
            if is_arrest_stable and overall_min > 0.05:
                return OutcomeClassificationResult(
                    outcome=EpisodeOutcome.UNRESOLVED_OBSTACLE,
                    reason=f"Route unverified/ambiguous, drone safely halted (min_c={overall_min:.4f}m)",
                    min_obstacle_clearance_m=min_obs,
                    min_boundary_clearance_m=min_bnd,
                    envelope_breach_detected=(min_obs < cls.SAFETY_ENVELOPE_THRESHOLD_M),
                    is_clean=False,
                    is_safe_rejection=False,
                    is_collision=False,
                    is_timeout=False,
                )
            else:
                return OutcomeClassificationResult(
                    outcome=EpisodeOutcome.UNSAFE_OR_UNSTABLE_ARREST,
                    reason="Unresolved route with unstable stopping behavior",
                    min_obstacle_clearance_m=min_obs,
                    min_boundary_clearance_m=min_bnd,
                    envelope_breach_detected=True,
                    is_clean=False,
                    is_safe_rejection=False,
                    is_collision=False,
                    is_timeout=False,
                )

        # 3. Goal Arrival Check
        if term_str == "success":
            # Must satisfy minimum obstacle safety envelope (0.150 m)
            if min_obs >= cls.SAFETY_ENVELOPE_THRESHOLD_M:
                return OutcomeClassificationResult(
                    outcome=EpisodeOutcome.CLEAN_SUCCESS,
                    reason=f"Clean arrival at goal with valid safety envelope (min_obs={min_obs:.4f}m >= 0.15m)",
                    min_obstacle_clearance_m=min_obs,
                    min_boundary_clearance_m=min_bnd,
                    envelope_breach_detected=False,
                    is_clean=True,
                    is_safe_rejection=False,
                    is_collision=False,
                    is_timeout=False,
                )
            else:
                # Arrived but violated safety envelope (e.g., Seed 20334 with 0.0892m)
                return OutcomeClassificationResult(
                    outcome=EpisodeOutcome.ENVELOPE_BREACH_ARRIVAL,
                    reason=f"Goal arrived but breached 0.15m obstacle envelope (min_obs={min_obs:.4f}m < 0.15m)",
                    min_obstacle_clearance_m=min_obs,
                    min_boundary_clearance_m=min_bnd,
                    envelope_breach_detected=True,
                    is_clean=False,
                    is_safe_rejection=False,
                    is_collision=False,
                    is_timeout=False,
                )

        # 4. Timeout Check
        if term_str == "timeout":
            return OutcomeClassificationResult(
                outcome=EpisodeOutcome.TIMEOUT,
                reason="Maximum simulation steps reached without valid completion or rejection",
                min_obstacle_clearance_m=min_obs,
                min_boundary_clearance_m=min_bnd,
                envelope_breach_detected=(min_obs < cls.SAFETY_ENVELOPE_THRESHOLD_M),
                is_clean=False,
                is_safe_rejection=False,
                is_collision=False,
                is_timeout=True,
            )

        # 5. Other Failure
        return OutcomeClassificationResult(
            outcome=EpisodeOutcome.OTHER_FAILURE,
            reason=other_failure_reason or f"Unclassified termination reason: {term_str}",
            min_obstacle_clearance_m=min_obs,
            min_boundary_clearance_m=min_bnd,
            envelope_breach_detected=(min_obs < cls.SAFETY_ENVELOPE_THRESHOLD_M),
            is_clean=False,
            is_safe_rejection=False,
            is_collision=False,
            is_timeout=False,
        )


@dataclass
class Obstacle3D:
    """Analytical representation of a 3D obstacle inside the corridor."""
    center: np.ndarray       # [x, y, z] center (m)
    radius: float            # Collision radius (m) (for cylindrical pillar or sphere)
    height: float = 2.0      # Height (m)
    obstacle_type: str = "cylinder" # "cylinder" (vertical pillar) or "sphere"

    def distance_to_point(self, point: np.ndarray) -> float:
        """Returns Euclidean clearance from point to obstacle surface."""
        p = np.asarray(point, dtype=np.float64)
        c = np.asarray(self.center, dtype=np.float64)
        if self.obstacle_type == "cylinder":
            # Distance in XY plane
            xy_dist = np.linalg.norm(p[:2] - c[:2]) - self.radius
            # Check Z bounds
            z_min = c[2] - self.height / 2.0
            z_max = c[2] + self.height / 2.0
            if z_min <= p[2] <= z_max:
                return float(xy_dist)
            elif p[2] < z_min:
                return float(np.sqrt(xy_dist**2 + (z_min - p[2])**2))
            else:
                return float(np.sqrt(xy_dist**2 + (p[2] - z_max)**2))
        else:
            return float(np.linalg.norm(p - c) - self.radius)


@dataclass
class Corridor3D:
    """3D rectangular corridor segment bounded by planar walls, floor, and ceiling."""
    name: str = "initial_benchmark_10m"
    x_start: float = 0.0          # Starting X coordinate (m)
    x_end: float = 10.0           # Exit X coordinate (m)
    width: float = 2.0            # Full corridor width (Y axis, from -width/2 to +width/2)
    height: float = 2.0           # Full corridor height (Z axis)
    z_floor: float = 0.0          # Floor level (m)
    z_ceiling: float = 2.0        # Ceiling level (m)
    y_center: float = 0.0         # Centerline Y position (m)
    start_position: Optional[np.ndarray] = None # Drone start [x, y, z]
    goal_position: Optional[np.ndarray] = None  # Target position [x_goal, y_goal, z_goal]
    goal_radius: float = 0.35     # Radius of acceptance sphere at destination (m)
    drone_radius: float = 0.15    # Drone bounding radius (m)
    pillar_x_positions: List[float] = field(default_factory=lambda: [3.0, 5.0, 7.0])
    pillar_y_range: float = 0.35  # Maximum lateral offset (+/-) for randomized pillars
    pillar_radius: float = 0.20   # Pillar cylinder radius
    max_steps: int = 500          # Maximum episode steps (dt=0.05s -> 25s)

    def __post_init__(self):
        if self.start_position is None:
            self.start_position = np.array([self.x_start + 0.5, self.y_center, (self.z_floor + self.z_ceiling) / 2.0], dtype=np.float32)
        else:
            self.start_position = np.asarray(self.start_position, dtype=np.float32)
        if self.goal_position is None:
            self.goal_position = np.array([self.x_end - 0.5, self.y_center, (self.z_floor + self.z_ceiling) / 2.0], dtype=np.float32)
        else:
            self.goal_position = np.asarray(self.goal_position, dtype=np.float32)

    @classmethod
    def get_10m_benchmark(cls) -> "Corridor3D":
        """Agreed initial benchmark: 10 m x 2 m x 2 m, start (0.5, 0, 1.0), goal (9.5, 0, 1.0)."""
        return cls(
            name="initial_benchmark_10m",
            x_start=0.0,
            x_end=10.0,
            width=2.0,
            height=2.0,
            z_floor=0.0,
            z_ceiling=2.0,
            y_center=0.0,
            start_position=np.array([0.5, 0.0, 1.0], dtype=np.float32),
            goal_position=np.array([9.5, 0.0, 1.0], dtype=np.float32),
            goal_radius=0.35,
            drone_radius=0.15,
            pillar_x_positions=[3.0, 5.0, 7.0],
            pillar_y_range=0.35,
            pillar_radius=0.20,
            max_steps=500,
        )

    @classmethod
    def get_20m_extended(cls) -> "Corridor3D":
        """Optional extended scenario: 20 m x 2.4 m x 2 m, start (1.0, 0, 1.0), goal (19.0, 0, 1.0)."""
        return cls(
            name="extended_scenario_20m",
            x_start=0.0,
            x_end=20.0,
            width=2.4,
            height=2.0,
            z_floor=0.0,
            z_ceiling=2.0,
            y_center=0.0,
            start_position=np.array([1.0, 0.0, 1.0], dtype=np.float32),
            goal_position=np.array([19.0, 0.0, 1.0], dtype=np.float32),
            goal_radius=0.40,
            drone_radius=0.15,
            pillar_x_positions=[4.0, 8.0, 12.0, 16.0],
            pillar_y_range=0.50,
            pillar_radius=0.20,
            max_steps=800,
        )

    @classmethod
    def from_scenario(cls, scenario: Union[str, "Corridor3D"]) -> "Corridor3D":
        if isinstance(scenario, Corridor3D):
            return scenario
        s = str(scenario).lower().strip()
        if "level_1" in s or "basic" in s:
            from core.physical_dimensions import ScaledEnvironmentFactory, DifficultyLevel
            return ScaledEnvironmentFactory.get_environment_config(DifficultyLevel.LEVEL_1_BASIC)
        if "level_2" in s or "standard_65" in s or "scaled" in s:
            from core.physical_dimensions import ScaledEnvironmentFactory, DifficultyLevel
            return ScaledEnvironmentFactory.get_environment_config(DifficultyLevel.LEVEL_2_STANDARD)
        if "level_3" in s or "challenging" in s:
            from core.physical_dimensions import ScaledEnvironmentFactory, DifficultyLevel
            return ScaledEnvironmentFactory.get_environment_config(DifficultyLevel.LEVEL_3_CHALLENGING)
        if "level_4" in s or "boundary" in s:
            from core.physical_dimensions import ScaledEnvironmentFactory, DifficultyLevel
            return ScaledEnvironmentFactory.get_environment_config(DifficultyLevel.LEVEL_4_FEASIBILITY_BOUNDARY)
        if "20" in s or "ext" in s:
            return cls.get_20m_extended()
        return cls.get_10m_benchmark()

    @property
    def y_min(self) -> float:
        """Right wall coordinate (negative Y)."""
        return self.y_center - (self.width / 2.0)

    @property
    def y_max(self) -> float:
        """Left wall coordinate (positive Y)."""
        return self.y_center + (self.width / 2.0)

    def boundary_distance(self, position: np.ndarray) -> float:
        """Computes analytical minimum clearance from drone center to nearest corridor boundary.
        
        Returns signed clearance (positive = inside, negative = penetrated boundary).
        """
        p = np.asarray(position, dtype=np.float64)
        d_left = self.y_max - p[1]
        d_right = p[1] - self.y_min
        d_ceil = self.z_ceiling - p[2]
        d_floor = p[2] - self.z_floor
        return float(min(d_left, d_right, d_ceil, d_floor) - self.drone_radius)

    def boundary_clearances(self, position: np.ndarray) -> Tuple[float, float, float, float]:
        """Returns individual clearances to [left_wall, right_wall, ceiling, floor]."""
        p = np.asarray(position, dtype=np.float64)
        d_left = float(self.y_max - p[1] - self.drone_radius)
        d_right = float(p[1] - self.y_min - self.drone_radius)
        d_ceil = float(self.z_ceiling - p[2] - self.drone_radius)
        d_floor = float(p[2] - self.z_floor - self.drone_radius)
        return (d_left, d_right, d_ceil, d_floor)

    def goal_progress(self, pos_prev: np.ndarray, pos_curr: np.ndarray) -> float:
        """Calculates forward progress projected along the corridor X axis (m)."""
        return float(pos_curr[0] - pos_prev[0])

    def distance_to_goal(self, position: np.ndarray) -> float:
        """Euclidean distance from current position to goal center."""
        return float(np.linalg.norm(np.asarray(position) - self.goal_position))

    def obstacle_clearance(self, position: np.ndarray, obstacles: List[Obstacle3D]) -> float:
        """Returns distance to nearest obstacle minus drone radius."""
        if not obstacles:
            return 10.0  # Large clearance when no obstacles
        min_dist = min(obs.distance_to_point(position) for obs in obstacles)
        return float(min_dist - self.drone_radius)

    def localization_uncertainty(self, covariance: np.ndarray) -> float:
        """Calculates scalar position uncertainty index sqrt(tr(Cov))."""
        if covariance is None or covariance.size == 0:
            return 0.0
        cov = np.asarray(covariance)
        # Position diagonal variance
        pos_var = cov[0, 0] + cov[1, 1] + cov[2, 2]
        return float(np.sqrt(max(0.0, pos_var)))

    def check_termination(
        self,
        position: np.ndarray,
        velocity: np.ndarray,
        obstacles: List[Obstacle3D],
        step_count: int,
        max_steps: int = 500,
        max_landing_speed: float = 0.35,
    ) -> Tuple[TerminationReason, str]:
        """Evaluates safe termination conditions rigorously."""
        pos = np.asarray(position)
        vel = np.asarray(velocity)

        # 1. Boundary crash check
        d_b = self.boundary_distance(pos)
        if d_b <= 0.0:
            return TerminationReason.CRASH_BOUNDARY, f"Boundary penetrated (clearance={d_b:.3f}m)"

        # 2. Obstacle collision check
        d_obs = self.obstacle_clearance(pos, obstacles)
        if d_obs <= 0.0:
            return TerminationReason.CRASH_OBSTACLE, f"Obstacle collision (clearance={d_obs:.3f}m)"

        # 3. Destination arrival check
        d_goal = self.distance_to_goal(pos)
        if d_goal <= self.goal_radius:
            speed = np.linalg.norm(vel)
            if speed <= max_landing_speed:
                return TerminationReason.SUCCESS, f"Arrived at goal (d={d_goal:.3f}m, speed={speed:.3f}m/s)"
            else:
                # Arrived but moving too fast
                pass

        # 4. Timeout check
        if step_count >= max_steps:
            return TerminationReason.TIMEOUT, f"Max steps ({max_steps}) exceeded"

        return TerminationReason.CONTINUE, "In flight"
