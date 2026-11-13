"""experimental_supervisor.py - Isolated Proactive Lateral Guidance Supervisor.

Phase 4J-F experimental supervisor designed to resolve centered-obstacle deadlocks
via deployable observation-driven lateral bypass guidance without modifying the canonical
supervisor or weakening invariant safety barriers.
"""

from collections import deque
from enum import Enum
import json
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple, Union
import numpy as np

from controllers.safety_supervisor import SafetyState, SafetySupervisor, SafetySupervisorConfig
from core.contracts import (
    ActionLimits,
    DeployableObservation,
    VelocityAction,
    compute_directional_clearances_m,
    compute_forward_clearance_m,
)
from core.geometry import Corridor3D


class GuidanceState(Enum):
    """Internal state of the proactive lateral guidance module."""
    IDLE = "idle"
    MONITORING = "monitoring"
    DEADLOCK_DETECTED = "deadlock_detected"
    GUIDANCE_ACTIVE_LEFT = "guidance_active_left"
    GUIDANCE_ACTIVE_RIGHT = "guidance_active_right"
    FALLBACK_BLOCKED = "fallback_blocked"


class IsolatedLateralGuidanceSupervisor:
    """Isolated experimental supervisor with deadlock-breaking lateral guidance.

    Operates in two strictly isolated modes:
    1. 'baseline': Identical to Phase 4J-E optimized supervisor (d_caut = 0.50m).
    2. 'lateral_guidance': Proactive lateral bypass direction selection upon detected stagnation.
    """

    def __init__(
        self,
        corridor: Corridor3D,
        mode: str = "lateral_guidance",
        caution_distance_m: float = 0.50,
        limits: ActionLimits = ActionLimits(),
        config_path: Optional[Union[str, Path]] = None,
    ):
        if mode not in ("baseline", "lateral_guidance"):
            raise ValueError(f"Invalid mode '{mode}': must be 'baseline' or 'lateral_guidance'")

        self.mode = mode
        self.corridor = corridor
        self.limits = limits
        self.caution_distance_m = float(caution_distance_m)

        # Invariant safety configuration
        self.safety_config = SafetySupervisorConfig(
            boundary_buffer_m=0.25,
            critical_obstacle_dist_m=0.30,
            emergency_reverse_dist_m=0.15,
            obstacle_caution_dist_m=self.caution_distance_m,
            critical_lateral_dist_m=0.32,
            caution_lateral_dist_m=0.55,
        )

        self._supervisor = SafetySupervisor(
            corridor=corridor,
            limits=limits,
            config=self.safety_config,
        )

        # Load or set default guidance hyperparameters
        if config_path is not None and Path(config_path).is_file():
            cfg_data = json.loads(Path(config_path).read_text())
            dd = cfg_data["deadlock_detection_parameters"]
            lg = cfg_data["lateral_guidance_parameters"]
            self.window_steps = int(dd["detection_window_steps"])
            self.max_delta_x_m = float(dd["max_forward_displacement_m"])
            self.max_mean_abs_vy_mps = float(dd["max_mean_abs_lateral_velocity_mps"])
            self.trigger_front_dist_m = float(dd["front_obstacle_trigger_dist_m"])
            self.lat_bias_speed_mps = float(lg["lateral_bias_speed_mps"])
            self.fwd_crawl_speed_mps = float(lg["forward_crawl_speed_mps"])
            self.commit_duration_steps = int(lg["commitment_duration_steps"])
            self.diff_margin_m = float(lg["opening_difference_margin_m"])
            self.min_wall_headroom_m = float(lg["min_wall_headroom_m"])
            self.wall_abort_threshold_m = float(lg["wall_abort_threshold_m"])
            self.obstacle_cleared_dist_m = float(lg["obstacle_cleared_threshold_m"])
        else:
            self.window_steps = 30
            self.max_delta_x_m = 0.06
            self.max_mean_abs_vy_mps = 0.08
            self.trigger_front_dist_m = 0.45
            self.lat_bias_speed_mps = 0.25
            self.fwd_crawl_speed_mps = 0.18
            self.commit_duration_steps = 18
            self.diff_margin_m = 0.05
            self.min_wall_headroom_m = 0.28
            self.wall_abort_threshold_m = 0.26
            self.obstacle_cleared_dist_m = 0.60

        # State tracking structures
        self.guidance_state: GuidanceState = GuidanceState.IDLE
        self.x_history: Deque[float] = deque(maxlen=self.window_steps + 1)
        self.vy_history: Deque[float] = deque(maxlen=self.window_steps + 1)
        self.commit_dir: int = 0  # +1 = Left, -1 = Right, 0 = None
        self.commit_remaining_steps: int = 0
        self.last_guidance_reason: str = ""

        # Counters for diagnostics
        self.total_deadlock_detections: int = 0
        self.total_commitments: int = 0
        self.total_direction_changes: int = 0
        self.total_fallback_events: int = 0
        self.prev_committed_dir: int = 0
        self.state_step_counts: Dict[str, int] = {s.value: 0 for s in GuidanceState}

    @property
    def loss_timer_s(self) -> float:
        return self._supervisor.loss_timer_s

    @loss_timer_s.setter
    def loss_timer_s(self, value: float):
        self._supervisor.loss_timer_s = value

    @property
    def stalled_timer_s(self) -> float:
        return self._supervisor.stalled_timer_s

    @stalled_timer_s.setter
    def stalled_timer_s(self, value: float):
        self._supervisor.stalled_timer_s = value

    @property
    def last_safe_action(self) -> VelocityAction:
        return self._supervisor.last_safe_action

    @last_safe_action.setter
    def last_safe_action(self, value: VelocityAction):
        self._supervisor.last_safe_action = value

    def reset_state(self):
        """Full reset of supervisor timers, rolling histories, and guidance commitments."""
        self._supervisor.loss_timer_s = 0.0
        self._supervisor.stalled_timer_s = 0.0
        self._supervisor.last_safe_action = VelocityAction(0.0, 0.0, 0.0, 0.0)

        self.guidance_state = GuidanceState.IDLE
        self.x_history.clear()
        self.vy_history.clear()
        self.commit_dir = 0
        self.commit_remaining_steps = 0
        self.last_guidance_reason = ""
        self.total_deadlock_detections = 0
        self.total_commitments = 0
        self.total_direction_changes = 0
        self.total_fallback_events = 0
        self.prev_committed_dir = 0
        self.state_step_counts = {s.value: 0 for s in GuidanceState}

    def detect_blocked_progress(self, front_dist: float, current_pos: np.ndarray, current_vel: np.ndarray) -> bool:
        """Evaluates whether UAV forward progress is blocked by a centered obstacle over rolling window."""
        self.x_history.append(float(current_pos[0]))
        self.vy_history.append(float(current_vel[1]))

        if len(self.x_history) < self.window_steps:
            return False

        delta_x = self.x_history[-1] - self.x_history[0]
        mean_abs_vy = float(np.mean(np.abs(list(self.vy_history))))

        # Criteria: in front caution zone, minimal net progress over 1.5s window, near-zero lateral steering
        is_blocked = (
            front_dist < self.trigger_front_dist_m
            and delta_x < self.max_delta_x_m
            and mean_abs_vy < self.max_mean_abs_vy_mps
        )
        return is_blocked

    def evaluate_lateral_openings(
        self,
        observation: np.ndarray,
        current_pos: np.ndarray,
    ) -> Tuple[int, str]:
        """Evaluates deployable rangefinder rays to select the wider, feasible bypass direction."""
        # Rangefinder rays (normalized in [0:12], multiply by 4.0m)
        rays = np.asarray(observation[:12], dtype=np.float32) * 4.0

        r_fl = float(rays[1])  # Front-Left (+30 deg)
        r_fr = float(rays[2])  # Front-Right (-30 deg)
        r_sl = float(rays[3])  # Side-Left (+60 deg)
        r_sr = float(rays[4])  # Side-Right (-60 deg)

        # Boundary clearances from current estimated position
        d_l_wall = float(1.0 - current_pos[1] - self.corridor.drone_radius)
        d_r_wall = float(current_pos[1] - (-1.0) - self.corridor.drone_radius)

        # Projected opening width accounting for forward diagonal clearance and wall buffer
        open_left = min(r_fl * np.cos(np.deg2rad(30)), d_l_wall)
        open_right = min(r_fr * np.cos(np.deg2rad(30)), d_r_wall)

        # Feasibility requires adequate wall clearance headroom (> 0.28m)
        left_feasible = (d_l_wall >= self.min_wall_headroom_m and open_left >= 0.22)
        right_feasible = (d_r_wall >= self.min_wall_headroom_m and open_right >= 0.22)

        if left_feasible and right_feasible:
            if open_left > open_right + self.diff_margin_m:
                return +1, f"Left opening superior ({open_left:.2f}m > {open_right:.2f}m)"
            elif open_right > open_left + self.diff_margin_m:
                return -1, f"Right opening superior ({open_right:.2f}m > {open_left:.2f}m)"
            else:
                # Deterministic tie-breaker: steer toward corridor center or positive Y if on centerline
                tb_dir = +1 if current_pos[1] <= 0.0 else -1
                side_str = "Left" if tb_dir == +1 else "Right"
                return tb_dir, f"Symmetric encounter -> Deterministic tie-breaker: {side_str}"
        elif left_feasible:
            return +1, f"Left opening feasible ({open_left:.2f}m), right constrained"
        elif right_feasible:
            return -1, f"Right opening feasible ({open_right:.2f}m), left constrained"
        else:
            return 0, "Both lateral sides constrained -> Fallback standstill"

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
        """Filters action with optional proactive lateral guidance and invariant safety barriers."""
        obs_vec = observation.to_vector() if isinstance(observation, DeployableObservation) else observation

        # Active Guidance Pre-Arbitration (only in lateral_guidance mode)
        adjusted_action = VelocityAction(
            vx=proposed_action.vx,
            vy=proposed_action.vy,
            vz=proposed_action.vz,
            yaw_rate=proposed_action.yaw_rate,
        )

        guidance_active = False
        guidance_tag = ""

        if self.mode == "lateral_guidance" and obs_vec is not None:
            # Wall distances
            d_l_wall = 1.0 - current_position[1] - self.corridor.drone_radius
            d_r_wall = current_position[1] - (-1.0) - self.corridor.drone_radius

            # 1. Check if progress is blocked
            is_blocked = self.detect_blocked_progress(min_front_range, current_position, current_velocity)
            if is_blocked and self.commit_remaining_steps <= 0:
                self.total_deadlock_detections += 1
                chosen_dir, reason = self.evaluate_lateral_openings(obs_vec, current_position)
                if chosen_dir != 0:
                    self.commit_dir = chosen_dir
                    self.commit_remaining_steps = self.commit_duration_steps
                    self.total_commitments += 1
                    if self.prev_committed_dir != 0 and self.commit_dir != self.prev_committed_dir:
                        self.total_direction_changes += 1
                    self.prev_committed_dir = self.commit_dir
                    self.last_guidance_reason = reason
                    self.guidance_state = (
                        GuidanceState.GUIDANCE_ACTIVE_LEFT if chosen_dir == +1 else GuidanceState.GUIDANCE_ACTIVE_RIGHT
                    )
                else:
                    self.guidance_state = GuidanceState.FALLBACK_BLOCKED
                    self.total_fallback_events += 1
                    self.last_guidance_reason = reason

            # 2. Maintain active commitment or check abort conditions
            if self.commit_remaining_steps > 0:
                self.commit_remaining_steps -= 1

                # Abort conditions: obstacle cleared or chosen wall constraint violated
                if min_front_range >= self.obstacle_cleared_dist_m:
                    self.commit_dir = 0
                    self.commit_remaining_steps = 0
                    self.guidance_state = GuidanceState.IDLE
                    guidance_tag = "[Guidance Released: Obstacle Cleared]"
                elif self.commit_dir == +1 and d_l_wall <= self.wall_abort_threshold_m:
                    self.commit_dir = 0
                    self.commit_remaining_steps = 0
                    self.guidance_state = GuidanceState.IDLE
                    guidance_tag = "[Guidance Aborted: Left Wall Limit]"
                elif self.commit_dir == -1 and d_r_wall <= self.wall_abort_threshold_m:
                    self.commit_dir = 0
                    self.commit_remaining_steps = 0
                    self.guidance_state = GuidanceState.IDLE
                    guidance_tag = "[Guidance Aborted: Right Wall Limit]"
                else:
                    # Inject proactive lateral bypass velocity
                    guidance_active = True
                    adjusted_action.vy = float(self.commit_dir * self.lat_bias_speed_mps)
                    # Maintain gentle forward crawl if safe
                    if adjusted_action.vx < self.fwd_crawl_speed_mps and min_front_range > self.safety_config.critical_obstacle_dist_m:
                        adjusted_action.vx = float(self.fwd_crawl_speed_mps)
                    dir_str = "Left" if self.commit_dir == +1 else "Right"
                    guidance_tag = f"[Lateral Guidance Active: Steering {dir_str} (vy={adjusted_action.vy:+.2f})]"

        # Strict Invariant Safety Filtering via Canonical Supervisor Core
        safe_act, s_state, s_reason = self._supervisor.filter_action(
            proposed_action=adjusted_action,
            current_position=current_position,
            current_velocity=current_velocity,
            min_front_range=min_front_range,
            dt=dt,
            is_localization_healthy=is_localization_healthy,
            telemetry_age_s=telemetry_age_s,
            observation=observation,
            range_rays=range_rays,
        )

        # Append guidance diagnostics to reason string
        if guidance_tag:
            s_reason = f"{s_reason}; {guidance_tag}"

        self.state_step_counts[self.guidance_state.value] = self.state_step_counts.get(self.guidance_state.value, 0) + 1

        return safe_act, s_state, s_reason
