"""eval_engine.py - Simulation Evaluation Engine & Episode Replay Buffer.

Manages synchronized stepping, repeatable evaluation over fixed held-out seeds,
episode recording for visual replay, and true measured metrics logging.
"""

from dataclasses import asdict, dataclass
import threading
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from controllers.registry import ControllerRegistry, IController
from controllers.safety_supervisor import SafetyState, SafetySupervisor, SafetySupervisorConfig
from core.contracts import (
    ActionLimits,
    DeployableObservation,
    VelocityAction,
    compute_forward_clearance_m,
)
from core.geometry import Corridor3D, TerminationReason
from localization.interface import LocalizationManager
from localization.sim_source import SimGroundTruthSource
from sim.corridor_env import CorridorSimEnv


@dataclass
class EvaluationRecord:
    """True measured outcome of a single evaluation episode."""
    episode_id: int
    seed: int
    controller_id: str
    checkpoint_id: str
    status: str
    steps: int
    flight_time_s: float
    distance_travelled_m: float
    safety_interventions: int
    min_boundary_distance_m: float
    min_obstacle_clearance_m: float
    final_distance_to_goal_m: float
    timestamp_iso: str


class EvaluationEngine:
    """Coordinates simulation rollouts, controller prediction, safety filtering, and metrics."""

    def __init__(self, seed: int = 100, scenario: str = "level_2"):
        self.lock = threading.RLock()
        self.env = CorridorSimEnv(scenario=scenario, gui=False, seed=seed)
        self.corridor = self.env.corridor
        # Safety supervisor automatically configures for 65cm UAV if drone_radius > 0.25
        self.safety_supervisor = SafetySupervisor(corridor=self.corridor)

        self.sim_loc_source = SimGroundTruthSource()
        self.loc_manager = LocalizationManager(primary_source=self.sim_loc_source, allow_dynamic_switching=False)

        self.registry = ControllerRegistry()
        self.active_controller_id = "reactive_baseline"
        self.active_controller: IController = self.registry.get_controller(self.active_controller_id)

        self.current_seed = seed
        self.episode_counter = 0
        self.is_running = False
        self.playback_speed = 1.0

        # Live episode state & trajectory trail
        self.trajectory_trail: List[List[float]] = []
        self.current_episode_snapshots: List[Dict[str, Any]] = []
        self.current_episode_status = "running"
        self.last_safety_state = SafetyState.NOMINAL.value
        self.last_safety_reason = "Nominal flight"
        self.last_proposed_action = VelocityAction(0.0, 0.0, 0.0, 0.0)
        self.last_applied_action = VelocityAction(0.0, 0.0, 0.0, 0.0)
        self.last_norm_action = np.zeros(4, dtype=np.float32)
        self.running_safety_interventions = 0

        # Evaluation history & recording
        self.evaluation_records: List[EvaluationRecord] = []
        self.recorded_episodes: Dict[int, List[Dict[str, Any]]] = {}

        self.reset(seed=seed)

    def select_scenario(self, scenario_id: str) -> Tuple[bool, str]:
        """Switches corridor scenario between Level 1-4 scaled and legacy benchmarks."""
        with self.lock:
            allowed = ("level_1", "level_2", "level_3", "level_4", "scaled", "10m", "20m")
            if scenario_id not in allowed:
                return False, f"Unknown scenario '{scenario_id}'. Choose from: {list(allowed)}."
            self.env.close()
            self.env = CorridorSimEnv(scenario=scenario_id, gui=False, seed=self.current_seed)
            self.corridor = self.env.corridor
            self.safety_supervisor = SafetySupervisor(corridor=self.corridor)
            if hasattr(self.active_controller, "update_environment"):
                self.active_controller.update_environment(self.corridor, self.env.obstacles)
            self.reset(seed=self.current_seed)
            return True, f"Scenario set to {self.corridor.name} ({self.corridor.x_end:.1f}m x {self.corridor.width:.1f}m x {self.corridor.height:.1f}m, drone_r={self.corridor.drone_radius:.3f}m)"

    def select_controller(self, controller_id: str) -> Tuple[bool, str]:
        """Switches active controller if implemented."""
        with self.lock:
            ctrl = self.registry.get_controller(controller_id)
            if ctrl is None:
                return False, f"Controller '{controller_id}' not found."
            if not ctrl.metadata.is_implemented:
                return False, f"Controller '{ctrl.metadata.display_name}' is planned for {ctrl.metadata.milestone_tag} and not yet implemented."

            self.active_controller_id = controller_id
            self.active_controller = ctrl
            if hasattr(self.active_controller, "update_environment"):
                self.active_controller.update_environment(self.corridor, self.env.obstacles)
            return True, f"Active controller set to: {ctrl.metadata.display_name} ({ctrl.metadata.checkpoint_id})"

    def reset(self, seed: Optional[int] = None) -> Dict[str, Any]:
        """Resets the simulation environment."""
        with self.lock:
            if seed is not None:
                self.current_seed = seed
            self.episode_counter += 1
            self.obs_vec, self.info = self.env.reset(seed=self.current_seed)
            if hasattr(self.active_controller, "update_environment"):
                self.active_controller.update_environment(self.corridor, self.env.obstacles)

            self.trajectory_trail = []
            self.current_episode_snapshots = []
            self.current_episode_status = "running"
            self.last_safety_state = SafetyState.NOMINAL.value
            self.last_safety_reason = "Initialized"
            self.last_proposed_action = VelocityAction(0.0, 0.0, 0.0, 0.0)
            self.last_applied_action = VelocityAction(0.0, 0.0, 0.0, 0.0)
            self.last_norm_action = np.zeros(4, dtype=np.float32)
            self.running_safety_interventions = 0

            # Reset supervisor episode-dependent state
            if self.safety_supervisor is not None:
                self.safety_supervisor.loss_timer_s = 0.0
                self.safety_supervisor.stalled_timer_s = 0.0
                self.safety_supervisor.last_safe_action = VelocityAction(0.0, 0.0, 0.0, 0.0)

            # Record initial snapshot
            snap = self._capture_snapshot()
            self.current_episode_snapshots.append(snap)
            self.trajectory_trail.append(snap["position"])
            return snap

    def step(self) -> Dict[str, Any]:
        """Executes a single synchronized simulation step."""
        with self.lock:
            if self.current_episode_status != "running":
                return self._capture_snapshot()

            # 1. Update Localization Layer
            priv_state = self.info["privileged_state"]
            if hasattr(self.active_controller, "set_state"):
                self.active_controller.set_state(priv_state.global_position, priv_state.global_velocity)

            self.sim_loc_source.update_from_sim(
                priv_state.global_position,
                priv_state.global_velocity,
                priv_state.global_orientation_quat,
                sim_time_s=self.env.current_sim_time
            )
            pkt, health = self.loc_manager.get_state(current_time_s=self.env.current_sim_time)

            # 2. Controller Inference
            proposed_action = self.active_controller.predict(self.obs_vec)
            self.last_proposed_action = proposed_action

            # 3. Multi-beam front clearance across forward-facing rays (indices 0..4)
            front_dist = compute_forward_clearance_m(
                self.obs_vec, max_range_m=self.env.sensor_config.max_range_m
            )

            # 4. Safety Supervisor Filtering
            safe_action, s_state, s_reason = self.safety_supervisor.filter_action(
                proposed_action=proposed_action,
                current_position=pkt.position if pkt else priv_state.global_position,
                current_velocity=pkt.velocity if pkt else priv_state.global_velocity,
                min_front_range=front_dist,
                dt=self.env.dt,
                is_localization_healthy=health.is_healthy,
                telemetry_age_s=health.packet_age_s,
                observation=self.obs_vec,
            )

            self.last_applied_action = safe_action
            self.last_safety_state = s_state.value
            self.last_safety_reason = s_reason
            if s_state != SafetyState.NOMINAL:
                self.running_safety_interventions += 1

            # 5. Physics Step
            norm_action = safe_action.to_normalized(self.env.action_limits)
            self.last_norm_action = norm_action
            self.obs_vec, reward, terminated, truncated, self.info = self.env.step(norm_action)

            # 6. Capture Snapshot & Trajectory
            snap = self._capture_snapshot()
            self.current_episode_snapshots.append(snap)
            self.trajectory_trail.append(snap["position"])

            # 7. Check Episode Termination
            term_reason = self.info.get("termination_reason", "continue")
            if terminated or truncated:
                self.current_episode_status = term_reason
                self._record_completed_episode()

            return snap

    def _capture_snapshot(self) -> Dict[str, Any]:
        """Captures full visual and telemetry snapshot directly from the simulation."""
        vis = self.env.get_visual_state()
        raw_obs = [round(float(x), 4) for x in self.obs_vec]

        # Breakdown of 23-float deployable observation into contract groups
        obs_groups = {
            "distance_rays_norm": raw_obs[0:12],
            "distance_rays_m": [round(raw_obs[i] * 4.0, 2) for i in range(min(12, len(raw_obs)))],
            "rel_goal_direction_body": raw_obs[12:15] if len(raw_obs) >= 15 else [0.0, 0.0, 0.0],
            "body_velocity_mps": raw_obs[15:18] if len(raw_obs) >= 18 else [0.0, 0.0, 0.0],
            "attitude_trig": raw_obs[18:22] if len(raw_obs) >= 22 else [0.0, 1.0, 0.0, 1.0],
            "yaw_rate_radps": raw_obs[22] if len(raw_obs) > 22 else 0.0,
        }

        # Commanded actions
        prop_act = {
            "vx": round(float(self.last_proposed_action.vx), 3),
            "vy": round(float(self.last_proposed_action.vy), 3),
            "vz": round(float(self.last_proposed_action.vz), 3),
            "yaw_rate": round(float(self.last_proposed_action.yaw_rate), 3),
        }
        applied_act = {
            "vx": round(float(self.last_applied_action.vx), 3),
            "vy": round(float(self.last_applied_action.vy), 3),
            "vz": round(float(self.last_applied_action.vz), 3),
            "yaw_rate": round(float(self.last_applied_action.yaw_rate), 3),
        }
        norm_act = [round(float(x), 3) for x in self.last_norm_action]

        euler_deg = [
            round(float(vis["orientation_euler"][0]) * 180.0 / np.pi, 1),
            round(float(vis["orientation_euler"][1]) * 180.0 / np.pi, 1),
            round(float(vis["orientation_euler"][2]) * 180.0 / np.pi, 1),
        ]

        flags = {
            "running": self.current_episode_status == "running",
            "success": self.current_episode_status == "success",
            "crash_obstacle": self.current_episode_status == "crash_obstacle",
            "crash_boundary": self.current_episode_status == "crash_boundary",
            "timeout": self.current_episode_status == "timeout",
        }

        # Ground truth state (clearly distinguished as simulation truth, NOT policy input)
        priv_truth = {
            "obstacle_clearance_m": round(float(vis["obstacle_clearance"]), 3),
            "boundary_distance_m": round(float(vis["boundary_distance"]), 3),
            "world_position_m": [round(float(x), 3) for x in vis["position"]],
            "world_velocity_mps": [round(float(x), 3) for x in vis["velocity"]],
            "obstacles": [
                {"center": [round(float(c), 3) for c in o.center], "radius": round(float(o.radius), 3)}
                for o in self.env.obstacles
            ],
        }

        return {
            "step": vis["step_count"],
            "sim_time_s": round(float(vis["sim_time_s"]), 2),
            "position": vis["position"],
            "velocity": vis["velocity"],
            "orientation_euler": vis["orientation_euler"],
            "orientation_euler_deg": euler_deg,
            "speed_mps": round(float(vis["speed_mps"]), 3),
            "boundary_distance": round(float(vis["boundary_distance"]), 3),
            "obstacle_clearance": round(float(vis["obstacle_clearance"]), 3),
            "goal_distance": round(float(vis["goal_distance"]), 3),
            "goal_position": [round(float(x), 3) for x in self.corridor.goal_position],
            "rays": vis["rays"],
            "episode_status": self.current_episode_status,
            "safety_state": self.last_safety_state,
            "safety_reason": self.last_safety_reason,
            "safety_interventions_count": self.running_safety_interventions,
            "controller_id": self.active_controller.metadata.controller_id,
            "checkpoint_id": self.active_controller.metadata.checkpoint_id,
            "controller_display_name": self.active_controller.metadata.display_name,
            "seed": self.current_seed,
            "scenario": self.corridor.name,
            "proposed_action": prop_act,
            "applied_action": applied_act,
            "normalized_action": norm_act,
            "deployable_observation_23": raw_obs,
            "observation_groups": obs_groups,
            "flags": flags,
            "privileged_simulation_ground_truth": priv_truth,
        }

    def _record_completed_episode(self) -> None:
        """Stores metrics and snapshot buffer of finished episode."""
        snaps = self.current_episode_snapshots
        min_b = min(s["boundary_distance"] for s in snaps) if snaps else 0.0
        min_obs = min(s["obstacle_clearance"] for s in snaps) if snaps else 0.0
        final_d = snaps[-1]["goal_distance"] if snaps else 0.0

        # Calculate true cumulative distance travelled
        dist_m = 0.0
        for i in range(1, len(snaps)):
            p0 = np.array(snaps[i - 1]["position"])
            p1 = np.array(snaps[i]["position"])
            dist_m += float(np.linalg.norm(p1 - p0))

        # Count safety supervisor interventions
        interventions = sum(1 for s in snaps if s.get("safety_state", "nominal") != "nominal")

        rec = EvaluationRecord(
            episode_id=self.episode_counter,
            seed=self.current_seed,
            controller_id=self.active_controller.metadata.controller_id,
            checkpoint_id=self.active_controller.metadata.checkpoint_id,
            status=self.current_episode_status,
            steps=len(snaps),
            flight_time_s=float(self.env.current_sim_time),
            distance_travelled_m=round(dist_m, 2),
            safety_interventions=interventions,
            min_boundary_distance_m=float(min_b),
            min_obstacle_clearance_m=float(min_obs),
            final_distance_to_goal_m=float(final_d),
            timestamp_iso=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        )
        self.evaluation_records.append(rec)
        self.recorded_episodes[self.episode_counter] = list(snaps)

    def run_eval_batch(self, seeds: List[int]) -> Dict[str, Any]:
        """Runs repeatable evaluation across fixed held-out seeds."""
        batch_results = []
        for s in seeds:
            self.reset(seed=s)
            while self.current_episode_status == "running":
                self.step()
            batch_results.append(self.evaluation_records[-1])

        return self.get_summary_metrics()

    def get_summary_metrics(self) -> Dict[str, Any]:
        """Aggregates real measured metrics across all completed evaluation runs."""
        with self.lock:
            if not self.evaluation_records:
                return {
                    "total_episodes": 0,
                    "has_data": False,
                    "message": "No evaluation episodes run yet. Use 'Run Eval Batch' or run episodes to generate real metrics."
                }

            total = len(self.evaluation_records)
            successes = sum(1 for r in self.evaluation_records if r.status == "success")
            crashes_obs = sum(1 for r in self.evaluation_records if r.status == "crash_obstacle")
            crashes_bound = sum(1 for r in self.evaluation_records if r.status == "crash_boundary")
            timeouts = sum(1 for r in self.evaluation_records if r.status == "timeout")

            flight_times = [r.flight_time_s for r in self.evaluation_records]
            distances = [r.distance_travelled_m for r in self.evaluation_records]
            interventions = [r.safety_interventions for r in self.evaluation_records]
            min_clearances = [min(r.min_boundary_distance_m, r.min_obstacle_clearance_m) for r in self.evaluation_records]

            return {
                "has_data": True,
                "total_episodes": total,
                "success_count": successes,
                "success_rate_pct": float((successes / total) * 100.0),
                "collision_count": crashes_obs + crashes_bound,
                "collision_rate_pct": float(((crashes_obs + crashes_bound) / total) * 100.0),
                "boundary_violation_count": crashes_bound,
                "boundary_violation_pct": float((crashes_bound / total) * 100.0),
                "timeout_count": timeouts,
                "timeout_rate_pct": float((timeouts / total) * 100.0),
                "mean_flight_time_s": float(np.mean(flight_times)),
                "mean_distance_travelled_m": float(np.mean(distances)),
                "mean_safety_interventions": float(np.mean(interventions)),
                "mean_min_clearance_m": float(np.mean(min_clearances)),
                "active_controller": self.active_controller.metadata.display_name,
                "active_checkpoint": self.active_controller.metadata.checkpoint_id,
                "records": [asdict(r) for r in self.evaluation_records[-20:]], # Latest 20
            }

    def get_replay_data(self, episode_id: int) -> Optional[List[Dict[str, Any]]]:
        """Returns recorded trajectory buffer for visual scrubbing."""
        with self.lock:
            return self.recorded_episodes.get(episode_id)
