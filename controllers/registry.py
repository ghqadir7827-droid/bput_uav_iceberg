"""registry.py - Controller Registry & Model Checkpoint Management.

Provides a unified interface for loading baseline controllers and future neural network
checkpoints without requiring modifications to the viewer or evaluation engine.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np

from core.contracts import ActionLimits, VelocityAction


@dataclass
class ControllerMetadata:
    """Metadata describing a controller and its specific checkpoint."""
    controller_id: str
    checkpoint_id: str
    display_name: str
    is_implemented: bool
    milestone_tag: str
    description: str
    file_path: Optional[str] = None


class IController(ABC):
    """Abstract interface for all navigation controllers."""

    @property
    @abstractmethod
    def metadata(self) -> ControllerMetadata:
        pass

    @abstractmethod
    def predict(self, observation: np.ndarray) -> VelocityAction:
        """Predicts high-level velocity action from deployable observation."""
        pass


class NominalBaselineController(IController):
    """Milestone 1 baseline: commands forward progress with proportional centering."""

    def __init__(self, forward_speed: float = 0.40):
        self.forward_speed = forward_speed
        self._meta = ControllerMetadata(
            controller_id="nominal_baseline",
            checkpoint_id="m1_nominal_baseline_v1",
            display_name="Nominal Baseline (Forward + Centering)",
            is_implemented=True,
            milestone_tag="Milestone 1",
            description="Constant forward transit with lateral goal tracking, filtered by Safety Supervisor."
        )

    @property
    def metadata(self) -> ControllerMetadata:
        return self._meta

    def predict(self, observation: np.ndarray) -> VelocityAction:
        # observation format: [0:12]=rays, [12:15]=rel_goal_body, [15:18]=vel_body, [18:22]=att, [22]=yaw_rate
        rel_goal_y = float(observation[13]) if len(observation) >= 14 else 0.0
        rel_goal_z = float(observation[14]) if len(observation) >= 15 else 0.0

        # Gentle proportional alignment to corridor centerline / target
        vy = float(np.clip(0.6 * rel_goal_y, -0.3, 0.3))
        vz = float(np.clip(0.6 * rel_goal_z, -0.2, 0.2))

        return VelocityAction(vx=self.forward_speed, vy=vy, vz=vz, yaw_rate=0.0)


class ReactiveBaselineController(IController):
    """Milestone 1 baseline: reactive obstacle avoidance using deployable multi-beam rangefinders."""

    def __init__(self, cruise_speed: float = 0.70, drone_radius: float = 0.15):
        self.cruise_speed = float(cruise_speed)
        self.drone_radius = float(drone_radius)
        self._meta = ControllerMetadata(
            controller_id="reactive_baseline",
            checkpoint_id="m1_reactive_baseline_v1",
            display_name="Reactive Obstacle Avoidance Baseline",
            is_implemented=True,
            milestone_tag="Milestone 1",
            description="Active lateral evasion from multi-beam rangefinders with goal tracking and terminal deceleration."
        )

    @property
    def metadata(self) -> ControllerMetadata:
        return self._meta

    def predict(self, observation: np.ndarray) -> VelocityAction:
        rays = observation[0:12] * 4.0  # Physical meters
        rel_goal = observation[12:15]   # [gx, gy, gz]
        gx, gy, gz = float(rel_goal[0]), float(rel_goal[1]), float(rel_goal[2])

        d_front = float(rays[0])
        d_fl = float(min(rays[1], rays[3]))
        d_fr = float(min(rays[2], rays[4]))
        d_left = float(rays[5])
        d_right = float(rays[6])
        d_up = float(rays[10])
        d_down = float(rays[11])

        front_min = min(d_front, d_fl, d_fr)

        # Dynamic thresholds scaled to vehicle radius (bit-identical at default 0.15m)
        obs_slow_dist = self.drone_radius + 1.25   # 1.40m at r=0.15, 1.575m at r=0.325
        obs_evade_dist = self.drone_radius + 1.35  # 1.50m at r=0.15, 1.675m at r=0.325
        wall_buffer = self.drone_radius + 0.50     # 0.65m at r=0.15, 0.825m at r=0.325
        vert_buffer = self.drone_radius + 0.35     # 0.50m at r=0.15, 0.675m at r=0.325

        # Longitudinal cruise speed with goal and obstacle slowing
        if gx < 0:
            vx = -0.15  # Gently reverse if overshot destination
        elif front_min < obs_slow_dist:
            vx = float(np.clip(0.30 + 0.35 * (front_min - (self.drone_radius + 0.25)) / 1.0, 0.20, self.cruise_speed))
        else:
            vx = float(self.cruise_speed * np.clip(gx, 0.25, 1.0))

        # Lateral / vertical goal tracking
        vy = float(0.5 * gy)
        vz = float(0.5 * gz)

        # Obstacle avoidance steering: evade toward side with more open space
        if front_min < obs_evade_dist:
            left_score = d_fl + 0.6 * d_left
            right_score = d_fr + 0.6 * d_right
            repel_strength = float(np.clip((obs_evade_dist - front_min) * 0.65, 0.0, 0.60))
            if left_score > right_score:
                vy += repel_strength
            else:
                vy -= repel_strength

        # Corridor wall centering buffers
        if d_left < wall_buffer:
            vy -= float((wall_buffer - d_left) * 0.5)
        if d_right < wall_buffer:
            vy += float((wall_buffer - d_right) * 0.5)

        # Floor / ceiling buffers
        if d_down < vert_buffer:
            vz += float((vert_buffer - d_down) * 0.4)
        if d_up < vert_buffer:
            vz -= float((vert_buffer - d_up) * 0.4)

        yaw_rate = float(np.clip(1.2 * vy, -0.4, 0.4))

        return VelocityAction(
            vx=float(np.clip(vx, -0.2, 0.8)),
            vy=float(np.clip(vy, -0.6, 0.6)),
            vz=float(np.clip(vz, -0.3, 0.3)),
            yaw_rate=yaw_rate
        )


class RandomBaselineController(IController):
    """Milestone 1 baseline: uniform random velocity exploration."""

    def __init__(self, seed: int = 42):
        self._rng = np.random.default_rng(seed)
        self._meta = ControllerMetadata(
            controller_id="random_baseline",
            checkpoint_id="m1_random_baseline_v1",
            display_name="Random Exploration Baseline",
            is_implemented=True,
            milestone_tag="Milestone 1",
            description="Uniformly samples velocity commands across physical bounds."
        )

    @property
    def metadata(self) -> ControllerMetadata:
        return self._meta

    def predict(self, observation: np.ndarray) -> VelocityAction:
        vx = float(self._rng.uniform(0.0, 0.5))
        vy = float(self._rng.uniform(-0.3, 0.3))
        vz = float(self._rng.uniform(-0.2, 0.2))
        yaw_rate = float(self._rng.uniform(-0.3, 0.3))
        return VelocityAction(vx=vx, vy=vy, vz=vz, yaw_rate=yaw_rate)


class HoverHoldController(IController):
    """Milestone 1 test controller: zero-velocity hover setpoint."""

    def __init__(self):
        self._meta = ControllerMetadata(
            controller_id="hover_hold",
            checkpoint_id="m1_hover_hold_v1",
            display_name="Zero-Velocity Hover Hold",
            is_implemented=True,
            milestone_tag="Milestone 1",
            description="Commands zero linear and angular velocities to test equilibrium."
        )

    @property
    def metadata(self) -> ControllerMetadata:
        return self._meta

    def predict(self, observation: np.ndarray) -> VelocityAction:
        return VelocityAction(vx=0.0, vy=0.0, vz=0.0, yaw_rate=0.0)


class ControllerRegistry:
    """Discovers, registers, and serves active controllers and checkpoints."""

    def __init__(self, checkpoints_dir: str = "models/checkpoints"):
        self.checkpoints_dir = Path(checkpoints_dir)
        self._controllers: Dict[str, IController] = {}
        self._planned_specs: List[ControllerMetadata] = []

        # Register Milestone 1 controllers
        self.register(ReactiveBaselineController())
        self.register(NominalBaselineController())
        self.register(RandomBaselineController())
        self.register(HoverHoldController())

        # Register Milestone 2 expert controller
        try:
            from controllers.apf_expert import ScriptedAPFExpertController
            self.register(ScriptedAPFExpertController())
        except ImportError:
            pass

        # Register Milestone 3 BC policy controller if checkpoint exists
        bc_registered = False
        try:
            from controllers.bc_policy import BehaviorCloningPolicyController
            bc_ctrl = BehaviorCloningPolicyController()
            if bc_ctrl.metadata.is_implemented:
                self.register(bc_ctrl)
                bc_registered = True
        except Exception:
            pass

        # Register Phase 6 Stateless Clearance Guidance controller if base checkpoint exists
        try:
            from controllers.guidance_controller import StatelessClearanceGuidanceController
            guidance_ctrl = StatelessClearanceGuidanceController()
            if guidance_ctrl.metadata.is_implemented:
                self.register(guidance_ctrl)
        except Exception:
            pass

        # Register Scaled 65cm UAV Guided BC Controllers (v1 and v2 Winner)
        try:
            from controllers.guidance_controller import StatelessClearanceGuidanceController
            from core.guidance import GuidanceConfig

            scaled_v1_path = Path("models/checkpoints/bc_policy_scaled_v1.pt")
            if scaled_v1_path.exists():
                g_cfg_v1 = GuidanceConfig.for_65cm_uav(drone_radius=0.325)
                ctrl_v1 = StatelessClearanceGuidanceController(
                    base_model=str(scaled_v1_path),
                    guidance_config=g_cfg_v1,
                )
                ctrl_v1._meta = ControllerMetadata(
                    controller_id="scaled_guided_bc_v1",
                    checkpoint_id="bc_policy_scaled_v1",
                    display_name="65cm UAV Guided BC v1 (Baseline)",
                    is_implemented=True,
                    milestone_tag="Phase 4 Baseline",
                    description="Scaled BC v1 baseline with 3-ray guidance.",
                    file_path=str(scaled_v1_path),
                )
                self.register(ctrl_v1)

            scaled_v2_path = Path("models/checkpoints/bc_policy_scaled_v2.pt")
            if scaled_v2_path.exists():
                g_cfg_v2 = GuidanceConfig.for_65cm_uav_v2(drone_radius=0.325)
                ctrl_v2 = StatelessClearanceGuidanceController(
                    base_model=str(scaled_v2_path),
                    guidance_config=g_cfg_v2,
                )
                ctrl_v2._meta = ControllerMetadata(
                    controller_id="scaled_guided_bc_v2",
                    checkpoint_id="bc_policy_scaled_v2",
                    display_name="65cm UAV Guided BC v2 (Exceed 90% Winner)",
                    is_implemented=True,
                    milestone_tag="Phase 5 Winner",
                    description="Retrained BC v2 with recovery perturbations and 5-ray harmonized guidance (94.0% success).",
                    file_path=str(scaled_v2_path),
                )
                self.register(ctrl_v2)

                g_cfg_v3 = GuidanceConfig.for_65cm_uav_v3(drone_radius=0.325)
                ctrl_v3 = StatelessClearanceGuidanceController(
                    base_model=str(scaled_v2_path),
                    guidance_config=g_cfg_v3,
                )
                ctrl_v3._meta = ControllerMetadata(
                    controller_id="scaled_guided_bc_v3",
                    checkpoint_id="bc_policy_scaled_v2_v3guidance",
                    display_name="65cm UAV Guided BC v3 (Persistent Guidance)",
                    is_implemented=True,
                    milestone_tag="Phase 6 Candidate",
                    description="Frozen BC v2 with directional-persistence tie-breaking guidance to eliminate chatter.",
                    file_path=str(scaled_v2_path),
                )
                self.register(ctrl_v3)

                g_cfg_v4 = GuidanceConfig.for_65cm_uav_v4_adaptive(drone_radius=0.325)
                ctrl_v4 = StatelessClearanceGuidanceController(
                    base_model=str(scaled_v2_path),
                    guidance_config=g_cfg_v4,
                )
                ctrl_v4._meta = ControllerMetadata(
                    controller_id="scaled_guided_bc_v4",
                    checkpoint_id="bc_policy_scaled_v2_v4adaptive",
                    display_name="65cm UAV Guided BC v4 (Corridor-Adaptive Guidance)",
                    is_implemented=True,
                    milestone_tag="Adaptive Candidate",
                    description="Frozen BC v2 with corridor-width-adaptive lateral guidance and persistent tie-breaking.",
                    file_path=str(scaled_v2_path),
                )
                self.register(ctrl_v4)

                g_cfg_v5 = GuidanceConfig.for_65cm_uav_v5_clearance_aware(drone_radius=0.325)
                ctrl_v5 = StatelessClearanceGuidanceController(
                    base_model=str(scaled_v2_path),
                    guidance_config=g_cfg_v5,
                )
                ctrl_v5._meta = ControllerMetadata(
                    controller_id="scaled_guided_bc_v5",
                    checkpoint_id="bc_policy_scaled_v2_v5clearance",
                    display_name="65cm UAV Guided BC v5 (Clearance-Aware Continuous Guidance)",
                    is_implemented=True,
                    milestone_tag="Phase 4F Candidate",
                    description="Frozen BC v2 with continuous clearance-scaled lateral guidance (0.15-0.40 m/s) and safe-arrest integration.",
                    file_path=str(scaled_v2_path),
                )
                self.register(ctrl_v5)
        except Exception:
            pass

        # Register planned controllers (strictly flagged as is_implemented=False)
        if not bc_registered:
            self._planned_specs.append(
                ControllerMetadata(
                    controller_id="imitation_bc_policy",
                    checkpoint_id="pending_training",
                    display_name="Behavior Cloning Policy (Planned)",
                    is_implemented=False,
                    milestone_tag="Milestone 3",
                    description="Supervised neural network imitation of expert demonstrations."
                )
            )

        # Register Milestone 4 PPO policy controller if checkpoint exists
        ppo_registered = False
        try:
            from controllers.ppo_policy import PPOPolicyController
            ppo_ctrl = PPOPolicyController()
            if ppo_ctrl.metadata.is_implemented:
                self.register(ppo_ctrl)
                ppo_registered = True
        except Exception:
            pass

        # Register planned controllers (strictly flagged as is_implemented=False)
        if not ppo_registered:
            self._planned_specs.append(
                ControllerMetadata(
                    controller_id="ppo_fine_tuned",
                    checkpoint_id="pending_training",
                    display_name="PPO Fine-Tuned Policy (Planned)",
                    is_implemented=False,
                    milestone_tag="Milestone 4",
                    description="Curriculum PPO reinforcement learning with hard-negative replay."
                )
            )
        # Register Milestone 5 ONNX edge runner controller if onnx model exists
        try:
            from controllers.onnx_controller import ONNXEdgePolicyController
            onnx_ctrl = ONNXEdgePolicyController()
            if onnx_ctrl.metadata.is_implemented:
                self.register(onnx_ctrl)
        except Exception:
            pass

        self.scan_checkpoints()

    def register(self, controller: IController) -> None:
        self._controllers[controller.metadata.controller_id] = controller

    def get_controller(self, controller_id: str) -> Optional[IController]:
        return self._controllers.get(controller_id)

    def scan_checkpoints(self) -> None:
        """Scans models/checkpoints for future .onnx and .pt files."""
        if not self.checkpoints_dir.exists():
            self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
            return

        for f in self.checkpoints_dir.glob("*.*"):
            if f.suffix in (".onnx", ".pt", ".pth"):
                cid = f"checkpoint_{f.stem}"
                # In future milestones, this wraps an ONNX / PyTorch runner
                pass

    def list_all_metadata(self) -> List[Dict]:
        """Returns JSON-serializable list of all active and planned controllers."""
        result = []
        for c in self._controllers.values():
            m = c.metadata
            result.append({
                "controller_id": m.controller_id,
                "checkpoint_id": m.checkpoint_id,
                "display_name": m.display_name,
                "is_implemented": m.is_implemented,
                "milestone_tag": m.milestone_tag,
                "description": m.description,
            })
        for p in self._planned_specs:
            result.append({
                "controller_id": p.controller_id,
                "checkpoint_id": p.checkpoint_id,
                "display_name": p.display_name,
                "is_implemented": p.is_implemented,
                "milestone_tag": p.milestone_tag,
                "description": p.description,
            })
        return result
