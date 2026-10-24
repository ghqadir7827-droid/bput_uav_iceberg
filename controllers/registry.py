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

    def __init__(self, cruise_speed: float = 0.70):
        self.cruise_speed = cruise_speed
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

        # Longitudinal cruise speed with goal and obstacle slowing
        if gx < 0:
            vx = -0.15  # Gently reverse if overshot destination
        elif front_min < 1.4:
            vx = float(np.clip(0.30 + 0.35 * (front_min - 0.4) / 1.0, 0.20, self.cruise_speed))
        else:
            vx = float(self.cruise_speed * np.clip(gx, 0.25, 1.0))

        # Lateral / vertical goal tracking
        vy = float(0.5 * gy)
        vz = float(0.5 * gz)

        # Obstacle avoidance steering: evade toward side with more open space
        if front_min < 1.5:
            left_score = d_fl + 0.6 * d_left
            right_score = d_fr + 0.6 * d_right
            repel_strength = float(np.clip((1.5 - front_min) * 0.65, 0.0, 0.60))
            if left_score > right_score:
                vy += repel_strength
            else:
                vy -= repel_strength

        # Corridor wall centering buffers
        if d_left < 0.65:
            vy -= float((0.65 - d_left) * 0.5)
        if d_right < 0.65:
            vy += float((0.65 - d_right) * 0.5)

        # Floor / ceiling buffers
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
