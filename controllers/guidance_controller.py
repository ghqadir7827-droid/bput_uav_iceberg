"""guidance_controller.py - Stateless Observable Clearance Guidance Controller.

Production implementation of the symmetry-breaking clearance guidance layer:
- Ingests 23-float DeployableObservation (Student Observation vector).
- Evaluates observable rangefinder sectors (front, left, right) at 20 Hz (dt = 0.05s).
- Resolves centered bimodal symmetry collapse deterministically.
- Features wall-proximity margin arbitration to eliminate sequential corner-pinches.
- Integrates directional hysteresis to suppress 20 Hz chattering.
- Implements soft terminal landing release near the destination goal sphere.
- Sits between the frozen BC baseline and the canonical Safety Supervisor.
"""

from pathlib import Path
from typing import Optional, Tuple, Union
import numpy as np
import torch

from controllers.registry import ControllerMetadata, IController
from core.contracts import ActionLimits, DeployableObservation, VelocityAction
from core.guidance import GuidanceConfig, StatelessClearanceGuidanceArbitrator
from models.policy_net import NavigationPolicyMLP


class StatelessClearanceGuidanceController(IController):
    """Stateless Clearance Guidance Controller with Wall-Aware Margin Arbitration."""

    def __init__(
        self,
        base_model: Optional[Union[NavigationPolicyMLP, str]] = None,
        activation_threshold_m: float = 0.85,
        default_tie_breaker: str = "LEFT",
        device: str = "cpu",
        guidance_config: Optional[GuidanceConfig] = None,
    ):
        self.activation_threshold_m = float(activation_threshold_m)
        self.default_tie_breaker = default_tie_breaker.upper()
        self.device = torch.device(device)
        self.action_limits = ActionLimits()

        # Shared guidance arbitrator
        if guidance_config is not None:
            self.guidance_config = guidance_config
        else:
            self.guidance_config = GuidanceConfig(
                activation_threshold_m=self.activation_threshold_m,
                default_tie_breaker=self.default_tie_breaker,
            )
        self.arbitrator = StatelessClearanceGuidanceArbitrator(
            config=self.guidance_config,
            limits=self.action_limits,
        )
        self.last_dir: str = "NONE"

        # Load base policy if path or instance provided
        self.base_model: Optional[NavigationPolicyMLP] = None
        self._is_loaded = False

        self._meta = ControllerMetadata(
            controller_id="bc_clearance_guidance",
            checkpoint_id="bc_guidance_v1",
            display_name="Stateless Clearance Guidance Policy (BC + Guidance)",
            is_implemented=True,
            milestone_tag="Phase 6 Candidate",
            description=(
                "Frozen BC policy augmented with observable rangefinder clearance arbitration, "
                "wall-proximity margin tie-breaking, and terminal soft-landing release."
            ),
        )

        if base_model is None:
            default_ckpt = Path("models/checkpoints/bc_policy_best.pt")
            if default_ckpt.exists():
                self.load_base_checkpoint(str(default_ckpt))
        elif isinstance(base_model, str):
            self.load_base_checkpoint(base_model)
        elif isinstance(base_model, NavigationPolicyMLP):
            self.base_model = base_model
            self.base_model.eval()
            self._is_loaded = True

    @property
    def metadata(self) -> ControllerMetadata:
        return self._meta

    def load_base_checkpoint(self, path_str: str) -> bool:
        p = Path(path_str)
        if not p.exists():
            return False
        try:
            ckpt = torch.load(p, map_location=self.device, weights_only=False)
            obs_dim = ckpt.get("obs_dim", 23)
            act_dim = ckpt.get("act_dim", 4)
            self.base_model = NavigationPolicyMLP(obs_dim=obs_dim, act_dim=act_dim).to(self.device)
            self.base_model.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
            self.base_model.eval()
            self._is_loaded = True
            self._meta.file_path = str(p)
            return True
        except Exception as e:
            print(f"Warning: Failed to load base checkpoint from {path_str}: {e}")
            self._is_loaded = False
            return False

    def reset(self) -> None:
        """Resets persistent direction latch and filter states."""
        self.arbitrator.reset()
        self.last_dir = "NONE"

    def predict(self, observation: Union[np.ndarray, DeployableObservation]) -> VelocityAction:
        """Computes safe velocity setpoints with clearance guidance overrides."""
        act, _, _, _ = self.predict_with_diagnostics(observation)
        return act

    def predict_with_diagnostics(
        self, observation: Union[np.ndarray, DeployableObservation]
    ) -> Tuple[VelocityAction, str, float, float]:
        """Predicts action and returns guidance diagnostics (selected_dir, c_left, c_right)."""
        if isinstance(observation, DeployableObservation):
            obs_vec = observation.to_vector()
            policy_vec = observation.to_policy_vector()
        else:
            obs_vec = np.asarray(observation, dtype=np.float32)
            if len(obs_vec) > 23:
                policy_vec = np.concatenate([obs_vec[:12], obs_vec[-11:]])
            else:
                policy_vec = obs_vec

        # 1. Base Policy Forward Pass (strictly 23 floats for frozen policy contract)
        if self._is_loaded and self.base_model is not None:
            raw_act_arr = self.base_model.predict_action(policy_vec)
            raw_act = VelocityAction(
                vx=float(raw_act_arr[0]),
                vy=float(raw_act_arr[1]),
                vz=float(raw_act_arr[2]),
                yaw_rate=float(raw_act_arr[3]),
            )
        else:
            raw_act = VelocityAction(vx=0.40, vy=0.0, vz=0.0, yaw_rate=0.0)

        # 2. Delegate to Shared Stateless Clearance Guidance Engine
        mod_act, selected_dir, c_left, c_right = self.arbitrator.arbitrate(raw_act, obs_vec)
        self.last_dir = selected_dir
        return mod_act, selected_dir, float(c_left), float(c_right)

