"""pillar3_recovery_controller.py - Isolated Controller Candidate for Phase 6B.

Integrates the frozen BC policy with PostEvasionRecoveryArbitrator.
"""

from pathlib import Path
from typing import Optional, Tuple, Union
import numpy as np
import torch

from controllers.registry import ControllerMetadata, IController
from core.contracts import ActionLimits, DeployableObservation, VelocityAction
from experiments.phase6b.phase6b_pillar3_recovery_20261030_025800.recovery_guidance import (
    PostEvasionRecoveryArbitrator,
    RecoveryConfig,
)
from models.policy_net import NavigationPolicyMLP


class PostEvasionRecoveryController(IController):
    """Isolated Phase 6B Controller: Frozen BC + Stateless Clearance Guidance + Post-Evasion Recovery."""

    def __init__(
        self,
        base_model: Optional[Union[NavigationPolicyMLP, str]] = None,
        config: Optional[RecoveryConfig] = None,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.action_limits = ActionLimits()
        self.config = config or RecoveryConfig()
        self.arbitrator = PostEvasionRecoveryArbitrator(
            config=self.config,
            limits=self.action_limits,
        )
        self.last_dir: str = "NONE"

        self.base_model: Optional[NavigationPolicyMLP] = None
        self._is_loaded = False

        self._meta = ControllerMetadata(
            controller_id="bc_pillar3_recovery_candidate",
            checkpoint_id="bc_phase6b_candidate",
            display_name="Post-Evasion Recovery Guidance Policy (Phase 6B Candidate)",
            is_implemented=True,
            milestone_tag="Phase 6B Candidate",
            description=(
                "Frozen BC policy augmented with stateless clearance guidance and "
                "pure-NumPy post-evasion corridor centerline recovery."
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
        """Resets arbitrator and direction latches."""
        self.arbitrator.reset()
        self.last_dir = "NONE"

    def predict(self, observation: Union[np.ndarray, DeployableObservation]) -> VelocityAction:
        act, _, _, _ = self.predict_with_diagnostics(observation)
        return act

    def predict_with_diagnostics(
        self, observation: Union[np.ndarray, DeployableObservation]
    ) -> Tuple[VelocityAction, str, float, float]:
        if isinstance(observation, DeployableObservation):
            obs_vec = observation.to_vector()
        else:
            obs_vec = np.asarray(observation, dtype=np.float32)

        if self._is_loaded and self.base_model is not None:
            raw_act_arr = self.base_model.predict_action(obs_vec)
            raw_act = VelocityAction(
                vx=float(raw_act_arr[0]),
                vy=float(raw_act_arr[1]),
                vz=float(raw_act_arr[2]),
                yaw_rate=float(raw_act_arr[3]),
            )
        else:
            raw_act = VelocityAction(vx=0.40, vy=0.0, vz=0.0, yaw_rate=0.0)

        mod_act, selected_dir, c_left, c_right = self.arbitrator.arbitrate(raw_act, obs_vec)
        self.last_dir = selected_dir
        return mod_act, selected_dir, float(c_left), float(c_right)

