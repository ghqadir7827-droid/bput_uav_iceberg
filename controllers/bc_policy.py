"""bc_policy.py - Behavior Cloning Neural Network Policy Controller.

Implements the trained imitation learning policy controller:
- Loads the trained PyTorch checkpoint (models/checkpoints/bc_policy_best.pt).
- Ingests the 23-float DeployableObservation vector (rangefinder rays, FLU goal vector, body velocity, attitude trig, yaw rate).
- Outputs high-level continuous VelocityAction [vx, vy, vz, yaw_rate] in body FLU frame.
- Inference latency: < 0.3 ms on standard CPU / ARM64.
"""

from pathlib import Path
from typing import Optional
import numpy as np
import torch

from controllers.registry import ControllerMetadata, IController
from core.contracts import VelocityAction
from models.policy_net import NavigationPolicyMLP


class BehaviorCloningPolicyController(IController):
    """Supervised imitation learning controller evaluated against the expert baseline."""

    def __init__(self, checkpoint_path: str = "models/checkpoints/bc_policy_best.pt"):
        self.checkpoint_path = Path(checkpoint_path)
        self.device = torch.device("cpu") # Onboard controller runs on CPU for lowest jitter
        self.model: Optional[NavigationPolicyMLP] = None
        self._is_loaded = False

        self._meta = ControllerMetadata(
            controller_id="imitation_bc_policy",
            checkpoint_id="unloaded",
            display_name="Behavior Cloning Policy (Imitation)",
            is_implemented=False,
            milestone_tag="Milestone 3",
            description="Supervised neural network imitation of expert demonstrations.",
            file_path=str(self.checkpoint_path)
        )

        self.load_checkpoint(checkpoint_path)

    @property
    def metadata(self) -> ControllerMetadata:
        return self._meta

    def load_checkpoint(self, path_str: str) -> bool:
        p = Path(path_str)
        if not p.exists():
            return False

        try:
            ckpt = torch.load(p, map_location=self.device, weights_only=False)
            obs_dim = ckpt.get("obs_dim", 23)
            act_dim = ckpt.get("act_dim", 4)
            epoch = ckpt.get("epoch", "unknown")
            val_loss = ckpt.get("val_loss", 0.0)

            self.model = NavigationPolicyMLP(obs_dim=obs_dim, act_dim=act_dim).to(self.device)
            self.model.load_state_dict(ckpt["model_state_dict"])
            self.model.eval()

            self._is_loaded = True
            self._meta.is_implemented = True
            self._meta.checkpoint_id = f"bc_epoch_{epoch}_loss_{val_loss:.5f}"
            self._meta.description = f"Supervised neural network trained for {epoch} epochs (Val MSE: {val_loss:.6f})."
            return True
        except Exception as e:
            print(f"Warning: Failed to load BC checkpoint from {path_str}: {e}")
            self._is_loaded = False
            self._meta.is_implemented = False
            return False

    def predict(self, observation: np.ndarray) -> VelocityAction:
        """Predicts velocity action from deployable student observation."""
        if not self._is_loaded or self.model is None:
            # Fallback zero velocity if model not loaded
            return VelocityAction(vx=0.0, vy=0.0, vz=0.0, yaw_rate=0.0)

        with torch.no_grad():
            t_obs = torch.as_tensor(observation, dtype=torch.float32, device=self.device)
            if t_obs.ndim == 1:
                t_obs = t_obs.unsqueeze(0)
            act = self.model(t_obs, bounded=True).squeeze(0).cpu().numpy()

        return VelocityAction(
            vx=float(np.clip(act[0], -0.2, 0.8)),
            vy=float(np.clip(act[1], -0.6, 0.6)),
            vz=float(np.clip(act[2], -0.4, 0.4)),
            yaw_rate=float(np.clip(act[3], -0.6, 0.6))
        )
