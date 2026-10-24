"""ppo_policy.py - PPO Fine-Tuned Reinforcement Learning Controller.

Implements the fine-tuned reinforcement learning policy:
- Loads the trained PyTorch actor checkpoint (models/checkpoints/ppo_policy_best.pt).
- Ingests the 23-float DeployableObservation vector (rangefinders, relative goal, body velocity, attitude trig, yaw rate).
- Outputs continuous VelocityAction [vx, vy, vz, yaw_rate] bounded strictly within physical limits.
- Inference latency: < 0.1 ms on CPU / ARM64.
"""

from pathlib import Path
from typing import Optional
import numpy as np
import torch

from controllers.registry import ControllerMetadata, IController
from core.contracts import VelocityAction
from models.policy_net import NavigationPolicyMLP


class PPOPolicyController(IController):
    """PPO fine-tuned policy controller evaluated against baseline and expert."""

    def __init__(self, checkpoint_path: str = "models/checkpoints/ppo_policy_best.pt"):
        self.checkpoint_path = Path(checkpoint_path)
        self.device = torch.device("cpu")
        self.model: Optional[NavigationPolicyMLP] = None
        self._is_loaded = False

        self._meta = ControllerMetadata(
            controller_id="ppo_fine_tuned",
            checkpoint_id="unloaded",
            display_name="PPO Fine-Tuned Policy",
            is_implemented=False,
            milestone_tag="Milestone 4",
            description="Reinforcement learning fine-tuned policy with curriculum and hard-negative replay.",
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
            iteration = ckpt.get("iteration", "best")
            eval_rate = ckpt.get("eval_success_rate", 0.0)

            self.model = NavigationPolicyMLP(obs_dim=obs_dim, act_dim=act_dim).to(self.device)
            self.model.load_state_dict(ckpt["model_state_dict"])
            self.model.eval()

            self._is_loaded = True
            self._meta.is_implemented = True
            self._meta.checkpoint_id = f"ppo_iter_{iteration}_succ_{eval_rate:.1f}pct"
            self._meta.description = f"PPO policy fine-tuned for {iteration} iterations (Success: {eval_rate:.1f}%)."
            return True
        except Exception as e:
            print(f"Warning: Failed to load PPO checkpoint from {path_str}: {e}")
            self._is_loaded = False
            self._meta.is_implemented = False
            return False

    def predict(self, observation: np.ndarray) -> VelocityAction:
        """Predicts high-level velocity command from deployable observation."""
        if not self._is_loaded or self.model is None:
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
