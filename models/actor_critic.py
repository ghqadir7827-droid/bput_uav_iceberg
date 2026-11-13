"""actor_critic.py - Actor-Critic Architecture for PPO Fine-Tuning.

Implements the Actor-Critic network for continuous velocity control:
- Actor: NavigationPolicyMLP initialized from Behavior Cloning pre-trained checkpoint.
- Critic: ValueNetworkMLP estimating scalar state value V(s).
- Gaussian action distribution with learnable or state-independent log_std.
"""

from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

from models.policy_net import NavigationPolicyMLP


class ValueNetworkMLP(nn.Module):
    """Critic network mapping 23-float deployable observation to scalar state value V(s)."""

    def __init__(self, obs_dim: int = 23, hidden_dims: Tuple[int, ...] = (128, 128, 64)):
        super().__init__()
        layers = []
        in_dim = obs_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.LayerNorm(h_dim))
            layers.append(nn.GELU())
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs).squeeze(-1)


class ActorCriticPPO(nn.Module):
    """Actor-Critic model for continuous PPO fine-tuning."""

    def __init__(
        self,
        obs_dim: int = 23,
        act_dim: int = 4,
        init_log_std: float = -1.2,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        self.actor = NavigationPolicyMLP(obs_dim=obs_dim, act_dim=act_dim)
        self.critic = ValueNetworkMLP(obs_dim=obs_dim)

        # Diagonal Gaussian exploration
        self.log_std = nn.Parameter(torch.ones(act_dim) * init_log_std)

    def load_bc_warmstart(self, bc_checkpoint_path: str, device: torch.device) -> bool:
        """Initializes actor weights from pre-trained Behavior Cloning checkpoint."""
        p = Path(bc_checkpoint_path)
        if not p.exists():
            print(f"Warning: BC checkpoint not found at {bc_checkpoint_path}. Starting from scratch.")
            return False

        try:
            ckpt = torch.load(p, map_location=device, weights_only=False)
            self.actor.load_state_dict(ckpt["model_state_dict"])
            print(f"[PPO Warmstart] Successfully loaded BC weights from: {bc_checkpoint_path}")
            return True
        except Exception as e:
            print(f"Warning: Failed to load BC warmstart: {e}")
            return False

    def load_ppo_checkpoint(self, ppo_checkpoint_path: str, device: torch.device) -> bool:
        """Initializes actor and critic weights from an existing PPO checkpoint."""
        p = Path(ppo_checkpoint_path)
        if not p.exists():
            print(f"Warning: PPO checkpoint not found at {ppo_checkpoint_path}.")
            return False

        try:
            ckpt = torch.load(p, map_location=device, weights_only=False)
            obs_dim = ckpt.get("obs_dim", 23)
            act_dim = ckpt.get("act_dim", 4)
            if obs_dim != self.obs_dim or act_dim != self.act_dim:
                print(f"Error: Dimension mismatch in {ppo_checkpoint_path}: obs_dim={obs_dim} (expected {self.obs_dim}), act_dim={act_dim} (expected {self.act_dim})")
                return False

            self.actor.load_state_dict(ckpt["model_state_dict"])
            if "critic_state_dict" in ckpt:
                self.critic.load_state_dict(ckpt["critic_state_dict"])
                print(f"[PPO Checkpoint] Successfully loaded actor and critic from: {ppo_checkpoint_path}")
            else:
                print(f"[PPO Checkpoint] Loaded actor only from: {ppo_checkpoint_path} (no critic in checkpoint)")

            if "log_std" in ckpt:
                with torch.no_grad():
                    self.log_std.copy_(ckpt["log_std"].to(device))
                print(f"[PPO Checkpoint] Loaded exploration log_std from checkpoint.")
            else:
                print(f"[PPO Checkpoint] Exploration log_std retained at default ({float(self.log_std[0].item()):.2f}).")

            return True
        except Exception as e:
            print(f"Warning: Failed to load PPO checkpoint: {e}")
            return False


    def get_action_and_value(
        self,
        obs: torch.Tensor,
        action: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Samples action, computes log probability, entropy, and state value."""
        # Mean action with bounded activation
        action_mean = self.actor(obs, bounded=True)
        action_std = torch.exp(torch.clamp(self.log_std, -2.5, 0.2))
        dist = Normal(action_mean, action_std)

        if deterministic:
            act = action_mean
        else:
            if action is None:
                act = dist.rsample()
            else:
                act = action

        # Sum log probs across the 4 action dimensions
        log_prob = dist.log_prob(act).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        value = self.critic(obs)

        return act, log_prob, entropy, value
