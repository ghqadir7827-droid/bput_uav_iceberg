"""policy_net.py - Lightweight Navigation Policy Architecture for Onboard Inference.

Implements an efficient Multi-Layer Perceptron (MLP) designed for deployment
on resource-constrained onboard hardware (e.g. Raspberry Pi 4/5):
- 23-float DeployableObservation input vector
- 4-float continuous velocity command output [vx, vy, vz, yaw_rate]
- Bound-enforcing activation head guarantees outputs remain within physical limits.
- Inference latency: < 0.3 ms on standard CPU / ARM64.
"""

from typing import Dict, Tuple
import numpy as np
import torch
import torch.nn as nn


class NavigationPolicyMLP(nn.Module):
    """Feedforward navigation policy mapping rangefinders and state to velocity commands."""

    def __init__(
        self,
        obs_dim: int = 23,
        act_dim: int = 4,
        hidden_dims: Tuple[int, ...] = (128, 128, 64),
        dropout_p: float = 0.0,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        # Physical Action Limits (Body FLU frame)
        # vx: [-0.2, 0.8] m/s -> center 0.3, half 0.5
        # vy: [-0.6, 0.6] m/s -> center 0.0, half 0.6
        # vz: [-0.4, 0.4] m/s -> center 0.0, half 0.4
        # yaw_rate: [-0.6, 0.6] rad/s -> center 0.0, half 0.6
        self.register_buffer("act_center", torch.tensor([0.30, 0.00, 0.00, 0.00], dtype=torch.float32))
        self.register_buffer("act_half_range", torch.tensor([0.50, 0.60, 0.40, 0.60], dtype=torch.float32))

        layers = []
        in_dim = obs_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.LayerNorm(h_dim))
            layers.append(nn.GELU())
            if dropout_p > 0.0:
                layers.append(nn.Dropout(dropout_p))
            in_dim = h_dim

        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(in_dim, act_dim)

    def forward(self, obs: torch.Tensor, bounded: bool = True) -> torch.Tensor:
        """Forward pass.
        
        Args:
            obs: Observation tensor of shape (batch, 23).
            bounded: If True, applies scaled tanh to strictly enforce physical limits.
        """
        features = self.backbone(obs)
        raw_act = self.head(features)

        if bounded:
            # Scaled tanh: center + half_range * tanh(raw)
            return self.act_center + self.act_half_range * torch.tanh(raw_act)
        return raw_act

    def predict_action(self, obs_np: np.ndarray) -> np.ndarray:
        """Convenience method for single-sample numpy inference."""
        self.eval()
        with torch.no_grad():
            t_obs = torch.as_tensor(obs_np, dtype=torch.float32, device=self.act_center.device)
            if t_obs.ndim == 1:
                t_obs = t_obs.unsqueeze(0)
            act = self.forward(t_obs, bounded=True)
            return act.squeeze(0).cpu().numpy()
