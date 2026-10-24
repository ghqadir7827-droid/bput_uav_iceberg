"""dataset.py - Expert Demonstration Dataset Loader for Behavior Cloning."""

from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset


class ExpertDemonstrationDataset(Dataset):
    """PyTorch Dataset wrapping verified expert demonstrations from .npz archive."""

    def __init__(self, npz_path: str):
        path = Path(npz_path)
        if not path.exists():
            raise FileNotFoundError(f"Demonstration file not found: {npz_path}")

        data = np.load(path, allow_pickle=True)
        self.observations = torch.from_numpy(data["observations"].astype(np.float32))
        self.actions = torch.from_numpy(data["actions"].astype(np.float32))
        self.episode_starts = torch.from_numpy(data["episode_starts"])
        self.episode_ids = torch.from_numpy(data["episode_ids"])

        assert len(self.observations) == len(self.actions), "Mismatch between observations and actions"

    def __len__(self) -> int:
        return len(self.observations)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.observations[idx], self.actions[idx]
