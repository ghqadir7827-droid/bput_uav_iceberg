"""test_scaled_bc_pipeline.py - Regression & Verification Tests for Scaled BC Pipeline.

Validates:
1. Dataset contract compliance for scaled demonstrations.
2. Checkpoint integrity and non-corruption of original checkpoints.
3. BehaviorCloningPolicyController inference correctness and latency budget (< 3.0 ms).
"""

import hashlib
from pathlib import Path
import time
import numpy as np
import pytest
import torch

from controllers.bc_policy import BehaviorCloningPolicyController
from core.contracts import VelocityAction


ORIGINAL_BC_SHA256 = "93c4d7d1d8476b8d288a24fca1862d87133696f78b9603aa622bcc367afad44f"


def test_original_checkpoint_hash_unmodified():
    """Confirms legacy bc_policy_best.pt was not overwritten or corrupted."""
    orig_path = Path("models/checkpoints/bc_policy_best.pt")
    assert orig_path.exists(), "Original checkpoint missing"
    with open(orig_path, "rb") as f:
        file_hash = hashlib.sha256(f.read()).hexdigest()
    assert file_hash == ORIGINAL_BC_SHA256, f"Original BC checkpoint hash modified: {file_hash}"


def test_scaled_demonstration_datasets_exist_and_valid():
    """Validates existence, keys, and dimensions of scaled train/val datasets."""
    train_path = Path("data/demonstrations/bc_train_scaled.npz")
    val_path = Path("data/demonstrations/bc_val_scaled.npz")

    assert train_path.exists(), "Train dataset missing"
    assert val_path.exists(), "Val dataset missing"

    train_data = np.load(train_path)
    val_data = np.load(val_path)

    for key in ["observations", "actions", "episode_starts", "episode_ids"]:
        assert key in train_data, f"Missing key {key} in train dataset"
        assert key in val_data, f"Missing key {key} in val dataset"

    # Verify shapes
    assert train_data["observations"].shape[1] == 23
    assert train_data["actions"].shape[1] == 4
    assert len(train_data["observations"]) >= 100000
    assert len(val_data["observations"]) >= 25000


def test_scaled_bc_policy_inference_and_latency():
    """Validates loading, output contracts, and latency of bc_policy_scaled_v1.pt."""
    scaled_ckpt = "models/checkpoints/bc_policy_scaled_v1.pt"
    assert Path(scaled_ckpt).exists(), "Scaled BC checkpoint missing"

    ctrl = BehaviorCloningPolicyController(scaled_ckpt)
    assert ctrl.model is not None

    dummy_obs = np.zeros(23, dtype=np.float32)
    dummy_obs[0:12] = 0.5 # mid-range rangefinders
    dummy_obs[12] = 1.0   # goal forward

    act = ctrl.predict(dummy_obs)
    assert isinstance(act, VelocityAction)
    assert -0.2 <= act.vx <= 0.8
    assert -0.6 <= act.vy <= 0.6
    assert -0.4 <= act.vz <= 0.4
    assert -0.6 <= act.yaw_rate <= 0.6

    # Real-time latency benchmark (< 3.0 ms)
    latencies = []
    for _ in range(200):
        t0 = time.perf_counter()
        _ = ctrl.predict(dummy_obs)
        latencies.append((time.perf_counter() - t0) * 1000.0)

    p99 = np.percentile(latencies, 99)
    assert p99 < 3.0, f"P99 inference latency exceeded 3.0 ms budget: {p99:.3f} ms"

