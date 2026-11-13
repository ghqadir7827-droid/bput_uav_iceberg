"""train_bc_phase4h.py - Phase 4H Supervised Training for BC Candidate.

Trains NavigationPolicyMLP on the Phase 4H targeted expert demonstrations:
- Input: 23-float deployable student observation
- Output: 4-float continuous velocity setpoint [vx, vy, vz, yaw_rate]
- Loss: MSE loss between student and targeted expert actions
- Checkpoint target: experiments/phase4h/checkpoints/bc_phase4h_candidate.pt
- Strictly preserves canonical models/checkpoints/bc_policy_best.pt immutable.
"""

import argparse
import hashlib
import json
from pathlib import Path
import time
from typing import Dict, List
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.dataset import ExpertDemonstrationDataset
from models.policy_net import NavigationPolicyMLP


def sha256_file(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def train_bc_candidate(
    train_npz: str = "experiments/phase4h/data/bc_train_phase4h.npz",
    val_npz: str = "experiments/phase4h/data/bc_val_phase4h.npz",
    output_checkpoint: str = "experiments/phase4h/checkpoints/bc_phase4h_candidate.pt",
    epochs: int = 30,
    batch_size: int = 128,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
) -> Dict:
    print("=" * 80)
    print(">>> PHASE 4H BEHAVIOR CLONING (BC) CANDIDATE TRAINING <<<")
    print(f"Train Dataset:    {train_npz}")
    print(f"Val Dataset:      {val_npz}")
    print(f"Output Target:    {output_checkpoint}")
    print(f"Epochs:           {epochs}")
    print(f"Batch Size:       {batch_size}")
    print(f"Learning Rate:    {lr}")
    print("=" * 80)

    device = torch.device("cpu") # Deterministic CPU execution
    print(f"Compute Device:   {device}")

    train_ds = ExpertDemonstrationDataset(train_npz)
    val_ds = ExpertDemonstrationDataset(val_npz)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    print(f"Train Samples:    {len(train_ds):,} across {len(train_loader)} batches")
    print(f"Val Samples:      {len(val_ds):,} across {len(val_loader)} batches")

    model = NavigationPolicyMLP(obs_dim=23, act_dim=4).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model Parameters: {total_params:,}")

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    ckpt_path = Path(output_checkpoint)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")

    history: List[Dict] = []
    print(f"\n{'Epoch':<6} | {'Train MSE':<14} | {'Val MSE':<14} | {'Val MAE':<14} | {'Time (s)':<10}")
    print("-" * 65)

    t_train_start = time.time()

    for epoch in range(1, epochs + 1):
        t_epoch_start = time.time()

        model.train()
        train_loss_sum = 0.0
        train_count = 0

        for b_obs, b_acts in train_loader:
            b_obs = b_obs.to(device)
            b_acts = b_acts.to(device)

            pred_acts = model(b_obs, bounded=True)
            loss = criterion(pred_acts, b_acts)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss_sum += float(loss.item()) * len(b_obs)
            train_count += len(b_obs)

        scheduler.step()
        epoch_train_loss = train_loss_sum / max(1, train_count)

        model.eval()
        val_loss_sum = 0.0
        val_mae_sum = 0.0
        val_count = 0

        with torch.no_grad():
            for b_obs, b_acts in val_loader:
                b_obs = b_obs.to(device)
                b_acts = b_acts.to(device)

                pred_acts = model(b_obs, bounded=True)
                val_loss = criterion(pred_acts, b_acts)
                val_mae = torch.mean(torch.abs(pred_acts - b_acts))

                val_loss_sum += float(val_loss.item()) * len(b_obs)
                val_mae_sum += float(val_mae.item()) * len(b_obs)
                val_count += len(b_obs)

        epoch_val_loss = val_loss_sum / max(1, val_count)
        epoch_val_mae = val_mae_sum / max(1, val_count)
        epoch_time = time.time() - t_epoch_start

        is_best = epoch_val_loss < best_val_loss
        if is_best:
            best_val_loss = epoch_val_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": best_val_loss,
                "val_mae": epoch_val_mae,
                "obs_dim": 23,
                "act_dim": 4,
                "total_params": total_params,
                "timestamp_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }, ckpt_path)

        flag = " *" if is_best else ""
        print(f"{epoch:<6} | {epoch_train_loss:<14.6f} | {epoch_val_loss:<14.6f} | {epoch_val_mae:<14.4f} | {epoch_time:<6.2f} s{flag}")

        history.append({
            "epoch": epoch,
            "train_loss": round(epoch_train_loss, 6),
            "val_loss": round(epoch_val_loss, 6),
            "val_mae": round(epoch_val_mae, 4),
            "is_best": is_best,
        })

    total_train_time = time.time() - t_train_start

    # Benchmark CPU latency
    model.eval()
    dummy_input = torch.randn(1, 23, device=device)
    for _ in range(50):
        _ = model(dummy_input)
    latencies = []
    for _ in range(500):
        t0 = time.perf_counter()
        _ = model(dummy_input)
        latencies.append((time.perf_counter() - t0) * 1000.0)

    mean_latency_ms = float(np.mean(latencies))
    p99_latency_ms = float(np.percentile(latencies, 99))
    ckpt_hash = sha256_file(ckpt_path)

    print("\n" + "=" * 80)
    print(">>> PHASE 4H BC CANDIDATE TRAINING COMPLETE <<<")
    print(f"Total Training Time:          {total_train_time:.1f} s")
    print(f"Best Validation MSE Loss:     {best_val_loss:.6f}")
    print(f"Saved Checkpoint:             {ckpt_path.resolve()}")
    print(f"SHA-256 Checksum:             {ckpt_hash}")
    print(f"Single-Sample Latency (Mean): {mean_latency_ms:.3f} ms")
    print(f"Single-Sample Latency (P99):  {p99_latency_ms:.3f} ms")
    print("=" * 80)

    log_path = Path("experiments/phase4h/logs/bc_training_log.json")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_data = {
        "status": "TRAINING_COMPLETE",
        "best_epoch": next(h["epoch"] for h in history if h.get("is_best")),
        "best_val_loss": round(best_val_loss, 6),
        "checkpoint_path": str(ckpt_path),
        "checkpoint_sha256": ckpt_hash,
        "total_epochs": epochs,
        "total_train_time_s": round(total_train_time, 2),
        "inference_latency_ms": {
            "mean": round(mean_latency_ms, 3),
            "p99": round(p99_latency_ms, 3),
        },
        "history": history,
    }
    with open(log_path, "w") as f:
        json.dump(log_data, f, indent=2)

    return log_data


if __name__ == "__main__":
    train_bc_candidate()
