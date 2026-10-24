"""train_bc.py - Behavior Cloning Supervised Training Pipeline.

Trains the neural network policy on expert demonstrations via supervised imitation learning:
- Loads verified demonstration transitions (train and validation splits).
- Minimizes Mean Squared Error (MSE) between student and expert actions.
- Tracks train and validation loss per epoch.
- Saves best checkpoint to models/checkpoints/bc_policy_best.pt.
- Measures and reports single-sample inference latency.
"""

import argparse
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


def train_behavior_cloning(
    train_npz: str = "data/demonstrations/bc_train_10m.npz",
    val_npz: str = "data/demonstrations/bc_val_10m.npz",
    checkpoint_dir: str = "models/checkpoints",
    epochs: int = 25,
    batch_size: int = 128,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device_name: str = "auto",
) -> Dict:
    print("=" * 80)
    print(">>> BEHAVIOR CLONING (BC) SUPERVISED TRAINING PIPELINE <<<")
    print(f"Train Dataset:    {train_npz}")
    print(f"Val Dataset:      {val_npz}")
    print(f"Epochs:           {epochs}")
    print(f"Batch Size:       {batch_size}")
    print(f"Learning Rate:    {lr}")
    print("=" * 80)

    # Device selection
    if device_name == "auto":
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    else:
        device = torch.device(device_name)
    print(f"Compute Device:   {device}")

    # Datasets and Loaders
    train_ds = ExpertDemonstrationDataset(train_npz)
    val_ds = ExpertDemonstrationDataset(val_npz)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    print(f"Train Transitions: {len(train_ds):,} across {len(train_loader)} batches")
    print(f"Val Transitions:   {len(val_ds):,} across {len(val_loader)} batches")

    # Policy Network
    model = NavigationPolicyMLP(obs_dim=23, act_dim=4).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model Parameters: {total_params:,} (Lightweight MLP)")

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")
    best_checkpoint_path = ckpt_dir / "bc_policy_best.pt"

    history: List[Dict] = []
    print(f"\n{'Epoch':<6} | {'Train MSE Loss':<16} | {'Val MSE Loss':<14} | {'Val MAE (m/s)':<14} | {'Time (s)':<10}")
    print("-" * 68)

    t_train_start = time.time()

    for epoch in range(1, epochs + 1):
        t_epoch_start = time.time()

        # Training Phase
        model.train()
        train_loss_sum = 0.0
        train_samples = 0

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
            train_samples += len(b_obs)

        scheduler.step()
        epoch_train_loss = train_loss_sum / max(1, train_samples)

        # Validation Phase
        model.eval()
        val_loss_sum = 0.0
        val_mae_sum = 0.0
        val_samples = 0

        with torch.no_grad():
            for b_obs, b_acts in val_loader:
                b_obs = b_obs.to(device)
                b_acts = b_acts.to(device)

                pred_acts = model(b_obs, bounded=True)
                val_loss = criterion(pred_acts, b_acts)
                val_mae = torch.mean(torch.abs(pred_acts - b_acts))

                val_loss_sum += float(val_loss.item()) * len(b_obs)
                val_mae_sum += float(val_mae.item()) * len(b_obs)
                val_samples += len(b_obs)

        epoch_val_loss = val_loss_sum / max(1, val_samples)
        epoch_val_mae = val_mae_sum / max(1, val_samples)
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
            }, best_checkpoint_path)

        flag = " *" if is_best else ""
        print(f"{epoch:<6} | {epoch_train_loss:<16.6f} | {epoch_val_loss:<14.6f} | {epoch_val_mae:<14.4f} | {epoch_time:<6.2f} s{flag}")

        history.append({
            "epoch": epoch,
            "train_loss": round(epoch_train_loss, 6),
            "val_loss": round(epoch_val_loss, 6),
            "val_mae": round(epoch_val_mae, 4),
            "is_best": is_best,
        })

    total_train_time = time.time() - t_train_start

    # Benchmark CPU Inference Latency (Single-sample forward pass)
    model.eval()
    dummy_input = torch.randn(1, 23, device=device)
    # Warmup
    for _ in range(50):
        _ = model(dummy_input)

    latencies = []
    for _ in range(500):
        t0 = time.perf_counter()
        _ = model(dummy_input)
        latencies.append((time.perf_counter() - t0) * 1000.0) # in ms

    mean_latency_ms = float(np.mean(latencies))
    p99_latency_ms = float(np.percentile(latencies, 99))

    print("\n" + "=" * 80)
    print(">>> BEHAVIOR CLONING TRAINING COMPLETE <<<")
    print(f"Total Training Time:          {total_train_time:.1f} s")
    print(f"Best Validation MSE Loss:     {best_val_loss:.6f}")
    print(f"Best Checkpoint Saved:        {best_checkpoint_path.resolve()}")
    print(f"Single-Sample Latency (Mean): {mean_latency_ms:.3f} ms (Target: < 3.0 ms)")
    print(f"Single-Sample Latency (P99):  {p99_latency_ms:.3f} ms")
    print("=" * 80)

    # Save Training Log
    log_path = ckpt_dir / "bc_training_log.json"
    log_data = {
        "status": "TRAINING_COMPLETE",
        "best_epoch": next(h["epoch"] for h in history if h.get("is_best")),
        "best_val_loss": round(best_val_loss, 6),
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
    parser = argparse.ArgumentParser(description="Behavior Cloning Training Pipeline")
    parser.add_argument("--train-data", type=str, default="data/demonstrations/bc_train_10m.npz", help="Path to train .npz archive")
    parser.add_argument("--val-data", type=str, default="data/demonstrations/bc_val_10m.npz", help="Path to val .npz archive")
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    args = parser.parse_args()

    train_behavior_cloning(
        train_npz=args.train_data,
        val_npz=args.val_data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )
