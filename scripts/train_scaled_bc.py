"""train_scaled_bc.py - Supervised Imitation Learning on Scaled Demonstrations.

Trains NavigationPolicyMLP on the verified 65cm UAV scaled demonstrations:
- Train Dataset: data/demonstrations/bc_train_scaled.npz (200 episodes, 104,799 transitions)
- Val Dataset: data/demonstrations/bc_val_scaled.npz (50 episodes, 26,067 transitions)
- Network: NavigationPolicyMLP (23-dim deployable obs -> 4-dim bounded velocity action)
- Saves best checkpoint to models/checkpoints/bc_policy_scaled_v1.pt (PRESERVING legacy bc_policy_best.pt)
- Tracks overall MSE loss plus action component MSE losses (vx, vy, vz, yaw_rate)
- Benchmarks single-sample CPU inference latency
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


def train_scaled_bc(
    train_npz: str = "data/demonstrations/bc_train_scaled.npz",
    val_npz: str = "data/demonstrations/bc_val_scaled.npz",
    checkpoint_path: str = "models/checkpoints/bc_policy_scaled_v1.pt",
    log_path: str = "models/checkpoints/bc_scaled_training_log.json",
    epochs: int = 30,
    batch_size: int = 128,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device_name: str = "auto",
) -> Dict:
    print("=" * 80)
    print(">>> BEHAVIOR CLONING (BC) SUPERVISED TRAINING ON SCALED ENVIRONMENT <<<")
    print(f"Train Dataset:    {train_npz}")
    print(f"Val Dataset:      {val_npz}")
    print(f"Epochs:           {epochs}")
    print(f"Batch Size:       {batch_size}")
    print(f"Learning Rate:    {lr}")
    print(f"Target Checkpoint:{checkpoint_path}")
    print("=" * 80)

    # Device selection
    if device_name == "auto":
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    else:
        device = torch.device(device_name)
    print(f"Compute Device:   {device}")

    train_ds = ExpertDemonstrationDataset(train_npz)
    val_ds = ExpertDemonstrationDataset(val_npz)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    print(f"Train Transitions: {len(train_ds):,} across {len(train_loader)} batches")
    print(f"Val Transitions:   {len(val_ds):,} across {len(val_loader)} batches")

    # Policy Network
    model = NavigationPolicyMLP(obs_dim=23, act_dim=4).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model Parameters: {total_params:,} (MLP: [23 -> 128 -> 128 -> 4])")

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    ckpt_file = Path(checkpoint_path)
    ckpt_file.parent.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")

    history: List[Dict] = []
    print(f"\n{'Epoch':<6} | {'Train MSE':<12} | {'Val MSE':<12} | {'Val MAE':<10} | {'MSE vx':<10} | {'MSE vy':<10} | {'MSE vz':<10} | {'MSE yaw':<10} | {'Time (s)':<8}")
    print("-" * 98)

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

        # Validation Phase with Action Component Tracking
        model.eval()
        val_loss_sum = 0.0
        val_mae_sum = 0.0
        comp_loss_sums = np.zeros(4, dtype=np.float64)
        val_samples = 0

        with torch.no_grad():
            for b_obs, b_acts in val_loader:
                b_obs = b_obs.to(device)
                b_acts = b_acts.to(device)

                pred_acts = model(b_obs, bounded=True)
                val_loss = criterion(pred_acts, b_acts)
                val_mae = torch.mean(torch.abs(pred_acts - b_acts))

                # Component losses: (pred - target)^2 per channel
                comp_err = torch.mean((pred_acts - b_acts) ** 2, dim=0).cpu().numpy()
                comp_loss_sums += comp_err * len(b_obs)

                val_loss_sum += float(val_loss.item()) * len(b_obs)
                val_mae_sum += float(val_mae.item()) * len(b_obs)
                val_samples += len(b_obs)

        epoch_val_loss = val_loss_sum / max(1, val_samples)
        epoch_val_mae = val_mae_sum / max(1, val_samples)
        comp_losses = comp_loss_sums / max(1, val_samples)
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
                "component_losses": {
                    "mse_vx": float(comp_losses[0]),
                    "mse_vy": float(comp_losses[1]),
                    "mse_vz": float(comp_losses[2]),
                    "mse_yaw": float(comp_losses[3]),
                },
                "obs_dim": 23,
                "act_dim": 4,
                "total_params": total_params,
                "drone_radius_m": 0.325,
                "timestamp_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }, ckpt_file)

        flag = " *" if is_best else ""
        print(f"{epoch:<6} | {epoch_train_loss:<12.6f} | {epoch_val_loss:<12.6f} | {epoch_val_mae:<10.4f} | {comp_losses[0]:<10.6f} | {comp_losses[1]:<10.6f} | {comp_losses[2]:<10.6f} | {comp_losses[3]:<10.6f} | {epoch_time:<8.2f}{flag}")

        history.append({
            "epoch": epoch,
            "train_loss": round(epoch_train_loss, 6),
            "val_loss": round(epoch_val_loss, 6),
            "val_mae": round(epoch_val_mae, 4),
            "mse_vx": round(float(comp_losses[0]), 6),
            "mse_vy": round(float(comp_losses[1]), 6),
            "mse_vz": round(float(comp_losses[2]), 6),
            "mse_yaw": round(float(comp_losses[3]), 6),
            "is_best": is_best,
        })

    total_train_time = time.time() - t_train_start

    # Checkpoint SHA-256
    with open(ckpt_file, "rb") as f:
        ckpt_sha256 = hashlib.sha256(f.read()).hexdigest()

    # Latency benchmark on CPU (standard deployable target)
    model.to("cpu")
    model.eval()
    dummy_input = torch.randn(1, 23, device="cpu")
    for _ in range(50):
        _ = model(dummy_input)

    latencies = []
    for _ in range(1000):
        t0 = time.perf_counter()
        _ = model(dummy_input)
        latencies.append((time.perf_counter() - t0) * 1000.0)

    mean_lat_ms = float(np.mean(latencies))
    p99_lat_ms = float(np.percentile(latencies, 99))

    print("\n" + "=" * 80)
    print(">>> SCALED BEHAVIOR CLONING TRAINING COMPLETE <<<")
    print(f"Total Training Time:          {total_train_time:.1f} s")
    print(f"Best Validation MSE Loss:     {best_val_loss:.6f}")
    print(f"Best Checkpoint Saved:        {ckpt_file.resolve()}")
    print(f"Checkpoint SHA-256:           {ckpt_sha256}")
    print(f"Single-Sample Latency (Mean): {mean_lat_ms:.3f} ms (Target: < 3.0 ms)")
    print(f"Single-Sample Latency (P99):  {p99_lat_ms:.3f} ms")
    print("=" * 80)

    log_file = Path(log_path)
    log_data = {
        "status": "TRAINING_COMPLETE",
        "checkpoint_path": str(ckpt_file),
        "checkpoint_sha256": ckpt_sha256,
        "best_epoch": next(h["epoch"] for h in reversed(history) if h.get("is_best")),
        "best_val_loss": round(best_val_loss, 6),
        "total_epochs": epochs,
        "total_train_time_s": round(total_train_time, 2),
        "inference_latency_ms": {
            "mean": round(mean_lat_ms, 3),
            "p99": round(p99_lat_ms, 3),
        },
        "history": history,
    }
    with open(log_file, "w") as f:
        json.dump(log_data, f, indent=2)

    return log_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Scaled Behavior Cloning Policy")
    parser.add_argument("--train-data", type=str, default="data/demonstrations/bc_train_scaled.npz")
    parser.add_argument("--val-data", type=str, default="data/demonstrations/bc_val_scaled.npz")
    parser.add_argument("--checkpoint", type=str, default="models/checkpoints/bc_policy_scaled_v1.pt")
    parser.add_argument("--log", type=str, default="models/checkpoints/bc_scaled_training_log.json")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    train_scaled_bc(
        train_npz=args.train_data,
        val_npz=args.val_data,
        checkpoint_path=args.checkpoint,
        log_path=args.log,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )

