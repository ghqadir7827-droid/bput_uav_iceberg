"""train_curriculum_pilot.py - Phase 4K Isolated Curriculum Training Runner.

Trains the BC curriculum candidate on augmented near-obstacle recovery demonstrations:
- Inputs: 23-float deployable observations
- Loss: Mean Squared Error (MSE)
- Optimizer: AdamW, CosineAnnealingLR
- Checkpoint: experiments/phase4k/.../bc_curriculum_candidate.pt
- Outputs: training_metrics.csv
"""

import csv
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


def train_candidate():
    exp_dir = Path(__file__).resolve().parent
    data_dir = exp_dir / "data"
    train_npz = data_dir / "bc_curriculum_train.npz"
    val_npz = data_dir / "bc_curriculum_val.npz"
    candidate_ckpt_path = exp_dir / "bc_curriculum_candidate.pt"
    metrics_csv = exp_dir / "training_metrics.csv"

    epochs = 15
    batch_size = 128
    lr = 5e-4
    weight_decay = 1e-4

    print("=" * 80)
    print(">>> PHASE 4K EXPERIMENT D: ISOLATED CURRICULUM TRAINING PILOT <<<")
    print(f"Train Dataset:    {train_npz}")
    print(f"Val Dataset:      {val_npz}")
    print(f"Epochs:           {epochs}")
    print(f"Batch Size:       {batch_size}")
    print(f"Learning Rate:    {lr}")
    print("=" * 80)

    device = torch.device("cpu")

    train_ds = ExpertDemonstrationDataset(str(train_npz))
    val_ds = ExpertDemonstrationDataset(str(val_npz))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    print(f"Train Transitions: {len(train_ds):,} across {len(train_loader)} batches")
    print(f"Val Transitions:   {len(val_ds):,} across {len(val_loader)} batches")

    # Initialize from scratch to evaluate purely what the curriculum induces
    model = NavigationPolicyMLP(obs_dim=23, act_dim=4).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_val_loss = float("inf")
    history: List[Dict] = []

    print(f"\n{'Epoch':<6} | {'Train MSE Loss':<16} | {'Val MSE Loss':<14} | {'Val MAE (m/s)':<14} | {'Time (s)':<10}")
    print("-" * 68)

    t_train_start = time.time()

    for epoch in range(1, epochs + 1):
        t_epoch_start = time.time()

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
            }, candidate_ckpt_path)

        flag = " *" if is_best else ""
        print(f"{epoch:<6} | {epoch_train_loss:<16.6f} | {epoch_val_loss:<14.6f} | {epoch_val_mae:<14.4f} | {epoch_time:<6.2f} s{flag}")

        history.append({
            "epoch": epoch,
            "train_loss": round(epoch_train_loss, 6),
            "val_loss": round(epoch_val_loss, 6),
            "val_mae": round(epoch_val_mae, 4),
            "duration_s": round(epoch_time, 2),
            "is_best": is_best,
        })

    total_train_time = time.time() - t_train_start

    # Save training_metrics.csv
    with open(metrics_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss", "val_mae", "duration_s", "is_best"])
        writer.writeheader()
        for h in history:
            writer.writerow(h)

    print("\n[OK] Saved training metrics to:", metrics_csv)
    print(f"[OK] Saved candidate checkpoint to: {candidate_ckpt_path}")
    print(f"Total training time: {total_train_time:.1f} s")


if __name__ == "__main__":
    train_candidate()

