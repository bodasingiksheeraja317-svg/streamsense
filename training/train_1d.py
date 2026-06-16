"""
train_1d.py
Project STREAMSENSE — Track A
Epic A3.2 — Training script for StreamSenseNet1D (1D CNN baseline, STRETCH)

Mirrors the training configuration of train.py (2D model) for a fair
comparison:
    - Seed=42
    - Adam, lr=0.001, weight_decay=1e-4
    - ReduceLROnPlateau (factor=0.5, patience=3, min_lr=1e-6)
    - Early stopping patience=8
    - Time-domain augmentation only (no SpecAugment — N/A for raw waveform)

Outputs:
    checkpoints_1d/best_model_1d.pth
    checkpoints_1d/training_log_1d.csv

Run from C:\\STREAMSENSE\\training\\:
    python train_1d.py
"""

import csv
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model_1d import StreamSenseNet1D, count_parameters
from dataset_1d import StreamSenseDataset1D

# ── Config (mirrors train.py) ─────────────────────────────────────────────────
SEED          = 42
BATCH_SIZE    = 64
MAX_EPOCHS    = 60
LR            = 0.001
WEIGHT_DECAY  = 1e-4
LR_FACTOR     = 0.5
LR_PATIENCE   = 3
LR_MIN        = 1e-6
EARLY_STOP_PATIENCE = 8

# ── Root path — environment-aware (see dataset_1d.py for Colab setup) ────────
_DEFAULT_ROOT = r"C:\STREAMSENSE" if os.name == "nt" else "/content/STREAMSENSE"
ROOT       = Path(os.environ.get("STREAMSENSE_ROOT", _DEFAULT_ROOT))
CKPT_DIR   = ROOT / "checkpoints_1d"
CKPT_PATH  = CKPT_DIR / "best_model_1d.pth"
LOG_PATH   = CKPT_DIR / "training_log_1d.csv"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_epoch(model, loader, criterion, optimizer=None):
    """Run one epoch. If optimizer is None, runs in eval mode (no grad)."""
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for waveforms, labels in loader:
            waveforms = waveforms.to(DEVICE)  # [B, 1, 16000]
            labels    = labels.to(DEVICE)

            if is_train:
                optimizer.zero_grad()

            logits = model(waveforms)  # [B, 10]
            loss   = criterion(logits, labels)

            if is_train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * waveforms.size(0)
            preds = logits.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += waveforms.size(0)

    avg_loss = total_loss / total_samples
    accuracy = 100.0 * total_correct / total_samples
    return avg_loss, accuracy


def main():
    print("=" * 60)
    print("STREAMSENSE — train_1d.py (Epic A3.2, 1D CNN baseline)")
    print("=" * 60)

    set_seed(SEED)
    print(f"Device: {DEVICE}")
    print(f"Seed:   {SEED}")

    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Data ───────────────────────────────────────────────────────────────
    print("\nLoading datasets...")
    train_ds = StreamSenseDataset1D(split="train", augment=True)
    val_ds   = StreamSenseDataset1D(split="val",   augment=False)

    print(f"  train: {len(train_ds)} samples")
    print(f"  val  : {len(val_ds)} samples")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                               num_workers=0, pin_memory=(DEVICE.type == "cuda"))
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                               num_workers=0, pin_memory=(DEVICE.type == "cuda"))

    # ── Model ──────────────────────────────────────────────────────────────
    model = StreamSenseNet1D(n_classes=10).to(DEVICE)
    n_params = count_parameters(model)
    print(f"\nModel: StreamSenseNet1D")
    print(f"  Parameters: {n_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=LR_FACTOR, patience=LR_PATIENCE, min_lr=LR_MIN
    )

    # ── Training loop ──────────────────────────────────────────────────────
    best_val_acc = 0.0
    best_epoch   = 0
    epochs_no_improve = 0

    log_rows = []

    print(f"\nTraining (max {MAX_EPOCHS} epochs, early stop patience={EARLY_STOP_PATIENCE})...")
    print(f"{'Epoch':>6} {'TrainLoss':>10} {'TrainAcc':>9} {'ValLoss':>9} {'ValAcc':>8} {'LR':>10} {'Time':>6}")

    for epoch in range(1, MAX_EPOCHS + 1):
        t0 = time.time()

        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc     = run_epoch(model, val_loader, criterion, optimizer=None)

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        elapsed = time.time() - t0

        print(f"{epoch:>6} {train_loss:>10.4f} {train_acc:>8.2f}% "
              f"{val_loss:>9.4f} {val_acc:>7.2f}% {current_lr:>10.2e} {elapsed:>5.1f}s")

        log_rows.append({
            "epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
            "val_loss": val_loss, "val_acc": val_acc, "lr": current_lr, "time_s": elapsed
        })

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            epochs_no_improve = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_acc": val_acc,
                "val_loss": val_loss,
            }, CKPT_PATH)
            print(f"         -> new best (val_acc={val_acc:.2f}%), saved to {CKPT_PATH.name}")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= EARLY_STOP_PATIENCE:
            print(f"\nEarly stopping at epoch {epoch} "
                  f"(no improvement for {EARLY_STOP_PATIENCE} epochs)")
            break

    # ── Save training log ─────────────────────────────────────────────────
    with open(LOG_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=log_rows[0].keys())
        writer.writeheader()
        writer.writerows(log_rows)

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)
    print(f"  Best epoch    : {best_epoch}")
    print(f"  Best val_acc  : {best_val_acc:.2f}%")
    print(f"  Checkpoint    -> {CKPT_PATH}")
    print(f"  Training log  -> {LOG_PATH}")


if __name__ == "__main__":
    main()
