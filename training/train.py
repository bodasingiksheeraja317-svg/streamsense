"""
train.py
Project STREAMSENSE — Track A
MPIC v1.0 — Full training loop with validation, checkpointing, and logging.

Designed to run on Google Colab T4 GPU.
Falls back to CPU automatically if CUDA is not available.

Inputs:
    data/splits/train_files.txt      via dataset.py
    data/splits/val_files.txt        via dataset.py
    training/model.py                architecture

Outputs:
    checkpoints/best_model.pth       best checkpoint by val accuracy
    checkpoints/training_log.csv     epoch | train_loss | val_loss | val_acc | lr

SpecAugment is applied here (on tensors, after mel_pipeline).
Time-domain augmentations are applied inside dataset.py.

Usage (Colab):
    python train.py
    python train.py --epochs 40 --batch 64 --lr 0.001
"""

import sys
import csv
import time
import random
import argparse
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

# ── Local imports ─────────────────────────────────────────────────────────────
try:
    from model   import StreamSenseNet
    from dataset import get_dataloader
except ImportError as e:
    print(f"[ERROR] Import failed: {e}")
    print("        Ensure model.py and dataset.py are in the same directory.")
    sys.exit(1)

# ── Paths ─────────────────────────────────────────────────────────────────────
# These work on both Windows (local) and Colab (Linux).
# On Colab, mount your Drive and adjust BASE_DIR to match your mount point,
# or just run from the cloned repo root — relative paths will resolve.

BASE_DIR     = Path(__file__).resolve().parent.parent   # C:\STREAMSENSE or /content/STREAMSENSE
SPLITS_DIR   = BASE_DIR / "data"  / "splits"
CKPT_DIR     = BASE_DIR / "checkpoints"

TRAIN_SPLIT  = SPLITS_DIR / "train_files.txt"
VAL_SPLIT    = SPLITS_DIR / "val_files.txt"
BEST_CKPT    = CKPT_DIR   / "best_model.pth"
TRAIN_LOG    = CKPT_DIR   / "training_log.csv"

# ── Fixed hyperparameters ─────────────────────────────────────────────────────
SEED         = 42
NUM_CLASSES  = 10
NUM_WORKERS  = 2      # set to 0 if Colab multiprocessing issues

# ── SpecAugment parameters ────────────────────────────────────────────────────
# Applied to tensors [B, 1, 64, 97] during training forward pass.
# Frequency masking: mask up to F mel bins
# Time masking:      mask up to T time frames
FREQ_MASK_F  = 8     # max mel bins to mask  (out of 64)
TIME_MASK_T  = 15    # max time frames to mask (out of 97)
N_FREQ_MASKS = 1     # number of frequency masks per sample
N_TIME_MASKS = 1     # number of time masks per sample


# ── Reproducibility ───────────────────────────────────────────────────────────

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ── SpecAugment ───────────────────────────────────────────────────────────────

def spec_augment(
    x          : torch.Tensor,
    freq_mask_f: int = FREQ_MASK_F,
    time_mask_t: int = TIME_MASK_T,
    n_freq     : int = N_FREQ_MASKS,
    n_time     : int = N_TIME_MASKS,
) -> torch.Tensor:
    """
    Apply SpecAugment to a batch of mel spectrograms in-place.

    Args:
        x : Tensor [B, 1, 64, 97]  — normalized mel spectrogram batch.
            Modified in-place; a clone is made first to avoid mutating
            the original dataloader output.

    Returns:
        Augmented tensor [B, 1, 64, 97].

    Each sample in the batch gets independently sampled masks.
    Masked regions are filled with 0.0 (mean of normalized data ≈ 0).
    """
    x = x.clone()
    B, C, F, T = x.shape    # B, 1, 64, 97

    for b in range(B):
        # Frequency masks — mask mel bins
        for _ in range(n_freq):
            f  = random.randint(0, freq_mask_f)
            f0 = random.randint(0, max(F - f, 0))
            x[b, :, f0 : f0 + f, :] = 0.0

        # Time masks — mask time frames
        for _ in range(n_time):
            t  = random.randint(0, time_mask_t)
            t0 = random.randint(0, max(T - t, 0))
            x[b, :, :, t0 : t0 + t] = 0.0

    return x


# ── Training epoch ────────────────────────────────────────────────────────────

def train_one_epoch(
    model     : nn.Module,
    loader    : torch.utils.data.DataLoader,
    criterion : nn.Module,
    optimizer : torch.optim.Optimizer,
    device    : torch.device,
    epoch     : int,
) -> float:
    """
    Run one full training epoch.

    Returns:
        mean training loss over all batches.
    """
    model.train()
    total_loss = 0.0
    n_batches  = len(loader)

    for batch_idx, (tensors, labels) in enumerate(loader):
        # tensors: [B, 1, 1, 64, 97] — note extra dim from dataset collation
        # squeeze the redundant dim-1 to get [B, 1, 64, 97]
        x = tensors.squeeze(1).to(device)      # [B, 1, 64, 97]
        y = labels.to(device)                  # [B]

        # SpecAugment on the batch (training only)
        x = spec_augment(x)

        optimizer.zero_grad()
        logits = model(x)                      # [B, 10]
        loss   = criterion(logits, y)
        loss.backward()

        # Gradient clipping — prevents occasional large gradient spikes
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        total_loss += loss.item()

        # Progress print every 200 batches
        if (batch_idx + 1) % 200 == 0 or (batch_idx + 1) == n_batches:
            print(
                f"  Epoch {epoch:>3}  "
                f"[{batch_idx+1:>4}/{n_batches}]  "
                f"loss={loss.item():.4f}",
                flush=True,
            )

    return total_loss / n_batches


# ── Validation epoch ──────────────────────────────────────────────────────────

def validate(
    model    : nn.Module,
    loader   : torch.utils.data.DataLoader,
    criterion: nn.Module,
    device   : torch.device,
) -> tuple[float, float]:
    """
    Run one full validation pass.

    Returns:
        (mean_val_loss, val_accuracy_percent)
    """
    model.eval()
    total_loss = 0.0
    correct    = 0
    total      = 0

    with torch.no_grad():
        for tensors, labels in loader:
            x = tensors.squeeze(1).to(device)  # [B, 1, 64, 97]
            y = labels.to(device)

            logits = model(x)                  # [B, 10]
            loss   = criterion(logits, y)

            total_loss += loss.item()
            preds       = logits.argmax(dim=1)
            correct    += (preds == y).sum().item()
            total      += y.size(0)

    mean_loss = total_loss / len(loader)
    accuracy  = 100.0 * correct / total
    return mean_loss, accuracy


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def save_checkpoint(model: nn.Module, epoch: int, val_acc: float, path: Path):
    torch.save(
        {
            "epoch"        : epoch,
            "val_accuracy" : val_acc,
            "model_state"  : model.state_dict(),
            "num_classes"  : NUM_CLASSES,
            "mpic_version" : "1.0",
        },
        path,
    )


def load_checkpoint(model: nn.Module, path: Path) -> tuple[int, float]:
    """Load checkpoint into model. Returns (epoch, val_acc)."""
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    return ckpt["epoch"], ckpt["val_accuracy"]


# ── CSV logger ────────────────────────────────────────────────────────────────

def init_csv(path: Path):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "val_acc", "lr"])


def append_csv(path: Path, epoch: int, train_loss: float,
               val_loss: float, val_acc: float, lr: float):
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([epoch, f"{train_loss:.6f}", f"{val_loss:.6f}",
                         f"{val_acc:.4f}", f"{lr:.8f}"])


# ── Main training loop ────────────────────────────────────────────────────────

def train(args):
    # ── Seed ──────────────────────────────────────────────────────────────────
    set_seed(SEED)

    # ── Device ────────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 60)
    print("STREAMSENSE — train.py")
    print("=" * 60)
    print(f"\nDevice       : {device}")
    if device.type == "cuda":
        print(f"GPU          : {torch.cuda.get_device_name(0)}")
        print(f"VRAM         : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"Epochs       : {args.epochs}")
    print(f"Batch size   : {args.batch}")
    print(f"Learning rate: {args.lr}")
    print(f"Seed         : {SEED}")

    # ── Verify split files ────────────────────────────────────────────────────
    for p, name in [(TRAIN_SPLIT, "train_files.txt"), (VAL_SPLIT, "val_files.txt")]:
        if not p.exists():
            print(f"[ERROR] Split file not found: {p}")
            print("        On Colab: ensure the repo is cloned and data/splits/ exists.")
            sys.exit(1)

    # ── DataLoaders ───────────────────────────────────────────────────────────
    print(f"\nLoading datasets...")
    train_loader = get_dataloader(
        TRAIN_SPLIT,
        is_train    = True,
        batch_size  = args.batch,
        num_workers = NUM_WORKERS,
        verbose     = True,
    )
    val_loader = get_dataloader(
        VAL_SPLIT,
        is_train    = False,
        batch_size  = args.batch,
        num_workers = NUM_WORKERS,
        verbose     = True,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = StreamSenseNet(num_classes=NUM_CLASSES).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel        : StreamSenseNet")
    print(f"Parameters   : {total_params:,}")

    # ── Loss, optimizer, scheduler ────────────────────────────────────────────
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode     = "min",       # monitor val_loss
        factor   = 0.5,         # halve LR on plateau
        patience = 3,           # wait 3 epochs before reducing
        min_lr   = 1e-6,
        verbose  = True,
    )

    # ── Checkpoint dir + CSV ──────────────────────────────────────────────────
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    init_csv(TRAIN_LOG)

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_acc  = 0.0
    best_epoch    = 0
    epochs_no_imp = 0
    EARLY_STOP    = 8       # stop if val_acc doesn't improve for 8 epochs

    print(f"\n{'─'*60}")
    print(f"Starting training — max {args.epochs} epochs")
    print(f"Early stopping patience: {EARLY_STOP} epochs")
    print(f"Best checkpoint → {BEST_CKPT}")
    print(f"{'─'*60}\n")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # Train
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )

        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        # Scheduler step (on val_loss)
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        elapsed = time.time() - t0

        # ── Epoch summary ─────────────────────────────────────────────────────
        improved = val_acc > best_val_acc
        marker   = " ← best" if improved else ""
        print(
            f"Epoch {epoch:>3}/{args.epochs}  "
            f"train_loss={train_loss:.4f}  "
            f"val_loss={val_loss:.4f}  "
            f"val_acc={val_acc:.2f}%  "
            f"lr={current_lr:.2e}  "
            f"t={elapsed:.1f}s"
            f"{marker}",
            flush=True,
        )

        # ── Save best checkpoint ──────────────────────────────────────────────
        if improved:
            best_val_acc  = val_acc
            best_epoch    = epoch
            epochs_no_imp = 0
            save_checkpoint(model, epoch, val_acc, BEST_CKPT)
            print(f"  [SAVED] best_model.pth  (val_acc={val_acc:.2f}%)")
        else:
            epochs_no_imp += 1

        # ── Log to CSV ────────────────────────────────────────────────────────
        append_csv(TRAIN_LOG, epoch, train_loss, val_loss, val_acc, current_lr)

        # ── Early stopping ────────────────────────────────────────────────────
        if epochs_no_imp >= EARLY_STOP:
            print(
                f"\n[EARLY STOP] No improvement for {EARLY_STOP} epochs. "
                f"Best was epoch {best_epoch} ({best_val_acc:.2f}%)."
            )
            break

    # ── Final summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"  Best epoch     : {best_epoch}")
    print(f"  Best val acc   : {best_val_acc:.2f}%")
    print(f"  Checkpoint     : {BEST_CKPT}")
    print(f"  Training log   : {TRAIN_LOG}")
    print(f"\nNext step: python evaluate.py")


# ── Argument parser ───────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="STREAMSENSE train.py")
    parser.add_argument("--epochs", type=int,   default=30,    help="Max training epochs (default 30)")
    parser.add_argument("--batch",  type=int,   default=32,    help="Batch size (default 32)")
    parser.add_argument("--lr",     type=float, default=0.001, help="Initial learning rate (default 0.001)")
    return parser.parse_args()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = parse_args()
    train(args)
