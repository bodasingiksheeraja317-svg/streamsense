"""
train_source_count.py
Project STREAMSENSE — WA-3 Source Counting

Full training loop for SourceCountCNN: fine-tunes on top of the frozen
StreamSenseNet backbone (checkpoints/best_model.pth), reports per-class
(per-N) validation accuracy each epoch, and saves the best checkpoint by
overall validation accuracy.

Run:
    python training/train_source_count.py \
        --train_csv  data/source_count_splits/source_count_train.csv \
        --val_csv    data/source_count_splits/source_count_val.csv \
        --backbone   checkpoints/best_model.pth \
        --best_path  checkpoints/best_source_counter.pth \
        --epochs 30 --batch_size 64 --lr 1e-4 --wd 1e-4 --num_workers 2 --seed 42
"""

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from dataset_source_count import SourceCountDataset
from model_source_count import SourceCountCNN

DEFAULTS = dict(
    train_csv="data/source_count_splits/source_count_train.csv",
    val_csv="data/source_count_splits/source_count_val.csv",
    backbone="checkpoints/best_model.pth",
    best_path="checkpoints/best_source_counter.pth",
    epochs=30,
    batch_size=64,
    lr=1e-4,
    wd=1e-4,
    seed=42,
    num_workers=2,
    n_classes=8,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser(description="Train SourceCountCNN.")
    parser.add_argument("--train_csv", type=str, default=DEFAULTS["train_csv"])
    parser.add_argument("--val_csv", type=str, default=DEFAULTS["val_csv"])
    parser.add_argument("--backbone", type=str, default=DEFAULTS["backbone"])
    parser.add_argument("--best_path", type=str, default=DEFAULTS["best_path"])
    parser.add_argument("--epochs", type=int, default=DEFAULTS["epochs"])
    parser.add_argument("--batch_size", type=int, default=DEFAULTS["batch_size"])
    parser.add_argument("--lr", type=float, default=DEFAULTS["lr"])
    parser.add_argument("--wd", type=float, default=DEFAULTS["wd"])
    parser.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    parser.add_argument("--num_workers", type=int, default=DEFAULTS["num_workers"])
    parser.add_argument("--n_classes", type=int, default=DEFAULTS["n_classes"])
    return parser.parse_args()


def evaluate(model, val_loader, device, n_classes: int):
    model.eval()
    total_loss = 0.0
    correct_per_class = np.zeros(n_classes, dtype=np.int64)
    total_per_class = np.zeros(n_classes, dtype=np.int64)
    n_correct, n_total = 0, 0

    with torch.no_grad():
        for mel, label in val_loader:
            mel, label = mel.to(device), label.to(device)
            logits = model(mel)
            loss = F.cross_entropy(logits, label)
            total_loss += loss.item() * mel.size(0)

            preds = logits.argmax(dim=1)
            n_correct += (preds == label).sum().item()
            n_total += label.size(0)

            for c in range(n_classes):
                mask = label == c
                total_per_class[c] += mask.sum().item()
                correct_per_class[c] += (preds[mask] == c).sum().item()

    val_loss = total_loss / max(n_total, 1)
    overall_acc = n_correct / max(n_total, 1)
    per_class_acc = np.divide(
        correct_per_class, total_per_class,
        out=np.zeros_like(correct_per_class, dtype=np.float64),
        where=total_per_class > 0,
    )
    return val_loss, overall_acc, per_class_acc


def train():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    training_dir = Path(__file__).resolve().parent
    if str(training_dir) not in sys.path:
        sys.path.insert(0, str(training_dir))

    # ── Data ─────────────────────────────────────────────────────────────────
    train_ds = SourceCountDataset(args.train_csv, augment=True)
    val_ds = SourceCountDataset(args.val_csv, augment=False)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )
    print(f"[INFO] Train samples: {len(train_ds)}   Val samples: {len(val_ds)}")

    # ── Model ────────────────────────────────────────────────────────────────
    model = SourceCountCNN(n_classes=args.n_classes).to(device)
    model.load_backbone(args.backbone)

    # ── Optimiser ────────────────────────────────────────────────────────────
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    best_val_acc = 0.0
    best_path = Path(args.best_path)
    best_path.parent.mkdir(parents=True, exist_ok=True)

    header_cols = "  ".join(f"N{n+1}" for n in range(args.n_classes))
    print(f"\n{'Epoch':>5}  {'Loss':>7}  {'ValAcc':>7}  {header_cols}  {'Time':>6}")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        running_loss = 0.0
        n_seen = 0

        for mel, label in train_loader:
            mel, label = mel.to(device), label.to(device)
            optimizer.zero_grad()
            logits = model(mel)
            loss = F.cross_entropy(logits, label)
            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()

            running_loss += loss.item() * mel.size(0)
            n_seen += mel.size(0)

        scheduler.step()
        train_loss = running_loss / max(n_seen, 1)

        val_loss, overall_acc, per_class_acc = evaluate(model, val_loader, device, args.n_classes)
        elapsed = time.time() - t0

        per_class_str = "  ".join(f"{acc:.2f}" for acc in per_class_acc)
        marker = ""
        if overall_acc > best_val_acc:
            best_val_acc = overall_acc
            marker = " <- best"
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "val_acc": overall_acc,
                "n_classes": args.n_classes,
            }, best_path)

        print(
            f"{epoch:>5}  {train_loss:>7.4f}  {overall_acc:>7.4f}  {per_class_str}  {elapsed:>5.1f}s{marker}"
        )

    print("\n" + "=" * 60)
    print(f"Best val accuracy: {best_val_acc:.4f}")
    final_per_class_str = "  ".join(f"N{n+1}={acc:.2f}" for n, acc in enumerate(per_class_acc))
    print(f"Per class: {final_per_class_str}")
    print(f"Saved to: {best_path}")
    print("=" * 60)


if __name__ == "__main__":
    train()
