"""
evaluate.py
Project STREAMSENSE — Track A
MPIC v1.0 — Final test set evaluation.

RUN ONCE ONLY after model is accepted from training.

Inputs:
    checkpoints/best_model.pth
    data/splits/test_files.txt      via dataset.py

Outputs:
    evaluation/evaluation_report.txt
    evaluation/confusion_matrix.png

Usage:
    python evaluate.py                         (local CPU or Colab)
    python evaluate.py --ckpt path/to/model.pth
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — works on Colab and headless
import matplotlib.pyplot as plt

try:
    from sklearn.metrics import (
        classification_report,
        confusion_matrix,
        accuracy_score,
    )
except ImportError:
    print("[ERROR] scikit-learn not installed.")
    print("        Run: pip install scikit-learn")
    sys.exit(1)

try:
    from model   import StreamSenseNet
    from dataset import get_dataloader
except ImportError as e:
    print(f"[ERROR] Import failed: {e}")
    sys.exit(1)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).resolve().parent.parent
SPLITS_DIR  = BASE_DIR / "data"  / "splits"
CKPT_DIR    = BASE_DIR / "checkpoints"
EVAL_DIR    = BASE_DIR / "evaluation"
LABELS_PATH = BASE_DIR / "class_labels.json"

TEST_SPLIT  = SPLITS_DIR / "test_files.txt"
DEFAULT_CKPT= CKPT_DIR   / "best_model.pth"
REPORT_PATH = EVAL_DIR   / "evaluation_report.txt"
CM_PATH     = EVAL_DIR   / "confusion_matrix.png"

# ── Constants ─────────────────────────────────────────────────────────────────
NUM_CLASSES  = 10
BATCH_SIZE   = 32
NUM_WORKERS  = 2


# ── Load class labels ─────────────────────────────────────────────────────────

def load_class_labels() -> dict:
    if not LABELS_PATH.exists():
        # Fallback — hardcoded order matches dataset split indices
        return {0:"yes",1:"no",2:"up",3:"down",4:"left",
                5:"right",6:"on",7:"off",8:"stop",9:"go"}
    with open(LABELS_PATH, "r") as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}


# ── Confusion matrix plot ─────────────────────────────────────────────────────

def plot_confusion_matrix(
    cm          : np.ndarray,
    class_names : list[str],
    save_path   : Path,
):
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)

    ax.set(
        xticks     = np.arange(len(class_names)),
        yticks     = np.arange(len(class_names)),
        xticklabels= class_names,
        yticklabels= class_names,
        ylabel     = "True Label",
        xlabel     = "Predicted Label",
        title      = "STREAMSENSE — Confusion Matrix (Test Set)",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=10)

    # Annotate cells
    thresh = cm.max() / 2.0
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(
                j, i, f"{cm[i,j]}",
                ha="center", va="center", fontsize=9,
                color="white" if cm[i, j] > thresh else "black",
            )

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {save_path}")


# ── Main evaluation ───────────────────────────────────────────────────────────

def evaluate(args):
    ckpt_path = Path(args.ckpt)

    print("=" * 60)
    print("STREAMSENSE — evaluate.py")
    print("=" * 60)

    # ── Device ────────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice     : {device}")

    # ── Prerequisites ──────────────────────────────────────────────────────────
    for p, name in [(ckpt_path, "best_model.pth"), (TEST_SPLIT, "test_files.txt")]:
        if not p.exists():
            print(f"[ERROR] Not found: {p}")
            sys.exit(1)

    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load model ─────────────────────────────────────────────────────────────
    print(f"\nLoading checkpoint: {ckpt_path}")
    ckpt  = torch.load(ckpt_path, map_location="cpu")
    model = StreamSenseNet(num_classes=NUM_CLASSES)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    print(f"  Trained epoch  : {ckpt['epoch']}")
    print(f"  Val accuracy   : {ckpt['val_accuracy']:.2f}%")

    # ── Class labels ───────────────────────────────────────────────────────────
    class_labels = load_class_labels()
    class_names  = [class_labels[i] for i in range(NUM_CLASSES)]
    print(f"\nClasses    : {class_names}")

    # ── Test DataLoader ────────────────────────────────────────────────────────
    print(f"\nLoading test split: {TEST_SPLIT}")
    test_loader = get_dataloader(
        TEST_SPLIT,
        is_train    = False,
        batch_size  = BATCH_SIZE,
        num_workers = NUM_WORKERS,
        verbose     = True,
    )

    # ── Inference loop ─────────────────────────────────────────────────────────
    print(f"\nRunning inference on {len(test_loader.dataset)} test samples...")

    all_preds  = []
    all_labels = []
    criterion  = nn.CrossEntropyLoss()
    total_loss = 0.0

    with torch.no_grad():
        for batch_idx, (tensors, labels) in enumerate(test_loader):
            x = tensors.squeeze(1).to(device)   # [B, 1, 64, 97]
            y = labels.to(device)

            logits = model(x)                    # [B, 10]
            loss   = criterion(logits, y)
            total_loss += loss.item()

            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

            if (batch_idx + 1) % 30 == 0:
                print(f"  [{batch_idx+1:>3}/{len(test_loader)}] batches done...",
                      flush=True)

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)

    # ── Metrics ────────────────────────────────────────────────────────────────
    test_loss = total_loss / len(test_loader)
    test_acc  = 100.0 * accuracy_score(all_labels, all_preds)

    print(f"\n{'='*60}")
    print(f"TEST RESULTS")
    print(f"{'='*60}")
    print(f"  Test loss     : {test_loss:.4f}")
    print(f"  Test accuracy : {test_acc:.2f}%")

    # Per-class report
    report = classification_report(
        all_labels, all_preds,
        target_names = class_names,
        digits       = 4,
    )
    print(f"\nPer-class report:\n{report}")

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)

    # ── Per-class accuracy ─────────────────────────────────────────────────────
    print("Per-class accuracy:")
    print(f"  {'Class':<10} {'Correct':>8} {'Total':>8} {'Acc':>8}")
    print(f"  {'─'*38}")
    for i, name in enumerate(class_names):
        mask    = all_labels == i
        correct = (all_preds[mask] == i).sum()
        total   = mask.sum()
        acc     = 100.0 * correct / total if total > 0 else 0.0
        print(f"  {name:<10} {correct:>8} {total:>8} {acc:>7.2f}%")

    # ── Confusion matrix plot ──────────────────────────────────────────────────
    print(f"\nSaving confusion matrix...")
    plot_confusion_matrix(cm, class_names, CM_PATH)

    # ── Save evaluation report ─────────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    report_lines = [
        "=" * 60,
        "STREAMSENSE — Evaluation Report",
        "=" * 60,
        f"Timestamp       : {timestamp}",
        f"Checkpoint      : {ckpt_path}",
        f"Trained epoch   : {ckpt['epoch']}",
        f"Val accuracy    : {ckpt['val_accuracy']:.2f}%",
        f"Device          : {device}",
        f"Test samples    : {len(all_labels)}",
        "",
        f"Test loss       : {test_loss:.4f}",
        f"Test accuracy   : {test_acc:.2f}%",
        "",
        "Per-class report:",
        report,
        "",
        "Confusion matrix (rows=true, cols=predicted):",
        "Classes: " + ", ".join(f"{i}={n}" for i, n in enumerate(class_names)),
        str(cm),
        "",
        "Per-class accuracy:",
    ]

    for i, name in enumerate(class_names):
        mask    = all_labels == i
        correct = (all_preds[mask] == i).sum()
        total   = mask.sum()
        acc     = 100.0 * correct / total if total > 0 else 0.0
        report_lines.append(f"  {name:<10} {correct}/{total}  ({acc:.2f}%)")

    report_lines += [
        "",
        "MPIC version    : 1.0",
        "Architecture    : StreamSenseNet (VGG-style 2D CNN)",
        "Parameters      : 295,786",
        "Dataset         : Google Speech Commands v2 (10 classes)",
    ]

    report_text = "\n".join(report_lines)

    with open(REPORT_PATH, "w") as f:
        f.write(report_text)

    print(f"  Saved → {REPORT_PATH}")

    # ── Final summary ──────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"EVALUATION COMPLETE")
    print(f"{'='*60}")
    print(f"  Test accuracy  : {test_acc:.2f}%")
    print(f"  Report         : {REPORT_PATH}")
    print(f"  Confusion matrix: {CM_PATH}")
    print(f"\nNext step: python export_onnx.py")


# ── Argument parser ───────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="STREAMSENSE evaluate.py")
    parser.add_argument(
        "--ckpt", type=str,
        default=str(DEFAULT_CKPT),
        help="Path to checkpoint (default: checkpoints/best_model.pth)"
    )
    return parser.parse_args()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = parse_args()
    evaluate(args)
