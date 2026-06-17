"""
evaluate_qat.py
Project STREAMSENSE — Track A
Epic A5.2 — QAT test accuracy evaluation (STRETCH)

Evaluates StreamSenseNetQAT on the full test split (5,779 samples).
Produces a standalone evaluation report — does NOT modify evaluate_onnx.py.

Inputs:
    checkpoints_qat/best_model_qat.pth
    data/splits/test_files.txt

Outputs:
    evaluation_qat/evaluation_report_qat.txt    (full per-class report)
    evaluation_qat/confusion_matrix_qat.png     (confusion matrix)

Run:
    python evaluate_qat.py
"""

import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Local imports ──────────────────────────────────────────────────────────────
try:
    from model_qat   import StreamSenseNetQAT, count_parameters
    from dataset_1d  import StreamSenseDataset1D  # not used — use dataset.py style
    from dataset     import get_dataloader
except ImportError as e:
    print(f"[ERROR] Import failed: {e}")
    sys.exit(1)

# ── Paths ─────────────────────────────────────────────────────────────────────
_DEFAULT_ROOT = r"C:\STREAMSENSE" if os.name == "nt" else "/content/streamsense"
BASE_DIR = Path(os.environ.get("STREAMSENSE_ROOT", _DEFAULT_ROOT))

CKPT_PATH  = BASE_DIR / "checkpoints_qat" / "best_model_qat.pth"
TEST_SPLIT = BASE_DIR / "data" / "splits" / "test_files.txt"
OUT_DIR    = BASE_DIR / "evaluation_qat"

CLASS_NAMES = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]
NUM_CLASSES = 10
BATCH_SIZE  = 64
NUM_WORKERS = 2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Known FP32 results (from evaluate.py) for side-by-side comparison
FP32_TEST_ACC  = 95.97
FP32_PARAMS    = 295786


def plot_confusion_matrix(cm, class_names, save_path):
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(class_names, fontsize=9)
    ax.set_xlabel("Predicted", fontsize=10)
    ax.set_ylabel("True", fontsize=10)
    ax.set_title(f"StreamSenseNetQAT Confusion Matrix", fontsize=11)

    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(
                j, i, str(cm[i, j]),
                ha="center", va="center", fontsize=8,
                color="white" if cm[i, j] > cm.max() / 2 else "black",
            )

    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def main():
    print("=" * 60)
    print("STREAMSENSE — evaluate_qat.py (Epic A5.2 STRETCH)")
    print("=" * 60)

    # ── Check inputs ────────────────────────────────────────────────────────
    for p, name in [(CKPT_PATH, "best_model_qat.pth"), (TEST_SPLIT, "test_files.txt")]:
        if not p.exists():
            print(f"[ERROR] Not found: {p} ({name})")
            print("        Run train_qat.py first." if "qat" in str(p) else "")
            sys.exit(1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load model ─────────────────────────────────────────────────────────
    print(f"\nLoading QAT checkpoint: {CKPT_PATH.name}")
    model = StreamSenseNetQAT(num_classes=NUM_CLASSES).to(DEVICE)
    ckpt  = torch.load(CKPT_PATH, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    n_params    = count_parameters(model)
    qat_epoch   = ckpt.get("epoch", "?")
    qat_val_acc = ckpt.get("val_accuracy", 0.0)
    print(f"  QAT epoch     : {qat_epoch}")
    print(f"  QAT val_acc   : {qat_val_acc:.2f}%")
    print(f"  Parameters    : {n_params['total']:,}")
    print(f"  Device        : {DEVICE}")

    # ── Test dataloader ────────────────────────────────────────────────────
    print(f"\nLoading test split: {TEST_SPLIT.name}")
    test_loader = get_dataloader(
        TEST_SPLIT, is_train=False, batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS, verbose=True,
    )

    # ── Inference ──────────────────────────────────────────────────────────
    all_preds  = []
    all_labels = []
    total_loss = 0.0
    criterion  = torch.nn.CrossEntropyLoss()

    print("\nRunning inference on test set...")
    t0 = time.time()
    with torch.no_grad():
        for tensors, labels in test_loader:
            x = tensors.squeeze(1).to(DEVICE)  # [B, 1, 64, 97]
            y = labels.to(DEVICE)

            logits = model(x)
            loss   = criterion(logits, y)
            total_loss += loss.item() * x.size(0)

            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(y.cpu().numpy().tolist())

    elapsed     = time.time() - t0
    n_total     = len(all_labels)
    avg_loss    = total_loss / n_total
    overall_acc = 100.0 * np.mean(np.array(all_preds) == np.array(all_labels))

    print(f"\nInference time : {elapsed:.2f}s for {n_total} samples")
    print(f"                 ({1000*elapsed/n_total:.2f} ms/sample)")
    print(f"Test loss      : {avg_loss:.4f}")
    print(f"Test accuracy  : {overall_acc:.2f}%")
    print(f"  vs FP32      : {FP32_TEST_ACC:.2f}%  (delta {overall_acc - FP32_TEST_ACC:+.2f}%)")

    # ── Classification report + confusion matrix ────────────────────────────
    report = classification_report(all_labels, all_preds, target_names=CLASS_NAMES, digits=4)
    cm     = confusion_matrix(all_labels, all_preds)

    # Per-class accuracy
    per_class_acc = {}
    for i, name in enumerate(CLASS_NAMES):
        mask    = np.array(all_labels) == i
        correct = (np.array(all_preds)[mask] == i).sum()
        total   = mask.sum()
        per_class_acc[name] = 100.0 * correct / total

    # ── Confusion matrix plot ───────────────────────────────────────────────
    cm_path = OUT_DIR / "confusion_matrix_qat.png"
    plot_confusion_matrix(cm, CLASS_NAMES, cm_path)
    print(f"\nConfusion matrix → {cm_path}")

    # ── Evaluation report ──────────────────────────────────────────────────
    report_path = OUT_DIR / "evaluation_report_qat.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("STREAMSENSE — StreamSenseNetQAT Evaluation Report\n")
        f.write("Epic A5.2 — QAT (Brevitas) on test split (STRETCH)\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"QAT checkpoint epoch  : {qat_epoch}\n")
        f.write(f"QAT val accuracy      : {qat_val_acc:.2f}%\n")
        f.write(f"Parameters            : {n_params['total']:,}\n\n")
        f.write(f"Test samples          : {n_total}\n")
        f.write(f"Test loss             : {avg_loss:.4f}\n")
        f.write(f"Test accuracy (QAT)   : {overall_acc:.2f}%\n")
        f.write(f"Test accuracy (FP32)  : {FP32_TEST_ACC:.2f}%\n")
        f.write(f"Delta (QAT - FP32)    : {overall_acc - FP32_TEST_ACC:+.2f}%\n")
        f.write(f"Inference time        : {elapsed:.2f}s ({1000*elapsed/n_total:.2f} ms/sample)\n\n")

        f.write("Per-class accuracy (%):\n")
        f.write(f"{'Class':<8} {'QAT':>8} {'FP32':>8} {'Delta':>8}\n")
        f.write("-" * 36 + "\n")
        fp32_per_class = {
            "yes": 98.84, "no": 96.79, "up": 95.17, "down": 94.04, "left": 96.67,
            "right": 99.29, "on": 95.66, "off": 94.47, "stop": 93.80, "go": 94.85,
        }
        for name in CLASS_NAMES:
            qat_a  = per_class_acc[name]
            fp32_a = fp32_per_class[name]
            delta  = qat_a - fp32_a
            f.write(f"{name:<8} {qat_a:>7.2f}% {fp32_a:>7.2f}% {delta:>+7.2f}%\n")

        f.write("\nPer-class classification report:\n")
        f.write(report)
        f.write("\nConfusion Matrix (rows=true, cols=pred):\n")
        f.write(str(CLASS_NAMES) + "\n")
        f.write(str(cm) + "\n")

    print(f"Evaluation report → {report_path}")

    # ── Console per-class comparison ────────────────────────────────────────
    print(f"\n{'Class':<8} {'QAT':>8} {'FP32':>8} {'Delta':>8}")
    print("-" * 36)
    fp32_per_class = {
        "yes": 98.84, "no": 96.79, "up": 95.17, "down": 94.04, "left": 96.67,
        "right": 99.29, "on": 95.66, "off": 94.47, "stop": 93.80, "go": 94.85,
    }
    for name in CLASS_NAMES:
        qat_a  = per_class_acc[name]
        fp32_a = fp32_per_class[name]
        delta  = qat_a - fp32_a
        print(f"{name:<8} {qat_a:>7.2f}% {fp32_a:>7.2f}% {delta:>+7.2f}%")

    # ── Final verdict ──────────────────────────────────────────────────────
    drop = FP32_TEST_ACC - overall_acc
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"  FP32 test accuracy : {FP32_TEST_ACC:.2f}%")
    print(f"  QAT  test accuracy : {overall_acc:.2f}%")
    print(f"  Accuracy drop      : {drop:.2f}%  (budget: ≤1.0%)")
    if drop <= 1.0:
        print(f"  [PASS] Within 1.0% accuracy drop budget.")
    else:
        print(f"  [WARN] Exceeds 1.0% accuracy drop budget. Consider more QAT epochs.")
    print(f"\n[DONE] evaluate_qat.py complete.")


if __name__ == "__main__":
    main()
