"""
evaluate_1d_comparison.py
Project STREAMSENSE — Track A
Epic A3.2 — Evaluation + Comparison report (1D baseline vs 2D StreamSenseNet)

Evaluates StreamSenseNet1D on the test split, computes overall accuracy,
per-class precision/recall/F1, and confusion matrix — same metrics as
evaluate.py for the 2D model. Then produces a side-by-side comparison
table against the 2D model's known results (from evaluation_report.txt),
to support the ADR (A3.3) decision on which architecture to deploy.

Outputs:
    evaluation_1d/evaluation_report_1d.txt
    evaluation_1d/confusion_matrix_1d.png
    evaluation_1d/comparison_1d_vs_2d.txt

Run from C:\\STREAMSENSE\\training\\:
    python evaluate_1d_comparison.py
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

from model_1d import StreamSenseNet1D, count_parameters
from dataset_1d import StreamSenseDataset1D

_DEFAULT_ROOT = r"C:\STREAMSENSE" if os.name == "nt" else "/content/STREAMSENSE"
ROOT       = Path(os.environ.get("STREAMSENSE_ROOT", _DEFAULT_ROOT))
CKPT_PATH  = ROOT / "checkpoints_1d" / "best_model_1d.pth"
OUT_DIR    = ROOT / "evaluation_1d"

CLASS_NAMES = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64

# ── Known 2D model results (from evaluation_report.txt / model card) ─────────
RESULTS_2D = {
    "params": 295786,
    "test_acc": 95.97,
    "test_loss": 0.1273,
    "per_class_acc": {
        "yes": 98.84, "no": 96.79, "up": 95.17, "down": 94.04, "left": 96.67,
        "right": 99.29, "on": 95.66, "off": 94.47, "stop": 93.80, "go": 94.85,
    },
}


def main():
    print("=" * 60)
    print("STREAMSENSE — evaluate_1d_comparison.py (Epic A3.2)")
    print("=" * 60)

    if not CKPT_PATH.exists():
        print(f"[ERROR] Checkpoint not found: {CKPT_PATH}")
        print("Run train_1d.py first.")
        sys.exit(1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load model ─────────────────────────────────────────────────────────
    model = StreamSenseNet1D(n_classes=10).to(DEVICE)
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    n_params = count_parameters(model)
    print(f"Loaded checkpoint: epoch {ckpt['epoch']}, val_acc={ckpt['val_acc']:.2f}%")
    print(f"Parameters: {n_params:,}")

    # ── Test set ───────────────────────────────────────────────────────────
    test_ds = StreamSenseDataset1D(split="test", augment=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    print(f"Test samples: {len(test_ds)}")

    # ── Run inference ──────────────────────────────────────────────────────
    all_preds  = []
    all_labels = []
    total_loss = 0.0
    criterion  = torch.nn.CrossEntropyLoss()

    print("\nRunning inference...")
    t0 = time.time()
    with torch.no_grad():
        for waveforms, labels in test_loader:
            waveforms = waveforms.to(DEVICE)
            labels    = labels.to(DEVICE)

            logits = model(waveforms)
            loss = criterion(logits, labels)
            total_loss += loss.item() * waveforms.size(0)

            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    elapsed = time.time() - t0
    avg_loss = total_loss / len(test_ds)
    overall_acc = 100.0 * np.mean(np.array(all_preds) == np.array(all_labels))

    print(f"Inference time: {elapsed:.2f}s for {len(test_ds)} samples "
          f"({1000*elapsed/len(test_ds):.2f} ms/sample)")
    print(f"Test accuracy: {overall_acc:.2f}%")
    print(f"Test loss:     {avg_loss:.4f}")

    # ── Classification report ─────────────────────────────────────────────
    report = classification_report(
        all_labels, all_preds, target_names=CLASS_NAMES, digits=4
    )
    cm = confusion_matrix(all_labels, all_preds)

    # Per-class accuracy
    per_class_acc_1d = {}
    for i, name in enumerate(CLASS_NAMES):
        mask = np.array(all_labels) == i
        correct = (np.array(all_preds)[mask] == i).sum()
        total = mask.sum()
        per_class_acc_1d[name] = 100.0 * correct / total

    # ── Save evaluation report ────────────────────────────────────────────
    report_path = OUT_DIR / "evaluation_report_1d.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("STREAMSENSE — StreamSenseNet1D Evaluation Report\n")
        f.write("Epic A3.2 — 1D CNN Baseline on raw audio frames (STRETCH)\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Checkpoint epoch : {ckpt['epoch']}\n")
        f.write(f"Val accuracy     : {ckpt['val_acc']:.2f}%\n")
        f.write(f"Parameters       : {n_params:,}\n\n")
        f.write(f"Test samples     : {len(test_ds)}\n")
        f.write(f"Test loss        : {avg_loss:.4f}\n")
        f.write(f"Test accuracy    : {overall_acc:.2f}%\n")
        f.write(f"Inference time   : {elapsed:.2f}s "
                f"({1000*elapsed/len(test_ds):.2f} ms/sample)\n\n")
        f.write("Per-class report:\n")
        f.write(report)
        f.write("\nConfusion Matrix (rows=true, cols=pred):\n")
        f.write(f"{CLASS_NAMES}\n")
        f.write(str(cm))
        f.write("\n")

    print(f"\nSaved -> {report_path}")

    # ── Confusion matrix plot ─────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right")
    ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"StreamSenseNet1D Confusion Matrix (test_acc={overall_acc:.2f}%)")

    for i in range(10):
        for j in range(10):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=8)

    fig.colorbar(im)
    fig.tight_layout()
    cm_path = OUT_DIR / "confusion_matrix_1d.png"
    fig.savefig(cm_path, dpi=120)
    plt.close(fig)
    print(f"Saved -> {cm_path}")

    # ── Comparison report (1D vs 2D) ──────────────────────────────────────
    comparison_path = OUT_DIR / "comparison_1d_vs_2d.txt"
    with open(comparison_path, "w", encoding="utf-8") as f:
        f.write("STREAMSENSE — Architecture Comparison: 1D CNN (raw) vs 2D CNN (mel)\n")
        f.write("Supports Epic A3.3 (ADR — Architecture Decision Record)\n")
        f.write("=" * 70 + "\n\n")

        f.write(f"{'Metric':<28}{'2D StreamSenseNet':>20}{'1D StreamSenseNet1D':>22}\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'Parameters':<28}{RESULTS_2D['params']:>20,}{n_params:>22,}\n")
        f.write(f"{'Test accuracy':<28}{RESULTS_2D['test_acc']:>19.2f}%{overall_acc:>21.2f}%\n")
        f.write(f"{'Test loss':<28}{RESULTS_2D['test_loss']:>20.4f}{avg_loss:>22.4f}\n")
        f.write(f"{'Input representation':<28}{'log-mel [1,64,97]':>20}{'raw waveform [1,16000]':>22}\n")
        f.write("\n")

        f.write("Per-class accuracy (%):\n")
        f.write(f"{'Class':<10}{'2D':>10}{'1D':>10}{'Delta (1D-2D)':>16}\n")
        f.write("-" * 46 + "\n")
        for name in CLASS_NAMES:
            acc_2d = RESULTS_2D["per_class_acc"][name]
            acc_1d = per_class_acc_1d[name]
            delta = acc_1d - acc_2d
            f.write(f"{name:<10}{acc_2d:>9.2f}%{acc_1d:>9.2f}%{delta:>+15.2f}%\n")

        f.write("\n")
        acc_diff = overall_acc - RESULTS_2D["test_acc"]
        param_ratio = n_params / RESULTS_2D["params"]
        f.write(f"Overall accuracy delta (1D - 2D): {acc_diff:+.2f} percentage points\n")
        f.write(f"Parameter ratio (1D / 2D): {param_ratio:.2f}x\n")

        f.write("\nNotes for ADR (A3.3):\n")
        f.write("- 2D mel-spectrogram representation provides explicit time-frequency\n")
        f.write("  structure as input, which the 1D model must learn implicitly from\n")
        f.write("  raw waveform via its receptive field.\n")
        f.write("- Compare accuracy-per-parameter and inference latency when deciding\n")
        f.write("  between representations for the FPGA deployment target.\n")

    print(f"Saved -> {comparison_path}")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  2D StreamSenseNet : {RESULTS_2D['params']:,} params, "
          f"{RESULTS_2D['test_acc']:.2f}% test acc")
    print(f"  1D StreamSenseNet1D: {n_params:,} params, "
          f"{overall_acc:.2f}% test acc")
    print(f"\n[DONE] See {comparison_path.name} for full comparison.")


if __name__ == "__main__":
    main()
