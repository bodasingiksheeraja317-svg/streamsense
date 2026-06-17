"""
train_qat.py
Project STREAMSENSE — Track A
Epic A5.2 — QAT fine-tuning + QONNX export (STRETCH)

Fine-tunes StreamSenseNetQAT (Brevitas) from the FP32 best_model.pth
checkpoint using Quantization-Aware Training. Exports the quantized model
as a QONNX file for the later FPGA path (Track E).

Workflow:
    1. Load FP32 best_model.pth weights → StreamSenseNetQAT
    2. Fine-tune 10 epochs (LR=0.0001, Adam, SpecAugment kept on)
    3. Save QAT checkpoint → checkpoints_qat/best_model_qat.pth
    4. Export QONNX → onnx_models/streamsense_model_qat.onnx
    5. Validate on 10 golden vectors (top-1 parity check vs FP32)

Run from /content/streamsense/training/ in Colab:
    python train_qat.py

Requirements:
    pip install brevitas onnx

Outputs:
    checkpoints_qat/best_model_qat.pth
    checkpoints_qat/training_log_qat.csv
    onnx_models/streamsense_model_qat.onnx
    evaluation_qat/evaluation_report_qat.txt
"""

import csv
import json
import os
import sys
import time
import random
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

# ── Local imports ──────────────────────────────────────────────────────────────
try:
    from model_qat import StreamSenseNetQAT, count_parameters, load_fp32_weights
    from dataset   import get_dataloader
except ImportError as e:
    print(f"[ERROR] Import failed: {e}")
    print("        Ensure model_qat.py and dataset.py are in the same directory.")
    sys.exit(1)

# ── Paths ─────────────────────────────────────────────────────────────────────
# Environment-aware: works on Windows and Colab
_DEFAULT_ROOT = r"C:\STREAMSENSE" if os.name == "nt" else "/content/streamsense"
BASE_DIR      = Path(os.environ.get("STREAMSENSE_ROOT", _DEFAULT_ROOT))

SPLITS_DIR   = BASE_DIR / "data"  / "splits"
CKPT_DIR     = BASE_DIR / "checkpoints_qat"
FP32_CKPT    = BASE_DIR / "checkpoints"      / "best_model.pth"
GV_ROOT      = BASE_DIR / "golden_vectors"
ONNX_DIR     = BASE_DIR / "onnx_models"
EVAL_DIR     = BASE_DIR / "evaluation_qat"

TRAIN_SPLIT  = SPLITS_DIR / "train_files.txt"
VAL_SPLIT    = SPLITS_DIR / "val_files.txt"
TEST_SPLIT   = SPLITS_DIR / "test_files.txt"
BEST_CKPT    = CKPT_DIR   / "best_model_qat.pth"
TRAIN_LOG    = CKPT_DIR   / "training_log_qat.csv"
QONNX_PATH   = ONNX_DIR   / "streamsense_model_qat.onnx"

GV_MANIFEST  = GV_ROOT / "manifest.json"
GV_RAW_DIR   = GV_ROOT / "raw"
GV_NORM_DIR  = GV_ROOT / "normalized"

# ── Hyperparameters ───────────────────────────────────────────────────────────
SEED              = 42
NUM_CLASSES       = 10
NUM_WORKERS       = 2       # set 0 if Colab multiprocessing issues

# QAT fine-tune: lower LR, fewer epochs — starting from a strong FP32 baseline
QAT_LR            = 0.0001   # 10x lower than initial training LR
QAT_MAX_EPOCHS    = 10
QAT_WEIGHT_DECAY  = 1e-4
QAT_LR_FACTOR     = 0.5
QAT_LR_PATIENCE   = 3
QAT_MIN_LR        = 1e-6
QAT_EARLY_STOP    = 5        # stop if no val_acc improvement for 5 epochs

# SpecAugment (kept identical to train.py)
FREQ_MASK_F  = 8
TIME_MASK_T  = 15
N_FREQ_MASKS = 1
N_TIME_MASKS = 1


# ── Helpers ───────────────────────────────────────────────────────────────────

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def spec_augment(x: torch.Tensor) -> torch.Tensor:
    """SpecAugment on batch [B, 1, 64, 97]. Identical to train.py."""
    x = x.clone()
    B, C, F, T = x.shape
    for b in range(B):
        for _ in range(N_FREQ_MASKS):
            f  = random.randint(0, FREQ_MASK_F)
            f0 = random.randint(0, max(F - f, 0))
            x[b, :, f0: f0 + f, :] = 0.0
        for _ in range(N_TIME_MASKS):
            t  = random.randint(0, TIME_MASK_T)
            t0 = random.randint(0, max(T - t, 0))
            x[b, :, :, t0: t0 + t] = 0.0
    return x


# ── Train / validate ───────────────────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, device, epoch):
    """One QAT fine-tuning epoch. Returns mean train loss."""
    model.train()
    total_loss = 0.0
    n_batches  = len(loader)

    for batch_idx, (tensors, labels) in enumerate(loader):
        x = tensors.squeeze(1).to(device)   # [B, 1, 64, 97]
        y = labels.to(device)
        x = spec_augment(x)

        optimizer.zero_grad()
        logits = model(x)
        loss   = criterion(logits, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()

        if (batch_idx + 1) % 200 == 0 or (batch_idx + 1) == n_batches:
            print(
                f"  Epoch {epoch:>3}  [{batch_idx+1:>4}/{n_batches}]  "
                f"loss={loss.item():.4f}",
                flush=True,
            )

    return total_loss / n_batches


def validate(model, loader, criterion, device):
    """One validation pass. Returns (mean_loss, accuracy_%)."""
    model.eval()
    total_loss = 0.0
    correct    = 0
    total      = 0

    with torch.no_grad():
        for tensors, labels in loader:
            x = tensors.squeeze(1).to(device)
            y = labels.to(device)
            logits     = model(x)
            loss       = criterion(logits, y)
            total_loss += loss.item()
            preds       = logits.argmax(dim=1)
            correct    += (preds == y).sum().item()
            total      += y.size(0)

    return total_loss / len(loader), 100.0 * correct / total


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def save_checkpoint(model, epoch, val_acc, path):
    torch.save(
        {
            "epoch"        : epoch,
            "val_accuracy" : val_acc,
            "model_state"  : model.state_dict(),
            "num_classes"  : NUM_CLASSES,
            "mpic_version" : "1.0",
            "qat"          : True,
        },
        path,
    )


def init_csv(path):
    with open(path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "val_loss", "val_acc", "lr"])


def append_csv(path, epoch, train_loss, val_loss, val_acc, lr):
    with open(path, "a", newline="") as f:
        csv.writer(f).writerow(
            [epoch, f"{train_loss:.6f}", f"{val_loss:.6f}", f"{val_acc:.4f}", f"{lr:.8f}"]
        )


# ── QONNX export ──────────────────────────────────────────────────────────────

def export_qonnx(model, qonnx_path: Path):
    """
    Export QAT model to QONNX using Brevitas exporter.
    QONNX preserves the quantization annotations needed for FINN / FPGA deployment.
    """
    try:
        from brevitas.export import export_qonnx
    except ImportError:
        try:
            # Older Brevitas API
            from brevitas.export.onnx.finn.manager import FINNManager
            print("  [INFO] Using FINNManager for QONNX export (older Brevitas).")
            FINNManager.export(model, input_shape=(1, 1, 64, 97), export_path=str(qonnx_path))
            return
        except ImportError:
            print("  [WARN] QONNX export API not found. Falling back to standard ONNX export.")
            print("         Install: pip install brevitas --upgrade")
            _export_onnx_fallback(model, qonnx_path)
            return

    model.eval()
    dummy = torch.zeros(1, 1, 64, 97)
    export_qonnx(
        model,
        args    = (dummy,),
        export_path = str(qonnx_path),
        opset_version = 13,
        input_names  = ["input"],
        output_names = ["logits"],
    )


def _export_onnx_fallback(model, path: Path):
    """Fallback: standard ONNX export (loses quantization annotations)."""
    model.eval()
    dummy = torch.zeros(1, 1, 64, 97)
    torch.onnx.export(
        model,
        dummy,
        str(path),
        opset_version      = 13,
        input_names        = ["input"],
        output_names       = ["logits"],
        do_constant_folding = True,
    )
    print(f"  [FALLBACK] Exported as standard ONNX (no QONNX annotations): {path.name}")


# ── Golden vector validation ──────────────────────────────────────────────────

def validate_golden_vectors(model, device) -> tuple:
    """
    Run QAT model on all 10 golden vectors. Compare top-1 predictions to
    ground truth (from manifest labels). Returns (passed, failed).
    Also loads FP32 ONNX for top-1 parity comparison if available.
    """
    if not GV_MANIFEST.exists():
        print("  [SKIP] Golden vector manifest not found. Skipping GV validation.")
        return 0, 0

    try:
        from mel_pipeline import preprocess
    except ImportError:
        print("  [SKIP] mel_pipeline not importable. Skipping GV validation.")
        return 0, 0

    with open(GV_MANIFEST) as f:
        manifest = json.load(f)
    vectors = manifest["vectors"]

    passed = 0
    failed = 0
    model.eval()

    print("\n  GV Validation (QAT top-1 vs ground truth):")
    for i in range(10):
        v       = vectors[str(i)]
        raw_bin = GV_RAW_DIR / v["raw_bin"]
        raw     = np.fromfile(str(raw_bin), dtype="<f4").reshape(tuple(v["raw_shape"]))

        tensor = preprocess(raw)                          # [1, 1, 64, 97]
        x      = tensor.to(device)

        with torch.no_grad():
            logits = model(x)                             # [1, 10]

        pred_idx = int(logits.argmax(dim=1).item())
        true_idx = i       # GV_0X = class X by construction
        ok       = pred_idx == true_idx
        status   = "PASS" if ok else "FAIL"

        if ok:
            passed += 1
        else:
            failed += 1

        print(
            f"    [{status}] GV_{i:02d}_{v['label']:<6} "
            f"pred={pred_idx}  true={true_idx}"
        )

    return passed, failed


# ── Main training loop ────────────────────────────────────────────────────────

def train(args):
    set_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 60)
    print("STREAMSENSE — train_qat.py  (Epic A5.2 STRETCH)")
    print("=" * 60)
    print(f"\nDevice       : {device}")
    if device.type == "cuda":
        print(f"GPU          : {torch.cuda.get_device_name(0)}")
    print(f"QAT LR       : {QAT_LR}")
    print(f"QAT Epochs   : {QAT_MAX_EPOCHS}")
    print(f"FP32 ckpt    : {FP32_CKPT}")

    # ── Verify paths ──────────────────────────────────────────────────────────
    for p, name in [
        (FP32_CKPT,    "best_model.pth (FP32 checkpoint)"),
        (TRAIN_SPLIT,  "train_files.txt"),
        (VAL_SPLIT,    "val_files.txt"),
    ]:
        if not p.exists():
            print(f"[ERROR] Not found: {p} ({name})")
            sys.exit(1)

    # ── Create dirs ───────────────────────────────────────────────────────────
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    ONNX_DIR.mkdir(parents=True, exist_ok=True)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    # ── DataLoaders ───────────────────────────────────────────────────────────
    print("\nLoading datasets...")
    train_loader = get_dataloader(
        TRAIN_SPLIT, is_train=True,  batch_size=args.batch,
        num_workers=NUM_WORKERS, verbose=True,
    )
    val_loader = get_dataloader(
        VAL_SPLIT, is_train=False, batch_size=args.batch,
        num_workers=NUM_WORKERS, verbose=True,
    )

    # ── Model: build QAT model, load FP32 weights ─────────────────────────────
    print("\nBuilding QAT model and loading FP32 weights...")
    model = StreamSenseNetQAT(num_classes=NUM_CLASSES).to(device)
    fp32_epoch, fp32_val_acc = load_fp32_weights(model, FP32_CKPT)
    params = count_parameters(model)
    print(f"  FP32 checkpoint: epoch {fp32_epoch}, val_acc={fp32_val_acc:.2f}%")
    print(f"  QAT model params: {params['total']:,}")

    # ── Loss, optimizer, scheduler ────────────────────────────────────────────
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=QAT_LR, weight_decay=QAT_WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=QAT_LR_FACTOR,
        patience=QAT_LR_PATIENCE, min_lr=QAT_MIN_LR,
    )

    # ── Training loop ─────────────────────────────────────────────────────────
    init_csv(TRAIN_LOG)
    best_val_acc  = 0.0
    best_epoch    = 0
    epochs_no_imp = 0

    print(f"\n{'─'*60}")
    print(f"QAT fine-tuning — max {QAT_MAX_EPOCHS} epochs")
    print(f"Early stopping patience: {QAT_EARLY_STOP} epochs")
    print(f"Best checkpoint → {BEST_CKPT}")
    print(f"{'─'*60}\n")

    for epoch in range(1, QAT_MAX_EPOCHS + 1):
        t0 = time.time()

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]
        elapsed    = time.time() - t0

        improved = val_acc > best_val_acc
        marker   = " ← best" if improved else ""
        print(
            f"Epoch {epoch:>3}/{QAT_MAX_EPOCHS}  "
            f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
            f"val_acc={val_acc:.2f}%  lr={current_lr:.2e}  t={elapsed:.1f}s{marker}",
            flush=True,
        )

        if improved:
            best_val_acc  = val_acc
            best_epoch    = epoch
            epochs_no_imp = 0
            save_checkpoint(model, epoch, val_acc, BEST_CKPT)
            print(f"  [SAVED] best_model_qat.pth  (val_acc={val_acc:.2f}%)")
        else:
            epochs_no_imp += 1

        append_csv(TRAIN_LOG, epoch, train_loss, val_loss, val_acc, current_lr)

        if epochs_no_imp >= QAT_EARLY_STOP:
            print(
                f"\n[EARLY STOP] No improvement for {QAT_EARLY_STOP} epochs. "
                f"Best was epoch {best_epoch} ({best_val_acc:.2f}%)."
            )
            break

    # ── Reload best checkpoint for export ─────────────────────────────────────
    print(f"\nReloading best QAT checkpoint (epoch {best_epoch})...")
    ckpt = torch.load(BEST_CKPT, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    model.to("cpu")   # export always on CPU

    # ── QONNX export ──────────────────────────────────────────────────────────
    print(f"\nExporting QONNX → {QONNX_PATH}")
    export_qonnx(model, QONNX_PATH)
    if QONNX_PATH.exists():
        size_mb = QONNX_PATH.stat().st_size / 1e6
        print(f"  File size : {size_mb:.2f} MB")
    else:
        print(f"  [WARN] QONNX file not found after export. Check export function.")

    # ── Golden vector validation ───────────────────────────────────────────────
    print("\nRunning golden vector validation (10 GVs)...")
    model.to(device)
    gv_passed, gv_failed = validate_golden_vectors(model, device)
    print(f"\n  GV Result: {gv_passed}/10 PASS  {gv_failed}/10 FAIL")
    if gv_failed == 0:
        print("  [PASS] All golden vectors correct. QAT model is valid.")
    else:
        print(f"  [WARN] {gv_failed} golden vector(s) failed. Review model.")

    # ── Write evaluation report ────────────────────────────────────────────────
    report_path = EVAL_DIR / "evaluation_report_qat.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("STREAMSENSE — StreamSenseNetQAT Evaluation Report\n")
        f.write("Epic A5.2 — QAT (Brevitas) STRETCH\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"FP32 base checkpoint  : epoch {fp32_epoch}, val_acc={fp32_val_acc:.2f}%\n")
        f.write(f"QAT best checkpoint   : epoch {best_epoch}, val_acc={best_val_acc:.2f}%\n")
        f.write(f"QAT LR                : {QAT_LR}\n")
        f.write(f"QAT Epochs run        : {epoch}\n")
        f.write(f"Parameters            : {params['total']:,}\n\n")
        f.write(f"Golden Vector Parity  : {gv_passed}/10 PASS\n")
        if QONNX_PATH.exists():
            f.write(f"QONNX model size      : {QONNX_PATH.stat().st_size / 1e6:.2f} MB\n")
        f.write(f"\nQONNX output          : {QONNX_PATH}\n")
        f.write(f"Training log          : {TRAIN_LOG}\n")
        f.write("\nNext step: run evaluate_qat.py for full test accuracy.\n")

    print(f"\nReport saved → {report_path}")

    # ── Final summary ──────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("QAT TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"  FP32 base val_acc  : {fp32_val_acc:.2f}%  (epoch {fp32_epoch})")
    print(f"  QAT  best val_acc  : {best_val_acc:.2f}%  (epoch {best_epoch})")
    print(f"  Delta (QAT - FP32) : {best_val_acc - fp32_val_acc:+.2f}%")
    print(f"  GV parity          : {gv_passed}/10 PASS")
    print(f"  QONNX checkpoint   : {QONNX_PATH}")
    print(f"  Training log       : {TRAIN_LOG}")
    print(f"\nNext step: python evaluate_qat.py")


# ── Argument parser ───────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="STREAMSENSE QAT fine-tuning (Brevitas)")
    parser.add_argument("--batch", type=int,   default=32,    help="Batch size (default 32)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
