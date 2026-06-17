"""
evaluate_onnx.py
Project STREAMSENSE — Track A
MPIC v1.0

Evaluates both FP32 and INT8 ONNX models on the full test split.
Produces accuracy, per-class precision/recall/F1, confusion matrix,
and appends results to evaluation_report.txt.

Usage:
    python evaluate_onnx.py

Paths (edit if needed):
    ONNX models  : C:\STREAMSENSE\onnx_models\
    Test split   : C:\STREAMSENSE\data\splits\test_files.txt
    Class labels : C:\STREAMSENSE\class_labels.json
    Report out   : C:\STREAMSENSE\evaluation\evaluation_report.txt
"""

import sys
import json
import time
import numpy as np
from pathlib import Path
from datetime import datetime

import torch
import torchaudio
import onnxruntime as ort
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
)

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT            = Path(r"C:\STREAMSENSE")
FP32_ONNX       = ROOT / "onnx_models" / "streamsense_model_fp32.onnx"
INT8_ONNX       = ROOT / "onnx_models" / "streamsense_model_int8.onnx"
TEST_SPLIT      = ROOT / "data" / "splits" / "test_files.txt"
CLASS_LABELS    = ROOT / "class_labels.json"
STATS_FILE      = ROOT / "stats" / "normalization_stats.json"
REPORT_OUT      = ROOT / "evaluation" / "evaluation_report.txt"

# ── MPIC v1.0 frozen parameters ───────────────────────────────────────────────
SAMPLE_RATE   = 16000
FRAME_LEN     = 16000
N_FFT         = 512
HOP_LENGTH    = 160
N_MELS        = 64
CENTER        = False
POWER         = 2.0
LOG_EPS       = 1e-10
CLIP_FLOOR_DB = -80.0

BATCH_SIZE    = 64   # inference batch size

# ── Load stats & class labels ─────────────────────────────────────────────────
with open(STATS_FILE, "r") as f:
    _stats = json.load(f)
GLOBAL_MEAN = float(_stats["global_mean"])
GLOBAL_STD  = float(_stats["global_std"])

with open(CLASS_LABELS, "r") as f:
    _cl = json.load(f)
# Support both {idx: label} and {label: idx} formats
if isinstance(list(_cl.values())[0], int):
    # {label: idx} → invert
    IDX_TO_LABEL = {v: k for k, v in _cl.items()}
else:
    # {idx: label} or {"0": label}
    IDX_TO_LABEL = {int(k): v for k, v in _cl.items()}

NUM_CLASSES = len(IDX_TO_LABEL)
CLASS_NAMES = [IDX_TO_LABEL[i] for i in range(NUM_CLASSES)]

# ── Mel transform (built once) ────────────────────────────────────────────────
_mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate = SAMPLE_RATE,
    n_fft       = N_FFT,
    hop_length  = HOP_LENGTH,
    n_mels      = N_MELS,
    window_fn   = torch.hann_window,
    center      = CENTER,
    power       = POWER,
)

def preprocess(raw: np.ndarray) -> np.ndarray:
    """
    MPIC v1.0 pipeline. Input: float32 numpy [T]. Output: float32 numpy [1,1,64,97].
    """
    waveform = torch.from_numpy(raw.copy()).float()
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)            # [1, T]
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    # pad / crop
    L = waveform.shape[1]
    if L < FRAME_LEN:
        waveform = torch.nn.functional.pad(waveform, (0, FRAME_LEN - L))
    elif L > FRAME_LEN:
        waveform = waveform[:, :FRAME_LEN]
    mel = _mel_transform(waveform)                  # [1, 64, 97]
    mel = 10.0 * torch.log10(mel + LOG_EPS)
    mel = torch.clamp(mel, min=CLIP_FLOOR_DB)
    mel = (mel - GLOBAL_MEAN) / GLOBAL_STD
    mel = mel.unsqueeze(0)                          # [1, 1, 64, 97]
    return mel.numpy().astype(np.float32)

# ── Parse test split ──────────────────────────────────────────────────────────
def parse_split(split_file: Path):
    samples = []

    label_to_idx = {v: k for k, v in IDX_TO_LABEL.items()}

    with open(split_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            parts = [p.strip() for p in line.split("|")]

            if len(parts) == 3:
                wav_path = Path(parts[0])

                try:
                    class_idx = int(parts[2])
                except:
                    class_idx = label_to_idx.get(parts[1], -1)

            else:
                continue

            samples.append((wav_path, class_idx))

    return samples

# ── Inference on full test set ────────────────────────────────────────────────
def run_inference(onnx_path: Path, samples: list) -> tuple:
    """
    Runs ONNX model on all samples.
    Returns (all_preds: np.ndarray, all_labels: np.ndarray, elapsed_sec: float).
    """
    sess_opts = ort.SessionOptions()
    sess_opts.inter_op_num_threads = 4
    sess_opts.intra_op_num_threads = 4
    session = ort.InferenceSession(str(onnx_path), sess_opts=sess_opts)
    input_name  = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    all_preds  = []
    all_labels = []
    errors     = 0
    t0         = time.time()

    # batch inference
    batch_inputs = []
    batch_labels = []

    def flush_batch():
        if not batch_inputs:
            return
        x = np.concatenate(batch_inputs, axis=0)   # [B, 1, 64, 97]
        logits = session.run([output_name], {input_name: x})[0]
        preds  = np.argmax(logits, axis=1)
        all_preds.extend(preds.tolist())
        all_labels.extend(batch_labels)
        batch_inputs.clear()
        batch_labels.clear()

    for i, (wav_path, class_idx) in enumerate(samples):
        if (i + 1) % 500 == 0:
            print(f"  [{i+1}/{len(samples)}] processing...", flush=True)

        if not wav_path.exists():
            errors += 1
            continue
        try:
            waveform, sr = torchaudio.load(str(wav_path))
            raw = waveform.squeeze(0).numpy().astype(np.float32)
            inp = preprocess(raw)               # [1, 1, 64, 97]
            batch_inputs.append(inp)
            batch_labels.append(class_idx)
        except Exception as e:
            errors += 1
            continue

        if len(batch_inputs) >= BATCH_SIZE:
            flush_batch()

    flush_batch()
    elapsed = time.time() - t0

    if errors:
        print(f"  [WARN] Skipped {errors} files (missing or unreadable)")

    return np.array(all_preds), np.array(all_labels), elapsed

# ── Report builder ────────────────────────────────────────────────────────────
def build_report_block(model_name: str, onnx_path: Path,
                        preds: np.ndarray, labels: np.ndarray,
                        elapsed: float) -> str:
    acc   = accuracy_score(labels, preds)
    cm    = confusion_matrix(labels, preds, labels=list(range(NUM_CLASSES)))
    report = classification_report(
        labels, preds,
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0,
    )

    lines = []
    sep = "=" * 60

    lines.append(sep)
    lines.append(f"  Model        : {model_name}")
    lines.append(f"  ONNX file    : {onnx_path.name}")
    lines.append(f"  Timestamp    : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"  Test samples : {len(labels)}")
    lines.append(f"  Accuracy     : {acc*100:.2f}%  ({int(acc*len(labels))}/{len(labels)})")
    lines.append(f"  Elapsed      : {elapsed:.1f}s")
    lines.append(sep)
    lines.append("")
    lines.append("Per-class report:")
    lines.append(report)

    lines.append("Per-class accuracy:")
    for i, name in enumerate(CLASS_NAMES):
        mask    = labels == i
        correct = int((preds[mask] == labels[mask]).sum())
        total   = int(mask.sum())
        lines.append(f"  {name:<10} {correct}/{total}  ({correct/total*100:.2f}%)")

    lines.append("")
    lines.append(f"Confusion matrix (rows=true, cols=predicted):")
    lines.append(f"Classes: " + ", ".join(f"{i}={n}" for i, n in enumerate(CLASS_NAMES)))
    for row in cm:
        lines.append("  " + str(row.tolist()))

    lines.append("")
    lines.append(f"MPIC version   : 1.0")
    lines.append(f"Architecture   : StreamSenseNet (VGG-style 2D CNN)")
    lines.append(f"Parameters     : 295,786")
    lines.append(f"Dataset        : Google Speech Commands v2 (10 classes)")
    lines.append(sep)

    return "\n".join(lines)

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("STREAMSENSE — ONNX Evaluation (FP32 + INT8)")
    print("=" * 60)

    # Validate paths
    for p, name in [
        (FP32_ONNX,    "streamsense_model_fp32.onnx"),
        (INT8_ONNX,    "streamsense_model_int8.onnx"),
        (TEST_SPLIT,   "test_files.txt"),
        (STATS_FILE,   "normalization_stats.json"),
        (CLASS_LABELS, "class_labels.json"),
    ]:
        if not p.exists():
            print(f"[ERROR] Not found: {p}")
            sys.exit(1)

    REPORT_OUT.parent.mkdir(parents=True, exist_ok=True)

    samples = parse_split(TEST_SPLIT)
    print(f"Test samples loaded : {len(samples)}")
    print(f"Classes             : {CLASS_NAMES}")
    print(f"Global mean         : {GLOBAL_MEAN:.6f} dB")
    print(f"Global std          : {GLOBAL_STD:.6f} dB")
    print()

    all_blocks = []
    results    = {}

    for model_name, onnx_path in [
        ("StreamSenseNet FP32", FP32_ONNX),
        ("StreamSenseNet INT8", INT8_ONNX),
    ]:
        print(f"{'─'*60}")
        print(f"Evaluating: {model_name}")
        print(f"  ONNX : {onnx_path.name}")
        preds, labels, elapsed = run_inference(onnx_path, samples)
        acc = accuracy_score(labels, preds)
        print(f"  Accuracy : {acc*100:.2f}%  ({int(acc*len(labels))}/{len(labels)})")
        print(f"  Elapsed  : {elapsed:.1f}s")

        block = build_report_block(model_name, onnx_path, preds, labels, elapsed)
        all_blocks.append(block)
        results[model_name] = acc

    # ── Comparison summary ────────────────────────────────────────────────────
    fp32_acc = results["StreamSenseNet FP32"]
    int8_acc = results["StreamSenseNet INT8"]
    drop     = (fp32_acc - int8_acc) * 100

    summary_lines = [
        "",
        "=" * 60,
        "  QUANTIZATION ACCURACY SUMMARY",
        "=" * 60,
        f"  FP32 accuracy  : {fp32_acc*100:.2f}%",
        f"  INT8 accuracy  : {int8_acc*100:.2f}%",
        f"  Accuracy drop  : {drop:+.2f}%",
        f"  INT8 budget    : {'PASS' if abs(drop) <= 1.0 else 'FAIL'}  (threshold: ≤1.0%)",
        "=" * 60,
        "",
    ]
    summary = "\n".join(summary_lines)

    # ── Write to report file ──────────────────────────────────────────────────
    full_report = "\n\n".join(all_blocks) + "\n" + summary

    # Append to existing report (training section stays intact)
    with open(REPORT_OUT, "a", encoding="utf-8") as f:
        f.write("\n\n")
        f.write("=" * 60 + "\n")
        f.write("  ONNX EVALUATION (appended by evaluate_onnx.py)\n")
        f.write("=" * 60 + "\n\n")
        f.write(full_report)

    # Also print summary to console
    print()
    print(summary)
    print(f"[DONE] Results appended to: {REPORT_OUT}")

if __name__ == "__main__":
    main()
