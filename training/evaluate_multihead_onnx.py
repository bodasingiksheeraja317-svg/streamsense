"""
evaluate_multihead_onnx.py
Project STREAMSENSE — Track A
Scope 2 / WA-4 Extension

Evaluates BOTH multi-head ONNX models on the full test split:
    onnx_models/streamsense_multihead_fp32.onnx
    onnx_models/streamsense_multihead_int8.onnx

For each model this script:
    1. Runs every test sample through the full MPIC v1.0 preprocessing pipeline.
    2. Feeds the resulting [1, 1, 64, 97] tensor through ORT.
    3. Extracts the 'logits' output ([1, 10]) for classification.
    4. Verifies that 'embedding' ([1, 128]) and 'novelty_score' ([1, 1]) are
       also present and correctly shaped (hard assert — fails loudly if broken).
    5. Computes top-1 accuracy, per-class precision / recall / F1, and confusion
       matrix, and prints a full report.
    6. Compares FP32 vs INT8 accuracy and checks that the INT8 drop is ≤ 1.0%.
    7. Appends a timestamped result block to
       evaluation/multihead_onnx_evaluation_report.txt

Output contract verified per ERR v1.0:
    logits        float32  [1, 10]   — classification head
    embedding     float32  [1, 128]  — projection head
    novelty_score float32  [1,  1]   — must be exactly 2-D

Usage (from project root):
    python training/evaluate_multihead_onnx.py

Optional overrides:
    --fp32   PATH   FP32 multihead ONNX (default: onnx_models/streamsense_multihead_fp32.onnx)
    --int8   PATH   INT8 multihead ONNX (default: onnx_models/streamsense_multihead_int8.onnx)
    --test   PATH   Test split file    (default: data/splits/test_files.txt)
    --stats  PATH   Normalization JSON (default: stats/normalization_stats.json)
    --labels PATH   Class labels JSON  (default: class_labels.json)
    --out    PATH   Report output      (default: evaluation/multihead_onnx_evaluation_report.txt)
    --batch  INT    Inference batch size (default: 64)
    --skip-int8     Skip INT8 evaluation (FP32 only)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torchaudio
import onnxruntime as ort
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

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

# Expected output shapes — ERR v1.0
EXPECTED_LOGITS_SHAPE        = (1, 10)
EXPECTED_EMBEDDING_SHAPE     = (1, 128)
EXPECTED_NOVELTY_SHAPE       = (1, 1)    # must be exactly 2-D

# INT8 budget — same as Scope 1 baseline
INT8_ACCURACY_DROP_BUDGET = 1.0   # percentage points

# ── Mel transform (built once, CPU, reused for every file) ────────────────────
_mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=SAMPLE_RATE,
    n_fft=N_FFT,
    hop_length=HOP_LENGTH,
    n_mels=N_MELS,
    window_fn=torch.hann_window,
    center=CENTER,
    power=POWER,
)


# ── MPIC v1.0 preprocessing pipeline ─────────────────────────────────────────

def _build_preprocessor(global_mean: float, global_std: float):
    """
    Returns a preprocess(raw) -> np.ndarray [1, 1, 64, 97] callable
    that implements the full 9-step MPIC v1.0 pipeline with the
    supplied normalization statistics.
    """
    def preprocess(raw: np.ndarray) -> np.ndarray:
        """
        Input : float32 numpy array, shape [T] or [C, T], any length
        Output: float32 numpy array, shape [1, 1, 64, 97]

        Steps:
            1-2. Accept and downmix to mono.
            3.   Pad (zeros right) or crop to exactly FRAME_LEN samples.
            4.   MelSpectrogram  → [1, 64, 97]
            5.   10 * log10(mel + LOG_EPS)
            6.   clamp ≥ CLIP_FLOOR_DB
            7.   (mel - global_mean) / global_std
            8.   unsqueeze batch → [1, 1, 64, 97]
        """
        waveform = torch.from_numpy(raw.copy()).float()

        # Step 1-2: ensure [1, T] mono
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)          # [1, T]
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)   # [1, T]

        # Step 3: pad or crop to exactly FRAME_LEN
        length = waveform.shape[1]
        if length < FRAME_LEN:
            waveform = torch.nn.functional.pad(waveform, (0, FRAME_LEN - length))
        elif length > FRAME_LEN:
            waveform = waveform[:, :FRAME_LEN]

        # Steps 4-6: mel spectrogram + log scaling + floor clamp
        mel = _mel_transform(waveform)                # [1, 64, 97]
        mel = 10.0 * torch.log10(mel + LOG_EPS)
        mel = torch.clamp(mel, min=CLIP_FLOOR_DB)

        # Step 7: global normalisation (MPIC v1.0 frozen stats)
        mel = (mel - global_mean) / global_std

        # Step 8: add batch dimension → [1, 1, 64, 97]
        mel = mel.unsqueeze(0)

        return mel.numpy().astype(np.float32)

    return preprocess


# ── Split file parser ─────────────────────────────────────────────────────────

def _parse_split(split_file: Path, idx_to_label: dict[int, str]) -> list[tuple[Path, int]]:
    """
    Parse test_files.txt.  Expected line format:
        C:\\STREAMSENSE\\data\\raw\\yes\\file.wav | yes | 0
    Returns list of (wav_path, class_index) tuples.
    Only keeps entries whose class_index is in idx_to_label.
    """
    label_to_idx = {v: k for k, v in idx_to_label.items()}
    samples: list[tuple[Path, int]] = []
    skipped = 0

    with open(split_file, "r", encoding="utf-8") as fh:
        for lineno, raw_line in enumerate(fh, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            parts = [p.strip() for p in line.split("|")]
            if len(parts) != 3:
                skipped += 1
                continue

            wav_path = Path(parts[0])
            try:
                class_idx = int(parts[2])
            except ValueError:
                class_idx = label_to_idx.get(parts[1], -1)

            if class_idx not in idx_to_label:
                skipped += 1
                continue

            samples.append((wav_path, class_idx))

    if skipped:
        print(f"  [WARN] {skipped} line(s) skipped in {split_file.name} "
              f"(malformed or out-of-range class).")
    return samples


# ── Shape gate — ERR v1.0 contract verification ───────────────────────────────

def _verify_output_contract(session: ort.InferenceSession, model_label: str) -> None:
    """
    Runs a single zero-valued dummy input through the session and asserts
    all three output heads are present with the correct shapes.
    Hard sys.exit(1) on any failure — a broken output contract means Track B,
    C, D, E cannot integrate against this model.
    """
    dummy = np.zeros((1, 1, 64, 97), dtype=np.float32)
    output_names = [o.name for o in session.get_outputs()]
    input_name   = session.get_inputs()[0].name

    outputs = session.run(output_names, {input_name: dummy})
    output_map = dict(zip(output_names, outputs))

    passed = True
    sep = "─" * 54

    print(f"\n  {sep}")
    print(f"  ERR v1.0 output contract — {model_label}")
    print(f"  {sep}")

    def _check(name: str, expected_shape: tuple) -> None:
        nonlocal passed
        if name not in output_map:
            print(f"  [FAIL]  '{name}' : MISSING from model outputs")
            passed = False
            return
        actual = output_map[name].shape
        ok = actual == expected_shape
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}]  '{name}' : {actual}  (expected {expected_shape})")
        if not ok:
            passed = False

    _check("logits",        EXPECTED_LOGITS_SHAPE)
    _check("embedding",     EXPECTED_EMBEDDING_SHAPE)
    _check("novelty_score", EXPECTED_NOVELTY_SHAPE)

    if not passed:
        print(f"\n  [ABORT] Output contract FAILED for {model_label}.")
        print("          ERR v1.0 requires all three outputs with exact shapes.")
        print("          Re-export the model and re-run this script.")
        sys.exit(1)

    print(f"  {sep}")
    print(f"  Output contract: PASS — all three heads present and correctly shaped.")
    print(f"  {sep}\n")


# ── Batched inference ─────────────────────────────────────────────────────────

def _run_inference(
    onnx_path: Path,
    samples: list[tuple[Path, int]],
    preprocess,
    batch_size: int,
    model_label: str,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Runs the multihead ONNX model on all samples.

    Returns:
        preds   : np.ndarray[int]   top-1 predicted class index per sample
        labels  : np.ndarray[int]   ground-truth class index per sample
        elapsed : float             wall-clock seconds
    """
    sess_opts = ort.SessionOptions()
    sess_opts.inter_op_num_threads = 4
    sess_opts.intra_op_num_threads = 4
    session = ort.InferenceSession(str(onnx_path), sess_opts=sess_opts)

    # Verify output contract before running the full dataset
    _verify_output_contract(session, model_label)

    input_name   = session.get_inputs()[0].name
    # We only need logits for accuracy; collect its name robustly
    logits_name  = None
    for out in session.get_outputs():
        if out.name == "logits":
            logits_name = out.name
            break
    if logits_name is None:
        # Fallback: first output (already caught by contract gate above, but be safe)
        logits_name = session.get_outputs()[0].name

    all_preds:  list[int] = []
    all_labels: list[int] = []
    errors = 0
    t0 = time.time()

    # The multihead model has a static batch dimension of 1 (frozen by MPIC v1.0
    # output contract [1,1,64,97] → outputs [1,10]/[1,128]/[1,1]).
    # ORT rejects any batch size other than 1, so we run one sample at a time.
    # The `batch_size` arg is accepted for CLI compatibility but is not used here.
    total = len(samples)
    for i, (wav_path, class_idx) in enumerate(samples):
        if (i + 1) % 500 == 0 or (i + 1) == total:
            pct = 100.0 * (i + 1) / total
            print(f"    [{i+1:>5}/{total}]  {pct:5.1f}%", flush=True)

        if not wav_path.exists():
            errors += 1
            continue

        try:
            waveform, sr = torchaudio.load(str(wav_path))
            raw = waveform.squeeze(0).numpy().astype(np.float32)
            inp = preprocess(raw)           # [1, 1, 64, 97]  — batch=1
            logits = session.run([logits_name], {input_name: inp})[0]  # [1, 10]
            pred   = int(np.argmax(logits, axis=1)[0])
            all_preds.append(pred)
            all_labels.append(class_idx)
        except Exception as exc:
            print(f"    [WARN] Error on {wav_path.name}: {exc}")
            errors += 1
            continue
    elapsed = time.time() - t0

    if errors:
        print(f"  [WARN] Skipped {errors}/{total} files (missing or unreadable).")

    return np.array(all_preds, dtype=int), np.array(all_labels, dtype=int), elapsed


# ── Report block builder ──────────────────────────────────────────────────────

def _build_report_block(
    model_label: str,
    onnx_path: Path,
    preds: np.ndarray,
    labels: np.ndarray,
    elapsed: float,
    class_names: list[str],
    num_classes: int,
    timestamp: str,
) -> str:
    acc = accuracy_score(labels, preds)
    cm  = confusion_matrix(labels, preds, labels=list(range(num_classes)))
    cls_report = classification_report(
        labels, preds,
        target_names=class_names,
        digits=4,
        zero_division=0,
    )

    lines: list[str] = []
    sep = "=" * 60
    lines.append(sep)
    lines.append(f"  Model        : {model_label}")
    lines.append(f"  ONNX file    : {onnx_path.name}")
    lines.append(f"  Timestamp    : {timestamp}")
    lines.append(f"  Test samples : {len(labels)}")
    lines.append(f"  Accuracy     : {acc*100:.2f}%  ({int(acc*len(labels))}/{len(labels)})")
    lines.append(f"  Elapsed      : {elapsed:.1f}s")
    lines.append(sep)
    lines.append("")
    lines.append("Per-class report:")
    lines.append(cls_report)
    lines.append("Per-class accuracy:")
    for i, name in enumerate(class_names):
        correct = int(cm[i, i])
        support = int(cm[i].sum())
        pct = 100.0 * correct / support if support > 0 else 0.0
        lines.append(f"  {name:<10} {correct}/{support}  ({pct:.2f}%)")
    lines.append("")
    lines.append(f"Confusion matrix (rows=true, cols=predicted):")
    lines.append(f"Classes: " + ", ".join(f"{i}={n}" for i, n in enumerate(class_names)))
    for row in cm:
        lines.append("  " + str(row.tolist()))
    lines.append("")
    lines.append(f"MPIC version   : 1.0")
    lines.append(f"Architecture   : StreamSenseWrapper (multi-head, Scope 2 WA-4)")
    lines.append(sep)

    return "\n".join(lines)


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    # Resolve project root: this file lives in training/, root is one level up.
    this_dir = Path(__file__).resolve().parent
    root     = this_dir.parent

    p = argparse.ArgumentParser(
        description=(
            "STREAMSENSE — evaluate both multi-head ONNX models on the test split. "
            "Verifies ERR v1.0 output contract (3 heads, exact shapes) and reports "
            "per-class accuracy, confusion matrix, and FP32 vs INT8 accuracy gap."
        )
    )
    p.add_argument(
        "--fp32",
        type=Path,
        default=root / "onnx_models" / "streamsense_multihead_fp32.onnx",
        help="Path to FP32 multihead ONNX model.",
    )
    p.add_argument(
        "--int8",
        type=Path,
        default=root / "onnx_models" / "streamsense_multihead_int8.onnx",
        help="Path to INT8 QDQ multihead ONNX model.",
    )
    p.add_argument(
        "--test",
        type=Path,
        default=root / "data" / "splits" / "test_files.txt",
        help="Test split file (pipe-delimited: path | label | index).",
    )
    p.add_argument(
        "--stats",
        type=Path,
        default=root / "stats" / "normalization_stats.json",
        help="Normalization stats JSON (global_mean, global_std).",
    )
    p.add_argument(
        "--labels",
        type=Path,
        default=root / "class_labels.json",
        help='Class labels JSON (e.g. {"0": "yes", ...}).',
    )
    p.add_argument(
        "--out",
        type=Path,
        default=root / "evaluation" / "multihead_onnx_evaluation_report.txt",
        help="Output report file (appended, not overwritten).",
    )
    p.add_argument(
        "--batch",
        type=int,
        default=64,
        help="Inference batch size (default: 64).",
    )
    p.add_argument(
        "--skip-int8",
        action="store_true",
        help="Skip INT8 evaluation and run FP32 only.",
    )
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args = _parse_args()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    sep  = "=" * 60
    sep2 = "─" * 60
    print(sep)
    print("  STREAMSENSE — Multi-Head ONNX Test-Set Evaluation")
    print("  Scope 2 / WA-4  |  MPIC v1.0  |  ERR v1.0")
    print(f"  Timestamp : {timestamp}")
    print(sep)

    # ── Load normalization stats ───────────────────────────────────────────────
    if not args.stats.exists():
        print(f"[ERROR] Normalization stats not found: {args.stats}")
        sys.exit(1)
    with open(args.stats, "r") as fh:
        stats = json.load(fh)
    global_mean = float(stats["global_mean"])
    global_std  = float(stats["global_std"])
    print(f"\nNormalization stats loaded from: {args.stats.name}")
    print(f"  global_mean : {global_mean:.6f} dB")
    print(f"  global_std  : {global_std:.6f} dB")

    # ── Build MPIC v1.0 preprocessor ──────────────────────────────────────────
    preprocess = _build_preprocessor(global_mean, global_std)

    # ── Load class labels ─────────────────────────────────────────────────────
    if not args.labels.exists():
        print(f"[ERROR] Class labels not found: {args.labels}")
        sys.exit(1)
    with open(args.labels, "r") as fh:
        raw_labels = json.load(fh)
    # Support both {"0": "yes"} and {"yes": 0} formats
    first_val = list(raw_labels.values())[0]
    if isinstance(first_val, int):
        idx_to_label: dict[int, str] = {v: k for k, v in raw_labels.items()}
    else:
        idx_to_label = {int(k): v for k, v in raw_labels.items()}

    num_classes = len(idx_to_label)
    class_names = [idx_to_label[i] for i in range(num_classes)]
    print(f"\nClasses ({num_classes}): {', '.join(class_names)}")

    # ── Parse test split ───────────────────────────────────────────────────────
    if not args.test.exists():
        print(f"[ERROR] Test split file not found: {args.test}")
        sys.exit(1)
    samples = _parse_split(args.test, idx_to_label)
    if not samples:
        print(f"[ERROR] No valid samples parsed from {args.test}.")
        sys.exit(1)
    print(f"\nTest split     : {args.test.name}")
    print(f"Total samples  : {len(samples)}")

    # ── Determine which models to evaluate ────────────────────────────────────
    models_to_run: list[tuple[str, Path]] = []

    if not args.fp32.exists():
        print(f"\n[ERROR] FP32 multihead ONNX not found: {args.fp32}")
        print("        Run training/export_multihead_onnx.py first (WA-4).")
        sys.exit(1)
    models_to_run.append(("StreamSenseWrapper FP32 (multihead)", args.fp32))

    if not args.skip_int8:
        if not args.int8.exists():
            print(f"\n[WARN] INT8 multihead ONNX not found: {args.int8}")
            print("       Skipping INT8 evaluation.")
        else:
            models_to_run.append(("StreamSenseWrapper INT8 (multihead)", args.int8))

    # ── Run evaluation for each model ─────────────────────────────────────────
    results: dict[str, float]  = {}
    report_blocks: list[str]   = []

    for model_label, onnx_path in models_to_run:
        print(f"\n{sep2}")
        print(f"  Evaluating : {model_label}")
        print(f"  ONNX       : {onnx_path.name}")
        print(f"  File size  : {onnx_path.stat().st_size / 1024:.1f} KB")
        print(f"{sep2}")

        preds, labels, elapsed = _run_inference(
            onnx_path=onnx_path,
            samples=samples,
            preprocess=preprocess,
            batch_size=args.batch,
            model_label=model_label,
        )

        acc = accuracy_score(labels, preds)
        correct = int(acc * len(labels))
        print(f"\n  Accuracy : {acc*100:.2f}%  ({correct}/{len(labels)})")
        print(f"  Elapsed  : {elapsed:.1f}s  "
              f"({elapsed / len(labels) * 1000:.1f} ms/sample)")

        results[model_label] = acc

        block = _build_report_block(
            model_label=model_label,
            onnx_path=onnx_path,
            preds=preds,
            labels=labels,
            elapsed=elapsed,
            class_names=class_names,
            num_classes=num_classes,
            timestamp=timestamp,
        )
        report_blocks.append(block)

        # Print per-class breakdown to console
        print(f"\n  Per-class accuracy:")
        cm = confusion_matrix(labels, preds, labels=list(range(num_classes)))
        for i, name in enumerate(class_names):
            correct_i = int(cm[i, i])
            support_i = int(cm[i].sum())
            pct_i     = 100.0 * correct_i / support_i if support_i > 0 else 0.0
            bar       = "█" * int(pct_i / 5)   # ASCII bar, 20-char max
            print(f"    {name:<8}  {correct_i:>4}/{support_i:<4}  {pct_i:6.2f}%  {bar}")

    # ── FP32 vs INT8 comparison (only when both were evaluated) ───────────────
    fp32_key  = "StreamSenseWrapper FP32 (multihead)"
    int8_key  = "StreamSenseWrapper INT8 (multihead)"
    has_both  = fp32_key in results and int8_key in results

    summary_lines: list[str] = [
        "",
        sep,
        "  MULTI-HEAD ONNX ACCURACY SUMMARY",
        sep,
    ]
    for label, acc in results.items():
        summary_lines.append(f"  {label:<42} : {acc*100:.2f}%")

    if has_both:
        fp32_acc = results[fp32_key]
        int8_acc = results[int8_key]
        drop     = (fp32_acc - int8_acc) * 100
        budget_ok = abs(drop) <= INT8_ACCURACY_DROP_BUDGET
        summary_lines.append("")
        summary_lines.append(f"  Accuracy drop (FP32 → INT8) : {drop:+.2f}%")
        summary_lines.append(
            f"  INT8 budget (≤{INT8_ACCURACY_DROP_BUDGET:.1f}%)           : "
            f"{'PASS' if budget_ok else 'FAIL'}"
        )
        if not budget_ok:
            summary_lines.append(
                f"  [WARN] INT8 drop ({abs(drop):.2f}%) exceeds {INT8_ACCURACY_DROP_BUDGET:.1f}% budget. "
                "Recalibrate PTQ."
            )

    summary_lines.append(sep)
    summary_lines.append("")
    summary = "\n".join(summary_lines)
    print(summary)

    # ── ERR v1.0 contract reminder ────────────────────────────────────────────
    print("  ERR v1.0 output contract (verified for all models above):")
    print(f"    logits        float32  {EXPECTED_LOGITS_SHAPE}   — logits head")
    print(f"    embedding     float32  {EXPECTED_EMBEDDING_SHAPE} — embed head")
    print(f"    novelty_score float32  {EXPECTED_NOVELTY_SHAPE}   — novelty head (2-D enforced)")
    print()

    # ── Write report file ─────────────────────────────────────────────────────
    args.out.parent.mkdir(parents=True, exist_ok=True)
    full_report = (
        f"\n\n{sep}\n"
        f"  MULTI-HEAD ONNX EVALUATION (evaluate_multihead_onnx.py)\n"
        f"  Timestamp : {timestamp}\n"
        f"{sep}\n\n"
        + "\n\n".join(report_blocks)
        + "\n"
        + summary
    )

    with open(args.out, "a", encoding="utf-8") as fh:
        fh.write(full_report)

    print(f"[DONE] Results appended to: {args.out}")


if __name__ == "__main__":
    main()
