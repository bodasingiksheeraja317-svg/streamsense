"""
run_gv_regression_1000.py
Project STREAMSENSE — Track A
End-to-end Golden Vector regression test on golden_vectors_1000/.

For each of the 1000 golden vectors:
    1. Load raw waveform from raw/GV1K_NNNN_label.bin           [16000] float32
    2. Run through mel_pipeline.preprocess() (Steps 1-8)         -> [1,1,64,97]
    3. Compare resulting normalized tensor against the
       precomputed normalized/GV1K_NNNN_label_norm.bin           [64,97] float32
       (max abs error must be <= tolerance from manifest.json)
    4. Run the [1,1,64,97] tensor through the ONNX model (FP32 by default)
    5. argmax -> predicted class, compare against label in labels/GV1K_NNNN_label_label.txt

Prints colored PASS/FAIL per-vector progress (summarized every 100), then a
final summary table: pipeline parity pass rate, model accuracy, max error
stats, and an overall PASS/FAIL verdict against manifest tolerance.

This is the script intended for the joint end-to-end GV regression session
with Kavish (Track B) — run on your machine against golden_vectors_1000/
while he runs his C++ equivalent against the same .bin files, then compare
results live.

Run from C:\\STREAMSENSE\\training\\:
    python run_gv_regression_1000.py
    python run_gv_regression_1000.py --model ..\\onnx_models\\streamsense_model_int8.onnx
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import onnxruntime as ort

from mel_pipeline import preprocess, OUTPUT_SHAPE

# ── ANSI colors (works in VS Code integrated terminal) ────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
RESET  = "\033[0m"
BOLD   = "\033[1m"

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT          = Path(r"C:\STREAMSENSE")
GV_DIR        = ROOT / "golden_vectors_1000"
RAW_DIR       = GV_DIR / "raw"
NORM_DIR      = GV_DIR / "normalized"
LABEL_DIR     = GV_DIR / "labels"
MANIFEST_PATH = GV_DIR / "manifest.json"

CLASS_LABELS_FILE = ROOT / "class_labels.json"
DEFAULT_MODEL     = ROOT / "onnx_models" / "streamsense_model_fp32.onnx"

FRAME_LEN = 16000
N_MELS    = 64


def load_class_labels(path: Path):
    with open(path, "r") as f:
        raw = json.load(f)
    labels = [None] * len(raw)
    for k, v in raw.items():
        labels[int(k)] = v
    return labels


def load_bin(path: Path, shape, dtype="<f4"):
    arr = np.fromfile(str(path), dtype=dtype)
    return arr.reshape(shape)


def main():
    parser = argparse.ArgumentParser(
        description="STREAMSENSE end-to-end GV regression on golden_vectors_1000/"
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL,
                         help=f"ONNX model path (default: {DEFAULT_MODEL})")
    parser.add_argument("--limit", type=int, default=None,
                         help="Limit number of vectors tested (default: all 1000)")
    args = parser.parse_args()

    print("=" * 64)
    print(f"{BOLD}STREAMSENSE — End-to-End GV Regression (golden_vectors_1000){RESET}")
    print("=" * 64)

    # ── Validate inputs ────────────────────────────────────────────────────
    for p, name in [(MANIFEST_PATH, "golden_vectors_1000/manifest.json"),
                     (args.model,    "ONNX model"),
                     (CLASS_LABELS_FILE, "class_labels.json")]:
        if not p.exists():
            print(f"{RED}[ERROR] Not found: {p} ({name}){RESET}")
            sys.exit(1)

    with open(MANIFEST_PATH, "r") as f:
        manifest = json.load(f)

    tolerance = float(manifest["tolerance_max_abs_error"])
    vectors   = manifest["vectors"]
    mel_shape = tuple(manifest["vectors"][next(iter(vectors))]["norm_shape"])  # [64,97]

    class_labels = load_class_labels(CLASS_LABELS_FILE)

    n_vectors = len(vectors)
    if args.limit is not None:
        n_vectors = min(n_vectors, args.limit)

    print(f"Model       : {args.model.name}")
    print(f"Vectors     : {n_vectors}")
    print(f"Tolerance   : {tolerance}")
    print(f"Mel shape   : {mel_shape}")
    print()

    # ── Load ONNX session ──────────────────────────────────────────────────
    sess = ort.InferenceSession(str(args.model), providers=["CPUExecutionProvider"])
    input_name  = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    print(f"ONNX input='{input_name}'  output='{output_name}'")
    print()

    # ── Run regression ─────────────────────────────────────────────────────
    keys = sorted(vectors.keys(), key=lambda k: int(k))[:n_vectors]

    pipeline_pass = 0
    pipeline_fail = 0
    accuracy_correct = 0

    max_errors = []
    fail_details = []

    REPORT_EVERY = 100

    for i, key in enumerate(keys):
        v = vectors[key]
        gv_name   = v["gv_name"]
        class_idx = v["class_idx"]

        raw_path   = RAW_DIR  / v["raw_bin"]
        norm_path  = NORM_DIR / v["norm_bin"]
        label_path = LABEL_DIR / f"{gv_name}_label.txt"

        # 1. Load raw waveform
        raw = load_bin(raw_path, (FRAME_LEN,))  # [16000] float32

        # 2. Run through mel_pipeline (Steps 1-8)
        mel_tensor = preprocess(raw)  # [1,1,64,97] torch.Tensor

        if tuple(mel_tensor.shape) != OUTPUT_SHAPE:
            pipeline_fail += 1
            fail_details.append((gv_name, "shape", f"got {tuple(mel_tensor.shape)}"))
            continue

        computed_norm = mel_tensor.squeeze(0).squeeze(0).numpy()  # [64,97]

        # 3. Compare against precomputed normalized .bin
        precomputed_norm = load_bin(norm_path, mel_shape)  # [64,97]
        abs_diff = np.abs(computed_norm - precomputed_norm)
        max_err  = float(abs_diff.max())
        max_errors.append(max_err)

        parity_ok = max_err <= tolerance
        if parity_ok:
            pipeline_pass += 1
        else:
            pipeline_fail += 1
            fail_details.append((gv_name, "parity", f"max_err={max_err:.6e} > {tolerance}"))

        # 4. Run ONNX inference
        input_array = mel_tensor.numpy().astype(np.float32)
        logits = sess.run([output_name], {input_name: input_array})[0][0]  # [10]
        pred_idx = int(np.argmax(logits))

        if pred_idx == class_idx:
            accuracy_correct += 1

        # Progress
        if (i + 1) % REPORT_EVERY == 0 or i == 0:
            status_color = GREEN if (parity_ok and pred_idx == class_idx) else RED
            print(f"  [{i+1:>4}/{n_vectors}]  {gv_name:<22} "
                  f"max_err={max_err:.2e}  "
                  f"pred={class_labels[pred_idx]:<6} true={class_labels[class_idx]:<6} "
                  f"{status_color}{'OK' if (parity_ok and pred_idx == class_idx) else 'CHECK'}{RESET}")

    # ── Summary ─────────────────────────────────────────────────────────────
    max_errors_arr = np.array(max_errors) if max_errors else np.array([0.0])
    overall_max_err = float(max_errors_arr.max())
    overall_mean_err = float(max_errors_arr.mean())

    accuracy = 100.0 * accuracy_correct / n_vectors if n_vectors > 0 else 0.0
    parity_pct = 100.0 * pipeline_pass / n_vectors if n_vectors > 0 else 0.0

    print()
    print("=" * 64)
    print(f"{BOLD}SUMMARY{RESET}")
    print("=" * 64)
    print(f"  Vectors tested        : {n_vectors}")
    print(f"  Pipeline parity PASS  : {pipeline_pass}/{n_vectors}  ({parity_pct:.2f}%)")
    print(f"  Pipeline parity FAIL  : {pipeline_fail}/{n_vectors}")
    print(f"  Max abs error (worst) : {overall_max_err:.6e}  (tolerance {tolerance})")
    print(f"  Mean abs error        : {overall_mean_err:.6e}")
    print(f"  Model accuracy        : {accuracy_correct}/{n_vectors}  ({accuracy:.2f}%)")

    if fail_details:
        print(f"\n  {YELLOW}First failures:{RESET}")
        for gv_name, kind, detail in fail_details[:10]:
            print(f"    [{kind}] {gv_name}: {detail}")
        if len(fail_details) > 10:
            print(f"    ... and {len(fail_details) - 10} more")

    print()
    overall_pass = (pipeline_fail == 0)
    if overall_pass:
        print(f"{GREEN}{BOLD}[PASS] All {n_vectors} golden vectors within tolerance "
              f"({tolerance}). End-to-end GV regression PASSED.{RESET}")
    else:
        print(f"{RED}{BOLD}[FAIL] {pipeline_fail}/{n_vectors} golden vectors exceeded "
              f"tolerance ({tolerance}). Review failures above.{RESET}")

    print("=" * 64)

    sys.exit(0 if overall_pass else 1)


if __name__ == "__main__":
    main()
