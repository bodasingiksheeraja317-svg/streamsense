"""
evaluate_qonnx.py
Project STREAMSENSE — Track A
Scope 2 / QAT Extension — QONNX Golden-Vector Evaluation

Loads streamsense_multihead.qonnx and evaluates it against all 1000 GV1K
normalized vectors using the qonnx runtime (required for onnx.brevitas custom ops).

Key facts:
  - GV1K vectors are already-normalized mel spectrograms stored as flat
    float32 little-endian binary, shape [64 x 97] = 6208 floats = 24832 bytes.
  - Fed DIRECTLY to the model — NO additional preprocessing applied.
  - Reshape to [1, 1, 64, 97] float32 before feeding.
  - Label parsed from filename stem: GV1K_NNNN_<label>_norm
    parts = stem.split("_")  ->  label = parts[2].lower()
  - Class map: yes=0, no=1, up=2, down=3, left=4, right=5, on=6, off=7, stop=8, go=9
  - Minimum passing threshold: 90.0% top-1 accuracy.
  - ERR v1.0 output order: logits [1,10], embedding [1,128], novelty_score [1,1]
    NOTE: qonnx export may give outputs auto-generated names (e.g. '143', '147').
          This script accesses them by index (0, 1, 2), not by name string.

Usage (from project root):
    python training/evaluate_qonnx.py

Optional overrides:
    --qonnx  PATH   QONNX model file   (default: onnx_models/streamsense_multihead.qonnx)
    --gvk    PATH   GV1K normalized dir (default: golden_vectors_1000/normalized)
    --out    PATH   Report output file  (default: evaluation/qonnx_evaluation_report.txt)
    --pass-threshold FLOAT  Min top-1%% to pass (default: 90.0)

Requirements:
    pip install qonnx
    (onnxruntime is also required as a qonnx dependency, but do NOT use it directly
     to load .qonnx files — it cannot handle onnx.brevitas custom ops.)
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

# ── qonnx runtime import ──────────────────────────────────────────────────────
# IMPORTANT: Do NOT use onnxruntime directly to load .qonnx files.
# .qonnx uses onnx.brevitas custom ops (Quant, BipolarQuant, etc.) that are
# not registered in vanilla onnxruntime. Use qonnx's own executor instead.
try:
    from qonnx.core.modelwrapper import ModelWrapper
    from qonnx.core.onnx_exec import execute_onnx
    from qonnx.transformation.infer_shapes import InferShapes
except ImportError:
    print("[ERROR] qonnx is not installed.")
    print()
    print("  Install it with:")
    print("  c:\\STREAMSENSE\\streamsense-env-win\\Scripts\\python.exe -m pip install qonnx")
    print()
    print("  Then re-run this script.")
    sys.exit(1)

# ── Project root (this file lives in training/) ───────────────────────────────
_THIS_DIR = Path(__file__).resolve().parent
_ROOT     = _THIS_DIR.parent

# ── Constants ─────────────────────────────────────────────────────────────────
TARGET_CLASSES = {
    "yes": 0, "no": 1, "up": 2, "down": 3,
    "left": 4, "right": 5, "on": 6, "off": 7,
    "stop": 8, "go": 9,
}
IDX_TO_LABEL = {v: k for k, v in TARGET_CLASSES.items()}
NUM_CLASSES  = 10
CLASS_NAMES  = [IDX_TO_LABEL[i] for i in range(NUM_CLASSES)]

# GV1K binary spec
GV1K_FLOATS = 64 * 97         # 6208
GV1K_BYTES  = GV1K_FLOATS * 4 # 24832
GV1K_SHAPE  = (1, 1, 64, 97)  # model input shape

# ERR v1.0 expected output shapes — in index order
# NOTE: output names in the .qonnx graph may be auto-generated ('143', '147', etc.)
# We verify shapes by index, not by name.
EXPECTED_SHAPES_BY_INDEX = [
    (1, 10),   # index 0 — logits
    (1, 128),  # index 1 — embedding
    (1, 1),    # index 2 — novelty_score
]
OUTPUT_LABELS = ["logits", "embedding", "novelty_score"]


# ── Label parser ──────────────────────────────────────────────────────────────

def _parse_label(stem: str) -> int | None:
    """
    Pattern: GV1K_NNNN_<label>_norm
    parts = stem.split("_")  ->  label = parts[2].lower()
    """
    parts = stem.split("_")
    if len(parts) < 4:
        return None
    return TARGET_CLASSES.get(parts[2].lower(), None)


# ── Load and prepare model ────────────────────────────────────────────────────

def load_qonnx_model(qonnx_path: Path) -> tuple[ModelWrapper, list[str], str]:
    """
    Load the .qonnx model, run InferShapes (required before execute_onnx),
    and return (model_wrapper, output_names, input_name).

    InferShapes must be called before execute_onnx — qonnx's executor
    requires all tensor shapes to be annotated in the graph.
    """
    print(f"  Loading QONNX : {qonnx_path}")
    print(f"  File size     : {qonnx_path.stat().st_size / 1024:.1f} KB")

    model = ModelWrapper(str(qonnx_path))
    model = model.transform(InferShapes())  # mandatory before execute_onnx

    input_name   = model.graph.input[0].name
    output_names = [o.name for o in model.graph.output]

    print(f"  Input node    : {input_name!r}")
    print(f"  Output nodes  : {output_names}")
    print(f"  (Outputs accessed by index 0/1/2, not by name string)")

    return model, output_names, input_name


# ── ERR v1.0 output contract gate ────────────────────────────────────────────

def verify_output_contract(
    model       : ModelWrapper,
    output_names: list[str],
    input_name  : str,
    model_label : str,
) -> None:
    """
    Feed a zero tensor and check all three output heads are present
    with exactly the right shapes. Hard sys.exit(1) on any failure.
    Uses qonnx execute_onnx, not onnxruntime.
    """
    dummy = np.zeros(GV1K_SHAPE, dtype=np.float32)
    odict = execute_onnx(model, {input_name: dummy})

    sep = "─" * 54
    print(f"\n  {sep}")
    print(f"  ERR v1.0 output contract — {model_label}")
    print(f"  {sep}")

    if len(output_names) < 3:
        print(f"  [FAIL] Expected 3 outputs, got {len(output_names)}: {output_names}")
        print(f"  [ABORT] Output contract FAILED. Re-export the QONNX.")
        sys.exit(1)

    passed = True
    for idx, (label, expected) in enumerate(zip(OUTPUT_LABELS, EXPECTED_SHAPES_BY_INDEX)):
        out_key = output_names[idx]
        if out_key not in odict:
            print(f"  [FAIL]  output[{idx}] '{out_key}' ({label}) : MISSING from odict")
            passed = False
        else:
            actual = odict[out_key].shape
            ok     = actual == expected
            print(f"  {'[PASS]' if ok else '[FAIL]'}  output[{idx}] ({label:<13}) : "
                  f"{actual}  (expected {expected})")
            if not ok:
                passed = False

    print(f"  {sep}")
    if not passed:
        print(f"\n  [ABORT] Output contract FAILED for {model_label}.")
        print("          Re-export the QONNX and re-run.")
        sys.exit(1)
    print(f"  Output contract: PASS\n")


# ── Main evaluation ───────────────────────────────────────────────────────────

def evaluate(
    qonnx_path    : Path,
    gvk_dir       : Path,
    pass_threshold: float,
) -> dict:
    """
    Run every GV1K .bin file through the QONNX model.
    Returns a result dict with full metrics.
    """
    # Load model once — InferShapes runs here
    model, output_names, input_name = load_qonnx_model(qonnx_path)

    # ERR v1.0 contract check
    verify_output_contract(model, output_names, input_name, qonnx_path.name)

    # Logits are always at index 0
    logits_key = output_names[0]

    # Collect GV1K files
    bin_files = sorted(gvk_dir.glob("*_norm.bin"))
    if not bin_files:
        print(f"[ERROR] No *_norm.bin files found in {gvk_dir}")
        sys.exit(1)
    print(f"  GV1K vectors  : {len(bin_files)} files found in {gvk_dir.name}/")

    # Per-class accumulators
    per_class_correct = [0] * NUM_CLASSES
    per_class_total   = [0] * NUM_CLASSES
    confusion         = [[0] * NUM_CLASSES for _ in range(NUM_CLASSES)]

    all_preds : list[int] = []
    all_labels: list[int] = []

    correct = 0
    wrong   = 0
    skipped = 0

    # Inference loop
    total = len(bin_files)
    for i, bf in enumerate(bin_files):
        if (i + 1) % 100 == 0 or (i + 1) == total:
            print(f"    [{i+1:>4}/{total}]  {100*(i+1)/total:5.1f}%", flush=True)

        true_idx = _parse_label(bf.stem)
        if true_idx is None:
            skipped += 1
            continue

        raw = np.fromfile(str(bf), dtype="<f4")
        if raw.size != GV1K_FLOATS:
            print(f"  [WARN] {bf.name}: expected {GV1K_FLOATS} floats, "
                  f"got {raw.size} — skipping")
            skipped += 1
            continue

        inp   = raw.reshape(GV1K_SHAPE).astype(np.float32)
        odict = execute_onnx(model, {input_name: inp})

        logits = odict[logits_key]              # [1, 10]
        pred   = int(np.argmax(logits, axis=1)[0])

        all_preds.append(pred)
        all_labels.append(true_idx)
        per_class_total[true_idx] += 1
        confusion[true_idx][pred] += 1

        if pred == true_idx:
            correct += 1
            per_class_correct[true_idx] += 1
        else:
            wrong += 1

    total_checked = correct + wrong
    top1_acc      = 100.0 * correct / total_checked if total_checked > 0 else 0.0

    return {
        "total_files"       : total,
        "total_checked"     : total_checked,
        "correct"           : correct,
        "wrong"             : wrong,
        "skipped"           : skipped,
        "top1_acc"          : top1_acc,
        "per_class_correct" : per_class_correct,
        "per_class_total"   : per_class_total,
        "confusion"         : confusion,
        "all_preds"         : all_preds,
        "all_labels"        : all_labels,
        "pass_threshold"    : pass_threshold,
        "passed"            : top1_acc >= pass_threshold,
    }


# ── Report builder ────────────────────────────────────────────────────────────

def print_and_write_report(
    r          : dict,
    qonnx_path : Path,
    gvk_dir    : Path,
    out_path   : Path,
    timestamp  : str,
) -> None:
    sep  = "=" * 60
    sep2 = "─" * 60

    lines: list[str] = []

    def ln(s: str = "") -> None:
        lines.append(s)
        print(s)

    ln(sep)
    ln("  STREAMSENSE -- QONNX GV1K Evaluation")
    ln("  Scope 2 / QAT Extension  |  ERR v1.0")
    ln(f"  Timestamp  : {timestamp}")
    ln(f"  Model      : {qonnx_path.name}")
    ln(f"  File size  : {qonnx_path.stat().st_size / 1024:.1f} KB")
    ln(f"  GV1K dir   : {gvk_dir}")
    ln(sep)
    ln()
    ln(f"  Total .bin files    : {r['total_files']}")
    ln(f"  Vectors checked     : {r['total_checked']}")
    ln(f"  Skipped (bad files) : {r['skipped']}")
    ln(f"  Correct             : {r['correct']}")
    ln(f"  Wrong               : {r['wrong']}")
    ln(f"  Top-1 Accuracy      : {r['top1_acc']:.2f}%  "
       f"({r['correct']}/{r['total_checked']})")
    ln(f"  Pass threshold      : {r['pass_threshold']:.1f}%")
    ln(f"  Gate result         : {'PASS' if r['passed'] else 'FAIL'}")
    ln()
    ln(sep2)
    ln("  Per-class accuracy")
    ln(sep2)

    for i, name in enumerate(CLASS_NAMES):
        c   = r["per_class_correct"][i]
        t   = r["per_class_total"][i]
        pct = 100.0 * c / t if t > 0 else 0.0
        bar = "#" * int(pct / 5)
        ln(f"  {name:<8}  {c:>3}/{t:<3}  {pct:6.2f}%  {bar}")

    ln()
    ln(sep2)
    ln("  Confusion matrix  (rows=true, cols=predicted)")
    ln(f"  Classes: {', '.join(f'{i}={n}' for i, n in enumerate(CLASS_NAMES))}")
    ln(sep2)
    for i, row in enumerate(r["confusion"]):
        ln(f"  {CLASS_NAMES[i]:<8}  {row}")

    ln()
    ln(sep2)
    ln("  ERR v1.0 output contract (verified before inference)")
    ln("    output[0] logits        float32  (1, 10)   -- classification head")
    ln("    output[1] embedding     float32  (1, 128)  -- projection head")
    ln("    output[2] novelty_score float32  (1, 1)    -- novelty head (2-D enforced)")
    ln("  Note: output node names in .qonnx may be auto-generated integers.")
    ln("        Shapes verified by index, names logged above for reference.")
    ln(sep2)
    ln()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "a", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"[DONE] Report appended to: {out_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="STREAMSENSE -- evaluate QONNX model on GV1K golden vectors."
    )
    p.add_argument(
        "--qonnx",
        type    = Path,
        default = _ROOT / "onnx_models" / "streamsense_multihead.qonnx",
        help    = "Path to QONNX model (default: onnx_models/streamsense_multihead.qonnx)",
    )
    p.add_argument(
        "--gvk",
        type    = Path,
        default = _ROOT / "golden_vectors_1000" / "normalized",
        help    = "GV1K normalized directory (default: golden_vectors_1000/normalized)",
    )
    p.add_argument(
        "--out",
        type    = Path,
        default = _ROOT / "evaluation" / "qonnx_evaluation_report.txt",
        help    = "Output report file (default: evaluation/qonnx_evaluation_report.txt)",
    )
    p.add_argument(
        "--pass-threshold",
        type    = float,
        default = 90.0,
        help    = "Minimum top-1%% to pass the gate (default: 90.0)",
    )
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args      = _parse_args()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print("=" * 60)
    print("  STREAMSENSE -- QONNX GV1K Evaluation")
    print("  Scope 2 / QAT Extension")
    print(f"  Timestamp : {timestamp}")
    print("=" * 60)

    if not args.qonnx.exists():
        print(f"\n[ERROR] QONNX model not found: {args.qonnx}")
        print("        Run the export cell in notebooks/qat_colab.ipynb first.")
        sys.exit(1)

    if not args.gvk.exists():
        print(f"\n[ERROR] GV1K directory not found: {args.gvk}")
        sys.exit(1)

    results = evaluate(
        qonnx_path     = args.qonnx,
        gvk_dir        = args.gvk,
        pass_threshold = args.pass_threshold,
    )

    print_and_write_report(
        r          = results,
        qonnx_path = args.qonnx,
        gvk_dir    = args.gvk,
        out_path   = args.out,
        timestamp  = timestamp,
    )

    if not results["passed"]:
        print(f"\n[FAIL] GV1K gate FAILED -- "
              f"{results['top1_acc']:.2f}% < {args.pass_threshold:.1f}% minimum.")
        print("       Do not promote this QONNX to Track E.")
        sys.exit(1)
    else:
        print(f"\n[PASS] GV1K gate PASSED -- "
              f"{results['top1_acc']:.2f}% >= {args.pass_threshold:.1f}%")
        print("       QONNX is deployment-grade. Safe to hand to Track E.")


if __name__ == "__main__":
    main()
