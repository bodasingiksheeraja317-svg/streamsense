"""
populate_gv_top1.py
Project STREAMSENSE — Track A
Scope 2 / QONNX Extension — Golden Vector Manifest Top-1 Population

PURPOSE
-------
Fills the `expected_top1_index` fields that are currently `null` in
golden_vectors/manifest.json.  Those fields were intentionally left as None
in generate_golden.py (line: "expected_top1_index": None, # filled after training)
because the model did not yet exist when the GVs were generated.

This script:
  1. Loads the existing golden_vectors/manifest.json (read-only).
  2. Reads the 10 pre-normalized GV binary files from golden_vectors/normalized/.
  3. Runs inference with up to three models, in order of authority:
       Model A  —  streamsense_multihead_fp32.onnx   (OnnxRuntime, canonical ERR v1.0)
       Model B  —  streamsense_multihead.qonnx        (qonnx executor, QAT/Brevitas)
       Model C  —  streamsense_model_fp32.onnx        (OnnxRuntime, single-head Scope 1)
  4. Compares predictions across models and logs any disagreement.
  5. Writes golden_vectors/manifest_with_top1.json (safe new file — does NOT overwrite).
  6. Prints a human-readable summary including:
       - Which field was populated for each GV
       - What value was written
       - Which model produced that value
       - Whether any model disagreed
  7. With --write-inplace, overwrites golden_vectors/manifest.json after
     explicit user confirmation.

AUTHORITY HIERARCHY FOR expected_top1_index
-------------------------------------------
The canonical value comes from Model A (FP32 multihead, ORT).  This is the
same model that populate the WA4 handover and the GV1K regression gate.
Model B (QONNX) is logged for comparison.  Model C (single-head) is a fallback
if Models A and B are both absent.

IMPORTANT: the QONNX model uses qonnx's execute_onnx, not onnxruntime.
           The FP32/INT8 ONNX models use onnxruntime directly.
           Do not mix the two runtimes.

PREREQUISITES
-------------
  pip install onnxruntime qonnx numpy

USAGE (from project root C:\\STREAMSENSE\\)
------------------------------------------
  # Write to a new file (safe — does not touch manifest.json):
  python training/populate_gv_top1.py

  # Also overwrite manifest.json in-place (confirms before writing):
  python training/populate_gv_top1.py --write-inplace

  # Override paths if running from a different location:
  python training/populate_gv_top1.py \\
      --manifest  golden_vectors/manifest.json \\
      --norm-dir  golden_vectors/normalized \\
      --fp32-mh   onnx_models/streamsense_multihead_fp32.onnx \\
      --qonnx     onnx_models/streamsense_multihead.qonnx \\
      --fp32-sh   onnx_models/streamsense_model_fp32.onnx \\
      --out       golden_vectors/manifest_with_top1.json
"""

from __future__ import annotations

import argparse
import json
import sys
import os
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import numpy as np

# ── Project-root resolution ───────────────────────────────────────────────────
_THIS_DIR = Path(__file__).resolve().parent
_ROOT     = _THIS_DIR.parent   # one level above training/

# ── Class map (MPIC v1.0 / class_labels.json) ────────────────────────────────
TARGET_CLASSES = {
    "yes": 0, "no": 1, "up": 2, "down": 3,
    "left": 4, "right": 5, "on": 6, "off": 7,
    "stop": 8, "go": 9,
}
IDX_TO_LABEL = {v: k for k, v in TARGET_CLASSES.items()}

# GV binary spec (matches generate_golden.py and evaluate_qonnx.py)
GV_FLOATS = 64 * 97           # 6208
GV_BYTES  = GV_FLOATS * 4     # 24832
GV_SHAPE  = (1, 1, 64, 97)   # model input shape

# ERR v1.0 expected output shapes (by index — for QONNX whose node names are ints)
EXPECTED_SHAPES = [(1, 10), (1, 128), (1, 1)]   # logits, embedding, novelty


# ─────────────────────────────────────────────────────────────────────────────
# OnnxRuntime loader (for .onnx models)
# ─────────────────────────────────────────────────────────────────────────────

def _load_ort_session(path: Path, label: str):
    """
    Load an ONNX model via onnxruntime.  Returns (session, input_name, logits_name).
    Returns None if onnxruntime is not installed or the file is missing.
    """
    if not path.exists():
        print(f"  [SKIP] {label}: file not found at {path}")
        return None

    try:
        import onnxruntime as ort
    except ImportError:
        print("  [WARN] onnxruntime not installed — skipping ORT models.")
        print("         pip install onnxruntime")
        return None

    opts = ort.SessionOptions()
    opts.inter_op_num_threads = 2
    opts.intra_op_num_threads = 2
    session = ort.InferenceSession(str(path), sess_options=opts,
                                   providers=["CPUExecutionProvider"])

    input_name = session.get_inputs()[0].name

    # Resolve logits output by name; fall back to index 0.
    logits_name = None
    for out in session.get_outputs():
        if out.name == "logits":
            logits_name = out.name
            break
    if logits_name is None:
        logits_name = session.get_outputs()[0].name

    print(f"  [LOADED] {label}")
    print(f"           input='{input_name}'  logits='{logits_name}'")
    print(f"           outputs: {[o.name for o in session.get_outputs()]}")
    return session, input_name, logits_name


def _ort_infer(session_info, inp: np.ndarray) -> int:
    """Run one forward pass via ORT.  Returns argmax int."""
    session, input_name, logits_name = session_info
    logits = session.run([logits_name], {input_name: inp})[0]   # [1, 10]
    return int(np.argmax(logits, axis=1)[0])


# ─────────────────────────────────────────────────────────────────────────────
# QONNX loader (for .qonnx model, Brevitas custom ops)
# ─────────────────────────────────────────────────────────────────────────────

def _load_qonnx_model(path: Path):
    """
    Load the QONNX model using qonnx's own executor.
    Returns (model_wrapper, input_name, logits_key) or None.

    CRITICAL: Do NOT use onnxruntime to load .qonnx files.
    The Brevitas Quant / BipolarQuant custom ops are not registered in ORT.
    """
    if not path.exists():
        print(f"  [SKIP] QONNX model: file not found at {path}")
        return None

    try:
        from qonnx.core.modelwrapper import ModelWrapper
        from qonnx.core.onnx_exec import execute_onnx
        from qonnx.transformation.infer_shapes import InferShapes
    except ImportError:
        print("  [WARN] qonnx is not installed — skipping QONNX model.")
        print("         pip install qonnx")
        return None

    model = ModelWrapper(str(path))
    model = model.transform(InferShapes())   # mandatory before execute_onnx

    input_name   = model.graph.input[0].name
    output_names = [o.name for o in model.graph.output]

    print(f"  [LOADED] QONNX: {path.name}")
    print(f"           input='{input_name}'")
    print(f"           outputs={output_names}  (accessed by index 0/1/2, not name)")

    # Verify ERR v1.0 contract with a dummy pass
    dummy = np.zeros(GV_SHAPE, dtype=np.float32)
    odict = execute_onnx(model, {input_name: dummy})
    for idx, (label, expected) in enumerate(
        zip(["logits", "embedding", "novelty_score"], EXPECTED_SHAPES)
    ):
        out_key = output_names[idx]
        actual  = odict[out_key].shape
        ok      = actual == expected
        print(f"           output[{idx}] {label:<15}: {actual}  "
              f"{'✓' if ok else '✗ MISMATCH expected ' + str(expected)}")
        if not ok:
            print(f"  [ERROR] QONNX output contract FAILED for '{label}'. "
                  f"Re-export the QONNX.")
            return None

    # logits_key is the name of output[0] (may be an auto-generated integer string)
    logits_key = output_names[0]
    return model, input_name, logits_key, execute_onnx


def _qonnx_infer(qonnx_info, inp: np.ndarray) -> int:
    """Run one forward pass via qonnx executor.  Returns argmax int."""
    model, input_name, logits_key, execute_onnx = qonnx_info
    odict  = execute_onnx(model, {input_name: inp})
    logits = odict[logits_key]                              # [1, 10]
    return int(np.argmax(logits, axis=1)[0])


# ─────────────────────────────────────────────────────────────────────────────
# Load one normalized GV binary
# ─────────────────────────────────────────────────────────────────────────────

def _load_gv_norm(norm_path: Path) -> np.ndarray | None:
    """
    Load a pre-normalized GV binary (float32 little-endian, shape [64,97])
    and return it reshaped to [1,1,64,97] float32.

    Returns None on size mismatch.
    """
    raw = np.fromfile(str(norm_path), dtype="<f4")
    if raw.size != GV_FLOATS:
        print(f"  [ERROR] {norm_path.name}: expected {GV_FLOATS} floats "
              f"({GV_BYTES} bytes), got {raw.size}")
        return None
    return raw.reshape(GV_SHAPE).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Main population logic
# ─────────────────────────────────────────────────────────────────────────────

def run(args: argparse.Namespace) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    sep  = "=" * 70
    sep2 = "─" * 70

    print(sep)
    print("  STREAMSENSE — populate_gv_top1.py")
    print("  Populating expected_top1_index in golden_vectors/manifest.json")
    print(f"  Timestamp : {timestamp}")
    print(sep)

    # ── Load manifest ─────────────────────────────────────────────────────────
    if not args.manifest.exists():
        print(f"\n[ERROR] Manifest not found: {args.manifest}")
        print("        Run training/generate_golden.py first.")
        sys.exit(1)

    with open(args.manifest, "r", encoding="utf-8") as fh:
        manifest = json.load(fh)

    vectors = manifest.get("vectors", {})
    if not vectors:
        print("[ERROR] manifest.json has no 'vectors' key or it is empty.")
        sys.exit(1)

    print(f"\nManifest loaded: {args.manifest}")
    print(f"  mpic_version : {manifest.get('mpic_version', '?')}")
    print(f"  Vectors      : {len(vectors)}")

    # Check which entries already have expected_top1_index filled
    already_filled = [k for k, v in vectors.items()
                      if v.get("expected_top1_index") is not None]
    null_entries   = [k for k, v in vectors.items()
                      if v.get("expected_top1_index") is None]

    if already_filled:
        print(f"\n  [NOTE] {len(already_filled)} vector(s) already have "
              f"expected_top1_index: {already_filled}")
    print(f"  [INFO] {len(null_entries)} vector(s) have expected_top1_index=null "
          f"→ will be populated")

    # ── Check norm-dir ────────────────────────────────────────────────────────
    if not args.norm_dir.exists():
        print(f"\n[ERROR] Normalized GV directory not found: {args.norm_dir}")
        print("        Run training/generate_golden.py first.")
        sys.exit(1)

    # ── Load models ───────────────────────────────────────────────────────────
    print(f"\n{sep2}")
    print("  Loading models")
    print(sep2)

    fp32_mh  = _load_ort_session(args.fp32_mh,  "FP32 multihead ONNX (canonical, ERR v1.0)")
    qonnx    = _load_qonnx_model(args.qonnx)
    fp32_sh  = _load_ort_session(args.fp32_sh,  "FP32 single-head ONNX (Scope 1 fallback)")

    # Determine authority source for expected_top1_index
    if fp32_mh is not None:
        authority_label = "streamsense_multihead_fp32.onnx (ORT)"
        authority_key   = "fp32_mh"
    elif qonnx is not None:
        authority_label = "streamsense_multihead.qonnx (qonnx executor)"
        authority_key   = "qonnx"
    elif fp32_sh is not None:
        authority_label = "streamsense_model_fp32.onnx (ORT, single-head fallback)"
        authority_key   = "fp32_sh"
    else:
        print("\n[ERROR] No model could be loaded. Cannot populate expected_top1_index.")
        print("        Ensure at least one of the following exists and is importable:")
        print(f"          {args.fp32_mh}")
        print(f"          {args.qonnx}")
        print(f"          {args.fp32_sh}")
        sys.exit(1)

    print(f"\n  Authority source (for expected_top1_index) : {authority_label}")
    if authority_key != "fp32_mh":
        print(f"  [WARN] FP32 multihead ONNX not available — using fallback.")

    # ── Inference loop over all 10 GVs ────────────────────────────────────────
    print(f"\n{sep2}")
    print("  Running inference on all 10 golden vectors")
    print(sep2)

    results: dict[str, dict] = {}   # key → {true_idx, fp32_mh, qonnx, fp32_sh, authority_pred}

    for gv_key in sorted(vectors.keys(), key=lambda x: int(x)):
        entry   = vectors[gv_key]
        gv_name = entry["gv_name"]
        label   = entry["label"]
        true_idx = TARGET_CLASSES.get(label, None)

        if true_idx is None:
            print(f"\n  [WARN] GV key={gv_key} label='{label}' not in TARGET_CLASSES — skipping")
            continue

        # Locate the norm binary using the manifest's norm_bin field
        norm_bin_name = entry.get("norm_bin")
        if not norm_bin_name:
            print(f"  [WARN] {gv_name}: no 'norm_bin' field in manifest — skipping")
            continue

        norm_path = args.norm_dir / norm_bin_name
        if not norm_path.exists():
            print(f"\n  [WARN] {gv_name}: norm binary not found: {norm_path}")
            continue

        inp = _load_gv_norm(norm_path)
        if inp is None:
            continue

        # Run inference with each available model
        pred_fp32_mh = _ort_infer(fp32_mh, inp)    if fp32_mh else None
        pred_qonnx   = _qonnx_infer(qonnx, inp)    if qonnx   else None
        pred_fp32_sh = _ort_infer(fp32_sh, inp)    if fp32_sh else None

        # Determine the value to write (authority source)
        if   authority_key == "fp32_mh": authority_pred = pred_fp32_mh
        elif authority_key == "qonnx":   authority_pred = pred_qonnx
        else:                            authority_pred = pred_fp32_sh

        results[gv_key] = {
            "gv_name"        : gv_name,
            "true_idx"       : true_idx,
            "true_label"     : label,
            "pred_fp32_mh"   : pred_fp32_mh,
            "pred_qonnx"     : pred_qonnx,
            "pred_fp32_sh"   : pred_fp32_sh,
            "authority_pred" : authority_pred,
        }

        # Console output
        fp32_mh_str = f"{pred_fp32_mh}={IDX_TO_LABEL.get(pred_fp32_mh,'?')}" if pred_fp32_mh is not None else "n/a"
        qonnx_str   = f"{pred_qonnx}={IDX_TO_LABEL.get(pred_qonnx,'?')}"     if pred_qonnx   is not None else "n/a"
        fp32_sh_str = f"{pred_fp32_sh}={IDX_TO_LABEL.get(pred_fp32_sh,'?')}" if pred_fp32_sh is not None else "n/a"

        correct = (authority_pred == true_idx)

        # Check for disagreements among models that were run
        preds_available = [p for p in [pred_fp32_mh, pred_qonnx, pred_fp32_sh]
                           if p is not None]
        disagree = len(set(preds_available)) > 1

        status = "CORRECT" if correct else "WRONG"
        flag   = "  [DISAGREE!]" if disagree else ""

        print(
            f"\n  {gv_name:<18}  true={true_idx}({label})"
            f"\n    fp32_mh: {fp32_mh_str:<12}  qonnx: {qonnx_str:<12}"
            f"  fp32_sh: {fp32_sh_str}"
            f"\n    → writing expected_top1_index = {authority_pred} "
            f"({IDX_TO_LABEL.get(authority_pred,'?')})  [{status}]{flag}"
        )

    # ── Build updated manifest ────────────────────────────────────────────────
    print(f"\n{sep2}")
    print("  Building updated manifest")
    print(sep2)

    updated_manifest = deepcopy(manifest)

    # Metadata about this population run (stored at top level)
    updated_manifest["top1_population_meta"] = {
        "populated_by"        : "training/populate_gv_top1.py",
        "timestamp"           : timestamp,
        "authority_model"     : authority_label,
        "authority_model_path": str(args.fp32_mh if authority_key == "fp32_mh"
                                    else args.qonnx if authority_key == "qonnx"
                                    else args.fp32_sh),
        "models_run": {
            "fp32_multihead": str(args.fp32_mh)  if fp32_mh else "not loaded",
            "qonnx"         : str(args.qonnx)    if qonnx   else "not loaded",
            "fp32_singlehd" : str(args.fp32_sh)  if fp32_sh else "not loaded",
        },
        "note": (
            "expected_top1_index comes from the authority model listed above. "
            "All available models are run and logged; disagreements are flagged. "
            "This field was previously null in the original generate_golden.py output."
        ),
    }

    n_filled       = 0
    n_disagreed    = 0
    n_correct      = 0
    disagreements  = []

    for gv_key, r in results.items():
        gv_entry = updated_manifest["vectors"][gv_key]

        # Per-vector inference log (stored alongside existing fields)
        gv_entry["expected_top1_index"]   = r["authority_pred"]
        gv_entry["expected_top1_label"]   = IDX_TO_LABEL.get(r["authority_pred"], "unknown")
        gv_entry["top1_source_model"]     = authority_label
        gv_entry["top1_correct_vs_label"] = (r["authority_pred"] == r["true_idx"])
        gv_entry["top1_inference_detail"] = {
            "fp32_multihead_pred" : r["pred_fp32_mh"],
            "qonnx_pred"          : r["pred_qonnx"],
            "fp32_singlehd_pred"  : r["pred_fp32_sh"],
        }

        preds_available = [p for p in [r["pred_fp32_mh"], r["pred_qonnx"], r["pred_fp32_sh"]]
                           if p is not None]
        disagree = len(set(preds_available)) > 1

        if disagree:
            n_disagreed += 1
            disagreements.append(gv_key)

        if r["authority_pred"] == r["true_idx"]:
            n_correct += 1

        n_filled += 1

    # ── Write output manifest ─────────────────────────────────────────────────
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(updated_manifest, fh, indent=2)

    print(f"\n  [WRITTEN] {args.out}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{sep}")
    print("  SUMMARY")
    print(sep)
    print(f"  Vectors processed         : {n_filled} / 10")
    print(f"  expected_top1_index filled: {n_filled}")
    print(f"  Correct vs. true label    : {n_correct} / {n_filled}")
    print(f"  Model disagreements       : {n_disagreed}"
          + (f"  (GV keys: {disagreements})" if disagreements else ""))
    print(f"\n  Authority source : {authority_label}")
    print(f"\n  Fields written per vector:")
    print(f"    expected_top1_index      — integer (argmax of logits from authority model)")
    print(f"    expected_top1_label      — string label for readability")
    print(f"    top1_source_model        — which model produced the value")
    print(f"    top1_correct_vs_label    — whether prediction matches the GV's true label")
    print(f"    top1_inference_detail    — predictions from all models that ran")
    print(f"\n  Output file : {args.out}")
    print(f"  Original    : {args.manifest}  (NOT modified)")

    if n_disagreed:
        print(f"\n  [WARN] {n_disagreed} disagreement(s) detected between models.")
        print("         This may indicate a QONNX export issue or quantization drift.")
        print("         Review the top1_inference_detail fields in the output manifest.")

    if n_correct < n_filled:
        print(f"\n  [WARN] {n_filled - n_correct} GV(s) were misclassified by the "
              f"authority model.")
        print("         This is unexpected for a 95.97% model on carefully selected GVs.")
        print("         Check that the norm binary paths are correct and the model is "
              "the production checkpoint.")

    # ── In-place write ────────────────────────────────────────────────────────
    if args.write_inplace:
        print(f"\n{sep2}")
        print("  --write-inplace requested")
        print(f"  Target: {args.manifest}")

        if n_disagreed > 0 or n_correct < n_filled:
            print("\n  [ABORT] In-place write blocked: disagreements or misclassifications "
                  "detected above.")
            print("          Investigate before overwriting the manifest.")
            sys.exit(1)

        ans = input(f"\n  Overwrite {args.manifest} with the populated version? [yes/no]: ")
        if ans.strip().lower() == "yes":
            with open(args.manifest, "w", encoding="utf-8") as fh:
                json.dump(updated_manifest, fh, indent=2)
            print(f"  [DONE] {args.manifest} updated in-place.")
        else:
            print("  [SKIP] In-place write cancelled.")

    print(f"\n{sep}")
    print("  DONE")
    print(sep)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "STREAMSENSE — populate expected_top1_index in golden_vectors/manifest.json. "
            "Runs inference on all 10 GVs with FP32 ONNX (authority) and QONNX (comparison). "
            "Writes golden_vectors/manifest_with_top1.json (safe new file by default)."
        )
    )
    p.add_argument(
        "--manifest",
        type    = Path,
        default = _ROOT / "golden_vectors" / "manifest.json",
        help    = "Input manifest.json (default: golden_vectors/manifest.json)",
    )
    p.add_argument(
        "--norm-dir",
        type    = Path,
        default = _ROOT / "golden_vectors" / "normalized",
        help    = "Directory containing GV *_norm.bin files (default: golden_vectors/normalized/)",
    )
    p.add_argument(
        "--fp32-mh",
        type    = Path,
        default = _ROOT / "onnx_models" / "streamsense_multihead_fp32.onnx",
        help    = "FP32 multihead ONNX (canonical authority; default: onnx_models/streamsense_multihead_fp32.onnx)",
    )
    p.add_argument(
        "--qonnx",
        type    = Path,
        default = _ROOT / "onnx_models" / "streamsense_multihead.qonnx",
        help    = "QONNX model for comparison (default: onnx_models/streamsense_multihead.qonnx)",
    )
    p.add_argument(
        "--fp32-sh",
        type    = Path,
        default = _ROOT / "onnx_models" / "streamsense_model_fp32.onnx",
        help    = "FP32 single-head ONNX fallback (default: onnx_models/streamsense_model_fp32.onnx)",
    )
    p.add_argument(
        "--out",
        type    = Path,
        default = _ROOT / "golden_vectors" / "manifest_with_top1.json",
        help    = "Output manifest file (default: golden_vectors/manifest_with_top1.json)",
    )
    p.add_argument(
        "--write-inplace",
        action  = "store_true",
        default = False,
        help    = "After writing --out, also overwrite manifest.json in-place "
                  "(requires interactive confirmation; blocked on disagreements or misclassifications)",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(args)
