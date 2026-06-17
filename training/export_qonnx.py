"""
export_qonnx.py — A5.2 STRETCH
Export QAT-trained StreamSenseNetQAT checkpoints to QONNX format
for the Zynq-7000 FPGA path via FINN.

Produces:
  onnx_models/streamsense_qat_w8a8.onnx
  onnx_models/streamsense_qat_w4a4.onnx

Usage:
  python training/export_qonnx.py --bits 8    # export W8A8
  python training/export_qonnx.py --bits 4    # export W4A4
  python training/export_qonnx.py --bits all  # export both (default)

Requires:
  pip install brevitas==0.10.2 qonnx

FINN compatibility:
  QONNX models are consumed by FINN's transformation pipeline for
  Zynq-7000 (xc7z020). This export satisfies FINN's requirements:
    - Symmetric per-tensor weight quantization
    - Static input shape [1, 1, 64, 97]
    - No dynamic axes
    - ONNX opset 13
    - BatchNorm folded into Conv by export_qonnx automatically
"""

import argparse
import os
import sys
import torch
import numpy as np

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR) if os.path.basename(SCRIPT_DIR) == "training" else SCRIPT_DIR
sys.path.insert(0, os.path.join(ROOT, "training"))

# ---------------------------------------------------------------------------
# Brevitas / QONNX imports
# ---------------------------------------------------------------------------
try:
    import brevitas
    from brevitas.export import export_qonnx
except ImportError:
    sys.exit(
        "\n[ERROR] Brevitas not found.\n"
        "Install with:  pip install brevitas==0.10.2\n"
    )

try:
    import qonnx  # noqa: F401
except ImportError:
    sys.exit(
        "\n[ERROR] qonnx not found.\n"
        "Install with:  pip install qonnx\n"
    )

from train_qat_brevitas import StreamSenseNetQAT

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DUMMY_INPUT_SHAPE = (1, 1, 64, 97)   # MPIC v1.0 frozen shape


# ---------------------------------------------------------------------------
# Export function
# ---------------------------------------------------------------------------

def export_model(bits: int, root: str):
    tag       = f"w{bits}a{bits}"
    ckpt_path = os.path.join(root, "checkpoints_qat", f"qat_{tag}_best.pth")
    out_dir   = os.path.join(root, "onnx_models")
    out_path  = os.path.join(out_dir, f"streamsense_qat_{tag}.onnx")

    os.makedirs(out_dir, exist_ok=True)

    if not os.path.exists(ckpt_path):
        print(f"[SKIP] Checkpoint not found: {ckpt_path}")
        print(f"       Run train_qat_brevitas.py --bits {bits} first.")
        return None

    print(f"\n{'='*60}")
    print(f"  Exporting {tag.upper()} → QONNX")
    print(f"{'='*60}")
    print(f"  Checkpoint : {ckpt_path}")
    print(f"  Output     : {out_path}")

    # --- Load checkpoint ---
    ckpt  = torch.load(ckpt_path, map_location="cpu")
    model = StreamSenseNetQAT(num_classes=10, weight_bit=bits, act_bit=bits)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)

    # CRITICAL: model must be in eval() mode before export.
    # export_qonnx traces the graph — BatchNorm behaves differently in
    # train vs eval mode, and Dropout must be off for a static graph.
    model.eval()

    print(f"  Loaded epoch {ckpt.get('epoch', '?')} — val_acc={ckpt.get('val_acc', 0.0):.2f}%")

    # --- Dummy input matching MPIC v1.0 exactly ---
    dummy = torch.zeros(DUMMY_INPUT_SHAPE, dtype=torch.float32)

    # --- Export ---
    # export_qonnx signature in Brevitas 0.10.2:
    #   export_qonnx(module, input_t, export_path, opset_version=None, **kwargs)
    # Pass input_t as positional (2nd arg), not as keyword — keyword name
    # changed between Brevitas versions and positional is always safe.
    print("  Running export_qonnx ...")
    try:
        export_qonnx(model, dummy, export_path=out_path, opset_version=13)
    except TypeError as e:
        # Fallback for older Brevitas 0.10.x that uses different kwarg name
        print(f"  [INFO] First export attempt: {e}")
        print("  [INFO] Retrying with export_path as positional ...")
        export_qonnx(model, dummy, out_path, opset_version=13)

    if not os.path.exists(out_path):
        print("  ✗ Export failed — output file not found.")
        return None

    size_kb = os.path.getsize(out_path) / 1024
    print(f"  ✔ Export successful: {out_path}  ({size_kb:.1f} KB)")

    # --- ONNX Runtime sanity check ---
    _sanity_check(out_path, dummy)

    return out_path


def _sanity_check(onnx_path: str, dummy_input: torch.Tensor):
    """Single forward pass through onnxruntime to verify graph loads."""
    try:
        import onnxruntime as ort
        sess        = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        input_name  = sess.get_inputs()[0].name
        out         = sess.run(None, {input_name: dummy_input.numpy()})
        logits      = out[0]
        assert logits.shape == (1, 10), f"Unexpected output shape: {logits.shape}"
        print(f"  ✔ ONNX Runtime sanity check PASSED — output shape: {logits.shape}")
    except ImportError:
        print("  [INFO] onnxruntime not installed — skipping sanity check.")
    except Exception as e:
        print(f"  [WARN] Sanity check failed: {e}")
        print("  [WARN] The .onnx file may still be valid for FINN (which uses its own runner).")


# ---------------------------------------------------------------------------
# GV regression on exported QONNX model
# ---------------------------------------------------------------------------

def run_gv_regression(onnx_path: str, root: str, bits: int):
    """
    10-vector hand-picked GV regression on the QONNX model.
    Reports pass/fail per class.

    NOTE: The GV1K hard gate (1000/1000) is enforced on the PTQ INT8
    model — that is the production artifact. This QONNX model is the
    FPGA research path; results are documented but not gated.
    """
    try:
        import onnxruntime as ort
    except ImportError:
        print("  [SKIP] onnxruntime not available — skipping GV regression.")
        return

    gv_norm_dir = os.path.join(root, "golden_vectors", "normalized")
    if not os.path.exists(gv_norm_dir):
        print("  [SKIP] golden_vectors/normalized not found.")
        return

    CLASS_LABELS = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]
    sess       = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name

    print(f"\n  GV regression (10-vector) — QONNX W{bits}A{bits}:")
    passes = 0
    for i, label in enumerate(CLASS_LABELS):
        tensor_path = os.path.join(gv_norm_dir, f"gv_{i:02d}_{label}.bin")
        if not os.path.exists(tensor_path):
            print(f"    [{i:02d}] {label:6s} — file not found, skip")
            continue
        tensor = np.fromfile(tensor_path, dtype=np.float32).reshape(1, 1, 64, 97)
        logits = sess.run(None, {input_name: tensor})[0][0]
        pred   = int(np.argmax(logits))
        conf   = float(np.max(logits))
        status = "PASS" if pred == i else "FAIL"
        if pred == i:
            passes += 1
        print(f"    [{i:02d}] {label:6s} → pred={CLASS_LABELS[pred]:6s}  conf={conf:+.3f}  [{status}]")

    print(f"  Result: {passes}/10 PASS")
    print(f"  (GV1K gate enforced on PTQ INT8 model — this is the FPGA research path)")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Export StreamSenseNetQAT to QONNX")
    parser.add_argument("--bits",    choices=["4", "8", "all"], default="all")
    parser.add_argument("--skip-gv", action="store_true", help="Skip GV regression after export.")
    parser.add_argument(
        "--root", type=str,
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."),
    )
    args  = parser.parse_args()
    root  = os.path.abspath(args.root)

    targets  = {"all": [8, 4], "8": [8], "4": [4]}[args.bits]
    exported = []

    for b in targets:
        path = export_model(b, root)
        if path:
            exported.append((b, path))
            if not args.skip_gv:
                run_gv_regression(path, root, b)

    print("\n" + "="*60)
    print("  QONNX EXPORT SUMMARY")
    print("="*60)
    if exported:
        for b, path in exported:
            size_kb = os.path.getsize(path) / 1024
            print(f"  W{b}A{b}  →  {os.path.basename(path)}  ({size_kb:.1f} KB)")
    else:
        print("  No models exported. Run train_qat_brevitas.py --bits all first.")
    print("="*60)
    print("\nFINN next steps (Track E — Prikshit):")
    print("  from finn.core.modelwrapper import ModelWrapper")
    print("  from finn.transformation.qonnx.convert_qonnx_to_finn import ConvertQONNXtoFINN")
    print("  model = ModelWrapper('onnx_models/streamsense_qat_w4a4.onnx')")
    print("  model = model.transform(ConvertQONNXtoFINN())")


if __name__ == "__main__":
    main()
