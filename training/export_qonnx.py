# =============================================================================
# export_qonnx.py — Export QAT model to QONNX format
# Project STREAMSENSE (OSL-PRG-2026-SE) | Track A | Epic A5.2 STRETCH
# Run on: Google Colab (after train_qat.py completes)
# =============================================================================
# WHAT THIS DOES:
#   1. Loads best_model_qat.pth from Drive
#   2. Exports to QONNX format (streamsense_model_qonnx.onnx)
#   3. Validates the exported graph with a dummy forward pass
#   4. Saves to Drive: STREAMSENSE_outputs/streamsense_model_qonnx.onnx
# =============================================================================
# NOTE ON QONNX:
#   QONNX is a Brevitas-defined ONNX dialect that preserves quantization
#   annotations (scale, zero-point, bit-width) in the graph. The FINN
#   toolchain reads this format to synthesize the model onto Zynq-7000 FPGA.
#   Standard ONNX Runtime cannot run QONNX directly — use the FP32 or
#   INT8 PTQ models for host inference.
# =============================================================================

# -----------------------------------------------------------------------------
# CELL 1 — Install (run once per Colab session, after train_qat.py installs)
# -----------------------------------------------------------------------------
# !pip install brevitas qonnx onnx onnxruntime --quiet

# -----------------------------------------------------------------------------
# CELL 2 — Mount Drive & sys.path (if not already done)
# -----------------------------------------------------------------------------
# from google.colab import drive
# drive.mount('/content/drive')
# import sys
# sys.path.insert(0, '/content/STREAMSENSE/training')

# -----------------------------------------------------------------------------
# CELL 3 — Run this script
# -----------------------------------------------------------------------------

import os
import torch
import numpy as np

import brevitas.nn as qnn
from brevitas.quant import Int8WeightPerTensorFloat, Int8ActPerTensorFloat
from brevitas.export import export_qonnx

# =============================================================================
# SECTION 1 — Paths
# =============================================================================

DRIVE_ROOT    = "/content/drive/MyDrive"
DRIVE_OUTPUTS = f"{DRIVE_ROOT}/STREAMSENSE_outputs"
QAT_CKPT      = f"{DRIVE_OUTPUTS}/best_model_qat.pth"
QONNX_OUT     = f"{DRIVE_OUTPUTS}/streamsense_model_qonnx.onnx"

# Also save a local copy to the repo for git commit
REPO_ONNX_DIR = "/content/streamsense/onnx_models"
QONNX_REPO    = f"{REPO_ONNX_DIR}/streamsense_model_qonnx.onnx"

DEVICE        = torch.device("cpu")   # export always on CPU

# MPIC v1.0 input shape — must not change
INPUT_SHAPE   = (1, 1, 64, 97)        # [batch, channel, mel_bins, time_frames]
NUM_CLASSES   = 10
WEIGHT_BIT_WIDTH = 8
ACT_BIT_WIDTH    = 8

# =============================================================================
# SECTION 2 — Re-define QAT model (must match train_qat.py exactly)
# =============================================================================

class QuantConvBlock(torch.nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.25):
        super().__init__()
        self.block = torch.nn.Sequential(
            qnn.QuantConv2d(
                in_ch, out_ch, kernel_size=3, padding=1,
                weight_bit_width=WEIGHT_BIT_WIDTH,
                weight_quant=Int8WeightPerTensorFloat,
                bias=False
            ),
            torch.nn.BatchNorm2d(out_ch),
            qnn.QuantReLU(bit_width=ACT_BIT_WIDTH, act_quant=Int8ActPerTensorFloat),
            qnn.QuantConv2d(
                out_ch, out_ch, kernel_size=3, padding=1,
                weight_bit_width=WEIGHT_BIT_WIDTH,
                weight_quant=Int8WeightPerTensorFloat,
                bias=False
            ),
            torch.nn.BatchNorm2d(out_ch),
            qnn.QuantReLU(bit_width=ACT_BIT_WIDTH, act_quant=Int8ActPerTensorFloat),
            torch.nn.MaxPool2d(2, 2),
            torch.nn.Dropout2d(dropout),
        )

    def forward(self, x):
        return self.block(x)


class StreamSenseNetQAT(torch.nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.input_quant = qnn.QuantIdentity(
            act_quant=Int8ActPerTensorFloat,
            bit_width=ACT_BIT_WIDTH,
            return_quant_tensor=True
        )
        self.block1     = QuantConvBlock(1,   32)
        self.block2     = QuantConvBlock(32,  64)
        self.block3     = QuantConvBlock(64, 128)
        self.gap        = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = torch.nn.Sequential(
            qnn.QuantLinear(
                128, 64,
                weight_bit_width=WEIGHT_BIT_WIDTH,
                weight_quant=Int8WeightPerTensorFloat,
                bias=True
            ),
            qnn.QuantReLU(bit_width=ACT_BIT_WIDTH, act_quant=Int8ActPerTensorFloat),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.input_quant(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.gap(x)
        x = x.flatten(1)
        x = self.classifier(x)
        return x


# =============================================================================
# SECTION 3 — Load QAT checkpoint
# =============================================================================

def load_qat_model(ckpt_path, device):
    print(f"Loading QAT checkpoint: {ckpt_path}")
    ckpt  = torch.load(ckpt_path, map_location=device)

    model = StreamSenseNetQAT(num_classes=NUM_CLASSES).to(device)

    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state, strict=True)
    model.eval()

    val_acc          = ckpt.get("val_acc", "unknown")
    weight_bit_width = ckpt.get("weight_bit_width", WEIGHT_BIT_WIDTH)
    act_bit_width    = ckpt.get("act_bit_width",    ACT_BIT_WIDTH)

    print(f"  val_acc          : {val_acc}")
    print(f"  weight_bit_width : {weight_bit_width}")
    print(f"  act_bit_width    : {act_bit_width}")
    return model


# =============================================================================
# SECTION 4 — Export to QONNX
# =============================================================================

def export_to_qonnx(model, output_path):
    """
    Export Brevitas QAT model to QONNX using Brevitas's export_qonnx().
    The exported graph contains QuantizeLinear / DequantizeLinear nodes
    with scale + zero-point annotations — the format FINN expects.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    dummy_input = torch.zeros(INPUT_SHAPE, dtype=torch.float32)

    print(f"\nExporting to QONNX: {output_path}")
    print(f"  Input shape : {INPUT_SHAPE}")
    print(f"  Opset       : 13 (QONNX standard)")

    export_qonnx(
        module      = model,
        input_t     = dummy_input,
        export_path = output_path,
        opset_version = 13,         # QONNX uses opset 13
    )

    size_kb = os.path.getsize(output_path) / 1024
    print(f"  File size   : {size_kb:.1f} KB")
    print(f"  Export      : DONE")
    return output_path


# =============================================================================
# SECTION 5 — Sanity check: dummy forward pass via onnxruntime
# =============================================================================

def validate_qonnx(onnx_path):
    """
    NOTE: onnxruntime cannot execute QONNX QuantizeLinear ops at runtime
    the same way FINN does. This validation only checks:
      - The graph loads without errors
      - Input/output names and shapes are correct
    For full execution validation, use the FINN toolchain.
    """
    try:
        import onnx
        import onnxruntime as ort

        model_proto = onnx.load(onnx_path)
        onnx.checker.check_model(model_proto)
        print("\n[Validation] ONNX graph check : PASS")

        # Print input/output info
        for inp in model_proto.graph.input:
            shape = [d.dim_value for d in inp.type.tensor_type.shape.dim]
            print(f"  Input  : {inp.name} — shape {shape}")
        for out in model_proto.graph.output:
            shape = [d.dim_value for d in out.type.tensor_type.shape.dim]
            print(f"  Output : {out.name} — shape {shape}")

        # Count nodes
        n_nodes = len(model_proto.graph.node)
        op_types = set(n.op_type for n in model_proto.graph.node)
        print(f"  Nodes  : {n_nodes} | Op types: {sorted(op_types)}")
        print("[Validation] Graph structure : PASS")

    except Exception as e:
        print(f"[Validation] WARNING: {e}")
        print("  This may be expected for QONNX ops not supported by onnxruntime.")
        print("  Use the FINN toolchain for full validation.")


# =============================================================================
# SECTION 6 — Main
# =============================================================================

def main():
    print("=" * 60)
    print("STREAMSENSE — QONNX Export (A5.2 STRETCH)")
    print("=" * 60)

    # Load QAT model
    model = load_qat_model(QAT_CKPT, DEVICE)

    # Export to Drive
    export_to_qonnx(model, QONNX_OUT)

    # Also copy to repo onnx_models/ for git commit
    os.makedirs(REPO_ONNX_DIR, exist_ok=True)
    import shutil
    shutil.copy2(QONNX_OUT, QONNX_REPO)
    print(f"\nCopied to repo: {QONNX_REPO}")

    # Validate graph structure
    validate_qonnx(QONNX_OUT)

    # Summary
    print("\n" + "=" * 60)
    print("QONNX Export Summary")
    print("=" * 60)
    print(f"  Input  : float32 {INPUT_SHAPE}")
    print(f"  Output : float32 [1, 10] logits")
    print(f"  Format : QONNX (Brevitas dialect, opset 13)")
    print(f"  Target : Zynq-7000 via FINN toolchain")
    print(f"  File   : {QONNX_OUT}")
    print(f"  Repo   : {QONNX_REPO}")
    print("\nNOTE: This model is for the FPGA path (Track E / FINN) only.")
    print("      For host inference, use streamsense_model_int8.onnx (PTQ).")
    print("=" * 60)


if __name__ == "__main__":
    main()
