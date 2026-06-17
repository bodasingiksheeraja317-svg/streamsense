"""
model_qat.py
Project STREAMSENSE — Track A
Epic A5.2 — QAT / Brevitas 2D CNN (STRETCH)

Quantization-Aware Training (QAT) version of StreamSenseNet using Brevitas.
Mirrors model.py architecture exactly, replacing standard PyTorch layers with
Brevitas quantized equivalents for 8-bit symmetric INT8 weight + activation
quantization.

Architecture:
    Input  : [B, 1, 64, 97]  float32 (normalized mel spectrogram — MPIC v1.0)
    Output : [B, 10]          float32 logits
    Same VGG-style structure as StreamSenseNet in model.py.

Used by: train_qat.py
Export:  QONNX via brevitas.export.export_qonnx_onnx()

Install: pip install brevitas
"""

import torch
import torch.nn as nn

import brevitas.nn as qnn
from brevitas.quant import Int8ActPerTensorFixedPoint, Int8WeightPerTensorFloat


# ── Quantization configs ────────────────────────────────────────────────────────
# Per-tensor INT8 — matches the PTQ configuration from quantize_ptq.ipynb.
#
# ACT_QUANT uses Int8ActPerTensorFixedPoint (NOT Int8ActPerTensorFloat).
# Reason: Int8ActPerTensorFloat stores its scale as a running-statistics
# *buffer* that Brevitas initialises on CPU and does not always move to the
# correct device when .to(cuda) is called, causing:
#   RuntimeError: Expected all tensors to be on the same device
# Int8ActPerTensorFixedPoint stores its scale as an nn.Parameter, which
# moves correctly with .to(device) and is still fully differentiable for QAT.

WEIGHT_QUANT  = Int8WeightPerTensorFloat    # INT8 symmetric, per-tensor, weights
ACT_QUANT     = Int8ActPerTensorFixedPoint  # INT8 per-tensor activations (param-scale)

# ── Architecture constants ─────────────────────────────────────────────────────
NUM_CLASSES = 10


# ── Quantized conv block ───────────────────────────────────────────────────────

def _qconv_block(in_ch: int, out_ch: int, dropout: float = 0.25) -> nn.Sequential:
    """
    QAT conv block — mirrors _conv_block() in model.py:
        QuantConv2d → BN → QuantReLU → QuantConv2d → BN → QuantReLU
        → MaxPool2d → Dropout2d

    Quantization applied to:
        - Conv weights  : Int8WeightPerTensorFloat
        - ReLU outputs  : Int8ActPerTensorFixedPoint
    MaxPool2d and BatchNorm2d are NOT quantized (fused at export time).

    Both QuantReLU layers use return_quant_tensor=False.
    Reason: return_quant_tensor=True outputs a QuantTensor, which causes the
    *next* QuantConv2d to activate its internal input act_quant — a separate
    Brevitas quantizer whose buffer lives on CPU and triggers:
        RuntimeError: Expected all tensors to be on the same device
    With return_quant_tensor=False, each QuantReLU emits a plain float32
    tensor; QuantConv2d only applies weight quantization (no device mismatch).

    Args:
        in_ch   : Input channels
        out_ch  : Output channels
        dropout : Dropout2d probability (default 0.25, matches model.py)
    """
    return nn.Sequential(
        qnn.QuantConv2d(
            in_channels  = in_ch,
            out_channels = out_ch,
            kernel_size  = 3,
            padding      = 1,
            bias         = False,
            weight_quant = WEIGHT_QUANT,
        ),
        nn.BatchNorm2d(out_ch),
        qnn.QuantReLU(act_quant=ACT_QUANT, return_quant_tensor=False),

        qnn.QuantConv2d(
            in_channels  = out_ch,
            out_channels = out_ch,
            kernel_size  = 3,
            padding      = 1,
            bias         = False,
            weight_quant = WEIGHT_QUANT,
        ),
        nn.BatchNorm2d(out_ch),
        qnn.QuantReLU(act_quant=ACT_QUANT, return_quant_tensor=False),

        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Dropout2d(p=dropout),
    )


# ── QAT Model ─────────────────────────────────────────────────────────────────

class StreamSenseNetQAT(nn.Module):
    """
    Brevitas QAT version of StreamSenseNet (2D CNN).

    Architecture is IDENTICAL to model.py — same layer order, same spatial
    shape evolution, same classifier head. Only the Conv2d and ReLU layers
    are replaced with Brevitas quantized equivalents.

    Spatial shape evolution (B = batch, matches model.py):
        Input          [B,   1, 64, 97]
        After block1   [B,  32, 32, 48]
        After block2   [B,  64, 16, 24]
        After block3   [B, 128,  8, 12]
        After GAP      [B, 128]
        After FC head  [B,  10]   ← logits

    Args:
        num_classes : Number of output classes (default 10)
    """

    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()

        # ── Quantized conv blocks ─────────────────────────────────────────────
        self.block1 = _qconv_block(in_ch=1,   out_ch=32,  dropout=0.25)
        self.block2 = _qconv_block(in_ch=32,  out_ch=64,  dropout=0.25)
        self.block3 = _qconv_block(in_ch=64,  out_ch=128, dropout=0.25)

        # ── Global Average Pooling ─────────────────────────────────────────────
        self.gap = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        # ── Classifier head (quantized linear layers) ─────────────────────────
        # QuantLinear for weights; standard ReLU + Dropout (not on the critical
        # inference path for edge deployment).
        self.classifier = nn.Sequential(
            qnn.QuantLinear(
                in_features  = 128,
                out_features = 64,
                bias         = True,
                weight_quant = WEIGHT_QUANT,
            ),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            qnn.QuantLinear(
                in_features  = 64,
                out_features = num_classes,
                bias         = True,
                weight_quant = WEIGHT_QUANT,
            ),
        )

        # ── Weight initialisation (same as model.py) ──────────────────────────
        self._init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : Tensor [B, 1, 64, 97] float32

        Returns:
            logits : Tensor [B, num_classes] float32
        """
        x = self.block1(x)          # [B,  32, 32, 48]
        x = self.block2(x)          # [B,  64, 16, 24]
        x = self.block3(x)          # [B, 128,  8, 12]
        x = self.gap(x)             # [B, 128,  1,  1]
        x = x.flatten(start_dim=1)  # [B, 128]
        x = self.classifier(x)      # [B, 10]
        return x

    def _init_weights(self):
        """Kaiming-normal for Conv; BN weight=1/bias=0. Matches model.py."""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, qnn.QuantConv2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)


# ── Parameter count ────────────────────────────────────────────────────────────

def count_parameters(model: nn.Module) -> dict:
    """Return total and trainable parameter counts."""
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


# ── Load FP32 weights into QAT model ─────────────────────────────────────────

def load_fp32_weights(qat_model: StreamSenseNetQAT, fp32_ckpt_path) -> tuple:
    """
    Transfer weights from a FP32 best_model.pth checkpoint into the QAT model.
    Layers match by name since the architecture is identical.

    Returns (epoch, val_accuracy) from the checkpoint.
    """
    ckpt = torch.load(fp32_ckpt_path, map_location="cpu")
    fp32_state = ckpt["model_state"]

    # Load with strict=False because Brevitas adds extra quantization state
    # tensors not present in the FP32 checkpoint. All weight/bias tensors
    # that DO exist in both are matched by name.
    missing, unexpected = qat_model.load_state_dict(fp32_state, strict=False)

    # Only Brevitas-specific quant state keys should be missing/unexpected.
    # Log them so the user knows the load was clean.
    quant_keys = [k for k in missing if "quant" in k or "scale" in k or "zero_point" in k]
    non_quant_missing = [k for k in missing if k not in quant_keys]
    if non_quant_missing:
        print(f"  [WARN] Non-quant keys missing from FP32 ckpt: {non_quant_missing}")
    print(f"  FP32 weights loaded. Brevitas-specific keys initialised from scratch: {len(quant_keys)}")

    return ckpt.get("epoch", 0), ckpt.get("val_accuracy", 0.0)


# ── Smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("StreamSenseNetQAT — smoke test")
    print("=" * 60)

    try:
        import brevitas
        print(f"Brevitas version : {brevitas.__version__}")
    except ImportError:
        print("[ERROR] Brevitas not installed. Run: pip install brevitas")
        raise

    model = StreamSenseNetQAT(num_classes=10)
    model.eval()

    params = count_parameters(model)
    print(f"Total parameters : {params['total']:,}")

    dummy = torch.zeros(1, 1, 64, 97)
    with torch.no_grad():
        out = model(dummy)

    print(f"Input  shape     : {tuple(dummy.shape)}")
    print(f"Output shape     : {tuple(out.shape)}")

    assert tuple(out.shape) == (1, 10), f"Expected (1,10), got {tuple(out.shape)}"
    assert out.dtype == torch.float32

    # Batch test
    batch = torch.zeros(8, 1, 64, 97)
    with torch.no_grad():
        batch_out = model(batch)
    assert tuple(batch_out.shape) == (8, 10)

    print("\n[PASS] Smoke test OK. Ready for train_qat.py")
