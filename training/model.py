"""
model.py
Project STREAMSENSE — Track A
MPIC v1.0 — CNN architecture definition.

Input  : Tensor [B, 1, 64, 97]  float32  (batch of normalized mel spectrograms)
Output : Tensor [B, 10]         float32  (raw logits, one per class)

Architecture: small VGG-style 2D CNN
  3 × Conv blocks (32 → 64 → 128 filters)
  Global Average Pooling
  2 × Fully Connected layers

No audio-specific layers. Fully portable to ONNX opset 17.

Used by: train.py, evaluate.py, export_onnx.py
"""

import torch
import torch.nn as nn


# ── Architecture constants ────────────────────────────────────────────────────
NUM_CLASSES   = 10
IN_CHANNELS   = 1       # single-channel mel spectrogram
INPUT_H       = 64      # mel bins
INPUT_W       = 97      # time frames


# ── Conv block helper ─────────────────────────────────────────────────────────

def _conv_block(in_ch: int, out_ch: int, dropout: float = 0.25) -> nn.Sequential:
    """
    Standard conv block used in all three stages:
        Conv2d → BN → ReLU → Conv2d → BN → ReLU → MaxPool(2,2) → Dropout

    Both Conv2d layers use kernel=3, padding=1 (same-padding).
    MaxPool(2,2) halves both spatial dimensions.
    """
    return nn.Sequential(
        nn.Conv2d(in_ch,  out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Dropout2d(p=dropout),
    )


# ── Model ─────────────────────────────────────────────────────────────────────

class StreamSenseNet(nn.Module):
    """
    VGG-style 2D CNN for 1-second mel spectrogram classification.

    Spatial evolution (H × W):
        Input          :  1 × 64 × 97
        After block 1  : 32 × 32 × 48
        After block 2  : 64 × 16 × 24
        After block 3  : 128 × 8 × 12
        After GAP      : 128

    Args:
        num_classes : Number of output classes (default 10).
    """

    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()

        # ── Conv block 1: 1 → 32 ─────────────────────────────────────────────
        # Input  [B,  1, 64, 97]
        # Output [B, 32, 32, 48]
        self.block1 = _conv_block(in_ch=1, out_ch=32, dropout=0.25)

        # ── Conv block 2: 32 → 64 ────────────────────────────────────────────
        # Input  [B, 32, 32, 48]
        # Output [B, 64, 16, 24]
        self.block2 = _conv_block(in_ch=32, out_ch=64, dropout=0.25)

        # ── Conv block 3: 64 → 128 ───────────────────────────────────────────
        # Input  [B, 64, 16, 24]
        # Output [B, 128, 8, 12]
        self.block3 = _conv_block(in_ch=64, out_ch=128, dropout=0.25)

        # ── Global Average Pooling ────────────────────────────────────────────
        # Collapses [B, 128, 8, 12] → [B, 128]
        # Replaces Flatten + large FC; fewer parameters, less overfitting
        self.gap = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        # ── Classifier head ───────────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(64, num_classes),
        )

        # ── Weight initialisation ─────────────────────────────────────────────
        self._init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : Tensor [B, 1, 64, 97] float32

        Returns:
            logits : Tensor [B, num_classes] float32
                     Raw logits — apply softmax externally for probabilities.
        """
        x = self.block1(x)          # [B, 32, 32, 48]
        x = self.block2(x)          # [B, 64, 16, 24]
        x = self.block3(x)          # [B, 128, 8, 12]
        x = self.gap(x)             # [B, 128, 1, 1]
        x = x.flatten(start_dim=1)  # [B, 128]
        x = self.classifier(x)      # [B, 10]
        return x

    def _init_weights(self):
        """
        Kaiming (He) init for Conv2d layers (ReLU gain).
        BatchNorm layers initialised to weight=1, bias=0.
        Linear layers use default PyTorch init (adequate for this scale).
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)


# ── Parameter count helper ────────────────────────────────────────────────────

def count_parameters(model: nn.Module) -> dict:
    """
    Return total and trainable parameter counts.
    """
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


# ── Smoke test ────────────────────────────────────────────────────────────────

def _smoke_test():
    """
    Instantiate the model, run a forward pass with random input,
    and verify output shape, dtype, and parameter count.
    Runs entirely on CPU — no GPU required.
    """
    print("=" * 60)
    print("STREAMSENSE — model.py smoke test")
    print("=" * 60)

    model = StreamSenseNet(num_classes=NUM_CLASSES)
    model.eval()

    # ── Forward pass ──────────────────────────────────────────────────────────
    dummy = torch.zeros(1, 1, 64, 97)          # single sample, batch=1
    with torch.no_grad():
        logits = model(dummy)

    params = count_parameters(model)

    print(f"\nArchitecture summary:")
    print(f"  Input shape   : [1, 1, 64, 97]")
    print(f"  Output shape  : {tuple(logits.shape)}")
    print(f"  Output dtype  : {logits.dtype}")
    print(f"  Total params  : {params['total']:,}")
    print(f"  Trainable     : {params['trainable']:,}")

    # ── Per-block shape trace ─────────────────────────────────────────────────
    print(f"\nLayer-by-layer shape trace (batch=1):")
    x = dummy
    x = model.block1(x);  print(f"  After block1  : {tuple(x.shape)}")
    x = model.block2(x);  print(f"  After block2  : {tuple(x.shape)}")
    x = model.block3(x);  print(f"  After block3  : {tuple(x.shape)}")
    x = model.gap(x);     print(f"  After GAP     : {tuple(x.shape)}")
    x = x.flatten(1);     print(f"  After flatten : {tuple(x.shape)}")
    x = model.classifier(x); print(f"  After FC head : {tuple(x.shape)}")

    # ── Batch size test ───────────────────────────────────────────────────────
    print(f"\nBatch size test (B=32):")
    batch = torch.zeros(32, 1, 64, 97)
    with torch.no_grad():
        out = model(batch)
    print(f"  Input  : {tuple(batch.shape)}")
    print(f"  Output : {tuple(out.shape)}")

    # ── PASS / FAIL ───────────────────────────────────────────────────────────
    shape_ok = tuple(logits.shape) == (1, NUM_CLASSES)
    dtype_ok = logits.dtype == torch.float32
    batch_ok = tuple(out.shape) == (32, NUM_CLASSES)

    print(f"\nChecks:")
    print(f"  Output shape (1, 10)  : {'PASS' if shape_ok else 'FAIL'}")
    print(f"  Output dtype float32  : {'PASS' if dtype_ok else 'FAIL'}")
    print(f"  Batch shape  (32, 10) : {'PASS' if batch_ok else 'FAIL'}")

    if shape_ok and dtype_ok and batch_ok:
        print(f"\n[PASS] model.py verified. Ready for train.py")
    else:
        print(f"\n[FAIL] One or more checks failed.")

    return model


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    _smoke_test()
