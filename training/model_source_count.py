"""
model_source_count.py
Project STREAMSENSE — WA-3 Source Counting

CNN that takes a normalised mel spectrogram [B, 1, 64, 97] and outputs
8-class logits (predicted_count = argmax(logits) + 1, i.e. 1..8 speakers).

The convolutional backbone (block1/block2/block3) mirrors StreamSenseNet's
architecture exactly (see training/model.py) so that ImageNet-style transfer
of the frozen keyword-spotting backbone weights is possible via
load_backbone(). The counting head replaces StreamSenseNet's 10-class
classifier with an 8-class head.

Input  : [B, 1, 64, 97]  float32
Output : [B, 8]          float32 raw logits
"""

import torch
import torch.nn as nn


class _ConvBlock(nn.Module):
    """
    Conv2d -> BN -> ReLU -> Conv2d -> BN -> ReLU -> MaxPool(2,2) -> Dropout2d

    Wrapped in self.block (an nn.Sequential) rather than assigned directly,
    so state_dict keys look like "block1.block.0.weight". StreamSenseNet's
    checkpoint (best_model.pth) stores unwrapped keys like "block1.0.weight"
    — load_backbone() below remaps between the two conventions.
    """

    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.25):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=dropout),
        )

    def forward(self, x):
        return self.block(x)


class SourceCountCNN(nn.Module):
    """
    Spatial evolution (H x W), identical to StreamSenseNet:
        Input          :   1 x 64 x 97
        After block1   :  32 x 32 x 48
        After block2   :  64 x 16 x 24
        After block3   : 128 x  8 x 12
        After GAP      : 128

    Args:
        n_classes : number of count classes (default 8, for counts 1..8)
    """

    def __init__(self, n_classes: int = 8):
        super().__init__()

        # ── backbone — same structure as StreamSenseNet ──────────────────────
        self.block1 = _ConvBlock(1, 32)
        self.block2 = _ConvBlock(32, 64)
        self.block3 = _ConvBlock(64, 128)
        self.gap = nn.AdaptiveAvgPool2d(output_size=(1, 1))   # -> [B, 128, 1, 1]

        # ── counting head — new, replaces StreamSenseNet's classifier ────────
        self.head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(64, n_classes),
        )

    def load_backbone(self, ckpt_path: str) -> None:
        """
        Loads conv-block weights from checkpoints/best_model.pth (a
        StreamSenseNet checkpoint saved by train.py as
        {"epoch", "val_accuracy", "model_state", "num_classes", "mpic_version"}).

        StreamSenseNet's blocks are plain nn.Sequential assigned directly to
        self.block1/2/3, so its keys look like "block1.0.weight". This
        model's blocks are wrapped in _ConvBlock (self.block1.block.0...), so
        we insert ".block." after the block-name prefix before matching.
        Only backbone keys (block1/block2/block3/gap) are transferred; the
        counting head is always freshly initialised.
        """
        raw = torch.load(ckpt_path, map_location="cpu")
        src_sd = raw["model_state"] if "model_state" in raw else raw

        dst_sd = self.state_dict()
        backbone_prefixes = {"block1", "block2", "block3", "gap"}
        loaded, skipped = 0, 0

        for k, v in src_sd.items():
            parts = k.split(".")
            if parts[0] in backbone_prefixes:
                remapped = parts[0] + ".block." + ".".join(parts[1:]) if len(parts) > 1 else k
            else:
                remapped = k

            if remapped in dst_sd and dst_sd[remapped].shape == v.shape:
                dst_sd[remapped] = v
                loaded += 1
            else:
                skipped += 1

        self.load_state_dict(dst_sd, strict=False)
        print(f"[SourceCountCNN] Backbone: {loaded} loaded, {skipped} skipped.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)          # [B, 32, 32, 48]
        x = self.block2(x)          # [B, 64, 16, 24]
        x = self.block3(x)          # [B, 128, 8, 12]
        x = self.gap(x)             # [B, 128, 1, 1]
        x = x.flatten(start_dim=1)  # [B, 128]
        x = self.head(x)            # [B, 8]
        return x


if __name__ == "__main__":
    # Smoke test — no checkpoint required.
    model = SourceCountCNN(n_classes=8)
    dummy = torch.randn(4, 1, 64, 97)
    out = model(dummy)
    print(f"Output shape: {tuple(out.shape)}  (expected (4, 8))")
    assert tuple(out.shape) == (4, 8), "Unexpected output shape."
    print("[PASS] model_source_count.py smoke test.")
