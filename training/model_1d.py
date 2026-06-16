"""
model_1d.py
Project STREAMSENSE — Track A
Epic A3.2 — 1D CNN Baseline on raw audio frames (STRETCH)

Compares against StreamSenseNet (model.py, 2D CNN on log-mel spectrograms).
This model takes the RAW WAVEFORM directly as input:

    Input  : [B, 1, 16000]  float32  (raw audio, NOT mel spectrogram)
    Output : [B, 10]        float32  (logits)

Architecture: 4 Conv1D blocks with progressive downsampling (stride=4 each,
4^4 = 256x reduction: 16000 -> ~62 timesteps), then Global Average Pooling
and the SAME dense classifier head shape as StreamSenseNet (128 -> 64 -> 10)
for a fair comparison of "does the 2D mel representation help, holding the
classifier head constant?"

Run directly for a smoke test:
    python model_1d.py
"""

import torch
import torch.nn as nn


class ConvBlock1D(nn.Module):
    """Conv1D -> BN -> ReLU -> Conv1D -> BN -> ReLU -> MaxPool1D -> Dropout"""

    def __init__(self, in_channels, out_channels, pool_stride=4, dropout=0.25):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=9,
                                stride=1, padding=4, bias=False)
        self.bn1   = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=9,
                                stride=1, padding=4, bias=False)
        self.bn2   = nn.BatchNorm1d(out_channels)
        self.relu  = nn.ReLU(inplace=True)
        self.pool  = nn.MaxPool1d(kernel_size=pool_stride, stride=pool_stride)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.drop(x)
        return x


class StreamSenseNet1D(nn.Module):
    """
    1D CNN baseline on raw 16000-sample waveforms.

    Shape evolution (B = batch size):
        Input            : [B, 1, 16000]
        ConvBlock1D #1   : [B, 32,  4000]   (16000 / 4)
        ConvBlock1D #2   : [B, 64,  1000]   (4000  / 4)
        ConvBlock1D #3   : [B, 128, 250]    (1000  / 4)
        ConvBlock1D #4   : [B, 128, 62]     (250   / 4, floor)
        GAP              : [B, 128, 1] -> [B, 128]
        FC(128,64) ReLU  : [B, 64]
        Dropout(0.5)
        FC(64,10)        : [B, 10]   (logits)
    """

    def __init__(self, n_classes=10):
        super().__init__()

        # 4 blocks, each downsampling by 4x: 16000 -> 4000 -> 1000 -> 250 -> 62
        self.block1 = ConvBlock1D(1,   32,  pool_stride=4, dropout=0.25)
        self.block2 = ConvBlock1D(32,  64,  pool_stride=4, dropout=0.25)
        self.block3 = ConvBlock1D(64,  128, pool_stride=4, dropout=0.25)
        self.block4 = ConvBlock1D(128, 128, pool_stride=4, dropout=0.25)

        self.gap = nn.AdaptiveAvgPool1d(1)

        self.fc1 = nn.Linear(128, 64)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, n_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        """
        Args:
            x: [B, 1, 16000] float32 raw waveform (NOT mel spectrogram)
        Returns:
            [B, 10] float32 logits
        """
        x = self.block1(x)   # [B, 32,  4000]
        x = self.block2(x)   # [B, 64,  1000]
        x = self.block3(x)   # [B, 128, 250]
        x = self.block4(x)   # [B, 128, 62]

        x = self.gap(x)      # [B, 128, 1]
        x = x.squeeze(-1)    # [B, 128]

        x = self.relu(self.fc1(x))  # [B, 64]
        x = self.dropout(x)
        x = self.fc2(x)              # [B, 10]
        return x


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    print("=" * 60)
    print("StreamSenseNet1D — smoke test")
    print("=" * 60)

    model = StreamSenseNet1D(n_classes=10)
    model.eval()

    n_params = count_parameters(model)
    print(f"Total parameters: {n_params:,}")

    # Smoke test with a batch of 4
    x = torch.randn(4, 1, 16000)
    with torch.no_grad():
        out = model(x)

    print(f"Input  shape: {tuple(x.shape)}")
    print(f"Output shape: {tuple(out.shape)}")

    assert tuple(out.shape) == (4, 10), f"Expected (4,10), got {tuple(out.shape)}"
    assert out.dtype == torch.float32

    # Compare param count to StreamSenseNet (2D, 295,786 params)
    print(f"\nStreamSenseNet (2D)  : 295,786 params")
    print(f"StreamSenseNet1D (1D): {n_params:,} params")

    print("\n[PASS] Smoke test OK.")
