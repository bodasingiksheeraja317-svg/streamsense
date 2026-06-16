"""
dataset_1d.py
Project STREAMSENSE — Track A
Epic A3.2 — Dataset adapter for the 1D CNN baseline (STRETCH)

Returns RAW WAVEFORMS [1, 16000] instead of mel spectrograms [1, 64, 97].
Reuses the same split files (train_files.txt / val_files.txt / test_files.txt)
and the SAME time-domain augmentations as the 2D pipeline (dataset.py), for
a fair comparison — only the representation (raw vs mel) differs, not the
augmentation strategy.

NOTE: SpecAugment (frequency/time masking) from the 2D pipeline does NOT
apply here, since it operates on the mel spectrogram. Only the time-domain
augmentations (time shift, Gaussian noise, amplitude scaling) are used.

Run directly for a smoke test:
    python dataset_1d.py
"""

import os
import random
from pathlib import Path

import torch
import torchaudio
from torch.utils.data import Dataset

# ── Root path — environment-aware ─────────────────────────────────────────
# On your Windows machine: defaults to C:\STREAMSENSE (unchanged behavior).
# On Colab: set the STREAMSENSE_ROOT environment variable before running,
# e.g. in a notebook cell:
#     import os
#     os.environ["STREAMSENSE_ROOT"] = "/content/STREAMSENSE"
# or pass it inline:
#     STREAMSENSE_ROOT=/content/STREAMSENSE python train_1d.py
_DEFAULT_ROOT = r"C:\STREAMSENSE" if os.name == "nt" else "/content/STREAMSENSE"
ROOT       = Path(os.environ.get("STREAMSENSE_ROOT", _DEFAULT_ROOT))
SPLITS_DIR = ROOT / "data" / "splits"

FRAME_LEN = 16000

# ── Time-domain augmentation parameters (match dataset.py) ────────────────────
TIME_SHIFT_PCT   = 0.20    # +/- 20% of frame length
NOISE_STD        = 0.005
AMPLITUDE_RANGE  = (0.8, 1.2)


def parse_split_line(line: str):
    parts     = line.strip().split("|")
    path      = Path(parts[0].strip())
    label     = parts[1].strip()
    class_idx = int(parts[2].strip())
    return path, label, class_idx


class StreamSenseDataset1D(Dataset):
    """
    Returns (waveform, class_idx) where waveform is [1, 16000] float32.

    Args:
        split: "train", "val", or "test"
        augment: if True, applies time-domain augmentations (train only)
    """

    def __init__(self, split: str, augment: bool = False):
        assert split in ("train", "val", "test"), f"Unknown split: {split}"

        self.split   = split
        self.augment = augment and (split == "train")

        split_file = SPLITS_DIR / f"{split}_files.txt"
        if not split_file.exists():
            raise FileNotFoundError(f"Split file not found: {split_file}")

        self.entries = []
        with open(split_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self.entries.append(parse_split_line(line))

    def __len__(self):
        return len(self.entries)

    def _load_waveform(self, path: Path) -> torch.Tensor:
        """Load WAV, mono, pad/crop to FRAME_LEN -> [1, 16000] float32"""
        waveform, sr = torchaudio.load(str(path))

        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        length = waveform.shape[1]
        if length < FRAME_LEN:
            waveform = torch.nn.functional.pad(waveform, (0, FRAME_LEN - length))
        elif length > FRAME_LEN:
            waveform = waveform[:, :FRAME_LEN]

        return waveform.float()  # [1, 16000]

    def _augment_waveform(self, waveform: torch.Tensor) -> torch.Tensor:
        """Apply time-domain augmentations: circular shift, noise, amplitude scale."""

        # Circular time shift +/- 20%
        max_shift = int(FRAME_LEN * TIME_SHIFT_PCT)
        shift = random.randint(-max_shift, max_shift)
        if shift != 0:
            waveform = torch.roll(waveform, shifts=shift, dims=-1)

        # Additive Gaussian noise
        noise = torch.randn_like(waveform) * NOISE_STD
        waveform = waveform + noise

        # Amplitude scaling
        scale = random.uniform(*AMPLITUDE_RANGE)
        waveform = waveform * scale

        return waveform

    def __getitem__(self, idx):
        path, label, class_idx = self.entries[idx]
        waveform = self._load_waveform(path)  # [1, 16000]

        if self.augment:
            waveform = self._augment_waveform(waveform)

        return waveform, class_idx


if __name__ == "__main__":
    print("=" * 60)
    print("StreamSenseDataset1D — smoke test")
    print("=" * 60)

    for split in ("train", "val", "test"):
        ds = StreamSenseDataset1D(split=split, augment=(split == "train"))
        x, y = ds[0]
        print(f"  {split:5s}: {len(ds):6d} samples  "
              f"sample0 shape={tuple(x.shape)} dtype={x.dtype} label={y}")

        assert tuple(x.shape) == (1, FRAME_LEN), f"Shape error: {x.shape}"
        assert x.dtype == torch.float32

    print("\n[PASS] Smoke test OK.")
