"""
dataset_source_count.py
Project STREAMSENSE — WA-3 Source Counting

PyTorch Dataset that loads pre-mixed .npy clips (produced by
build_source_count_dataset.py) and returns mel spectrograms via the frozen
mel_pipeline.preprocess() function.

Split CSV format (header row):
    filepath,label

Returns:
    (tensor [1, 64, 97] float32, label int64)

Used by: train_source_count.py
"""

import csv
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

# mel_pipeline must be importable from the same directory (training/)
try:
    import mel_pipeline
except ImportError as e:
    print(f"[ERROR] Cannot import mel_pipeline: {e}")
    print("        Ensure mel_pipeline.py is in the same directory (training/).")
    sys.exit(1)


class SourceCountDataset(Dataset):
    """
    Args:
        csv_path : path to source_count_train.csv / source_count_val.csv
        augment  : if True, adds small Gaussian noise to the mel spectrogram
    """

    def __init__(self, csv_path, augment: bool = False):
        self.csv_path = Path(csv_path)
        if not self.csv_path.exists():
            raise FileNotFoundError(f"Split file not found: {self.csv_path}")

        self.records = []
        with open(self.csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.records.append((row["filepath"], int(row["label"])))

        if len(self.records) == 0:
            raise RuntimeError(f"No records loaded from {self.csv_path}")

        self.augment = augment

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        filepath, label = self.records[idx]

        wav = np.load(filepath)                        # [16000] float32
        wav_tensor = torch.from_numpy(wav).float()      # [16000]

        # preprocess() already returns a torch.Tensor of shape [1, 1, 64, 97] —
        # do NOT wrap the output in torch.from_numpy() again.
        mel = mel_pipeline.preprocess(wav_tensor)        # [1, 1, 64, 97]
        mel = mel.squeeze(0)                              # [1, 64, 97]

        if self.augment:
            mel = mel + torch.randn_like(mel) * 0.01

        return mel, torch.tensor(label, dtype=torch.long)


if __name__ == "__main__":
    # Quick smoke test: python training/dataset_source_count.py <csv_path>
    if len(sys.argv) < 2:
        print("Usage: python dataset_source_count.py <path_to_source_count_train.csv>")
        sys.exit(1)

    ds = SourceCountDataset(sys.argv[1])
    mel, label = ds[0]
    print(f"Dataset size : {len(ds)}")
    print(f"mel shape    : {tuple(mel.shape)}  (expected (1, 64, 97))")
    print(f"label        : {label.item()}  (expected 0-7)")
