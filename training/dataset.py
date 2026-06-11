"""
dataset.py
Project STREAMSENSE — Track A
MPIC v1.0 — PyTorch Dataset and DataLoader for training, validation, and test.

Split file format (one line per sample):
    C:\\STREAMSENSE\\data\\raw\\yes\\file.wav | yes | 0

Returns:
    (tensor [1, 1, 64, 97] float32,  class_index int)

Used by: train.py, evaluate.py
"""

import sys
import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio
from pathlib import Path

# mel_pipeline must be on the Python path (same directory)
try:
    from mel_pipeline import preprocess
except ImportError as e:
    print(f"[ERROR] Cannot import mel_pipeline: {e}")
    print("        Ensure mel_pipeline.py is in the same directory.")
    sys.exit(1)

# ── Constants ─────────────────────────────────────────────────────────────────
NUM_CLASSES   = 10
BATCH_SIZE    = 32
NUM_WORKERS   = 2          # set to 0 on Windows if multiprocessing issues arise
EXPECTED_SHAPE = (1, 1, 64, 97)

# ── Split file parser ─────────────────────────────────────────────────────────

def _parse_split_file(split_path: Path) -> list[tuple[Path, int]]:
    """
    Read a split file and return a list of (wav_path, class_index) tuples.

    Line format:  C:\\STREAMSENSE\\data\\raw\\yes\\file.wav | yes | 0
    Fields:       [0] absolute Windows path  [1] label string  [2] class index

    Lines that are blank or start with '#' are skipped.
    Raises FileNotFoundError if the split file does not exist.
    """
    if not split_path.exists():
        raise FileNotFoundError(f"Split file not found: {split_path}")

    samples = []
    skipped = 0

    with open(split_path, "r") as f:
        for lineno, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split("|")
            if len(parts) != 3:
                print(
                    f"  [WARN] dataset.py: skipping malformed line {lineno} "
                    f"in {split_path.name}: '{line[:80]}'"
                )
                skipped += 1
                continue

            wav_path  = Path(parts[0].strip())
            class_idx = int(parts[2].strip())

            if class_idx < 0 or class_idx >= NUM_CLASSES:
                print(
                    f"  [WARN] dataset.py: class index {class_idx} out of range "
                    f"at line {lineno} — skipping."
                )
                skipped += 1
                continue

            samples.append((wav_path, class_idx))

    if skipped:
        print(f"  [WARN] {skipped} line(s) skipped in {split_path.name}")

    return samples


# ── Dataset ───────────────────────────────────────────────────────────────────

class StreamSenseDataset(Dataset):
    """
    PyTorch Dataset for STREAMSENSE.

    Each __getitem__ call:
        1. Loads the WAV file via torchaudio
        2. Passes raw waveform through mel_pipeline.preprocess()
        3. Returns (tensor [1, 1, 64, 97] float32, class_index int)

    Missing files are returned as a zero tensor with their original class_index
    and a warning is printed once per missing path (not on every epoch).

    Args:
        split_file  : Path to train_files.txt / val_files.txt / test_files.txt
        augment     : If True, apply time-domain augmentations before mel.
                      Set True only for training splits.
        verbose     : Print summary after loading the split file.
    """

    # Track missing paths across all instances to avoid repeated warnings
    _warned_missing: set = set()

    def __init__(
        self,
        split_file: Path | str,
        augment: bool = False,
        verbose: bool = True,
    ):
        self.split_file = Path(split_file)
        self.augment    = augment
        self.samples    = _parse_split_file(self.split_file)

        if verbose:
            print(
                f"[dataset] Loaded '{self.split_file.name}': "
                f"{len(self.samples)} samples  |  augment={augment}"
            )

    # ── Length ────────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.samples)

    # ── Single item ───────────────────────────────────────────────────────────

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        wav_path, class_idx = self.samples[idx]

        # ── Load WAV ──────────────────────────────────────────────────────────
        if not wav_path.exists():
            if wav_path not in StreamSenseDataset._warned_missing:
                print(f"  [WARN] Missing WAV: {wav_path}  → returning zero tensor")
                StreamSenseDataset._warned_missing.add(wav_path)
            zero = torch.zeros(1, 1, 64, 97, dtype=torch.float32)
            return zero, class_idx

        try:
            waveform, sr = torchaudio.load(str(wav_path))   # [C, T]
        except Exception as e:
            print(f"  [WARN] Cannot load {wav_path.name}: {e}  → returning zero tensor")
            zero = torch.zeros(1, 1, 64, 97, dtype=torch.float32)
            return zero, class_idx

        # ── Mono ──────────────────────────────────────────────────────────────
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)   # [1, T]

        # ── Extract float32 numpy array [T] ───────────────────────────────────
        raw = waveform.squeeze(0).numpy()                   # [T] float32

        # ── Time-domain augmentations (training only) ─────────────────────────
        if self.augment:
            raw = _augment_raw(raw)

        # ── mel_pipeline.preprocess() expects float32 array [16000] ──────────
        # It handles pad/crop internally.
        tensor = preprocess(raw)                            # [1, 1, 64, 97]

        return tensor, class_idx

    # ── Convenience repr ──────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"StreamSenseDataset("
            f"split='{self.split_file.name}', "
            f"n={len(self.samples)}, "
            f"augment={self.augment})"
        )


# ── Time-domain augmentations ─────────────────────────────────────────────────
# Applied in __getitem__ BEFORE mel_pipeline, training split only.
# SpecAugment (time/freq masking) is applied in train.py on the tensor output.

import numpy as np

FRAME_LEN     = 16000
SHIFT_MAX     = 0.20          # ±20% of frame length
NOISE_STD     = 0.005         # Gaussian noise standard deviation
AMP_MIN       = 0.80          # amplitude scale min
AMP_MAX       = 1.20          # amplitude scale max


def _augment_raw(raw: np.ndarray) -> np.ndarray:
    """
    Apply three time-domain augmentations in sequence.

    1. Time shift   — circular shift by ±20% of frame length
    2. Gaussian noise — add noise with std=0.005
    3. Amplitude scale — scale by random factor in [0.8, 1.2]

    Input/output: float32 numpy array, any length (pad/crop handled by mel_pipeline).
    """
    n = len(raw)

    # 1. Time shift (circular)
    max_shift = int(n * SHIFT_MAX)
    shift     = np.random.randint(-max_shift, max_shift + 1)
    raw       = np.roll(raw, shift)

    # 2. Gaussian noise
    raw = raw + np.random.normal(0.0, NOISE_STD, size=raw.shape).astype(np.float32)

    # 3. Amplitude scaling
    scale = np.random.uniform(AMP_MIN, AMP_MAX)
    raw   = (raw * scale).astype(np.float32)

    return raw


# ── DataLoader factory ────────────────────────────────────────────────────────

def get_dataloader(
    split_file : Path | str,
    is_train   : bool  = False,
    batch_size : int   = BATCH_SIZE,
    num_workers: int   = NUM_WORKERS,
    verbose    : bool  = True,
) -> DataLoader:
    """
    Build and return a DataLoader for the given split.

    Args:
        split_file  : Path to train_files.txt / val_files.txt / test_files.txt
        is_train    : True  → shuffle=True, augment=True  (training split)
                      False → shuffle=False, augment=False (val / test splits)
        batch_size  : Samples per batch (default 32)
        num_workers : Worker processes for data loading (default 2)
                      Set to 0 on Windows if you see multiprocessing errors.
        verbose     : Print DataLoader summary.

    Returns:
        torch.utils.data.DataLoader
    """
    dataset = StreamSenseDataset(
        split_file = Path(split_file),
        augment    = is_train,
        verbose    = verbose,
    )

    loader = DataLoader(
        dataset,
        batch_size  = batch_size,
        shuffle     = is_train,
        num_workers = num_workers,
        pin_memory  = False,     # CPU training — pin_memory gives no benefit
        drop_last   = False,
    )

    if verbose:
        n_batches = len(loader)
        print(
            f"[dataloader] '{Path(split_file).name}': "
            f"{len(dataset)} samples → {n_batches} batches "
            f"(batch={batch_size}, shuffle={is_train}, workers={num_workers})"
        )

    return loader


# ── Smoke test ────────────────────────────────────────────────────────────────

def _smoke_test(split_file: Path):
    """
    Load the first batch from a split and verify tensor shape and dtype.
    Prints PASS or FAIL. Does not require GPU.
    """
    print(f"\n{'='*60}")
    print(f"STREAMSENSE — dataset.py smoke test")
    print(f"{'='*60}")
    print(f"Split : {split_file}")

    loader = get_dataloader(split_file, is_train=False, batch_size=4, num_workers=0)

    tensors, labels = next(iter(loader))

    print(f"\nFirst batch:")
    print(f"  tensor shape : {tuple(tensors.shape)}")
    print(f"  tensor dtype : {tensors.dtype}")
    print(f"  labels       : {labels.tolist()}")
    print(f"  tensor min   : {tensors.min():.4f}")
    print(f"  tensor max   : {tensors.max():.4f}")

    expected = (4, 1, 1, 64, 97)
    if tuple(tensors.shape) == expected and tensors.dtype == torch.float32:
        print(f"\n[PASS] Output shape {tuple(tensors.shape)} and dtype float32 correct.")
    else:
        print(
            f"\n[FAIL] Expected shape {expected} and dtype float32, "
            f"got {tuple(tensors.shape)} {tensors.dtype}"
        )


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Run a quick smoke test on train_files.txt if it exists.
    # Usage: python dataset.py
    #    or: python dataset.py C:\STREAMSENSE\data\splits\val_files.txt

    if len(sys.argv) > 1:
        target = Path(sys.argv[1])
    else:
        target = Path(r"C:\STREAMSENSE\data\splits\train_files.txt")

    if not target.exists():
        print(f"[ERROR] Split file not found: {target}")
        print("Usage: python dataset.py [path_to_split_file.txt]")
        sys.exit(1)

    _smoke_test(target)
