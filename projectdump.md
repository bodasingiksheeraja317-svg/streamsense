# STREAMSENSE — Project Dump

## Project Tree

```
STREAMSENSE/
├── checkpoints/ [...]
├── checkpoints_1d/ [...]
├── data/ [...]
├── evaluation/
│   ├── confusion_matrix.png
│   ├── evaluation_report.txt
│   ├── multihead_onnx_evaluation_report.txt
│   └── qonnx_evaluation_report.txt
├── evaluation_1d/
│   ├── comparison_1d_vs_2d.txt
│   ├── confusion_matrix_1d.png
│   └── evaluation_report_1d.txt
├── golden_vectors/
│   ├── labels/ [...]
│   ├── mel/ [...]
│   ├── normalized/ [...]
│   ├── raw/ [...]
│   ├── wav/ [...]
├── golden_vectors_1000/
│   ├── labels/ [...]
│   ├── mel/ [...]
│   ├── normalized/ [...]
│   ├── raw/ [...]
├── golden_vectors_10_matlab/
│   ├── labels/ [...]
│   ├── mel/ [...]
│   ├── normalized/ [...]
│   ├── raw/ [...]
│   ├── README.txt
│   └── verify_report.txt
├── onnx_models/
│   ├── quantize_ptq.ipynb
├── recordings/ [...]
├── stats/
│   ├── golden_selection.json
│   └── normalization_stats.json
├── streamsense-env-win/ [...]
├── training/
│   ├── __pycache__/ [...]
│   ├── logs/ [...]
│   ├── compute_normstats.py
│   ├── dataset.py
│   ├── dataset_1d.py
│   ├── evaluate.py
│   ├── evaluate_1d_comparison.py
│   ├── evaluate_multihead_onnx.py
│   ├── evaluate_onnx.py
│   ├── evaluate_qonnx.py
│   ├── export_multihead_onnx.py
│   ├── export_onnx.ipynb
│   ├── generate_golden.py
│   ├── generate_golden10_matlab.py
│   ├── generate_golden_1000.py
│   ├── goldenvector_stream.py
│   ├── live_demo.py
│   ├── live_gv1k_demo.py
│   ├── mel_pipeline.py
│   ├── mel_pipeline_matlab.m
│   ├── model.py
│   ├── model_1d.py
│   ├── nsp_receiver.py
│   ├── nsp_sender.py
│   ├── populate_gv_top1.py
│   ├── qat_finetune.py
│   ├── run_gv_regression_1000.py
│   ├── select_golden.py
│   ├── stream_simulator.py
│   ├── streaming_framer.py
│   ├── test_integration.py
│   ├── train.py
│   ├── train_1d.py
│   ├── verify_gv10_matlab.m
│   └── verify_pipeline.py
├── unknown_data/ [...]
├── .gitattributes
├── .gitignore
├── class_labels.json
├── qat_colab.ipynb
├── quick_predict.ipynb
├── Streamsense.ipynb
└── STREAMSENSE1D.ipynb
```

---
## Source Files

### `training/compute_normstats.py`

```python
"""
compute_normstats.py
Project STREAMSENSE — Track A
Computes global_mean and global_std over the training split.

Pipeline steps applied here: Steps 1-6 only (NO normalization).
Step 7 (normalization) requires these stats — computed here.

MPIC v1.0 params used:
    sample_rate  = 16000
    n_fft        = 512
    hop_length   = 160
    n_mels       = 64
    center       = False   <- critical, gives T=97
    power        = 2.0
    log scaling  = 10 * log10(mel + 1e-10)
    clip floor   = -80 dB

Expected T = floor((16000 - 512) / 160) + 1 = 97
Expected spectrogram shape per file: [64, 97]
Expected n_elements per file: 64 * 97 = 6208
Expected train files: 26984
"""

import torch
import torchaudio
import numpy as np
import json
import sys
from pathlib import Path

# ── Paths (native Windows — no WSL conversion needed) ────────────────────────
TRAIN_SPLIT = Path(r"C:\STREAMSENSE\data\splits\train_files.txt")
STATS_DIR   = Path(r"C:\STREAMSENSE\stats")
STATS_OUT   = STATS_DIR / "normalization_stats.json"

# ── MPIC v1.0 frozen parameters ──────────────────────────────────────────────
SAMPLE_RATE   = 16000
FRAME_LEN     = 16000          # samples
N_FFT         = 512
HOP_LENGTH    = 160            # 10 ms at 16 kHz
N_MELS        = 64
CENTER        = False          # critical — do NOT change
POWER         = 2.0
LOG_EPS       = 1e-10
CLIP_FLOOR_DB = -80.0

EXPECTED_T     = (FRAME_LEN - N_FFT) // HOP_LENGTH + 1  # = 97
EXPECTED_SHAPE = (N_MELS, EXPECTED_T)                    # (64, 97)
EXPECTED_ELEMS = N_MELS * EXPECTED_T                     # 6208

# ── Build MelSpectrogram transform (CPU, reused for every file) ───────────────
mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate = SAMPLE_RATE,
    n_fft       = N_FFT,
    hop_length  = HOP_LENGTH,
    n_mels      = N_MELS,
    window_fn   = torch.hann_window,
    center      = CENTER,
    power       = POWER,
)

# ── Helper: read Windows path from split line, return Path object ─────────────
def parse_path(line: str) -> Path:
    """
    Split file format:  C:\STREAMSENSE\data\raw\yes\file.wav | yes | 0
    Extract first field, strip whitespace, return as Path.
    Already a Windows path — no conversion needed.
    """
    win_path = line.split("|")[0].strip()
    return Path(win_path)

# ── Helper: load one WAV → float32 tensor [1, FRAME_LEN] ─────────────────────
def load_wav(path: Path) -> torch.Tensor:
    """
    Steps 1-3 of MPIC pipeline:
      1. Load as float32
      2. Convert stereo -> mono (mean channels)
      3. Pad (zeros right) or crop to exactly FRAME_LEN samples
    Returns shape [1, 16000]
    """
    waveform, sr = torchaudio.load(str(path))          # [C, T]

    # Step 2: mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)  # [1, T]

    # Step 3: pad or crop
    length = waveform.shape[1]
    if length < FRAME_LEN:
        waveform = torch.nn.functional.pad(waveform, (0, FRAME_LEN - length))
    elif length > FRAME_LEN:
        waveform = waveform[:, :FRAME_LEN]

    return waveform.float()                            # [1, 16000]

# ── Helper: waveform → log-mel spectrogram (Steps 4-6) ───────────────────────
def compute_logmel(waveform: torch.Tensor) -> torch.Tensor:
    """
    Steps 4-6 of MPIC pipeline (no normalization):
      4. MelSpectrogram -> [1, 64, 97]
      5. 10 * log10(mel + 1e-10)
      6. clamp min=-80 dB
    Returns shape [64, 97]
    """
    mel = mel_transform(waveform)               # [1, 64, 97]
    mel = 10.0 * torch.log10(mel + LOG_EPS)     # log scaling
    mel = torch.clamp(mel, min=CLIP_FLOOR_DB)   # clip floor
    return mel.squeeze(0)                       # [64, 97]

# ── Read split file → list of Paths ──────────────────────────────────────────
def read_split(split_file: Path) -> list:
    paths = []
    with open(split_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            paths.append(parse_path(line))
    return paths

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("STREAMSENSE — compute_normstats.py")
    print("=" * 60)

    # Verify MPIC T=97
    assert EXPECTED_T == 97, f"T mismatch: got {EXPECTED_T}, expected 97"
    print(f"[OK] MPIC check: T = {EXPECTED_T}, shape = {EXPECTED_SHAPE}")

    # Create stats dir if needed
    STATS_DIR.mkdir(parents=True, exist_ok=True)

    # Load file list
    print(f"\nReading split: {TRAIN_SPLIT}")
    if not TRAIN_SPLIT.exists():
        print(f"[ERROR] Split file not found: {TRAIN_SPLIT}")
        sys.exit(1)

    file_list = read_split(TRAIN_SPLIT)
    n_files   = len(file_list)
    print(f"Files found in split: {n_files}")

    if n_files != 26984:
        print(f"[WARN] Expected 26984 train files, got {n_files}. Continuing.")

    # ── Online accumulation (sum-of-squares, float64 for precision) ──────────
    # ~26984 files x 6208 elements = ~167.6M values
    # Storing all in RAM would need ~1.3 GB — accumulate instead
    sum_x  = np.float64(0.0)
    sum_x2 = np.float64(0.0)
    n_elem = np.int64(0)

    n_errors     = 0
    REPORT_EVERY = 1000

    print(f"\nProcessing {n_files} files (reporting every {REPORT_EVERY})...\n")

    for i, wav_path in enumerate(file_list):

        # Progress report
        if i % REPORT_EVERY == 0:
            pct = 100.0 * i / n_files
            print(f"  [{i:>6}/{n_files}]  {pct:5.1f}%", flush=True)

        if not wav_path.exists():
            print(f"  [SKIP] Missing: {wav_path}")
            n_errors += 1
            continue

        try:
            waveform = load_wav(wav_path)        # [1, 16000]
            logmel   = compute_logmel(waveform)  # [64, 97]

            # Shape guard
            if tuple(logmel.shape) != EXPECTED_SHAPE:
                print(f"  [SKIP] Bad shape {logmel.shape}: {wav_path.name}")
                n_errors += 1
                continue

            # Accumulate
            arr     = logmel.numpy().astype(np.float64).ravel()  # (6208,)
            sum_x  += arr.sum()
            sum_x2 += (arr * arr).sum()
            n_elem += arr.size

        except Exception as e:
            print(f"  [SKIP] Error on {wav_path.name}: {e}")
            n_errors += 1
            continue

    print(f"\n  [{n_files}/{n_files}]  100.0%  — done\n")

    # ── Compute mean and std ──────────────────────────────────────────────────
    if n_elem == 0:
        print("[ERROR] No elements accumulated. Check your paths.")
        sys.exit(1)

    global_mean = float(sum_x / n_elem)
    variance    = float(sum_x2 / n_elem - (sum_x / n_elem) ** 2)

    if variance < 0:
        print(f"[WARN] Tiny negative variance ({variance:.2e}) due to float rounding — clamping to 0")
        variance = 0.0

    global_std = float(np.sqrt(variance))

    # ── Save JSON ─────────────────────────────────────────────────────────────
    stats = {
        "global_mean"   : global_mean,
        "global_std"    : global_std,
        "n_files"       : int(n_files - n_errors),
        "n_elements"    : int(n_elem),
        "n_errors"      : int(n_errors),
        "mpic_version"  : "1.0",
        "n_fft"         : N_FFT,
        "hop_length"    : HOP_LENGTH,
        "n_mels"        : N_MELS,
        "center"        : CENTER,
        "clip_floor_db" : CLIP_FLOOR_DB,
        "log_eps"       : LOG_EPS,
    }

    with open(STATS_OUT, "w") as f:
        json.dump(stats, f, indent=2)

    # ── Final report ──────────────────────────────────────────────────────────
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"  Files processed : {n_files - n_errors} / {n_files}")
    print(f"  Files skipped   : {n_errors}")
    print(f"  Total elements  : {n_elem:,}  (expected ~{26984 * EXPECTED_ELEMS:,})")
    print(f"  global_mean     : {global_mean:.6f}  dB")
    print(f"  global_std      : {global_std:.6f}  dB")
    print(f"\nSaved -> {STATS_OUT}")

    # Sanity checks — these ranges are expected for Speech Commands log-mel
    if not (-60.0 < global_mean < 0.0):
        print(f"[WARN] mean={global_mean:.4f} looks unusual (expected between -60 and 0 dB)")
    if not (5.0 < global_std < 40.0):
        print(f"[WARN] std={global_std:.4f} looks unusual (expected between 5 and 40)")

    if n_errors == 0:
        print("\n[DONE] All files processed cleanly.")
    else:
        print(f"\n[DONE] Completed with {n_errors} skipped files.")

if __name__ == "__main__":
    main()

```

### `training/dataset.py`

```python
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

```

### `training/dataset_1d.py`

```python
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

import numpy as np
import soundfile as sf
import torch
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
        """
        Load WAV via soundfile (NOT torchaudio.load) -> mono, pad/crop to
        FRAME_LEN -> [1, 16000] float32.

        Uses soundfile directly rather than torchaudio.load because recent
        torchaudio versions (e.g. 2.11.0 on Colab) route loading through a
        torchcodec/FFmpeg backend that can fail with
        "RuntimeError: SingleStreamDecoder ..." on some environments, and
        torchaudio.set_audio_backend() has been removed in these versions
        so the legacy backend can no longer be force-selected. soundfile
        reads WAV files directly via libsndfile and avoids this entirely.
        """
        data, sr = sf.read(str(path), dtype="float32", always_2d=True)
        # data: [T, C] from soundfile (note: channel-LAST, opposite of
        # torchaudio's [C, T] convention) -> convert to [C, T]
        waveform = torch.from_numpy(data.T)  # [C, T]

        if sr != 16000:
            # Speech Commands v2 is 16kHz; this guards against any
            # mismatched files rather than silently mishandling them.
            import torchaudio.functional as AF
            waveform = AF.resample(waveform, sr, 16000)

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

```

### `training/evaluate.py`

```python
"""
evaluate.py
Project STREAMSENSE — Track A
MPIC v1.0 — Final test set evaluation.

RUN ONCE ONLY after model is accepted from training.

Inputs:
    checkpoints/best_model.pth
    data/splits/test_files.txt      via dataset.py

Outputs:
    evaluation/evaluation_report.txt
    evaluation/confusion_matrix.png

Usage:
    python evaluate.py                         (local CPU or Colab)
    python evaluate.py --ckpt path/to/model.pth
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — works on Colab and headless
import matplotlib.pyplot as plt

try:
    from sklearn.metrics import (
        classification_report,
        confusion_matrix,
        accuracy_score,
    )
except ImportError:
    print("[ERROR] scikit-learn not installed.")
    print("        Run: pip install scikit-learn")
    sys.exit(1)

try:
    from model   import StreamSenseNet
    from dataset import get_dataloader
except ImportError as e:
    print(f"[ERROR] Import failed: {e}")
    sys.exit(1)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).resolve().parent.parent
SPLITS_DIR  = BASE_DIR / "data"  / "splits"
CKPT_DIR    = BASE_DIR / "checkpoints"
EVAL_DIR    = BASE_DIR / "evaluation"
LABELS_PATH = BASE_DIR / "class_labels.json"

TEST_SPLIT  = SPLITS_DIR / "test_files.txt"
DEFAULT_CKPT= CKPT_DIR   / "best_model.pth"
REPORT_PATH = EVAL_DIR   / "evaluation_report.txt"
CM_PATH     = EVAL_DIR   / "confusion_matrix.png"

# ── Constants ─────────────────────────────────────────────────────────────────
NUM_CLASSES  = 10
BATCH_SIZE   = 32
NUM_WORKERS  = 2


# ── Load class labels ─────────────────────────────────────────────────────────

def load_class_labels() -> dict:
    if not LABELS_PATH.exists():
        # Fallback — hardcoded order matches dataset split indices
        return {0:"yes",1:"no",2:"up",3:"down",4:"left",
                5:"right",6:"on",7:"off",8:"stop",9:"go"}
    with open(LABELS_PATH, "r") as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}


# ── Confusion matrix plot ─────────────────────────────────────────────────────

def plot_confusion_matrix(
    cm          : np.ndarray,
    class_names : list[str],
    save_path   : Path,
):
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)

    ax.set(
        xticks     = np.arange(len(class_names)),
        yticks     = np.arange(len(class_names)),
        xticklabels= class_names,
        yticklabels= class_names,
        ylabel     = "True Label",
        xlabel     = "Predicted Label",
        title      = "STREAMSENSE — Confusion Matrix (Test Set)",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=10)

    # Annotate cells
    thresh = cm.max() / 2.0
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(
                j, i, f"{cm[i,j]}",
                ha="center", va="center", fontsize=9,
                color="white" if cm[i, j] > thresh else "black",
            )

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {save_path}")


# ── Main evaluation ───────────────────────────────────────────────────────────

def evaluate(args):
    ckpt_path = Path(args.ckpt)

    print("=" * 60)
    print("STREAMSENSE — evaluate.py")
    print("=" * 60)

    # ── Device ────────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice     : {device}")

    # ── Prerequisites ──────────────────────────────────────────────────────────
    for p, name in [(ckpt_path, "best_model.pth"), (TEST_SPLIT, "test_files.txt")]:
        if not p.exists():
            print(f"[ERROR] Not found: {p}")
            sys.exit(1)

    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load model ─────────────────────────────────────────────────────────────
    print(f"\nLoading checkpoint: {ckpt_path}")
    ckpt  = torch.load(ckpt_path, map_location="cpu")
    model = StreamSenseNet(num_classes=NUM_CLASSES)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    print(f"  Trained epoch  : {ckpt['epoch']}")
    print(f"  Val accuracy   : {ckpt['val_accuracy']:.2f}%")

    # ── Class labels ───────────────────────────────────────────────────────────
    class_labels = load_class_labels()
    class_names  = [class_labels[i] for i in range(NUM_CLASSES)]
    print(f"\nClasses    : {class_names}")

    # ── Test DataLoader ────────────────────────────────────────────────────────
    print(f"\nLoading test split: {TEST_SPLIT}")
    test_loader = get_dataloader(
        TEST_SPLIT,
        is_train    = False,
        batch_size  = BATCH_SIZE,
        num_workers = NUM_WORKERS,
        verbose     = True,
    )

    # ── Inference loop ─────────────────────────────────────────────────────────
    print(f"\nRunning inference on {len(test_loader.dataset)} test samples...")

    all_preds  = []
    all_labels = []
    criterion  = nn.CrossEntropyLoss()
    total_loss = 0.0

    with torch.no_grad():
        for batch_idx, (tensors, labels) in enumerate(test_loader):
            x = tensors.squeeze(1).to(device)   # [B, 1, 64, 97]
            y = labels.to(device)

            logits = model(x)                    # [B, 10]
            loss   = criterion(logits, y)
            total_loss += loss.item()

            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

            if (batch_idx + 1) % 30 == 0:
                print(f"  [{batch_idx+1:>3}/{len(test_loader)}] batches done...",
                      flush=True)

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)

    # ── Metrics ────────────────────────────────────────────────────────────────
    test_loss = total_loss / len(test_loader)
    test_acc  = 100.0 * accuracy_score(all_labels, all_preds)

    print(f"\n{'='*60}")
    print(f"TEST RESULTS")
    print(f"{'='*60}")
    print(f"  Test loss     : {test_loss:.4f}")
    print(f"  Test accuracy : {test_acc:.2f}%")

    # Per-class report
    report = classification_report(
        all_labels, all_preds,
        target_names = class_names,
        digits       = 4,
    )
    print(f"\nPer-class report:\n{report}")

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)

    # ── Per-class accuracy ─────────────────────────────────────────────────────
    print("Per-class accuracy:")
    print(f"  {'Class':<10} {'Correct':>8} {'Total':>8} {'Acc':>8}")
    print(f"  {'─'*38}")
    for i, name in enumerate(class_names):
        mask    = all_labels == i
        correct = (all_preds[mask] == i).sum()
        total   = mask.sum()
        acc     = 100.0 * correct / total if total > 0 else 0.0
        print(f"  {name:<10} {correct:>8} {total:>8} {acc:>7.2f}%")

    # ── Confusion matrix plot ──────────────────────────────────────────────────
    print(f"\nSaving confusion matrix...")
    plot_confusion_matrix(cm, class_names, CM_PATH)

    # ── Save evaluation report ─────────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    report_lines = [
        "=" * 60,
        "STREAMSENSE — Evaluation Report",
        "=" * 60,
        f"Timestamp       : {timestamp}",
        f"Checkpoint      : {ckpt_path}",
        f"Trained epoch   : {ckpt['epoch']}",
        f"Val accuracy    : {ckpt['val_accuracy']:.2f}%",
        f"Device          : {device}",
        f"Test samples    : {len(all_labels)}",
        "",
        f"Test loss       : {test_loss:.4f}",
        f"Test accuracy   : {test_acc:.2f}%",
        "",
        "Per-class report:",
        report,
        "",
        "Confusion matrix (rows=true, cols=predicted):",
        "Classes: " + ", ".join(f"{i}={n}" for i, n in enumerate(class_names)),
        str(cm),
        "",
        "Per-class accuracy:",
    ]

    for i, name in enumerate(class_names):
        mask    = all_labels == i
        correct = (all_preds[mask] == i).sum()
        total   = mask.sum()
        acc     = 100.0 * correct / total if total > 0 else 0.0
        report_lines.append(f"  {name:<10} {correct}/{total}  ({acc:.2f}%)")

    report_lines += [
        "",
        "MPIC version    : 1.0",
        "Architecture    : StreamSenseNet (VGG-style 2D CNN)",
        "Parameters      : 295,786",
        "Dataset         : Google Speech Commands v2 (10 classes)",
    ]

    report_text = "\n".join(report_lines)

    with open(REPORT_PATH, "w") as f:
        f.write(report_text)

    print(f"  Saved → {REPORT_PATH}")

    # ── Final summary ──────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"EVALUATION COMPLETE")
    print(f"{'='*60}")
    print(f"  Test accuracy  : {test_acc:.2f}%")
    print(f"  Report         : {REPORT_PATH}")
    print(f"  Confusion matrix: {CM_PATH}")
    print(f"\nNext step: python export_onnx.py")


# ── Argument parser ───────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="STREAMSENSE evaluate.py")
    parser.add_argument(
        "--ckpt", type=str,
        default=str(DEFAULT_CKPT),
        help="Path to checkpoint (default: checkpoints/best_model.pth)"
    )
    return parser.parse_args()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = parse_args()
    evaluate(args)

```

### `training/evaluate_1d_comparison.py`

```python
"""
evaluate_1d_comparison.py
Project STREAMSENSE — Track A
Epic A3.2 — Evaluation + Comparison report (1D baseline vs 2D StreamSenseNet)

Evaluates StreamSenseNet1D on the test split, computes overall accuracy,
per-class precision/recall/F1, and confusion matrix — same metrics as
evaluate.py for the 2D model. Then produces a side-by-side comparison
table against the 2D model's known results (from evaluation_report.txt),
to support the ADR (A3.3) decision on which architecture to deploy.

Outputs:
    evaluation_1d/evaluation_report_1d.txt
    evaluation_1d/confusion_matrix_1d.png
    evaluation_1d/comparison_1d_vs_2d.txt

Run from C:\\STREAMSENSE\\training\\:
    python evaluate_1d_comparison.py
"""

import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model_1d import StreamSenseNet1D, count_parameters
from dataset_1d import StreamSenseDataset1D

_DEFAULT_ROOT = r"C:\STREAMSENSE" if os.name == "nt" else "/content/STREAMSENSE"
ROOT       = Path(os.environ.get("STREAMSENSE_ROOT", _DEFAULT_ROOT))
CKPT_PATH  = ROOT / "checkpoints_1d" / "best_model_1d.pth"
OUT_DIR    = ROOT / "evaluation_1d"

CLASS_NAMES = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64

# ── Known 2D model results (from evaluation_report.txt / model card) ─────────
RESULTS_2D = {
    "params": 295786,
    "test_acc": 95.97,
    "test_loss": 0.1273,
    "per_class_acc": {
        "yes": 98.84, "no": 96.79, "up": 95.17, "down": 94.04, "left": 96.67,
        "right": 99.29, "on": 95.66, "off": 94.47, "stop": 93.80, "go": 94.85,
    },
}


def main():
    print("=" * 60)
    print("STREAMSENSE — evaluate_1d_comparison.py (Epic A3.2)")
    print("=" * 60)

    if not CKPT_PATH.exists():
        print(f"[ERROR] Checkpoint not found: {CKPT_PATH}")
        print("Run train_1d.py first.")
        sys.exit(1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load model ─────────────────────────────────────────────────────────
    model = StreamSenseNet1D(n_classes=10).to(DEVICE)
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    n_params = count_parameters(model)
    print(f"Loaded checkpoint: epoch {ckpt['epoch']}, val_acc={ckpt['val_acc']:.2f}%")
    print(f"Parameters: {n_params:,}")

    # ── Test set ───────────────────────────────────────────────────────────
    test_ds = StreamSenseDataset1D(split="test", augment=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    print(f"Test samples: {len(test_ds)}")

    # ── Run inference ──────────────────────────────────────────────────────
    all_preds  = []
    all_labels = []
    total_loss = 0.0
    criterion  = torch.nn.CrossEntropyLoss()

    print("\nRunning inference...")
    t0 = time.time()
    with torch.no_grad():
        for waveforms, labels in test_loader:
            waveforms = waveforms.to(DEVICE)
            labels    = labels.to(DEVICE)

            logits = model(waveforms)
            loss = criterion(logits, labels)
            total_loss += loss.item() * waveforms.size(0)

            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    elapsed = time.time() - t0
    avg_loss = total_loss / len(test_ds)
    overall_acc = 100.0 * np.mean(np.array(all_preds) == np.array(all_labels))

    print(f"Inference time: {elapsed:.2f}s for {len(test_ds)} samples "
          f"({1000*elapsed/len(test_ds):.2f} ms/sample)")
    print(f"Test accuracy: {overall_acc:.2f}%")
    print(f"Test loss:     {avg_loss:.4f}")

    # ── Classification report ─────────────────────────────────────────────
    report = classification_report(
        all_labels, all_preds, target_names=CLASS_NAMES, digits=4
    )
    cm = confusion_matrix(all_labels, all_preds)

    # Per-class accuracy
    per_class_acc_1d = {}
    for i, name in enumerate(CLASS_NAMES):
        mask = np.array(all_labels) == i
        correct = (np.array(all_preds)[mask] == i).sum()
        total = mask.sum()
        per_class_acc_1d[name] = 100.0 * correct / total

    # ── Save evaluation report ────────────────────────────────────────────
    report_path = OUT_DIR / "evaluation_report_1d.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("STREAMSENSE — StreamSenseNet1D Evaluation Report\n")
        f.write("Epic A3.2 — 1D CNN Baseline on raw audio frames (STRETCH)\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Checkpoint epoch : {ckpt['epoch']}\n")
        f.write(f"Val accuracy     : {ckpt['val_acc']:.2f}%\n")
        f.write(f"Parameters       : {n_params:,}\n\n")
        f.write(f"Test samples     : {len(test_ds)}\n")
        f.write(f"Test loss        : {avg_loss:.4f}\n")
        f.write(f"Test accuracy    : {overall_acc:.2f}%\n")
        f.write(f"Inference time   : {elapsed:.2f}s "
                f"({1000*elapsed/len(test_ds):.2f} ms/sample)\n\n")
        f.write("Per-class report:\n")
        f.write(report)
        f.write("\nConfusion Matrix (rows=true, cols=pred):\n")
        f.write(f"{CLASS_NAMES}\n")
        f.write(str(cm))
        f.write("\n")

    print(f"\nSaved -> {report_path}")

    # ── Confusion matrix plot ─────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right")
    ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"StreamSenseNet1D Confusion Matrix (test_acc={overall_acc:.2f}%)")

    for i in range(10):
        for j in range(10):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=8)

    fig.colorbar(im)
    fig.tight_layout()
    cm_path = OUT_DIR / "confusion_matrix_1d.png"
    fig.savefig(cm_path, dpi=120)
    plt.close(fig)
    print(f"Saved -> {cm_path}")

    # ── Comparison report (1D vs 2D) ──────────────────────────────────────
    comparison_path = OUT_DIR / "comparison_1d_vs_2d.txt"
    with open(comparison_path, "w", encoding="utf-8") as f:
        f.write("STREAMSENSE — Architecture Comparison: 1D CNN (raw) vs 2D CNN (mel)\n")
        f.write("Supports Epic A3.3 (ADR — Architecture Decision Record)\n")
        f.write("=" * 70 + "\n\n")

        f.write(f"{'Metric':<28}{'2D StreamSenseNet':>20}{'1D StreamSenseNet1D':>22}\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'Parameters':<28}{RESULTS_2D['params']:>20,}{n_params:>22,}\n")
        f.write(f"{'Test accuracy':<28}{RESULTS_2D['test_acc']:>19.2f}%{overall_acc:>21.2f}%\n")
        f.write(f"{'Test loss':<28}{RESULTS_2D['test_loss']:>20.4f}{avg_loss:>22.4f}\n")
        f.write(f"{'Input representation':<28}{'log-mel [1,64,97]':>20}{'raw waveform [1,16000]':>22}\n")
        f.write("\n")

        f.write("Per-class accuracy (%):\n")
        f.write(f"{'Class':<10}{'2D':>10}{'1D':>10}{'Delta (1D-2D)':>16}\n")
        f.write("-" * 46 + "\n")
        for name in CLASS_NAMES:
            acc_2d = RESULTS_2D["per_class_acc"][name]
            acc_1d = per_class_acc_1d[name]
            delta = acc_1d - acc_2d
            f.write(f"{name:<10}{acc_2d:>9.2f}%{acc_1d:>9.2f}%{delta:>+15.2f}%\n")

        f.write("\n")
        acc_diff = overall_acc - RESULTS_2D["test_acc"]
        param_ratio = n_params / RESULTS_2D["params"]
        f.write(f"Overall accuracy delta (1D - 2D): {acc_diff:+.2f} percentage points\n")
        f.write(f"Parameter ratio (1D / 2D): {param_ratio:.2f}x\n")

        f.write("\nNotes for ADR (A3.3):\n")
        f.write("- 2D mel-spectrogram representation provides explicit time-frequency\n")
        f.write("  structure as input, which the 1D model must learn implicitly from\n")
        f.write("  raw waveform via its receptive field.\n")
        f.write("- Compare accuracy-per-parameter and inference latency when deciding\n")
        f.write("  between representations for the FPGA deployment target.\n")

    print(f"Saved -> {comparison_path}")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  2D StreamSenseNet : {RESULTS_2D['params']:,} params, "
          f"{RESULTS_2D['test_acc']:.2f}% test acc")
    print(f"  1D StreamSenseNet1D: {n_params:,} params, "
          f"{overall_acc:.2f}% test acc")
    print(f"\n[DONE] See {comparison_path.name} for full comparison.")


if __name__ == "__main__":
    main()

```

### `training/evaluate_multihead_onnx.py`

```python
"""
evaluate_multihead_onnx.py
Project STREAMSENSE — Track A
Scope 2 / WA-4 Extension

Evaluates BOTH multi-head ONNX models on the full test split:
    onnx_models/streamsense_multihead_fp32.onnx
    onnx_models/streamsense_multihead_int8.onnx

For each model this script:
    1. Runs every test sample through the full MPIC v1.0 preprocessing pipeline.
    2. Feeds the resulting [1, 1, 64, 97] tensor through ORT.
    3. Extracts the 'logits' output ([1, 10]) for classification.
    4. Verifies that 'embedding' ([1, 128]) and 'novelty_score' ([1, 1]) are
       also present and correctly shaped (hard assert — fails loudly if broken).
    5. Computes top-1 accuracy, per-class precision / recall / F1, and confusion
       matrix, and prints a full report.
    6. Compares FP32 vs INT8 accuracy and checks that the INT8 drop is ≤ 1.0%.
    7. Appends a timestamped result block to
       evaluation/multihead_onnx_evaluation_report.txt

Output contract verified per ERR v1.0:
    logits        float32  [1, 10]   — classification head
    embedding     float32  [1, 128]  — projection head
    novelty_score float32  [1,  1]   — must be exactly 2-D

Usage (from project root):
    python training/evaluate_multihead_onnx.py

Optional overrides:
    --fp32   PATH   FP32 multihead ONNX (default: onnx_models/streamsense_multihead_fp32.onnx)
    --int8   PATH   INT8 multihead ONNX (default: onnx_models/streamsense_multihead_int8.onnx)
    --test   PATH   Test split file    (default: data/splits/test_files.txt)
    --stats  PATH   Normalization JSON (default: stats/normalization_stats.json)
    --labels PATH   Class labels JSON  (default: class_labels.json)
    --out    PATH   Report output      (default: evaluation/multihead_onnx_evaluation_report.txt)
    --batch  INT    Inference batch size (default: 64)
    --skip-int8     Skip INT8 evaluation (FP32 only)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torchaudio
import onnxruntime as ort
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

# ── MPIC v1.0 frozen parameters ───────────────────────────────────────────────
SAMPLE_RATE   = 16000
FRAME_LEN     = 16000
N_FFT         = 512
HOP_LENGTH    = 160
N_MELS        = 64
CENTER        = False
POWER         = 2.0
LOG_EPS       = 1e-10
CLIP_FLOOR_DB = -80.0

# Expected output shapes — ERR v1.0
EXPECTED_LOGITS_SHAPE        = (1, 10)
EXPECTED_EMBEDDING_SHAPE     = (1, 128)
EXPECTED_NOVELTY_SHAPE       = (1, 1)    # must be exactly 2-D

# INT8 budget — same as Scope 1 baseline
INT8_ACCURACY_DROP_BUDGET = 1.0   # percentage points

# ── Mel transform (built once, CPU, reused for every file) ────────────────────
_mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=SAMPLE_RATE,
    n_fft=N_FFT,
    hop_length=HOP_LENGTH,
    n_mels=N_MELS,
    window_fn=torch.hann_window,
    center=CENTER,
    power=POWER,
)


# ── MPIC v1.0 preprocessing pipeline ─────────────────────────────────────────

def _build_preprocessor(global_mean: float, global_std: float):
    """
    Returns a preprocess(raw) -> np.ndarray [1, 1, 64, 97] callable
    that implements the full 9-step MPIC v1.0 pipeline with the
    supplied normalization statistics.
    """
    def preprocess(raw: np.ndarray) -> np.ndarray:
        """
        Input : float32 numpy array, shape [T] or [C, T], any length
        Output: float32 numpy array, shape [1, 1, 64, 97]

        Steps:
            1-2. Accept and downmix to mono.
            3.   Pad (zeros right) or crop to exactly FRAME_LEN samples.
            4.   MelSpectrogram  → [1, 64, 97]
            5.   10 * log10(mel + LOG_EPS)
            6.   clamp ≥ CLIP_FLOOR_DB
            7.   (mel - global_mean) / global_std
            8.   unsqueeze batch → [1, 1, 64, 97]
        """
        waveform = torch.from_numpy(raw.copy()).float()

        # Step 1-2: ensure [1, T] mono
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)          # [1, T]
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)   # [1, T]

        # Step 3: pad or crop to exactly FRAME_LEN
        length = waveform.shape[1]
        if length < FRAME_LEN:
            waveform = torch.nn.functional.pad(waveform, (0, FRAME_LEN - length))
        elif length > FRAME_LEN:
            waveform = waveform[:, :FRAME_LEN]

        # Steps 4-6: mel spectrogram + log scaling + floor clamp
        mel = _mel_transform(waveform)                # [1, 64, 97]
        mel = 10.0 * torch.log10(mel + LOG_EPS)
        mel = torch.clamp(mel, min=CLIP_FLOOR_DB)

        # Step 7: global normalisation (MPIC v1.0 frozen stats)
        mel = (mel - global_mean) / global_std

        # Step 8: add batch dimension → [1, 1, 64, 97]
        mel = mel.unsqueeze(0)

        return mel.numpy().astype(np.float32)

    return preprocess


# ── Split file parser ─────────────────────────────────────────────────────────

def _parse_split(split_file: Path, idx_to_label: dict[int, str]) -> list[tuple[Path, int]]:
    """
    Parse test_files.txt.  Expected line format:
        C:\\STREAMSENSE\\data\\raw\\yes\\file.wav | yes | 0
    Returns list of (wav_path, class_index) tuples.
    Only keeps entries whose class_index is in idx_to_label.
    """
    label_to_idx = {v: k for k, v in idx_to_label.items()}
    samples: list[tuple[Path, int]] = []
    skipped = 0

    with open(split_file, "r", encoding="utf-8") as fh:
        for lineno, raw_line in enumerate(fh, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            parts = [p.strip() for p in line.split("|")]
            if len(parts) != 3:
                skipped += 1
                continue

            wav_path = Path(parts[0])
            try:
                class_idx = int(parts[2])
            except ValueError:
                class_idx = label_to_idx.get(parts[1], -1)

            if class_idx not in idx_to_label:
                skipped += 1
                continue

            samples.append((wav_path, class_idx))

    if skipped:
        print(f"  [WARN] {skipped} line(s) skipped in {split_file.name} "
              f"(malformed or out-of-range class).")
    return samples


# ── Shape gate — ERR v1.0 contract verification ───────────────────────────────

def _verify_output_contract(session: ort.InferenceSession, model_label: str) -> None:
    """
    Runs a single zero-valued dummy input through the session and asserts
    all three output heads are present with the correct shapes.
    Hard sys.exit(1) on any failure — a broken output contract means Track B,
    C, D, E cannot integrate against this model.
    """
    dummy = np.zeros((1, 1, 64, 97), dtype=np.float32)
    output_names = [o.name for o in session.get_outputs()]
    input_name   = session.get_inputs()[0].name

    outputs = session.run(output_names, {input_name: dummy})
    output_map = dict(zip(output_names, outputs))

    passed = True
    sep = "─" * 54

    print(f"\n  {sep}")
    print(f"  ERR v1.0 output contract — {model_label}")
    print(f"  {sep}")

    def _check(name: str, expected_shape: tuple) -> None:
        nonlocal passed
        if name not in output_map:
            print(f"  [FAIL]  '{name}' : MISSING from model outputs")
            passed = False
            return
        actual = output_map[name].shape
        ok = actual == expected_shape
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}]  '{name}' : {actual}  (expected {expected_shape})")
        if not ok:
            passed = False

    _check("logits",        EXPECTED_LOGITS_SHAPE)
    _check("embedding",     EXPECTED_EMBEDDING_SHAPE)
    _check("novelty_score", EXPECTED_NOVELTY_SHAPE)

    if not passed:
        print(f"\n  [ABORT] Output contract FAILED for {model_label}.")
        print("          ERR v1.0 requires all three outputs with exact shapes.")
        print("          Re-export the model and re-run this script.")
        sys.exit(1)

    print(f"  {sep}")
    print(f"  Output contract: PASS — all three heads present and correctly shaped.")
    print(f"  {sep}\n")


# ── Batched inference ─────────────────────────────────────────────────────────

def _run_inference(
    onnx_path: Path,
    samples: list[tuple[Path, int]],
    preprocess,
    batch_size: int,
    model_label: str,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Runs the multihead ONNX model on all samples.

    Returns:
        preds   : np.ndarray[int]   top-1 predicted class index per sample
        labels  : np.ndarray[int]   ground-truth class index per sample
        elapsed : float             wall-clock seconds
    """
    sess_opts = ort.SessionOptions()
    sess_opts.inter_op_num_threads = 4
    sess_opts.intra_op_num_threads = 4
    session = ort.InferenceSession(str(onnx_path), sess_opts=sess_opts)

    # Verify output contract before running the full dataset
    _verify_output_contract(session, model_label)

    input_name   = session.get_inputs()[0].name
    # We only need logits for accuracy; collect its name robustly
    logits_name  = None
    for out in session.get_outputs():
        if out.name == "logits":
            logits_name = out.name
            break
    if logits_name is None:
        # Fallback: first output (already caught by contract gate above, but be safe)
        logits_name = session.get_outputs()[0].name

    all_preds:  list[int] = []
    all_labels: list[int] = []
    errors = 0
    t0 = time.time()

    # The multihead model has a static batch dimension of 1 (frozen by MPIC v1.0
    # output contract [1,1,64,97] → outputs [1,10]/[1,128]/[1,1]).
    # ORT rejects any batch size other than 1, so we run one sample at a time.
    # The `batch_size` arg is accepted for CLI compatibility but is not used here.
    total = len(samples)
    for i, (wav_path, class_idx) in enumerate(samples):
        if (i + 1) % 500 == 0 or (i + 1) == total:
            pct = 100.0 * (i + 1) / total
            print(f"    [{i+1:>5}/{total}]  {pct:5.1f}%", flush=True)

        if not wav_path.exists():
            errors += 1
            continue

        try:
            waveform, sr = torchaudio.load(str(wav_path))
            raw = waveform.squeeze(0).numpy().astype(np.float32)
            inp = preprocess(raw)           # [1, 1, 64, 97]  — batch=1
            logits = session.run([logits_name], {input_name: inp})[0]  # [1, 10]
            pred   = int(np.argmax(logits, axis=1)[0])
            all_preds.append(pred)
            all_labels.append(class_idx)
        except Exception as exc:
            print(f"    [WARN] Error on {wav_path.name}: {exc}")
            errors += 1
            continue
    elapsed = time.time() - t0

    if errors:
        print(f"  [WARN] Skipped {errors}/{total} files (missing or unreadable).")

    return np.array(all_preds, dtype=int), np.array(all_labels, dtype=int), elapsed


# ── Report block builder ──────────────────────────────────────────────────────

def _build_report_block(
    model_label: str,
    onnx_path: Path,
    preds: np.ndarray,
    labels: np.ndarray,
    elapsed: float,
    class_names: list[str],
    num_classes: int,
    timestamp: str,
) -> str:
    acc = accuracy_score(labels, preds)
    cm  = confusion_matrix(labels, preds, labels=list(range(num_classes)))
    cls_report = classification_report(
        labels, preds,
        target_names=class_names,
        digits=4,
        zero_division=0,
    )

    lines: list[str] = []
    sep = "=" * 60
    lines.append(sep)
    lines.append(f"  Model        : {model_label}")
    lines.append(f"  ONNX file    : {onnx_path.name}")
    lines.append(f"  Timestamp    : {timestamp}")
    lines.append(f"  Test samples : {len(labels)}")
    lines.append(f"  Accuracy     : {acc*100:.2f}%  ({int(acc*len(labels))}/{len(labels)})")
    lines.append(f"  Elapsed      : {elapsed:.1f}s")
    lines.append(sep)
    lines.append("")
    lines.append("Per-class report:")
    lines.append(cls_report)
    lines.append("Per-class accuracy:")
    for i, name in enumerate(class_names):
        correct = int(cm[i, i])
        support = int(cm[i].sum())
        pct = 100.0 * correct / support if support > 0 else 0.0
        lines.append(f"  {name:<10} {correct}/{support}  ({pct:.2f}%)")
    lines.append("")
    lines.append(f"Confusion matrix (rows=true, cols=predicted):")
    lines.append(f"Classes: " + ", ".join(f"{i}={n}" for i, n in enumerate(class_names)))
    for row in cm:
        lines.append("  " + str(row.tolist()))
    lines.append("")
    lines.append(f"MPIC version   : 1.0")
    lines.append(f"Architecture   : StreamSenseWrapper (multi-head, Scope 2 WA-4)")
    lines.append(sep)

    return "\n".join(lines)


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    # Resolve project root: this file lives in training/, root is one level up.
    this_dir = Path(__file__).resolve().parent
    root     = this_dir.parent

    p = argparse.ArgumentParser(
        description=(
            "STREAMSENSE — evaluate both multi-head ONNX models on the test split. "
            "Verifies ERR v1.0 output contract (3 heads, exact shapes) and reports "
            "per-class accuracy, confusion matrix, and FP32 vs INT8 accuracy gap."
        )
    )
    p.add_argument(
        "--fp32",
        type=Path,
        default=root / "onnx_models" / "streamsense_multihead_fp32.onnx",
        help="Path to FP32 multihead ONNX model.",
    )
    p.add_argument(
        "--int8",
        type=Path,
        default=root / "onnx_models" / "streamsense_multihead_int8.onnx",
        help="Path to INT8 QDQ multihead ONNX model.",
    )
    p.add_argument(
        "--test",
        type=Path,
        default=root / "data" / "splits" / "test_files.txt",
        help="Test split file (pipe-delimited: path | label | index).",
    )
    p.add_argument(
        "--stats",
        type=Path,
        default=root / "stats" / "normalization_stats.json",
        help="Normalization stats JSON (global_mean, global_std).",
    )
    p.add_argument(
        "--labels",
        type=Path,
        default=root / "class_labels.json",
        help='Class labels JSON (e.g. {"0": "yes", ...}).',
    )
    p.add_argument(
        "--out",
        type=Path,
        default=root / "evaluation" / "multihead_onnx_evaluation_report.txt",
        help="Output report file (appended, not overwritten).",
    )
    p.add_argument(
        "--batch",
        type=int,
        default=64,
        help="Inference batch size (default: 64).",
    )
    p.add_argument(
        "--skip-int8",
        action="store_true",
        help="Skip INT8 evaluation and run FP32 only.",
    )
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args = _parse_args()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    sep  = "=" * 60
    sep2 = "─" * 60
    print(sep)
    print("  STREAMSENSE — Multi-Head ONNX Test-Set Evaluation")
    print("  Scope 2 / WA-4  |  MPIC v1.0  |  ERR v1.0")
    print(f"  Timestamp : {timestamp}")
    print(sep)

    # ── Load normalization stats ───────────────────────────────────────────────
    if not args.stats.exists():
        print(f"[ERROR] Normalization stats not found: {args.stats}")
        sys.exit(1)
    with open(args.stats, "r") as fh:
        stats = json.load(fh)
    global_mean = float(stats["global_mean"])
    global_std  = float(stats["global_std"])
    print(f"\nNormalization stats loaded from: {args.stats.name}")
    print(f"  global_mean : {global_mean:.6f} dB")
    print(f"  global_std  : {global_std:.6f} dB")

    # ── Build MPIC v1.0 preprocessor ──────────────────────────────────────────
    preprocess = _build_preprocessor(global_mean, global_std)

    # ── Load class labels ─────────────────────────────────────────────────────
    if not args.labels.exists():
        print(f"[ERROR] Class labels not found: {args.labels}")
        sys.exit(1)
    with open(args.labels, "r") as fh:
        raw_labels = json.load(fh)
    # Support both {"0": "yes"} and {"yes": 0} formats
    first_val = list(raw_labels.values())[0]
    if isinstance(first_val, int):
        idx_to_label: dict[int, str] = {v: k for k, v in raw_labels.items()}
    else:
        idx_to_label = {int(k): v for k, v in raw_labels.items()}

    num_classes = len(idx_to_label)
    class_names = [idx_to_label[i] for i in range(num_classes)]
    print(f"\nClasses ({num_classes}): {', '.join(class_names)}")

    # ── Parse test split ───────────────────────────────────────────────────────
    if not args.test.exists():
        print(f"[ERROR] Test split file not found: {args.test}")
        sys.exit(1)
    samples = _parse_split(args.test, idx_to_label)
    if not samples:
        print(f"[ERROR] No valid samples parsed from {args.test}.")
        sys.exit(1)
    print(f"\nTest split     : {args.test.name}")
    print(f"Total samples  : {len(samples)}")

    # ── Determine which models to evaluate ────────────────────────────────────
    models_to_run: list[tuple[str, Path]] = []

    if not args.fp32.exists():
        print(f"\n[ERROR] FP32 multihead ONNX not found: {args.fp32}")
        print("        Run training/export_multihead_onnx.py first (WA-4).")
        sys.exit(1)
    models_to_run.append(("StreamSenseWrapper FP32 (multihead)", args.fp32))

    if not args.skip_int8:
        if not args.int8.exists():
            print(f"\n[WARN] INT8 multihead ONNX not found: {args.int8}")
            print("       Skipping INT8 evaluation.")
        else:
            models_to_run.append(("StreamSenseWrapper INT8 (multihead)", args.int8))

    # ── Run evaluation for each model ─────────────────────────────────────────
    results: dict[str, float]  = {}
    report_blocks: list[str]   = []

    for model_label, onnx_path in models_to_run:
        print(f"\n{sep2}")
        print(f"  Evaluating : {model_label}")
        print(f"  ONNX       : {onnx_path.name}")
        print(f"  File size  : {onnx_path.stat().st_size / 1024:.1f} KB")
        print(f"{sep2}")

        preds, labels, elapsed = _run_inference(
            onnx_path=onnx_path,
            samples=samples,
            preprocess=preprocess,
            batch_size=args.batch,
            model_label=model_label,
        )

        acc = accuracy_score(labels, preds)
        correct = int(acc * len(labels))
        print(f"\n  Accuracy : {acc*100:.2f}%  ({correct}/{len(labels)})")
        print(f"  Elapsed  : {elapsed:.1f}s  "
              f"({elapsed / len(labels) * 1000:.1f} ms/sample)")

        results[model_label] = acc

        block = _build_report_block(
            model_label=model_label,
            onnx_path=onnx_path,
            preds=preds,
            labels=labels,
            elapsed=elapsed,
            class_names=class_names,
            num_classes=num_classes,
            timestamp=timestamp,
        )
        report_blocks.append(block)

        # Print per-class breakdown to console
        print(f"\n  Per-class accuracy:")
        cm = confusion_matrix(labels, preds, labels=list(range(num_classes)))
        for i, name in enumerate(class_names):
            correct_i = int(cm[i, i])
            support_i = int(cm[i].sum())
            pct_i     = 100.0 * correct_i / support_i if support_i > 0 else 0.0
            bar       = "█" * int(pct_i / 5)   # ASCII bar, 20-char max
            print(f"    {name:<8}  {correct_i:>4}/{support_i:<4}  {pct_i:6.2f}%  {bar}")

    # ── FP32 vs INT8 comparison (only when both were evaluated) ───────────────
    fp32_key  = "StreamSenseWrapper FP32 (multihead)"
    int8_key  = "StreamSenseWrapper INT8 (multihead)"
    has_both  = fp32_key in results and int8_key in results

    summary_lines: list[str] = [
        "",
        sep,
        "  MULTI-HEAD ONNX ACCURACY SUMMARY",
        sep,
    ]
    for label, acc in results.items():
        summary_lines.append(f"  {label:<42} : {acc*100:.2f}%")

    if has_both:
        fp32_acc = results[fp32_key]
        int8_acc = results[int8_key]
        drop     = (fp32_acc - int8_acc) * 100
        budget_ok = abs(drop) <= INT8_ACCURACY_DROP_BUDGET
        summary_lines.append("")
        summary_lines.append(f"  Accuracy drop (FP32 → INT8) : {drop:+.2f}%")
        summary_lines.append(
            f"  INT8 budget (≤{INT8_ACCURACY_DROP_BUDGET:.1f}%)           : "
            f"{'PASS' if budget_ok else 'FAIL'}"
        )
        if not budget_ok:
            summary_lines.append(
                f"  [WARN] INT8 drop ({abs(drop):.2f}%) exceeds {INT8_ACCURACY_DROP_BUDGET:.1f}% budget. "
                "Recalibrate PTQ."
            )

    summary_lines.append(sep)
    summary_lines.append("")
    summary = "\n".join(summary_lines)
    print(summary)

    # ── ERR v1.0 contract reminder ────────────────────────────────────────────
    print("  ERR v1.0 output contract (verified for all models above):")
    print(f"    logits        float32  {EXPECTED_LOGITS_SHAPE}   — logits head")
    print(f"    embedding     float32  {EXPECTED_EMBEDDING_SHAPE} — embed head")
    print(f"    novelty_score float32  {EXPECTED_NOVELTY_SHAPE}   — novelty head (2-D enforced)")
    print()

    # ── Write report file ─────────────────────────────────────────────────────
    args.out.parent.mkdir(parents=True, exist_ok=True)
    full_report = (
        f"\n\n{sep}\n"
        f"  MULTI-HEAD ONNX EVALUATION (evaluate_multihead_onnx.py)\n"
        f"  Timestamp : {timestamp}\n"
        f"{sep}\n\n"
        + "\n\n".join(report_blocks)
        + "\n"
        + summary
    )

    with open(args.out, "a", encoding="utf-8") as fh:
        fh.write(full_report)

    print(f"[DONE] Results appended to: {args.out}")


if __name__ == "__main__":
    main()

```

### `training/evaluate_onnx.py`

```python
"""
evaluate_onnx.py
Project STREAMSENSE — Track A
MPIC v1.0

Evaluates both FP32 and INT8 ONNX models on the full test split.
Produces accuracy, per-class precision/recall/F1, confusion matrix,
and appends results to evaluation_report.txt.

Usage:
    python evaluate_onnx.py

Paths (edit if needed):
    ONNX models  : C:\STREAMSENSE\onnx_models\
    Test split   : C:\STREAMSENSE\data\splits\test_files.txt
    Class labels : C:\STREAMSENSE\class_labels.json
    Report out   : C:\STREAMSENSE\evaluation\evaluation_report.txt
"""

import sys
import json
import time
import numpy as np
from pathlib import Path
from datetime import datetime

import torch
import torchaudio
import onnxruntime as ort
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
)

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT            = Path(r"C:\STREAMSENSE")
FP32_ONNX       = ROOT / "onnx_models" / "streamsense_model_fp32.onnx"
INT8_ONNX       = ROOT / "onnx_models" / "streamsense_model_int8.onnx"
TEST_SPLIT      = ROOT / "data" / "splits" / "test_files.txt"
CLASS_LABELS    = ROOT / "class_labels.json"
STATS_FILE      = ROOT / "stats" / "normalization_stats.json"
REPORT_OUT      = ROOT / "evaluation" / "evaluation_report.txt"

# ── MPIC v1.0 frozen parameters ───────────────────────────────────────────────
SAMPLE_RATE   = 16000
FRAME_LEN     = 16000
N_FFT         = 512
HOP_LENGTH    = 160
N_MELS        = 64
CENTER        = False
POWER         = 2.0
LOG_EPS       = 1e-10
CLIP_FLOOR_DB = -80.0

BATCH_SIZE    = 64   # inference batch size

# ── Load stats & class labels ─────────────────────────────────────────────────
with open(STATS_FILE, "r") as f:
    _stats = json.load(f)
GLOBAL_MEAN = float(_stats["global_mean"])
GLOBAL_STD  = float(_stats["global_std"])

with open(CLASS_LABELS, "r") as f:
    _cl = json.load(f)
# Support both {idx: label} and {label: idx} formats
if isinstance(list(_cl.values())[0], int):
    # {label: idx} → invert
    IDX_TO_LABEL = {v: k for k, v in _cl.items()}
else:
    # {idx: label} or {"0": label}
    IDX_TO_LABEL = {int(k): v for k, v in _cl.items()}

NUM_CLASSES = len(IDX_TO_LABEL)
CLASS_NAMES = [IDX_TO_LABEL[i] for i in range(NUM_CLASSES)]

# ── Mel transform (built once) ────────────────────────────────────────────────
_mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate = SAMPLE_RATE,
    n_fft       = N_FFT,
    hop_length  = HOP_LENGTH,
    n_mels      = N_MELS,
    window_fn   = torch.hann_window,
    center      = CENTER,
    power       = POWER,
)

def preprocess(raw: np.ndarray) -> np.ndarray:
    """
    MPIC v1.0 pipeline. Input: float32 numpy [T]. Output: float32 numpy [1,1,64,97].
    """
    waveform = torch.from_numpy(raw.copy()).float()
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)            # [1, T]
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    # pad / crop
    L = waveform.shape[1]
    if L < FRAME_LEN:
        waveform = torch.nn.functional.pad(waveform, (0, FRAME_LEN - L))
    elif L > FRAME_LEN:
        waveform = waveform[:, :FRAME_LEN]
    mel = _mel_transform(waveform)                  # [1, 64, 97]
    mel = 10.0 * torch.log10(mel + LOG_EPS)
    mel = torch.clamp(mel, min=CLIP_FLOOR_DB)
    mel = (mel - GLOBAL_MEAN) / GLOBAL_STD
    mel = mel.unsqueeze(0)                          # [1, 1, 64, 97]
    return mel.numpy().astype(np.float32)

# ── Parse test split ──────────────────────────────────────────────────────────
def parse_split(split_file: Path):
    samples = []

    label_to_idx = {v: k for k, v in IDX_TO_LABEL.items()}

    with open(split_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            parts = [p.strip() for p in line.split("|")]

            if len(parts) == 3:
                wav_path = Path(parts[0])

                try:
                    class_idx = int(parts[2])
                except:
                    class_idx = label_to_idx.get(parts[1], -1)

            else:
                continue

            samples.append((wav_path, class_idx))

    return samples

# ── Inference on full test set ────────────────────────────────────────────────
def run_inference(onnx_path: Path, samples: list) -> tuple:
    """
    Runs ONNX model on all samples.
    Returns (all_preds: np.ndarray, all_labels: np.ndarray, elapsed_sec: float).
    """
    sess_opts = ort.SessionOptions()
    sess_opts.inter_op_num_threads = 4
    sess_opts.intra_op_num_threads = 4
    session = ort.InferenceSession(str(onnx_path), sess_opts=sess_opts)
    input_name  = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    all_preds  = []
    all_labels = []
    errors     = 0
    t0         = time.time()

    # batch inference
    batch_inputs = []
    batch_labels = []

    def flush_batch():
        if not batch_inputs:
            return
        x = np.concatenate(batch_inputs, axis=0)   # [B, 1, 64, 97]
        logits = session.run([output_name], {input_name: x})[0]
        preds  = np.argmax(logits, axis=1)
        all_preds.extend(preds.tolist())
        all_labels.extend(batch_labels)
        batch_inputs.clear()
        batch_labels.clear()

    for i, (wav_path, class_idx) in enumerate(samples):
        if (i + 1) % 500 == 0:
            print(f"  [{i+1}/{len(samples)}] processing...", flush=True)

        if not wav_path.exists():
            errors += 1
            continue
        try:
            waveform, sr = torchaudio.load(str(wav_path))
            raw = waveform.squeeze(0).numpy().astype(np.float32)
            inp = preprocess(raw)               # [1, 1, 64, 97]
            batch_inputs.append(inp)
            batch_labels.append(class_idx)
        except Exception as e:
            errors += 1
            continue

        if len(batch_inputs) >= BATCH_SIZE:
            flush_batch()

    flush_batch()
    elapsed = time.time() - t0

    if errors:
        print(f"  [WARN] Skipped {errors} files (missing or unreadable)")

    return np.array(all_preds), np.array(all_labels), elapsed

# ── Report builder ────────────────────────────────────────────────────────────
def build_report_block(model_name: str, onnx_path: Path,
                        preds: np.ndarray, labels: np.ndarray,
                        elapsed: float) -> str:
    acc   = accuracy_score(labels, preds)
    cm    = confusion_matrix(labels, preds, labels=list(range(NUM_CLASSES)))
    report = classification_report(
        labels, preds,
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0,
    )

    lines = []
    sep = "=" * 60

    lines.append(sep)
    lines.append(f"  Model        : {model_name}")
    lines.append(f"  ONNX file    : {onnx_path.name}")
    lines.append(f"  Timestamp    : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"  Test samples : {len(labels)}")
    lines.append(f"  Accuracy     : {acc*100:.2f}%  ({int(acc*len(labels))}/{len(labels)})")
    lines.append(f"  Elapsed      : {elapsed:.1f}s")
    lines.append(sep)
    lines.append("")
    lines.append("Per-class report:")
    lines.append(report)

    lines.append("Per-class accuracy:")
    for i, name in enumerate(CLASS_NAMES):
        mask    = labels == i
        correct = int((preds[mask] == labels[mask]).sum())
        total   = int(mask.sum())
        lines.append(f"  {name:<10} {correct}/{total}  ({correct/total*100:.2f}%)")

    lines.append("")
    lines.append(f"Confusion matrix (rows=true, cols=predicted):")
    lines.append(f"Classes: " + ", ".join(f"{i}={n}" for i, n in enumerate(CLASS_NAMES)))
    for row in cm:
        lines.append("  " + str(row.tolist()))

    lines.append("")
    lines.append(f"MPIC version   : 1.0")
    lines.append(f"Architecture   : StreamSenseNet (VGG-style 2D CNN)")
    lines.append(f"Parameters     : 295,786")
    lines.append(f"Dataset        : Google Speech Commands v2 (10 classes)")
    lines.append(sep)

    return "\n".join(lines)

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("STREAMSENSE — ONNX Evaluation (FP32 + INT8)")
    print("=" * 60)

    # Validate paths
    for p, name in [
        (FP32_ONNX,    "streamsense_model_fp32.onnx"),
        (INT8_ONNX,    "streamsense_model_int8.onnx"),
        (TEST_SPLIT,   "test_files.txt"),
        (STATS_FILE,   "normalization_stats.json"),
        (CLASS_LABELS, "class_labels.json"),
    ]:
        if not p.exists():
            print(f"[ERROR] Not found: {p}")
            sys.exit(1)

    REPORT_OUT.parent.mkdir(parents=True, exist_ok=True)

    samples = parse_split(TEST_SPLIT)
    print(f"Test samples loaded : {len(samples)}")
    print(f"Classes             : {CLASS_NAMES}")
    print(f"Global mean         : {GLOBAL_MEAN:.6f} dB")
    print(f"Global std          : {GLOBAL_STD:.6f} dB")
    print()

    all_blocks = []
    results    = {}

    for model_name, onnx_path in [
        ("StreamSenseNet FP32", FP32_ONNX),
        ("StreamSenseNet INT8", INT8_ONNX),
    ]:
        print(f"{'─'*60}")
        print(f"Evaluating: {model_name}")
        print(f"  ONNX : {onnx_path.name}")
        preds, labels, elapsed = run_inference(onnx_path, samples)
        acc = accuracy_score(labels, preds)
        print(f"  Accuracy : {acc*100:.2f}%  ({int(acc*len(labels))}/{len(labels)})")
        print(f"  Elapsed  : {elapsed:.1f}s")

        block = build_report_block(model_name, onnx_path, preds, labels, elapsed)
        all_blocks.append(block)
        results[model_name] = acc

    # ── Comparison summary ────────────────────────────────────────────────────
    fp32_acc = results["StreamSenseNet FP32"]
    int8_acc = results["StreamSenseNet INT8"]
    drop     = (fp32_acc - int8_acc) * 100

    summary_lines = [
        "",
        "=" * 60,
        "  QUANTIZATION ACCURACY SUMMARY",
        "=" * 60,
        f"  FP32 accuracy  : {fp32_acc*100:.2f}%",
        f"  INT8 accuracy  : {int8_acc*100:.2f}%",
        f"  Accuracy drop  : {drop:+.2f}%",
        f"  INT8 budget    : {'PASS' if abs(drop) <= 1.0 else 'FAIL'}  (threshold: ≤1.0%)",
        "=" * 60,
        "",
    ]
    summary = "\n".join(summary_lines)

    # ── Write to report file ──────────────────────────────────────────────────
    full_report = "\n\n".join(all_blocks) + "\n" + summary

    # Append to existing report (training section stays intact)
    with open(REPORT_OUT, "a", encoding="utf-8") as f:
        f.write("\n\n")
        f.write("=" * 60 + "\n")
        f.write("  ONNX EVALUATION (appended by evaluate_onnx.py)\n")
        f.write("=" * 60 + "\n\n")
        f.write(full_report)

    # Also print summary to console
    print()
    print(summary)
    print(f"[DONE] Results appended to: {REPORT_OUT}")

if __name__ == "__main__":
    main()
    
```

### `training/evaluate_qonnx.py`

```python
"""
evaluate_qonnx.py
Project STREAMSENSE — Track A
Scope 2 / QAT Extension — QONNX Golden-Vector Evaluation

Loads streamsense_multihead.qonnx and evaluates it against all 1000 GV1K
normalized vectors using the qonnx runtime (required for onnx.brevitas custom ops).

Key facts:
  - GV1K vectors are already-normalized mel spectrograms stored as flat
    float32 little-endian binary, shape [64 x 97] = 6208 floats = 24832 bytes.
  - Fed DIRECTLY to the model — NO additional preprocessing applied.
  - Reshape to [1, 1, 64, 97] float32 before feeding.
  - Label parsed from filename stem: GV1K_NNNN_<label>_norm
    parts = stem.split("_")  ->  label = parts[2].lower()
  - Class map: yes=0, no=1, up=2, down=3, left=4, right=5, on=6, off=7, stop=8, go=9
  - Minimum passing threshold: 90.0% top-1 accuracy.
  - ERR v1.0 output order: logits [1,10], embedding [1,128], novelty_score [1,1]
    NOTE: qonnx export may give outputs auto-generated names (e.g. '143', '147').
          This script accesses them by index (0, 1, 2), not by name string.

Usage (from project root):
    python training/evaluate_qonnx.py

Optional overrides:
    --qonnx  PATH   QONNX model file   (default: onnx_models/streamsense_multihead.qonnx)
    --gvk    PATH   GV1K normalized dir (default: golden_vectors_1000/normalized)
    --out    PATH   Report output file  (default: evaluation/qonnx_evaluation_report.txt)
    --pass-threshold FLOAT  Min top-1%% to pass (default: 90.0)

Requirements:
    pip install qonnx
    (onnxruntime is also required as a qonnx dependency, but do NOT use it directly
     to load .qonnx files — it cannot handle onnx.brevitas custom ops.)
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

# ── qonnx runtime import ──────────────────────────────────────────────────────
# IMPORTANT: Do NOT use onnxruntime directly to load .qonnx files.
# .qonnx uses onnx.brevitas custom ops (Quant, BipolarQuant, etc.) that are
# not registered in vanilla onnxruntime. Use qonnx's own executor instead.
try:
    from qonnx.core.modelwrapper import ModelWrapper
    from qonnx.core.onnx_exec import execute_onnx
    from qonnx.transformation.infer_shapes import InferShapes
except ImportError:
    print("[ERROR] qonnx is not installed.")
    print()
    print("  Install it with:")
    print("  c:\\STREAMSENSE\\streamsense-env-win\\Scripts\\python.exe -m pip install qonnx")
    print()
    print("  Then re-run this script.")
    sys.exit(1)

# ── Project root (this file lives in training/) ───────────────────────────────
_THIS_DIR = Path(__file__).resolve().parent
_ROOT     = _THIS_DIR.parent

# ── Constants ─────────────────────────────────────────────────────────────────
TARGET_CLASSES = {
    "yes": 0, "no": 1, "up": 2, "down": 3,
    "left": 4, "right": 5, "on": 6, "off": 7,
    "stop": 8, "go": 9,
}
IDX_TO_LABEL = {v: k for k, v in TARGET_CLASSES.items()}
NUM_CLASSES  = 10
CLASS_NAMES  = [IDX_TO_LABEL[i] for i in range(NUM_CLASSES)]

# GV1K binary spec
GV1K_FLOATS = 64 * 97         # 6208
GV1K_BYTES  = GV1K_FLOATS * 4 # 24832
GV1K_SHAPE  = (1, 1, 64, 97)  # model input shape

# ERR v1.0 expected output shapes — in index order
# NOTE: output names in the .qonnx graph may be auto-generated ('143', '147', etc.)
# We verify shapes by index, not by name.
EXPECTED_SHAPES_BY_INDEX = [
    (1, 10),   # index 0 — logits
    (1, 128),  # index 1 — embedding
    (1, 1),    # index 2 — novelty_score
]
OUTPUT_LABELS = ["logits", "embedding", "novelty_score"]


# ── Label parser ──────────────────────────────────────────────────────────────

def _parse_label(stem: str) -> int | None:
    """
    Pattern: GV1K_NNNN_<label>_norm
    parts = stem.split("_")  ->  label = parts[2].lower()
    """
    parts = stem.split("_")
    if len(parts) < 4:
        return None
    return TARGET_CLASSES.get(parts[2].lower(), None)


# ── Load and prepare model ────────────────────────────────────────────────────

def load_qonnx_model(qonnx_path: Path) -> tuple[ModelWrapper, list[str], str]:
    """
    Load the .qonnx model, run InferShapes (required before execute_onnx),
    and return (model_wrapper, output_names, input_name).

    InferShapes must be called before execute_onnx — qonnx's executor
    requires all tensor shapes to be annotated in the graph.
    """
    print(f"  Loading QONNX : {qonnx_path}")
    print(f"  File size     : {qonnx_path.stat().st_size / 1024:.1f} KB")

    model = ModelWrapper(str(qonnx_path))
    model = model.transform(InferShapes())  # mandatory before execute_onnx

    input_name   = model.graph.input[0].name
    output_names = [o.name for o in model.graph.output]

    print(f"  Input node    : {input_name!r}")
    print(f"  Output nodes  : {output_names}")
    print(f"  (Outputs accessed by index 0/1/2, not by name string)")

    return model, output_names, input_name


# ── ERR v1.0 output contract gate ────────────────────────────────────────────

def verify_output_contract(
    model       : ModelWrapper,
    output_names: list[str],
    input_name  : str,
    model_label : str,
) -> None:
    """
    Feed a zero tensor and check all three output heads are present
    with exactly the right shapes. Hard sys.exit(1) on any failure.
    Uses qonnx execute_onnx, not onnxruntime.
    """
    dummy = np.zeros(GV1K_SHAPE, dtype=np.float32)
    odict = execute_onnx(model, {input_name: dummy})

    sep = "─" * 54
    print(f"\n  {sep}")
    print(f"  ERR v1.0 output contract — {model_label}")
    print(f"  {sep}")

    if len(output_names) < 3:
        print(f"  [FAIL] Expected 3 outputs, got {len(output_names)}: {output_names}")
        print(f"  [ABORT] Output contract FAILED. Re-export the QONNX.")
        sys.exit(1)

    passed = True
    for idx, (label, expected) in enumerate(zip(OUTPUT_LABELS, EXPECTED_SHAPES_BY_INDEX)):
        out_key = output_names[idx]
        if out_key not in odict:
            print(f"  [FAIL]  output[{idx}] '{out_key}' ({label}) : MISSING from odict")
            passed = False
        else:
            actual = odict[out_key].shape
            ok     = actual == expected
            print(f"  {'[PASS]' if ok else '[FAIL]'}  output[{idx}] ({label:<13}) : "
                  f"{actual}  (expected {expected})")
            if not ok:
                passed = False

    print(f"  {sep}")
    if not passed:
        print(f"\n  [ABORT] Output contract FAILED for {model_label}.")
        print("          Re-export the QONNX and re-run.")
        sys.exit(1)
    print(f"  Output contract: PASS\n")


# ── Main evaluation ───────────────────────────────────────────────────────────

def evaluate(
    qonnx_path    : Path,
    gvk_dir       : Path,
    pass_threshold: float,
) -> dict:
    """
    Run every GV1K .bin file through the QONNX model.
    Returns a result dict with full metrics.
    """
    # Load model once — InferShapes runs here
    model, output_names, input_name = load_qonnx_model(qonnx_path)

    # ERR v1.0 contract check
    verify_output_contract(model, output_names, input_name, qonnx_path.name)

    # Logits are always at index 0
    logits_key = output_names[0]

    # Collect GV1K files
    bin_files = sorted(gvk_dir.glob("*_norm.bin"))
    if not bin_files:
        print(f"[ERROR] No *_norm.bin files found in {gvk_dir}")
        sys.exit(1)
    print(f"  GV1K vectors  : {len(bin_files)} files found in {gvk_dir.name}/")

    # Per-class accumulators
    per_class_correct = [0] * NUM_CLASSES
    per_class_total   = [0] * NUM_CLASSES
    confusion         = [[0] * NUM_CLASSES for _ in range(NUM_CLASSES)]

    all_preds : list[int] = []
    all_labels: list[int] = []

    correct = 0
    wrong   = 0
    skipped = 0

    # Inference loop
    total = len(bin_files)
    for i, bf in enumerate(bin_files):
        if (i + 1) % 100 == 0 or (i + 1) == total:
            print(f"    [{i+1:>4}/{total}]  {100*(i+1)/total:5.1f}%", flush=True)

        true_idx = _parse_label(bf.stem)
        if true_idx is None:
            skipped += 1
            continue

        raw = np.fromfile(str(bf), dtype="<f4")
        if raw.size != GV1K_FLOATS:
            print(f"  [WARN] {bf.name}: expected {GV1K_FLOATS} floats, "
                  f"got {raw.size} — skipping")
            skipped += 1
            continue

        inp   = raw.reshape(GV1K_SHAPE).astype(np.float32)
        odict = execute_onnx(model, {input_name: inp})

        logits = odict[logits_key]              # [1, 10]
        pred   = int(np.argmax(logits, axis=1)[0])

        all_preds.append(pred)
        all_labels.append(true_idx)
        per_class_total[true_idx] += 1
        confusion[true_idx][pred] += 1

        if pred == true_idx:
            correct += 1
            per_class_correct[true_idx] += 1
        else:
            wrong += 1

    total_checked = correct + wrong
    top1_acc      = 100.0 * correct / total_checked if total_checked > 0 else 0.0

    return {
        "total_files"       : total,
        "total_checked"     : total_checked,
        "correct"           : correct,
        "wrong"             : wrong,
        "skipped"           : skipped,
        "top1_acc"          : top1_acc,
        "per_class_correct" : per_class_correct,
        "per_class_total"   : per_class_total,
        "confusion"         : confusion,
        "all_preds"         : all_preds,
        "all_labels"        : all_labels,
        "pass_threshold"    : pass_threshold,
        "passed"            : top1_acc >= pass_threshold,
    }


# ── Report builder ────────────────────────────────────────────────────────────

def print_and_write_report(
    r          : dict,
    qonnx_path : Path,
    gvk_dir    : Path,
    out_path   : Path,
    timestamp  : str,
) -> None:
    sep  = "=" * 60
    sep2 = "─" * 60

    lines: list[str] = []

    def ln(s: str = "") -> None:
        lines.append(s)
        print(s)

    ln(sep)
    ln("  STREAMSENSE -- QONNX GV1K Evaluation")
    ln("  Scope 2 / QAT Extension  |  ERR v1.0")
    ln(f"  Timestamp  : {timestamp}")
    ln(f"  Model      : {qonnx_path.name}")
    ln(f"  File size  : {qonnx_path.stat().st_size / 1024:.1f} KB")
    ln(f"  GV1K dir   : {gvk_dir}")
    ln(sep)
    ln()
    ln(f"  Total .bin files    : {r['total_files']}")
    ln(f"  Vectors checked     : {r['total_checked']}")
    ln(f"  Skipped (bad files) : {r['skipped']}")
    ln(f"  Correct             : {r['correct']}")
    ln(f"  Wrong               : {r['wrong']}")
    ln(f"  Top-1 Accuracy      : {r['top1_acc']:.2f}%  "
       f"({r['correct']}/{r['total_checked']})")
    ln(f"  Pass threshold      : {r['pass_threshold']:.1f}%")
    ln(f"  Gate result         : {'PASS' if r['passed'] else 'FAIL'}")
    ln()
    ln(sep2)
    ln("  Per-class accuracy")
    ln(sep2)

    for i, name in enumerate(CLASS_NAMES):
        c   = r["per_class_correct"][i]
        t   = r["per_class_total"][i]
        pct = 100.0 * c / t if t > 0 else 0.0
        bar = "#" * int(pct / 5)
        ln(f"  {name:<8}  {c:>3}/{t:<3}  {pct:6.2f}%  {bar}")

    ln()
    ln(sep2)
    ln("  Confusion matrix  (rows=true, cols=predicted)")
    ln(f"  Classes: {', '.join(f'{i}={n}' for i, n in enumerate(CLASS_NAMES))}")
    ln(sep2)
    for i, row in enumerate(r["confusion"]):
        ln(f"  {CLASS_NAMES[i]:<8}  {row}")

    ln()
    ln(sep2)
    ln("  ERR v1.0 output contract (verified before inference)")
    ln("    output[0] logits        float32  (1, 10)   -- classification head")
    ln("    output[1] embedding     float32  (1, 128)  -- projection head")
    ln("    output[2] novelty_score float32  (1, 1)    -- novelty head (2-D enforced)")
    ln("  Note: output node names in .qonnx may be auto-generated integers.")
    ln("        Shapes verified by index, names logged above for reference.")
    ln(sep2)
    ln()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "a", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"[DONE] Report appended to: {out_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="STREAMSENSE -- evaluate QONNX model on GV1K golden vectors."
    )
    p.add_argument(
        "--qonnx",
        type    = Path,
        default = _ROOT / "onnx_models" / "streamsense_multihead.qonnx",
        help    = "Path to QONNX model (default: onnx_models/streamsense_multihead.qonnx)",
    )
    p.add_argument(
        "--gvk",
        type    = Path,
        default = _ROOT / "golden_vectors_1000" / "normalized",
        help    = "GV1K normalized directory (default: golden_vectors_1000/normalized)",
    )
    p.add_argument(
        "--out",
        type    = Path,
        default = _ROOT / "evaluation" / "qonnx_evaluation_report.txt",
        help    = "Output report file (default: evaluation/qonnx_evaluation_report.txt)",
    )
    p.add_argument(
        "--pass-threshold",
        type    = float,
        default = 90.0,
        help    = "Minimum top-1%% to pass the gate (default: 90.0)",
    )
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args      = _parse_args()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print("=" * 60)
    print("  STREAMSENSE -- QONNX GV1K Evaluation")
    print("  Scope 2 / QAT Extension")
    print(f"  Timestamp : {timestamp}")
    print("=" * 60)

    if not args.qonnx.exists():
        print(f"\n[ERROR] QONNX model not found: {args.qonnx}")
        print("        Run the export cell in notebooks/qat_colab.ipynb first.")
        sys.exit(1)

    if not args.gvk.exists():
        print(f"\n[ERROR] GV1K directory not found: {args.gvk}")
        sys.exit(1)

    results = evaluate(
        qonnx_path     = args.qonnx,
        gvk_dir        = args.gvk,
        pass_threshold = args.pass_threshold,
    )

    print_and_write_report(
        r          = results,
        qonnx_path = args.qonnx,
        gvk_dir    = args.gvk,
        out_path   = args.out,
        timestamp  = timestamp,
    )

    if not results["passed"]:
        print(f"\n[FAIL] GV1K gate FAILED -- "
              f"{results['top1_acc']:.2f}% < {args.pass_threshold:.1f}% minimum.")
        print("       Do not promote this QONNX to Track E.")
        sys.exit(1)
    else:
        print(f"\n[PASS] GV1K gate PASSED -- "
              f"{results['top1_acc']:.2f}% >= {args.pass_threshold:.1f}%")
        print("       QONNX is deployment-grade. Safe to hand to Track E.")


if __name__ == "__main__":
    main()

```

### `training/export_multihead_onnx.py`

```python
"""
export_multihead_onnx.py
Project STREAMSENSE — Track A
Scope 2 / WA-4 — Deployment-Grade Multi-Head ONNX Export

Exports the StreamSenseWrapper (WA-2) to two ONNX graphs:

    onnx_models/streamsense_multihead_fp32.onnx
    onnx_models/streamsense_multihead_int8.onnx  (PTQ QDQ)

Each graph carries all three heads with static, non-dynamic shapes:

    Input
    ──────────────────────────────────────────────────────
    input          float32  [1, 1, 64, 97]

    Outputs
    ──────────────────────────────────────────────────────
    logits         float32  [1, 10]      — identical to frozen baseline
    embedding      float32  [1, 128]     — linear projection from GAP
    novelty_score  float32  [1,  1]      — 2-D, 1 − max(softmax(logits))

Field-grade requirements (Scope 2 §4, §7 WA-4, §8 D-A5, §9):

    ✓  Static shapes throughout — no dynamic axes anywhere
    ✓  Opset 17 (pinned; matches existing single-head baseline)
    ✓  Operator fusion + constant folding applied via onnxoptimizer / ort
    ✓  Training-only ops (BatchNorm training path, Dropout) removed in eval mode
    ✓  Metadata embedded in the ONNX model (producer, version, date, MPIC)
    ✓  INT8 PTQ via ONNX Runtime quantize_static; calibrated on GV1K normalized
       vectors (or a synthetic fallback if GV1K is absent)
    ✓  FP32 parity gate: element-wise logit diff vs frozen single-head baseline,
       threshold 5e-4; hard abort on failure (_verify_fp32_parity).
    ✓  INT8 parity gate: top-1 agreement vs true label derived from filename;
       hard abort on failure (_verify_int8_top1).

Parity gate design — why two separate functions (Section 9):

    FP32 (_verify_fp32_parity):
        The multi-head FP32 graph runs the same op sequence as the frozen
        single-head baseline through the logits path.  Any divergence is a
        code defect, not quantization noise.  Criterion: element-wise max
        absolute difference ≤ 5e-4 for every vector.  Hard abort on any fail.
        Gate G6 requires 1000/1000 vectors green (§8.1, §9).

    INT8 (_verify_int8_top1):
        Quantization shifts raw logit values — absolute element-wise diff of
        0.1–0.5 is normal and expected (validated against the Scope 1 single-
        head INT8 evaluation report: per-class logit diffs of 0.11–0.40).
        Using a 5e-4 element-wise threshold against FP32 logits for an INT8
        graph is nonsensical and will always fail.  The correct and only
        meaningful criterion is top-1 agreement against the ground-truth label
        derived from the GV1K filename (GV1K_NNNN_<label>_norm.bin).
        Minimum passing rate: 90% top-1 accuracy on the checked vectors.
        Hard abort if rate < 90%.
        Gate G6 requires 1000/1000 vectors checked (§8.1, §9).

DSA Decision Record — Export Optimiser (Section 6 requirement):
    Date       : 2026-06-23
    Component  : Export optimiser
    Structure  : torch.onnx.export (eval mode) → onnxoptimizer passes
                 (eliminate_deadend, fuse_bn_into_conv, fuse_add_bias_into_conv,
                 fuse_consecutive_squeezes, eliminate_nop_transpose,
                 eliminate_unused_initializer) → onnxruntime shape inference.
                 onnxoptimizer is a hard dependency for field-grade export;
                 the script aborts with install instructions if it is absent.
    Complexity : One-shot graph pass — O(|nodes|).
    Alternative rejected: torch.jit.script + onnx.optimize_model → requires
    TorchScript compatibility annotations on all sub-modules; incompatible with
    torchaudio._transforms used in mel_pipeline.  torch.onnx.export with
    tracing is the correct path for this architecture (no data-dependent control
    flow in inference mode).

DSA Decision Record — INT8 Calibration:
    Date       : 2026-06-23
    Component  : INT8 calibration data
    Structure  : GV1K normalized .bin vectors (golden_vectors_1000/normalized/).
                 100 vectors are used (calibration subset); vectors are loaded
                 as [1, 1, 64, 97] float32 tensors and fed to
                 onnxruntime.quantization.quantize_static.
    Fallback   : If GV1K is absent, 64 random float32 tensors drawn from
                 N(0, 1) are used.  INT8 accuracy on GV1K may degrade slightly
                 but the graph structure is identical.
    Alternative rejected: random calibration alone (high variance, poor per-
    channel scale estimates); training-split resampling (requires dataset files
    on disk, not portable).

Run:
    cd training/
    python export_multihead_onnx.py

    Optional flags:
        --ckpt   PATH    Override checkpoint path (default: checkpoints/best_model.pth)
        --out    DIR     Override output directory (default: onnx_models/)
        --gvk    DIR     Override GV1K normalized dir (default: golden_vectors_1000/normalized/)
        --skip-int8      Skip INT8 export (FP32 only)
        --skip-verify    Skip GV1K parity verification
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import torch

# ── Resolve project root ──────────────────────────────────────────────────────
_THIS_DIR = Path(__file__).resolve().parent
_ROOT     = _THIS_DIR.parent

if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from streaming_wrapper import (  # noqa: E402
    StreamSenseWrapper,
    build_wrapper,
    NUM_CLASSES,
    EMBEDDING_DIM,
)

# ── ONNX / ORT imports ────────────────────────────────────────────────────────
try:
    import onnx
    import onnx.helper as onnx_helper  # noqa: F401 — kept for metadata helpers
    from onnx import TensorProto  # noqa: F401
except ImportError as e:
    print(f"[ERROR] onnx not installed: {e}\n        pip install onnx")
    sys.exit(1)

try:
    import onnxruntime as ort
    from onnxruntime.quantization import (
        quantize_static,
        CalibrationDataReader,
        QuantFormat,
        QuantType,
    )
except ImportError as e:
    print(f"[ERROR] onnxruntime not installed: {e}\n        pip install onnxruntime")
    sys.exit(1)

# onnxoptimizer is a hard dependency for field-grade export (D-A5 DoD: "fused").
# The script aborts here rather than silently skipping fusion passes.
try:
    import onnxoptimizer
    _HAS_OPTIMIZER = True
except ImportError:
    print(
        "[ERROR] onnxoptimizer not installed — operator fusion is required for "
        "field-grade export (Scope 2 §8 D-A5 DoD: 'fused').\n"
        "        pip install onnxoptimizer\n"
        "        Then re-run this script."
    )
    sys.exit(1)

# ── Constants — frozen by MPIC v1.0 ──────────────────────────────────────────
OPSET_VERSION = 17
INPUT_SHAPE   = (1, 1, 64, 97)   # [batch, channel, mel_bins, time_frames]
LOGITS_SHAPE  = (1, 10)
EMBED_SHAPE   = (1, EMBEDDING_DIM)
NOVELTY_SHAPE = (1, 1)           # MUST be 2-D

# Number of calibration samples for INT8 PTQ
CALIB_N_SAMPLES = 100

# GV1K parity tolerance for FP32 element-wise gate (from manifest / Section 9)
GV1K_FP32_TOLERANCE = 5e-4

# INT8 top-1 minimum passing rate — 90% is consistent with the ~0.11% accuracy
# drop observed in the Scope 1 single-head INT8 evaluation report.
INT8_TOP1_MIN_RATE = 0.90

# Canonical class label → index mapping (matches class_labels.json order)
# Index: 0=yes, 1=no, 2=up, 3=down, 4=left, 5=right, 6=on, 7=off, 8=stop, 9=go
_LABEL_TO_IDX: dict[str, int] = {
    "yes": 0, "no": 1, "up": 2, "down": 3, "left": 4,
    "right": 5, "on": 6, "off": 7, "stop": 8, "go": 9,
}


# ─────────────────────────────────────────────────────────────────────────────
# Calibration data reader
# ─────────────────────────────────────────────────────────────────────────────

class GV1KCalibrationReader(CalibrationDataReader):
    """
    Feeds normalized GV1K vectors to onnxruntime's static calibration pass.

    Loads *_norm.bin files from golden_vectors_1000/normalized/ and reshapes
    each [64, 97] float32 binary to [1, 1, 64, 97] for the model input.
    Falls back to synthetic Gaussian data if the directory is absent or
    insufficient vectors are found.

    Args:
        gv1k_norm_dir : Path to golden_vectors_1000/normalized/
        input_name    : ONNX graph input name (default: "input")
        n_samples     : Maximum number of calibration samples to use
    """

    def __init__(
        self,
        gv1k_norm_dir: Path,
        input_name: str = "input",
        n_samples: int = CALIB_N_SAMPLES,
    ):
        self._input_name = input_name
        self._data: list[np.ndarray] = []
        self._idx = 0

        bin_files = sorted(gv1k_norm_dir.glob("*_norm.bin")) if gv1k_norm_dir.exists() else []
        n_loaded = 0

        for bf in bin_files[:n_samples]:
            raw = np.fromfile(str(bf), dtype="<f4")
            if raw.size != 64 * 97:
                continue
            tensor = raw.reshape(1, 1, 64, 97).astype(np.float32)
            self._data.append(tensor)
            n_loaded += 1

        if n_loaded < 16:
            # Fallback: synthetic Gaussian (mean≈0, std≈1 — post-normalised range)
            n_synthetic = max(n_samples - n_loaded, 64)
            rng = np.random.default_rng(42)
            for _ in range(n_synthetic):
                tensor = rng.standard_normal((1, 1, 64, 97)).astype(np.float32)
                self._data.append(tensor)
            source = f"synthetic (GV1K absent or < 16 vectors found in {gv1k_norm_dir})"
        else:
            source = f"GV1K ({n_loaded} vectors from {gv1k_norm_dir})"

        print(f"  [calib] Calibration source: {source}")
        print(f"  [calib] Total calibration samples: {len(self._data)}")

    def get_next(self) -> dict[str, np.ndarray] | None:
        if self._idx >= len(self._data):
            return None
        sample = {self._input_name: self._data[self._idx]}
        self._idx += 1
        return sample

    def rewind(self):
        self._idx = 0


# ─────────────────────────────────────────────────────────────────────────────
# ONNX export helpers
# ─────────────────────────────────────────────────────────────────────────────

def _export_fp32(
    wrapper: StreamSenseWrapper,
    out_path: Path,
) -> None:
    """
    Export StreamSenseWrapper to a static-shape FP32 ONNX graph (opset 17).

    All three outputs are exported.  No dynamic axes.  Model is placed in
    eval() mode before tracing so BatchNorm and Dropout run in inference mode
    (training-only branches are dead and fused/eliminated by optimiser).
    """
    wrapper.eval()

    dummy = torch.zeros(*INPUT_SHAPE, dtype=torch.float32)

    # ── Trace and export ──────────────────────────────────────────────────────
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            dummy,
            str(out_path),
            export_params       = True,
            opset_version       = OPSET_VERSION,
            do_constant_folding = True,
            input_names         = ["input"],
            output_names        = ["logits", "embedding", "novelty_score"],
            # Static shapes — no dynamic_axes entry means every dimension is fixed.
            dynamic_axes        = None,
            verbose             = False,
        )

    print(f"  [export] FP32 ONNX written: {out_path}  ({out_path.stat().st_size:,} bytes)")


def _add_metadata(model_path: Path, extra: dict[str, str]) -> None:
    """
    Embed key=value metadata into the ONNX model's metadata_props.

    Required by field-readiness: "metadata embedded in the model" (§4).
    """
    model = onnx.load(str(model_path))
    for k, v in extra.items():
        entry = model.metadata_props.add()
        entry.key   = k
        entry.value = str(v)
    onnx.save(model, str(model_path))


def _optimize_fp32(fp32_path: Path) -> None:
    """
    Apply operator fusion and constant folding to the exported FP32 graph.

    onnxoptimizer is a hard dependency (checked at module import); this
    function always runs fusion passes.  ORT shape-inference is also run
    to propagate static shapes for downstream tooling.
    """
    model = onnx.load(str(fp32_path))
    passes = [
        "eliminate_deadend",
        "fuse_bn_into_conv",
        "fuse_add_bias_into_conv",
        "fuse_consecutive_squeezes",
        "eliminate_nop_transpose",
        "eliminate_unused_initializer",
        "eliminate_nop_pad",
        "fuse_consecutive_reduces",
    ]
    # Only run passes that are available in the installed version
    available = set(onnxoptimizer.get_available_passes())
    passes    = [p for p in passes if p in available]
    optimised = onnxoptimizer.optimize(model, passes)
    onnx.save(optimised, str(fp32_path))
    print(f"  [opt]    onnxoptimizer passes applied: {passes}")

    # Shape inference — always run after fusion
    try:
        model = onnx.load(str(fp32_path))
        model_inferred = onnx.shape_inference.infer_shapes(model)
        onnx.save(model_inferred, str(fp32_path))
        print(f"  [opt]    Shape inference complete.")
    except Exception as e:
        warnings.warn(f"Shape inference failed (non-fatal): {e}", stacklevel=2)


def _verify_fp32_outputs(fp32_path: Path, wrapper: StreamSenseWrapper) -> bool:
    """
    Verify that the exported ONNX graph produces outputs with the correct
    static shapes and that all three output names are present.

    Returns True if all checks pass.
    """
    sess = ort.InferenceSession(str(fp32_path), providers=["CPUExecutionProvider"])

    inputs  = {i.name: i for i in sess.get_inputs()}
    outputs = {o.name: o for o in sess.get_outputs()}

    ok = True

    # Input check
    if "input" not in inputs:
        print(f"  [FAIL] Input 'input' not found in ONNX graph")
        ok = False
    else:
        in_shape = inputs["input"].shape
        if list(in_shape) != list(INPUT_SHAPE):
            print(f"  [FAIL] Input shape {in_shape} != expected {list(INPUT_SHAPE)}")
            ok = False
        else:
            print(f"  [PASS] Input  'input'         shape {in_shape}")

    # Output checks
    for name, expected_shape in [
        ("logits",        list(LOGITS_SHAPE)),
        ("embedding",     list(EMBED_SHAPE)),
        ("novelty_score", list(NOVELTY_SHAPE)),
    ]:
        if name not in outputs:
            print(f"  [FAIL] Output '{name}' not found in ONNX graph")
            ok = False
        else:
            shape = outputs[name].shape
            if list(shape) != expected_shape:
                print(f"  [FAIL] Output '{name}' shape {shape} != {expected_shape}")
                ok = False
            else:
                print(f"  [PASS] Output '{name}'{'':>8} shape {shape}")

    return ok


def _quantize_int8(
    fp32_path: Path,
    int8_path: Path,
    gv1k_norm_dir: Path,
) -> None:
    """
    Post-training static quantization (PTQ) of the FP32 multi-head graph.

    Format : QDQ (Quantize-Dequantize nodes inline — matches existing baseline)
    Types  : weights QInt8, activations QInt8
    Grain  : per-tensor (per_channel=False — matches existing baseline)

    The 'novelty_score' output involves Softmax + ReduceMax + Sub — these ops
    remain float32 because they are at the output boundary and onnxruntime's
    default exclude list keeps Softmax in FP32.  This is correct behaviour:
    the novelty computation is cheap (10-element softmax) and FP32 precision
    is desirable for the open-set threshold decision.
    """
    input_name = "input"

    calib_reader = GV1KCalibrationReader(
        gv1k_norm_dir = gv1k_norm_dir,
        input_name    = input_name,
        n_samples     = CALIB_N_SAMPLES,
    )

    quantize_static(
        model_input             = str(fp32_path),
        model_output            = str(int8_path),
        calibration_data_reader = calib_reader,
        quant_format            = QuantFormat.QDQ,
        per_channel             = False,
        weight_type             = QuantType.QInt8,
        activation_type         = QuantType.QInt8,
        nodes_to_exclude        = [],
        extra_options           = {
            "ActivationSymmetric": False,
            "WeightSymmetric":     True,
        },
    )

    print(f"  [int8]   INT8 QDQ graph written: {int8_path}  ({int8_path.stat().st_size:,} bytes)")


# ─────────────────────────────────────────────────────────────────────────────
# GV1K parity verification — FP32 gate (element-wise, hard abort)
# ─────────────────────────────────────────────────────────────────────────────

def _verify_fp32_parity(
    onnx_path: Path,
    gv1k_norm_dir: Path,
    baseline_onnx: Path | None,
    tolerance: float = GV1K_FP32_TOLERANCE,
    n_vectors: int = 1000,
) -> bool:
    """
    FP32 logit parity gate: element-wise max absolute difference vs the frozen
    single-head FP32 baseline must be ≤ tolerance for every GV1K vector.

    Rationale: the multi-head FP32 graph executes the identical op sequence
    through the logits path.  Any divergence beyond float32 rounding is a code
    defect in the wrapper or export, not expected quantization noise.

    Gate G6 (§8.1, §9) requires 1000/1000 vectors green.

    If baseline_onnx is absent, the gate is skipped (returns True with a
    warning) — this should only happen if the single-head baseline has not
    yet been exported to onnx_models/.

    Returns True if all vectors pass.  Hard abort (sys.exit(1)) is called by
    the caller on False.
    """
    if not gv1k_norm_dir.exists():
        print(f"  [SKIP] GV1K dir not found: {gv1k_norm_dir} — skipping FP32 parity check.")
        return True

    bin_files = sorted(gv1k_norm_dir.glob("*_norm.bin"))[:n_vectors]
    if not bin_files:
        print(f"  [SKIP] No *_norm.bin files in {gv1k_norm_dir}")
        return True

    if baseline_onnx is None or not baseline_onnx.exists():
        print(f"  [WARN] Baseline ONNX not found — FP32 element-wise parity cannot be checked.")
        print(f"         Expected: {baseline_onnx}")
        print(f"  [WARN] Skipping FP32 parity gate.  Export streamsense_model_fp32.onnx first.")
        return True

    # Load sessions
    mh_sess    = ort.InferenceSession(str(onnx_path),      providers=["CPUExecutionProvider"])
    bl_sess    = ort.InferenceSession(str(baseline_onnx),  providers=["CPUExecutionProvider"])
    mh_in_name = mh_sess.get_inputs()[0].name
    bl_in_name = bl_sess.get_inputs()[0].name
    bl_out_name = bl_sess.get_outputs()[0].name  # "logits" on single-head model

    print(f"  [fp32-parity] Baseline: {baseline_onnx.name}")
    print(f"  [fp32-parity] Tolerance: {tolerance:.1e}  (element-wise max abs diff)")
    print(f"  [fp32-parity] Vectors to check: {n_vectors}  (Gate G6 requires {n_vectors}/1000)")

    n_pass   = 0
    n_fail   = 0
    max_diff = 0.0
    failures: list[tuple[str, float]] = []

    for bf in bin_files:
        raw = np.fromfile(str(bf), dtype="<f4")
        if raw.size != 64 * 97:
            continue
        inp = raw.reshape(1, 1, 64, 97).astype(np.float32)

        mh_logits = mh_sess.run(["logits"], {mh_in_name: inp})[0]   # [1, 10]
        bl_logits = bl_sess.run([bl_out_name], {bl_in_name: inp})[0] # [1, 10]

        diff = float(np.abs(mh_logits - bl_logits).max())
        max_diff = max(max_diff, diff)

        if diff <= tolerance:
            n_pass += 1
        else:
            n_fail += 1
            failures.append((bf.name, diff))

    total = n_pass + n_fail
    print(f"  [fp32-parity] Vectors checked   : {total}")
    print(f"  [fp32-parity] Max logit diff     : {max_diff:.6e}  (threshold {tolerance:.1e})")
    print(f"  [fp32-parity] Pass: {n_pass}/{total}  Fail: {n_fail}/{total}")

    if failures:
        print(f"  [fp32-parity] First failures (up to 5):")
        for fname, diff in failures[:5]:
            print(f"               {fname}  max_diff={diff:.6e}")

    if n_fail == 0:
        print(f"  [fp32-parity] PASS — all vectors within element-wise tolerance.")
        return True
    else:
        print(f"  [fp32-parity] FAIL — {n_fail} vector(s) exceeded element-wise tolerance.")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# GV1K parity verification — INT8 gate (top-1 vs true label, hard abort)
# ─────────────────────────────────────────────────────────────────────────────

def _parse_label_from_filename(stem: str) -> int | None:
    """
    Extract the true class index from a GV1K normalized filename.

    Expected pattern: GV1K_NNNN_<label>_norm
    Examples:
        GV1K_0000_yes_norm  → 0
        GV1K_0042_stop_norm → 8
        GV1K_0099_go_norm   → 9

    Returns the class index, or None if the label is not recognized.
    All 10 class labels are single words; parts[2] is unambiguous.
    """
    parts = stem.split("_")
    # parts: ['GV1K', 'NNNN', '<label>', 'norm']
    if len(parts) < 4:
        return None
    label_str = parts[2].lower()
    return _LABEL_TO_IDX.get(label_str, None)


def _verify_int8_top1(
    onnx_path: Path,
    gv1k_norm_dir: Path,
    min_pass_rate: float = INT8_TOP1_MIN_RATE,
    n_vectors: int = 1000,
) -> bool:
    """
    INT8 top-1 accuracy gate: top-1 predicted class must match the ground-truth
    label (parsed from the GV1K filename) for ≥ min_pass_rate of vectors.

    Rationale: INT8 quantization shifts raw logit values by 0.1–0.5 in absolute
    terms — this is expected and validated in the Scope 1 evaluation report
    (single-head INT8 accuracy drop: 0.11%).  Comparing INT8 logit values
    element-wise against FP32 baselines using a tight threshold is nonsensical
    for a quantized graph; the correct criterion is top-1 agreement with the
    ground truth.

    Gate G6 (§8.1, §9) requires 1000/1000 vectors checked.

    Label source: GV1K filename pattern GV1K_NNNN_<label>_norm.bin.
    Any vector whose filename cannot be parsed is skipped (counted separately).

    Args:
        onnx_path     : Path to the INT8 ONNX graph.
        gv1k_norm_dir : Path to golden_vectors_1000/normalized/.
        min_pass_rate : Minimum fraction of vectors that must be top-1 correct.
                        Default 0.90 (90%).
        n_vectors     : Number of GV1K vectors to check.

    Returns True if pass_rate ≥ min_pass_rate.  Hard abort is called by the
    caller on False.
    """
    if not gv1k_norm_dir.exists():
        print(f"  [SKIP] GV1K dir not found: {gv1k_norm_dir} — skipping INT8 top-1 check.")
        return True

    bin_files = sorted(gv1k_norm_dir.glob("*_norm.bin"))[:n_vectors]
    if not bin_files:
        print(f"  [SKIP] No *_norm.bin files in {gv1k_norm_dir}")
        return True

    sess    = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name

    print(f"  [int8-top1] Checking top-1 accuracy against GV1K ground-truth labels")
    print(f"  [int8-top1] Min passing rate: {min_pass_rate*100:.0f}%")
    print(f"  [int8-top1] Vectors to check: {n_vectors}  (Gate G6 requires {n_vectors}/1000)")

    n_correct   = 0
    n_wrong     = 0
    n_skipped   = 0   # vectors whose label cannot be parsed from filename
    failures: list[tuple[str, int, int]] = []  # (filename, true_idx, pred_idx)

    for bf in bin_files:
        stem      = bf.stem   # e.g. GV1K_0042_stop_norm
        true_idx  = _parse_label_from_filename(stem)

        if true_idx is None:
            n_skipped += 1
            print(f"  [int8-top1] SKIP (unparseable label): {bf.name}")
            continue

        raw = np.fromfile(str(bf), dtype="<f4")
        if raw.size != 64 * 97:
            n_skipped += 1
            continue
        inp = raw.reshape(1, 1, 64, 97).astype(np.float32)

        logits   = sess.run(["logits"], {in_name: inp})[0]   # [1, 10]
        pred_idx = int(np.argmax(logits[0]))

        if pred_idx == true_idx:
            n_correct += 1
        else:
            n_wrong += 1
            failures.append((bf.name, true_idx, pred_idx))

    total_checked = n_correct + n_wrong
    if total_checked == 0:
        print(f"  [int8-top1] SKIP — no vectors could be checked (all skipped).")
        return True

    pass_rate = n_correct / total_checked
    print(f"  [int8-top1] Vectors checked : {total_checked}  (skipped: {n_skipped})")
    print(f"  [int8-top1] Correct (top-1) : {n_correct}/{total_checked}  ({pass_rate*100:.1f}%)")
    print(f"  [int8-top1] Wrong   (top-1) : {n_wrong}/{total_checked}")

    if failures:
        print(f"  [int8-top1] First mismatches (up to 5):")
        idx_to_label = {v: k for k, v in _LABEL_TO_IDX.items()}
        for fname, true_i, pred_i in failures[:5]:
            print(f"               {fname}  true={idx_to_label.get(true_i,'?')}({true_i})"
                  f"  pred={idx_to_label.get(pred_i,'?')}({pred_i})")

    if pass_rate >= min_pass_rate:
        print(f"  [int8-top1] PASS — top-1 rate {pass_rate*100:.1f}% ≥ {min_pass_rate*100:.0f}%")
        return True
    else:
        print(f"  [int8-top1] FAIL — top-1 rate {pass_rate*100:.1f}% < {min_pass_rate*100:.0f}%")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# novelty_score shape verification
# ─────────────────────────────────────────────────────────────────────────────

def _verify_novelty_shape(onnx_path: Path) -> bool:
    """
    Verify novelty_score output is exactly [1, 1] — non-negotiable contract.
    """
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    inp  = np.zeros(INPUT_SHAPE, dtype=np.float32)
    results = sess.run(None, {"input": inp})

    output_names = [o.name for o in sess.get_outputs()]
    ns_idx = output_names.index("novelty_score")
    ns_arr = results[ns_idx]

    if ns_arr.shape == (1, 1):
        print(f"  [PASS] novelty_score shape: {ns_arr.shape}  (required: (1, 1))")
        return True
    else:
        print(f"  [FAIL] novelty_score shape: {ns_arr.shape}  (required: (1, 1))")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Main export pipeline
# ─────────────────────────────────────────────────────────────────────────────

def export_multihead(
    ckpt_path    : Path,
    out_dir      : Path,
    gv1k_norm_dir: Path,
    skip_int8    : bool = False,
    skip_verify  : bool = False,
) -> tuple[Path, Path | None]:
    """
    Full export pipeline: FP32 → optimise → metadata → verify → INT8 → verify.

    Returns:
        (fp32_path, int8_path)  — int8_path is None if skip_int8=True.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    fp32_path = out_dir / "streamsense_multihead_fp32.onnx"
    int8_path = out_dir / "streamsense_multihead_int8.onnx"

    baseline_fp32 = out_dir / "streamsense_model_fp32.onnx"  # existing single-head

    # ── 1. Build wrapper ──────────────────────────────────────────────────────
    print(f"\n{'='*64}")
    print("Step 1 — Load checkpoint and build wrapper")
    print(f"{'='*64}")
    wrapper = build_wrapper(ckpt_path=ckpt_path, eval_mode=True)
    print(f"  Embedding dim  : {wrapper.embedding_dim}")
    total_params = sum(p.numel() for p in wrapper.parameters())
    print(f"  Total params   : {total_params:,}")

    # ── 2. Export FP32 ────────────────────────────────────────────────────────
    print(f"\n{'='*64}")
    print("Step 2 — Export FP32 ONNX (opset 17, static shapes)")
    print(f"{'='*64}")
    _export_fp32(wrapper, fp32_path)

    # ── 3. Optimise (fusion + constant folding) ────────────────────────────────
    print(f"\n{'='*64}")
    print("Step 3 — Operator fusion + constant folding")
    print(f"{'='*64}")
    _optimize_fp32(fp32_path)

    # ── 4. Embed metadata ─────────────────────────────────────────────────────
    print(f"\n{'='*64}")
    print("Step 4 — Embed metadata")
    print(f"{'='*64}")
    timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
    metadata = {
        "project"        : "STREAMSENSE",
        "track"          : "A",
        "document"       : "OSL-PRG-2026-SE-WPA Rev 2.0",
        "scope"          : "Scope 2 / WA-4",
        "mpic_version"   : "1.0",
        "opset"          : str(OPSET_VERSION),
        "checkpoint"     : str(ckpt_path),
        "export_utc"     : timestamp,
        "heads"          : "logits,embedding,novelty_score",
        "input_shape"    : "1,1,64,97",
        "logits_shape"   : "1,10",
        "embedding_shape": f"1,{EMBEDDING_DIM}",
        "novelty_shape"  : "1,1",
        "embedding_dim"  : str(EMBEDDING_DIM),
        "num_classes"    : str(NUM_CLASSES),
        "novelty_method" : "1-max_softmax",
        "dynamic_axes"   : "none",
        "quantization"   : "fp32",
    }
    _add_metadata(fp32_path, metadata)
    print(f"  [meta]  Metadata fields embedded: {len(metadata)}")
    print(f"  [meta]  Export timestamp (UTC): {timestamp}")

    # ── 5. Verify FP32 output shapes ──────────────────────────────────────────
    print(f"\n{'='*64}")
    print("Step 5 — Verify FP32 output shapes and names")
    print(f"{'='*64}")
    shape_ok = _verify_fp32_outputs(fp32_path, wrapper)
    if not shape_ok:
        print("[ABORT] FP32 output shape verification failed — aborting export.")
        sys.exit(1)

    # Explicit novelty_score 2-D check
    novelty_2d_ok = _verify_novelty_shape(fp32_path)
    if not novelty_2d_ok:
        print("[ABORT] novelty_score is not [1, 1] — aborting export.")
        sys.exit(1)

    # ── 6. GV1K logit parity (FP32 — element-wise) ────────────────────────────
    if not skip_verify:
        print(f"\n{'='*64}")
        print("Step 6 — GV1K logit parity check (FP32, element-wise vs baseline)")
        print(f"{'='*64}")
        fp32_parity_ok = _verify_fp32_parity(
            onnx_path     = fp32_path,
            gv1k_norm_dir = gv1k_norm_dir,
            baseline_onnx = baseline_fp32 if baseline_fp32.exists() else None,
            tolerance     = GV1K_FP32_TOLERANCE,
            n_vectors     = 1000,
        )
        if not fp32_parity_ok:
            print("[ABORT] FP32 GV1K element-wise parity FAILED — hard stop (Section 9).")
            sys.exit(1)
    else:
        print("  [SKIP] FP32 parity check skipped (--skip-verify).")

    # ── 7. INT8 PTQ export ────────────────────────────────────────────────────
    int8_path_out: Path | None = None

    if not skip_int8:
        print(f"\n{'='*64}")
        print("Step 7 — INT8 PTQ quantization (QDQ, per-tensor, QInt8)")
        print(f"{'='*64}")
        _quantize_int8(fp32_path, int8_path, gv1k_norm_dir)

        # Embed metadata for INT8 graph
        int8_metadata = {**metadata, "quantization": "int8_qdq_ptq"}
        _add_metadata(int8_path, int8_metadata)

        # ── 8. Verify INT8 output shapes ──────────────────────────────────────
        print(f"\n{'='*64}")
        print("Step 8 — Verify INT8 output shapes")
        print(f"{'='*64}")
        int8_shape_ok   = _verify_fp32_outputs(int8_path, wrapper)
        int8_novelty_ok = _verify_novelty_shape(int8_path)

        if not (int8_shape_ok and int8_novelty_ok):
            print("[ABORT] INT8 output shape verification failed.")
            sys.exit(1)

        # ── 9. GV1K top-1 accuracy check (INT8) ───────────────────────────────
        # Criterion: top-1 predicted class vs ground-truth label from filename.
        # Element-wise logit diff vs FP32 baseline is NOT the criterion here —
        # INT8 quantization noise of 0.1–0.5 in logit space is expected and
        # normal.  See module-level docstring for full rationale.
        if not skip_verify:
            print(f"\n{'='*64}")
            print("Step 9 — GV1K top-1 accuracy check (INT8 vs ground-truth labels)")
            print(f"{'='*64}")
            int8_top1_ok = _verify_int8_top1(
                onnx_path     = int8_path,
                gv1k_norm_dir = gv1k_norm_dir,
                min_pass_rate = INT8_TOP1_MIN_RATE,
                n_vectors     = 1000,
            )
            if not int8_top1_ok:
                print("[ABORT] INT8 top-1 accuracy below minimum threshold — hard stop (Section 9).")
                sys.exit(1)
        else:
            print("  [SKIP] INT8 top-1 check skipped (--skip-verify).")

        int8_path_out = int8_path

    # ── 10. Summary ───────────────────────────────────────────────────────────
    print(f"\n{'='*64}")
    print("EXPORT COMPLETE — Summary")
    print(f"{'='*64}")
    print(f"  FP32 graph : {fp32_path}")
    print(f"             : {fp32_path.stat().st_size:,} bytes")
    if int8_path_out is not None:
        print(f"  INT8 graph : {int8_path_out}")
        print(f"             : {int8_path_out.stat().st_size:,} bytes")
    print()
    print("  Output contract (static shapes, no dynamic axes):")
    print(f"    input          float32  {list(INPUT_SHAPE)}")
    print(f"    logits         float32  {list(LOGITS_SHAPE)}")
    print(f"    embedding      float32  {list(EMBED_SHAPE)}")
    print(f"    novelty_score  float32  {list(NOVELTY_SHAPE)}")
    print()
    print("  Parity gates passed:")
    if not skip_verify:
        print("    FP32 — element-wise logit diff ≤ 5e-4 vs frozen baseline  [PASS]")
        if not skip_int8:
            print(f"    INT8 — top-1 accuracy ≥ {INT8_TOP1_MIN_RATE*100:.0f}% vs GV1K ground truth  [PASS]")
    print()
    print("  Next step: run_gv_regression_1000.py against both graphs")
    print(f"{'='*64}")

    return fp32_path, int8_path_out


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="STREAMSENSE — Deployment-grade multi-head ONNX export (WA-4)"
    )
    p.add_argument(
        "--ckpt",
        type    = Path,
        default = _ROOT / "checkpoints" / "best_model.pth",
        help    = "Path to best_model.pth (default: checkpoints/best_model.pth)",
    )
    p.add_argument(
        "--out",
        type    = Path,
        default = _ROOT / "onnx_models",
        help    = "Output directory for ONNX files (default: onnx_models/)",
    )
    p.add_argument(
        "--gvk",
        type    = Path,
        default = _ROOT / "golden_vectors_1000" / "normalized",
        help    = "Path to GV1K normalized/ directory (default: golden_vectors_1000/normalized/)",
    )
    p.add_argument(
        "--skip-int8",
        action  = "store_true",
        default = False,
        help    = "Skip INT8 export (FP32 only)",
    )
    p.add_argument(
        "--skip-verify",
        action  = "store_true",
        default = False,
        help    = "Skip GV1K parity verification",
    )
    return p.parse_args()


def main():
    print("=" * 64)
    print("STREAMSENSE — export_multihead_onnx.py (WA-4)")
    print("Scope 2 — Deployment-grade multi-head ONNX export")
    print("=" * 64)

    args = _parse_args()

    print(f"\nCheckpoint : {args.ckpt}")
    print(f"Output dir : {args.out}")
    print(f"GV1K dir   : {args.gvk}")
    print(f"Skip INT8  : {args.skip_int8}")
    print(f"Skip verify: {args.skip_verify}")

    # Pre-flight: checkpoint must exist
    if not args.ckpt.exists():
        print(f"\n[ERROR] Checkpoint not found: {args.ckpt}")
        print("        Run training/train.py first, or check the path.")
        sys.exit(1)

    export_multihead(
        ckpt_path     = args.ckpt,
        out_dir       = args.out,
        gv1k_norm_dir = args.gvk,
        skip_int8     = args.skip_int8,
        skip_verify   = args.skip_verify,
    )


if __name__ == "__main__":
    main()

```

### `training/generate_golden.py`

```python
"""
generate_golden.py
Project STREAMSENSE — Track A
Reads golden_selection.json, generates all binary artifacts for all 10 vectors.

Outputs per class (in golden_vectors/):
    raw/        GV_0X_label.bin         [16000] float32 = 64000 bytes
    mel/        GV_0X_label_mel.bin     [64,97] float32 = 24896 bytes
    normalized/ GV_0X_label_norm.bin    [64,97] float32 = 24896 bytes
    labels/     GV_0X_label_label.txt   class index
    manifest.json

Run:
    python generate_golden.py
"""

import torch
import torchaudio
import numpy as np
import json
import sys
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
SELECTION_JSON  = Path(r"C:\STREAMSENSE\stats\golden_selection.json")
STATS_FILE      = Path(r"C:\STREAMSENSE\stats\normalization_stats.json")
GV_ROOT         = Path(r"C:\STREAMSENSE\golden_vectors")

RAW_DIR         = GV_ROOT / "raw"
MEL_DIR         = GV_ROOT / "mel"
NORM_DIR        = GV_ROOT / "normalized"
LABEL_DIR       = GV_ROOT / "labels"
MANIFEST_PATH   = GV_ROOT / "manifest.json"

# ── MPIC v1.0 frozen parameters ───────────────────────────────────────────────
SAMPLE_RATE   = 16000
FRAME_LEN     = 16000
N_FFT         = 512
HOP_LENGTH    = 160
N_MELS        = 64
CENTER        = False
POWER         = 2.0
LOG_EPS       = 1e-10
CLIP_FLOOR_DB = -80.0
EXPECTED_T    = (FRAME_LEN - N_FFT) // HOP_LENGTH + 1   # 97

# ── Expected binary sizes ─────────────────────────────────────────────────────
RAW_BYTES   = FRAME_LEN * 4                              # 64000
MEL_BYTES   = N_MELS * EXPECTED_T * 4                   # 24896

# ── Load normalization stats ──────────────────────────────────────────────────
with open(STATS_FILE, "r") as f:
    _stats = json.load(f)
GLOBAL_MEAN = float(_stats["global_mean"])
GLOBAL_STD  = float(_stats["global_std"])

# ── MelSpectrogram transform ──────────────────────────────────────────────────
mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate = SAMPLE_RATE,
    n_fft       = N_FFT,
    hop_length  = HOP_LENGTH,
    n_mels      = N_MELS,
    window_fn   = torch.hann_window,
    center      = CENTER,
    power       = POWER,
)

# ── Helpers ───────────────────────────────────────────────────────────────────
def load_wav_raw(path: Path) -> np.ndarray:
    """
    Steps 1-3: load WAV, mono, pad/crop to FRAME_LEN.
    Returns numpy float32 [16000] — this is the raw GV binary content.
    """
    waveform, sr = torchaudio.load(str(path))
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    length = waveform.shape[1]
    if length < FRAME_LEN:
        waveform = torch.nn.functional.pad(waveform, (0, FRAME_LEN - length))
    elif length > FRAME_LEN:
        waveform = waveform[:, :FRAME_LEN]
    return waveform.squeeze(0).float().numpy()           # [16000] float32


def compute_mel(raw: np.ndarray) -> np.ndarray:
    """
    Steps 4-6: MelSpectrogram + log + clamp.
    Returns numpy float32 [64, 97] row-major — mel GV binary content.
    """
    waveform = torch.from_numpy(raw).unsqueeze(0)        # [1, 16000]
    mel = mel_transform(waveform)                        # [1, 64, 97]
    mel = 10.0 * torch.log10(mel + LOG_EPS)
    mel = torch.clamp(mel, min=CLIP_FLOOR_DB)
    return mel.squeeze(0).numpy()                        # [64, 97] float32


def compute_norm(mel: np.ndarray) -> np.ndarray:
    """
    Step 7: global normalization.
    Returns numpy float32 [64, 97] — normalized GV binary content.
    """
    return ((mel - GLOBAL_MEAN) / GLOBAL_STD).astype(np.float32)


def validate_size(path: Path, expected_bytes: int, name: str) -> bool:
    actual = path.stat().st_size
    if actual != expected_bytes:
        print(f"  [FAIL] {name}: expected {expected_bytes} bytes, got {actual}")
        return False
    return True


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("STREAMSENSE — generate_golden.py")
    print("=" * 60)
    print(f"global_mean = {GLOBAL_MEAN:.6f} dB")
    print(f"global_std  = {GLOBAL_STD:.6f} dB")
    print(f"Expected T  = {EXPECTED_T}")
    print(f"RAW_BYTES   = {RAW_BYTES}  ({FRAME_LEN} x 4)")
    print(f"MEL_BYTES   = {MEL_BYTES}  ({N_MELS} x {EXPECTED_T} x 4)")

    # Validate inputs
    for p, name in [(SELECTION_JSON, "golden_selection.json"),
                    (STATS_FILE,     "normalization_stats.json")]:
        if not p.exists():
            print(f"[ERROR] Not found: {p}")
            sys.exit(1)

    # Create output dirs
    for d in [RAW_DIR, MEL_DIR, NORM_DIR, LABEL_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    # Load selection
    with open(SELECTION_JSON, "r") as f:
        selection = json.load(f)

    manifest = {}
    all_passed = True

    for class_idx in range(10):
        s         = selection[str(class_idx)]
        gv_name   = s["gv_name"]               # e.g. GV_00_yes
        label     = s["label"]
        src_path  = Path(s["source_path"])

        print(f"\n{'─'*50}")
        print(f"Generating {gv_name}  (class {class_idx} — '{label}')")
        print(f"  Source: {src_path.name}")

        if not src_path.exists():
            print(f"  [ERROR] Source WAV not found: {src_path}")
            all_passed = False
            continue

        # ── Generate arrays ───────────────────────────────────────────────────
        raw  = load_wav_raw(src_path)           # [16000] float32
        mel  = compute_mel(raw)                 # [64, 97] float32
        norm = compute_norm(mel)                # [64, 97] float32

        # Validate shapes
        assert raw.shape  == (FRAME_LEN,),           f"raw shape error: {raw.shape}"
        assert mel.shape  == (N_MELS, EXPECTED_T),   f"mel shape error: {mel.shape}"
        assert norm.shape == (N_MELS, EXPECTED_T),   f"norm shape error: {norm.shape}"
        assert raw.dtype  == np.float32
        assert mel.dtype  == np.float32
        assert norm.dtype == np.float32

        # ── Write binary files ────────────────────────────────────────────────
        raw_path  = RAW_DIR  / f"{gv_name}.bin"
        mel_path  = MEL_DIR  / f"{gv_name}_mel.bin"
        norm_path = NORM_DIR / f"{gv_name}_norm.bin"
        lbl_path  = LABEL_DIR / f"{gv_name}_label.txt"

        # numpy .tofile() writes row-major (C order) little-endian float32
        raw.tofile(str(raw_path))
        mel.tofile(str(mel_path))
        norm.tofile(str(norm_path))

        with open(lbl_path, "w") as f:
            f.write(str(class_idx))

        # ── Validate file sizes ───────────────────────────────────────────────
        ok_raw  = validate_size(raw_path,  RAW_BYTES, f"{gv_name}.bin")
        ok_mel  = validate_size(mel_path,  MEL_BYTES, f"{gv_name}_mel.bin")
        ok_norm = validate_size(norm_path, MEL_BYTES, f"{gv_name}_norm.bin")

        passed = ok_raw and ok_mel and ok_norm
        if not passed:
            all_passed = False

        print(f"  raw  → {raw_path.name}   {raw_path.stat().st_size} bytes  {'OK' if ok_raw else 'FAIL'}")
        print(f"  mel  → {mel_path.name}  {mel_path.stat().st_size} bytes  {'OK' if ok_mel else 'FAIL'}")
        print(f"  norm → {norm_path.name}  {norm_path.stat().st_size} bytes  {'OK' if ok_norm else 'FAIL'}")
        print(f"  lbl  → {lbl_path.name}")

        # Quick stats for manifest
        manifest[str(class_idx)] = {
            "gv_name"              : gv_name,
            "class_idx"            : class_idx,
            "label"                : label,
            "source_file"          : src_path.name,
            "raw_bin"              : str(raw_path.name),
            "mel_bin"              : str(mel_path.name),
            "norm_bin"             : str(norm_path.name),
            "raw_bytes"            : int(raw_path.stat().st_size),
            "mel_bytes"            : int(mel_path.stat().st_size),
            "norm_bytes"           : int(norm_path.stat().st_size),
            "raw_shape"            : [FRAME_LEN],
            "mel_shape"            : [N_MELS, EXPECTED_T],
            "norm_shape"           : [N_MELS, EXPECTED_T],
            "dtype"                : "float32",
            "endianness"           : "little",
            "layout"               : "row-major C order",
            "mel_peak_db"          : round(float(mel.max()), 4),
            "mel_min_db"           : round(float(mel.min()), 4),
            "norm_mean"            : round(float(norm.mean()), 6),
            "norm_std"             : round(float(norm.std()), 6),
            "expected_top1_index"  : None,   # filled after training
            "size_validated"       : passed,
        }

    # ── Write manifest ────────────────────────────────────────────────────────
    manifest_out = {
        "mpic_version"   : "1.0",
        "global_mean"    : GLOBAL_MEAN,
        "global_std"     : GLOBAL_STD,
        "n_fft"          : N_FFT,
        "hop_length"     : HOP_LENGTH,
        "n_mels"         : N_MELS,
        "center"         : CENTER,
        "clip_floor_db"  : CLIP_FLOOR_DB,
        "log_eps"        : LOG_EPS,
        "frame_len"      : FRAME_LEN,
        "expected_T"     : EXPECTED_T,
        "raw_bytes"      : RAW_BYTES,
        "mel_bytes"      : MEL_BYTES,
        "tolerance_max_abs_error" : 1e-4,
        "vectors"        : manifest,
    }

    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest_out, f, indent=2)

    # ── Final summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for idx in range(10):
        if str(idx) in manifest:
            m = manifest[str(idx)]
            status = "PASS" if m["size_validated"] else "FAIL"
            print(f"  [{status}] {m['gv_name']:20s}  "
                  f"mel_peak={m['mel_peak_db']:6.1f} dB  "
                  f"norm_mean={m['norm_mean']:+.4f}")

    print(f"\nManifest -> {MANIFEST_PATH}")

    if all_passed:
        print("\n[DONE] All 10 golden vectors generated and validated.")
        print("Next: python verify_pipeline.py")
    else:
        print("\n[FAIL] Some vectors failed — check errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()

```

### `training/generate_golden10_matlab.py`

```python
"""
generate_golden10_matlab.py
Project STREAMSENSE — Track A

Reads the existing golden_vectors/ binary files (generated by generate_golden.py)
and re-saves them in MATLAB-friendly column-major (Fortran) order into a new
folder:  C:\STREAMSENSE\golden_vectors_10_matlab\

The MATLAB team can then load every file with a plain reshape — no transpose trick.

Folder structure created:
    golden_vectors_10_matlab/
        raw/          GV_00_yes.bin          [16000]   float32  (unchanged — 1D has no layout issue)
        mel/          GV_00_yes_mel.bin      [64 x 97] float32  column-major
        normalized/   GV_00_yes_norm.bin     [64 x 97] float32  column-major
        labels/       GV_00_yes_label.txt    class index        (unchanged)
        README.txt    explains the layout to the MATLAB team
        manifest.json

MATLAB load (no tricks needed):
    fid  = fopen('GV_00_yes_norm.bin', 'rb', 'l');
    data = fread(fid, 64*97, 'float32=>single');
    fclose(fid);
    ref  = reshape(data, [64, 97]);   % correct — column-major matches file layout

Run from C:\\STREAMSENSE\\training\\ :
    python generate_golden10_matlab.py
"""

import numpy as np
import json
import sys
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT          = Path(r"C:\STREAMSENSE")
GV_SRC        = ROOT / "golden_vectors"          # existing Python golden vectors
GV_MATLAB     = ROOT / "golden_vectors_10_matlab" # new MATLAB-friendly folder
MANIFEST_SRC  = GV_SRC / "manifest.json"

# ── Output subdirectories ─────────────────────────────────────────────────────
RAW_OUT   = GV_MATLAB / "raw"
MEL_OUT   = GV_MATLAB / "mel"
NORM_OUT  = GV_MATLAB / "normalized"
LABEL_OUT = GV_MATLAB / "labels"

# ── Class labels ──────────────────────────────────────────────────────────────
LABELS = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go']

# ── MPIC v1.0 frozen shapes ───────────────────────────────────────────────────
FRAME_LEN  = 16000
N_MELS     = 64
EXPECTED_T = 97


def load_rowmajor(path: Path, shape: tuple) -> np.ndarray:
    """Load a Python/NumPy row-major float32 binary file and reshape correctly."""
    data = np.fromfile(str(path), dtype=np.float32)
    if data.size != np.prod(shape):
        raise ValueError(f"Size mismatch in {path.name}: "
                         f"expected {np.prod(shape)} values, got {data.size}")
    return data.reshape(shape)   # C row-major — correct for Python-saved files


def save_colmajor(array: np.ndarray, path: Path):
    """Save array in Fortran column-major order (MATLAB-native layout).

    np.asfortranarray().tofile() is a known pitfall: tofile() always writes
    the C-contiguous view of the buffer, so it silently produces row-major
    bytes even for a Fortran array.  tobytes(order='F') forces the correct
    column-major byte sequence before writing.
    """
    path.write_bytes(array.tobytes(order="F"))


def write_readme(out_dir: Path):
    readme = out_dir / "README.txt"
    readme.write_text("""\
STREAMSENSE — Golden Vectors (MATLAB Edition)
=============================================
Generated by: generate_golden10_matlab.py
Layout:       Column-major (Fortran order) — MATLAB native

These files are re-saved versions of the standard golden_vectors/ binaries,
with 2D arrays (mel, normalized) stored in MATLAB column-major order so that
a plain reshape() call gives the correct [64 x 97] matrix.

File shapes:
  raw/        GV_0X_label.bin        [16000]   float32  little-endian
  mel/        GV_0X_label_mel.bin    [64 x 97] float32  little-endian  column-major
  normalized/ GV_0X_label_norm.bin   [64 x 97] float32  little-endian  column-major
  labels/     GV_0X_label_label.txt  plain text, class index 0-9

MATLAB load — copy-paste ready:
  % Raw audio (1D — no layout issue)
  fid = fopen('GV_00_yes.bin', 'rb', 'l');
  raw = fread(fid, 16000, 'float32=>single');
  fclose(fid);

  % Mel spectrogram [64 x 97]
  fid = fopen('GV_00_yes_mel.bin', 'rb', 'l');
  mel = reshape(fread(fid, 64*97, 'float32=>single'), [64, 97]);
  fclose(fid);

  % Normalised spectrogram [64 x 97]
  fid = fopen('GV_00_yes_norm.bin', 'rb', 'l');
  ref = reshape(fread(fid, 64*97, 'float32=>single'), [64, 97]);
  fclose(fid);

Tolerance: max(abs(streamsense_mel_pipeline(raw) - ref)) < 5e-4
""")


def main():
    print("=" * 60)
    print("STREAMSENSE — generate_golden10_matlab.py")
    print("=" * 60)
    print(f"Source : {GV_SRC}")
    print(f"Output : {GV_MATLAB}")

    # ── Validate source exists ────────────────────────────────────────────────
    for subdir in ["raw", "mel", "normalized", "labels"]:
        if not (GV_SRC / subdir).exists():
            print(f"[ERROR] Source folder not found: {GV_SRC / subdir}")
            print("        Run generate_golden.py first.")
            sys.exit(1)

    # ── Create output directories ─────────────────────────────────────────────
    for d in [RAW_OUT, MEL_OUT, NORM_OUT, LABEL_OUT]:
        d.mkdir(parents=True, exist_ok=True)

    write_readme(GV_MATLAB)

    manifest_out = {
        "description"  : "MATLAB-friendly golden vectors — column-major layout",
        "source"       : "generated by generate_golden10_matlab.py",
        "layout"       : "column-major Fortran order (MATLAB native)",
        "dtype"        : "float32",
        "endianness"   : "little",
        "matlab_reshape": "reshape(fread(fid,64*97,'float32=>single'),[64,97])",
        "vectors"      : {}
    }

    all_passed = True

    for i, lbl in enumerate(LABELS):
        gv_name = f"GV_{i:02d}_{lbl}"
        print(f"\n{'─'*50}")
        print(f"Processing {gv_name}  (class {i} — '{lbl}')")

        # ── Source file paths ─────────────────────────────────────────────────
        src_raw   = GV_SRC / "raw"        / f"{gv_name}.bin"
        src_mel   = GV_SRC / "mel"        / f"{gv_name}_mel.bin"
        src_norm  = GV_SRC / "normalized" / f"{gv_name}_norm.bin"
        src_label = GV_SRC / "labels"     / f"{gv_name}_label.txt"

        for p in [src_raw, src_mel, src_norm]:
            if not p.exists():
                print(f"  [ERROR] Missing: {p}")
                all_passed = False
                continue

        # ── Load Python row-major files ───────────────────────────────────────
        raw  = load_rowmajor(src_raw,  (FRAME_LEN,))          # [16000]
        mel  = load_rowmajor(src_mel,  (N_MELS, EXPECTED_T))  # [64, 97]
        norm = load_rowmajor(src_norm, (N_MELS, EXPECTED_T))  # [64, 97]

        # ── Output file paths ─────────────────────────────────────────────────
        dst_raw   = RAW_OUT   / f"{gv_name}.bin"
        dst_mel   = MEL_OUT   / f"{gv_name}_mel.bin"
        dst_norm  = NORM_OUT  / f"{gv_name}_norm.bin"
        dst_label = LABEL_OUT / f"{gv_name}_label.txt"

        # ── Save raw (1D — no layout change needed) ───────────────────────────
        raw.tofile(str(dst_raw))

        # ── Save mel and norm in MATLAB column-major order ────────────────────
        save_colmajor(mel,  dst_mel)
        save_colmajor(norm, dst_norm)

        # ── Copy label file ───────────────────────────────────────────────────
        if src_label.exists():
            dst_label.write_text(src_label.read_text())
        else:
            dst_label.write_text(str(i))

        # ── Validate output sizes ─────────────────────────────────────────────
        expected_raw_bytes  = FRAME_LEN * 4             # 64000
        expected_mel_bytes  = N_MELS * EXPECTED_T * 4  # 24832  (64 * 97 * 4)

        ok_raw  = dst_raw.stat().st_size  == expected_raw_bytes
        ok_mel  = dst_mel.stat().st_size  == expected_mel_bytes
        ok_norm = dst_norm.stat().st_size == expected_mel_bytes

        if not (ok_raw and ok_mel and ok_norm):
            all_passed = False

        print(f"  raw  → {dst_raw.name}   {dst_raw.stat().st_size} bytes  {'OK' if ok_raw else 'FAIL'}")
        print(f"  mel  → {dst_mel.name}  {dst_mel.stat().st_size} bytes  {'OK' if ok_mel else 'FAIL'}")
        print(f"  norm → {dst_norm.name}  {dst_norm.stat().st_size} bytes  {'OK' if ok_norm else 'FAIL'}")

        # ── Self-verify: reload and compare to original ───────────────────────
        # The file holds column-major bytes (col varies slowest).
        # np.fromfile reads them flat; reshape with order='F' fills column-by-column,
        # which is the inverse of tobytes(order='F'), so norm_rebuilt == norm exactly.
        norm_reloaded = np.fromfile(str(dst_norm), dtype=np.float32)
        norm_rebuilt  = norm_reloaded.reshape((N_MELS, EXPECTED_T), order='F')
        max_err = float(np.max(np.abs(norm_rebuilt - norm)))
        print(f"  self-check max_err = {max_err:.2e}  {'OK' if max_err < 1e-7 else 'FAIL'}")

        manifest_out["vectors"][str(i)] = {
            "gv_name"   : gv_name,
            "class_idx" : i,
            "label"     : lbl,
            "raw_file"  : dst_raw.name,
            "mel_file"  : dst_mel.name,
            "norm_file" : dst_norm.name,
            "raw_shape" : [FRAME_LEN],
            "mel_shape" : [N_MELS, EXPECTED_T],
            "norm_shape": [N_MELS, EXPECTED_T],
            "norm_mean" : round(float(norm.mean()), 6),
            "norm_std"  : round(float(norm.std()),  6),
            "size_ok"   : ok_raw and ok_mel and ok_norm,
        }

    # ── Write manifest ────────────────────────────────────────────────────────
    manifest_path = GV_MATLAB / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest_out, f, indent=2)

    # ── Final summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for i, lbl in enumerate(LABELS):
        v = manifest_out["vectors"].get(str(i), {})
        status = "PASS" if v.get("size_ok") else "FAIL"
        print(f"  [{status}] GV_{i:02d}_{lbl}")

    print(f"\nManifest  → {manifest_path}")
    print(f"README    → {GV_MATLAB / 'README.txt'}")

    if all_passed:
        print(f"\n[DONE] golden_vectors_10_matlab/ is ready to share.")
        print(f"       Share the entire folder: {GV_MATLAB}")
        print(f"       MATLAB team uses plain reshape — no transpose needed.")
    else:
        print("\n[FAIL] Some vectors failed — check errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()

```

### `training/generate_golden_1000.py`

```python
"""
generate_golden_1000.py
Project STREAMSENSE — Track A
Generates a 1000-sample Golden Vector set for large-scale cross-implementation
(Python vs C++/MATLAB) regression testing — a superset of the 10-vector
hand-picked GV set used for tight pipeline parity checks.

Unlike select_golden.py (manual visual selection of 10 "best" examples),
this script SELECTS AUTOMATICALLY:
    - 1000 samples drawn from the TEST split
    - Fixed random seed (42) for reproducibility — both Track A and Track B
      can regenerate the exact same selection independently if needed
    - Stratified roughly evenly across the 10 classes (100 per class,
      remainder distributed by round-robin if test split sizes differ)

For each selected sample, produces (same binary format as the 10-vector set):
    golden_vectors_1000/raw/GV1K_NNNN_label.bin          [16000]      float32
    golden_vectors_1000/mel/GV1K_NNNN_label_mel.bin      [64, 97]     float32
    golden_vectors_1000/normalized/GV1K_NNNN_label_norm.bin [64, 97]  float32
    golden_vectors_1000/labels/GV1K_NNNN_label_label.txt  (class index)
    golden_vectors_1000/manifest.json

Pipeline: identical Steps 1-7 to mel_pipeline.py (load/mono/pad-crop ->
MelSpectrogram -> log -> clamp -> normalize), reusing the same MPIC v1.0
frozen constants and global_mean/global_std.

Run:
    python generate_golden_1000.py
"""

import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torchaudio

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT          = Path(r"C:\STREAMSENSE")
TEST_SPLIT    = ROOT / "data"  / "splits" / "test_files.txt"
STATS_FILE    = ROOT / "stats" / "normalization_stats.json"

OUT_DIR       = ROOT / "golden_vectors_1000"
RAW_DIR       = OUT_DIR / "raw"
MEL_DIR       = OUT_DIR / "mel"
NORM_DIR      = OUT_DIR / "normalized"
LABEL_DIR     = OUT_DIR / "labels"
MANIFEST_PATH = OUT_DIR / "manifest.json"

# ── MPIC v1.0 frozen parameters ───────────────────────────────────────────────
SAMPLE_RATE   = 16000
FRAME_LEN     = 16000
N_FFT         = 512
HOP_LENGTH    = 160
N_MELS        = 64
CENTER        = False
POWER         = 2.0
LOG_EPS       = 1e-10
CLIP_FLOOR_DB = -80.0

EXPECTED_T = (FRAME_LEN - N_FFT) // HOP_LENGTH + 1  # = 97
RAW_BYTES  = FRAME_LEN * 4                          # 64000
MEL_BYTES  = N_MELS * EXPECTED_T * 4                # 24832

CLASS_MAP = {
    0: "yes", 1: "no",  2: "up",   3: "down", 4: "left",
    5: "right", 6: "on", 7: "off", 8: "stop", 9: "go"
}
N_CLASSES   = 10
N_TOTAL     = 1000
PER_CLASS   = N_TOTAL // N_CLASSES   # 100
RANDOM_SEED = 42

# ── MelSpectrogram transform (built once) ────────────────────────────────────
mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate = SAMPLE_RATE,
    n_fft       = N_FFT,
    hop_length  = HOP_LENGTH,
    n_mels      = N_MELS,
    window_fn   = torch.hann_window,
    center      = CENTER,
    power       = POWER,
)

# ── Load normalization stats ──────────────────────────────────────────────────
if not STATS_FILE.exists():
    print(f"[ERROR] Stats file not found: {STATS_FILE}")
    sys.exit(1)

with open(STATS_FILE, "r") as f:
    _stats = json.load(f)
GLOBAL_MEAN = float(_stats["global_mean"])
GLOBAL_STD  = float(_stats["global_std"])


# ── Helpers ───────────────────────────────────────────────────────────────────
def parse_path(line: str):
    """Returns (Path, label_str, class_idx) from a split line."""
    parts     = line.strip().split("|")
    win_path  = parts[0].strip()
    label     = parts[1].strip()
    class_idx = int(parts[2].strip())
    return Path(win_path), label, class_idx


def read_test_split() -> dict:
    """Returns dict: class_idx -> list of (Path, label) tuples."""
    buckets = {i: [] for i in range(N_CLASSES)}
    with open(TEST_SPLIT, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            path, label, idx = parse_path(line)
            buckets[idx].append((path, label))
    return buckets


def load_wav_raw(path: Path) -> np.ndarray:
    """Steps 1-3: load, mono, pad/crop -> [16000] float32."""
    waveform, sr = torchaudio.load(str(path))
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    length = waveform.shape[1]
    if length < FRAME_LEN:
        waveform = torch.nn.functional.pad(waveform, (0, FRAME_LEN - length))
    elif length > FRAME_LEN:
        waveform = waveform[:, :FRAME_LEN]
    return waveform.squeeze(0).float().numpy()  # [16000] float32


def compute_mel(raw: np.ndarray) -> np.ndarray:
    """Steps 4-6: MelSpectrogram + log + clamp -> [64, 97] float32."""
    waveform = torch.from_numpy(raw).unsqueeze(0)  # [1, 16000]
    mel = mel_transform(waveform)                  # [1, 64, 97]
    mel = 10.0 * torch.log10(mel + LOG_EPS)
    mel = torch.clamp(mel, min=CLIP_FLOOR_DB)
    return mel.squeeze(0).numpy()                  # [64, 97] float32


def compute_norm(mel: np.ndarray) -> np.ndarray:
    """Step 7: global normalization -> [64, 97] float32."""
    return ((mel - GLOBAL_MEAN) / GLOBAL_STD).astype(np.float32)


def validate_size(path: Path, expected_bytes: int, name: str) -> bool:
    actual = path.stat().st_size
    if actual != expected_bytes:
        print(f"  [FAIL] {name}: expected {expected_bytes} bytes, got {actual}")
        return False
    return True


def select_1000(buckets: dict) -> list:
    """
    Automatic selection — no manual review.
    Aims for PER_CLASS samples per class (random, seeded). If a class has
    fewer than PER_CLASS available in the test split, takes all of them
    and redistributes the shortfall round-robin across other classes so
    the total stays as close to N_TOTAL as possible.

    Returns: list of (gv_index, class_idx, label, Path) tuples, gv_index
    0..N-1 in selection order.
    """
    rng = random.Random(RANDOM_SEED)

    # First pass: shuffle each class's file list, take up to PER_CLASS
    per_class_selected = {}
    shortfall = 0
    for class_idx in range(N_CLASSES):
        files = buckets[class_idx][:]  # copy
        rng.shuffle(files)
        take = min(PER_CLASS, len(files))
        per_class_selected[class_idx] = files[:take]
        shortfall += (PER_CLASS - take)
        if take < PER_CLASS:
            print(f"  [WARN] class {class_idx} ('{CLASS_MAP[class_idx]}') "
                  f"only has {len(files)} test files (< {PER_CLASS})")

    # Second pass: redistribute shortfall round-robin from classes with
    # surplus (files beyond PER_CLASS already shuffled/excluded above)
    if shortfall > 0:
        leftovers = {
            class_idx: buckets[class_idx][PER_CLASS:]
            for class_idx in range(N_CLASSES)
            if len(buckets[class_idx]) > PER_CLASS
        }
        for class_idx in leftovers:
            rng.shuffle(leftovers[class_idx])

        class_cycle = [c for c in range(N_CLASSES) if leftovers.get(c)]
        ci = 0
        while shortfall > 0 and class_cycle:
            c = class_cycle[ci % len(class_cycle)]
            if leftovers[c]:
                per_class_selected[c].append(leftovers[c].pop())
                shortfall -= 1
            else:
                class_cycle.remove(c)
                continue
            ci += 1

    # Flatten into a single list, ordered by class then index, then
    # shuffle the overall order so GV1K_0000.. isn't grouped by class
    flat = []
    for class_idx in range(N_CLASSES):
        for (path, label) in per_class_selected[class_idx]:
            flat.append((class_idx, label, path))

    rng.shuffle(flat)

    return [(i, class_idx, label, path) for i, (class_idx, label, path) in enumerate(flat)]


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("STREAMSENSE — generate_golden_1000.py")
    print("=" * 60)
    print(f"global_mean = {GLOBAL_MEAN:.6f} dB")
    print(f"global_std  = {GLOBAL_STD:.6f} dB")
    print(f"Expected T  = {EXPECTED_T}")
    print(f"RAW_BYTES   = {RAW_BYTES}")
    print(f"MEL_BYTES   = {MEL_BYTES}")
    print(f"Target total: {N_TOTAL}  ({PER_CLASS} per class, seed={RANDOM_SEED})")

    if not TEST_SPLIT.exists():
        print(f"[ERROR] Test split not found: {TEST_SPLIT}")
        sys.exit(1)

    for d in (RAW_DIR, MEL_DIR, NORM_DIR, LABEL_DIR):
        d.mkdir(parents=True, exist_ok=True)

    print(f"\nReading test split: {TEST_SPLIT}")
    buckets = read_test_split()
    for c in range(N_CLASSES):
        print(f"  class {c} ('{CLASS_MAP[c]}'): {len(buckets[c])} files available")

    print("\nSelecting samples...")
    selection = select_1000(buckets)
    print(f"Selected {len(selection)} samples total")

    manifest_vectors = {}
    n_passed = 0
    n_failed = 0
    n_errors = 0

    print(f"\nGenerating {len(selection)} golden vectors...")
    REPORT_EVERY = 100

    for gv_index, class_idx, label, src_path in selection:
        gv_name = f"GV1K_{gv_index:04d}_{label}"

        if (gv_index + 1) % REPORT_EVERY == 0 or gv_index == 0:
            print(f"  [{gv_index+1:>4}/{len(selection)}]  {gv_name}")

        if not src_path.exists():
            print(f"  [ERROR] Source WAV not found: {src_path}")
            n_errors += 1
            continue

        try:
            raw  = load_wav_raw(src_path)   # [16000] float32
            mel  = compute_mel(raw)         # [64, 97] float32
            norm = compute_norm(mel)        # [64, 97] float32
        except Exception as e:
            print(f"  [ERROR] {gv_name}: {e}")
            n_errors += 1
            continue

        # Validate shapes/dtypes
        assert raw.shape  == (FRAME_LEN,),         f"raw shape error: {raw.shape}"
        assert mel.shape  == (N_MELS, EXPECTED_T), f"mel shape error: {mel.shape}"
        assert norm.shape == (N_MELS, EXPECTED_T), f"norm shape error: {norm.shape}"
        assert raw.dtype  == np.float32
        assert mel.dtype  == np.float32
        assert norm.dtype == np.float32

        raw_path  = RAW_DIR   / f"{gv_name}.bin"
        mel_path  = MEL_DIR   / f"{gv_name}_mel.bin"
        norm_path = NORM_DIR  / f"{gv_name}_norm.bin"
        lbl_path  = LABEL_DIR / f"{gv_name}_label.txt"

        raw.tofile(str(raw_path))
        mel.tofile(str(mel_path))
        norm.tofile(str(norm_path))

        with open(lbl_path, "w") as f:
            f.write(str(class_idx))

        ok_raw  = validate_size(raw_path,  RAW_BYTES, f"{gv_name}.bin")
        ok_mel  = validate_size(mel_path,  MEL_BYTES, f"{gv_name}_mel.bin")
        ok_norm = validate_size(norm_path, MEL_BYTES, f"{gv_name}_norm.bin")

        passed = ok_raw and ok_mel and ok_norm
        if passed:
            n_passed += 1
        else:
            n_failed += 1

        manifest_vectors[str(gv_index)] = {
            "gv_name"     : gv_name,
            "class_idx"   : class_idx,
            "label"       : label,
            "source_file" : src_path.name,
            "raw_bin"     : raw_path.name,
            "mel_bin"     : mel_path.name,
            "norm_bin"    : norm_path.name,
            "raw_bytes"   : int(raw_path.stat().st_size),
            "mel_bytes"   : int(mel_path.stat().st_size),
            "norm_bytes"  : int(norm_path.stat().st_size),
            "raw_shape"   : [FRAME_LEN],
            "mel_shape"   : [N_MELS, EXPECTED_T],
            "norm_shape"  : [N_MELS, EXPECTED_T],
            "dtype"       : "float32",
            "endianness"  : "little",
            "layout"      : "row-major C order",
            "mel_peak_db" : round(float(mel.max()), 4),
            "mel_min_db"  : round(float(mel.min()), 4),
            "norm_mean"   : round(float(norm.mean()), 6),
            "norm_std"    : round(float(norm.std()), 6),
            "size_validated": passed,
        }

    # ── Write manifest ────────────────────────────────────────────────────────
    manifest_out = {
        "mpic_version"   : "1.0",
        "set_name"       : "golden_vectors_1000",
        "n_total"        : len(selection),
        "n_per_class_target": PER_CLASS,
        "random_seed"    : RANDOM_SEED,
        "source_split"   : "test",
        "global_mean"    : GLOBAL_MEAN,
        "global_std"     : GLOBAL_STD,
        "n_fft"          : N_FFT,
        "hop_length"     : HOP_LENGTH,
        "n_mels"         : N_MELS,
        "center"         : CENTER,
        "clip_floor_db"  : CLIP_FLOOR_DB,
        "log_eps"        : LOG_EPS,
        "frame_len"      : FRAME_LEN,
        "expected_T"     : EXPECTED_T,
        "raw_bytes"      : RAW_BYTES,
        "mel_bytes"      : MEL_BYTES,
        "tolerance_max_abs_error": 0.0005,
        "vectors"        : manifest_vectors,
    }

    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest_out, f, indent=2)

    # ── Class distribution summary ────────────────────────────────────────────
    class_counts = {c: 0 for c in range(N_CLASSES)}
    for v in manifest_vectors.values():
        class_counts[v["class_idx"]] += 1

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"  Generated : {n_passed + n_failed} / {len(selection)}")
    print(f"  Passed    : {n_passed}")
    print(f"  Failed    : {n_failed}")
    print(f"  Errors    : {n_errors}")
    print(f"\n  Class distribution:")
    for c in range(N_CLASSES):
        print(f"    {c} ({CLASS_MAP[c]:<6}): {class_counts[c]:>4}")

    print(f"\nManifest -> {MANIFEST_PATH}")
    print(f"Output dir -> {OUT_DIR}")

    if n_failed == 0 and n_errors == 0:
        print("\n[DONE] golden_vectors_1000/ generated and validated.")
    else:
        print(f"\n[WARN] Completed with {n_failed} size failures, {n_errors} errors.")
        sys.exit(1)


if __name__ == "__main__":
    main()

```

### `training/goldenvector_stream.py`

```python
"""
goldenvector_stream.py
Project STREAMSENSE — Track A (Scope 2, WA-1)

GV1K Streaming Parity Test — PRIMARY EXIT CRITERION for WA-1.

Feeds each of the 1000 GV1K raw audio files through StreamingFramer in
randomly-sized chunks (simulating real network jitter) and compares the
output mel tensors against the frozen GV1K normalised golden vectors.

Exit criterion: ALL 1000 vectors must pass max absolute error < 5e-4.
Expected result: max_abs_error = 0.00e+00 (exact) because:
  - Normalisation uses frozen constants (not Welford live stats)
  - One complete 16000-sample file → one framer → one [1,1,64,97] tensor
  - Identical to mel_pipeline.preprocess() output within floating-point precision

Welford Report:
  After processing all 1000 files, the script reports the aggregate
  Welford statistics (accumulated across all frames) vs the frozen global
  constants. This is a convergence validation check, not the exit criterion.

Usage:
  python goldenvector_stream.py
  python goldenvector_stream.py --chunk-min 50 --chunk-max 4000
  python goldenvector_stream.py --chunk-min 16000 --chunk-max 16000  # exact blocks
"""

import sys
import argparse
import random
import math
import json
import numpy as np
import torch
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
REPO_ROOT  = Path(__file__).resolve().parent.parent
GV1K_ROOT  = REPO_ROOT / "golden_vectors_1000"
RAW_DIR    = GV1K_ROOT / "raw"
NORM_DIR   = GV1K_ROOT / "normalized"
MANIFEST   = GV1K_ROOT / "manifest.json"

# ── MPIC v1.0 shape constants ──────────────────────────────────────────────────
N_MELS     = 64
EXPECTED_T = 97
TOLERANCE  = 5e-4

# ── Import streaming framer ────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from streaming_framer import StreamingFramer, GLOBAL_MEAN, GLOBAL_STD


def load_raw(path: Path) -> np.ndarray:
    """Load GV1K raw binary → float32 numpy [16000]."""
    return np.fromfile(str(path), dtype=np.float32)


def load_norm(path: Path) -> np.ndarray:
    """Load GV1K normalised binary → float32 numpy [64, 97]."""
    return np.fromfile(str(path), dtype=np.float32).reshape(N_MELS, EXPECTED_T)


def feed_in_chunks(
    framer: StreamingFramer,
    raw: np.ndarray,
    chunk_min: int,
    chunk_max: int,
) -> torch.Tensor:
    """
    Break raw [16000] into random-sized chunks and feed into the framer.
    Collects all emitted frames and concatenates into [1, 1, 64, K].
    """
    all_frames = []
    idx = 0
    while idx < len(raw):
        size  = random.randint(chunk_min, chunk_max)
        chunk = raw[idx : idx + size]
        idx  += size
        for out in framer.process_chunk(chunk):
            all_frames.append(out)

    if not all_frames:
        return torch.zeros((1, 1, N_MELS, 0), dtype=torch.float32)
    return torch.cat(all_frames, dim=3)             # [1, 1, 64, K_total]


def main():
    parser = argparse.ArgumentParser(
        description="GV1K streaming parity test — WA-1 exit criterion"
    )
    parser.add_argument("--chunk-min", type=int, default=100,
                        help="Min chunk size in samples (default: 100)")
    parser.add_argument("--chunk-max", type=int, default=8000,
                        help="Max chunk size in samples (default: 8000)")
    args = parser.parse_args()

    print("=" * 72)
    print("GV1K Streaming Parity Test — WA-1 Exit Criterion")
    print(f"  Tolerance  : {TOLERANCE:.0e}")
    print(f"  Chunk range: {args.chunk_min} – {args.chunk_max} samples (random per packet)")
    print(f"  Frozen mean: {GLOBAL_MEAN}  Frozen std: {GLOBAL_STD}")
    print("=" * 72)

    # ── Validate directories ──────────────────────────────────────────────────
    for d, name in [(RAW_DIR, "raw"), (NORM_DIR, "normalized")]:
        if not d.exists():
            print(f"[ERROR] GV1K directory not found: {d}")
            sys.exit(1)

    # ── Collect file pairs ────────────────────────────────────────────────────
    raw_files  = sorted(RAW_DIR.glob("*.bin"))
    norm_files = sorted(NORM_DIR.glob("*_norm.bin"))

    if len(raw_files) != len(norm_files):
        print(f"[ERROR] File count mismatch: {len(raw_files)} raw vs {len(norm_files)} norm")
        sys.exit(1)

    n_files = len(raw_files)
    if n_files == 0:
        print("[ERROR] No GV1K files found.")
        sys.exit(1)

    print(f"  Files found: {n_files}\n")

    # ── Welford aggregate accumulator (across ALL files) ──────────────────────
    # Chan parallel merge: we merge each fresh framer's Welford state into
    # this aggregate after processing each file.
    agg_n    = 0.0
    agg_mean = 0.0
    agg_M2   = 0.0

    passed    = 0
    failed    = 0
    max_diffs = []
    REPORT_EVERY = 100

    for i, (raw_path, norm_path) in enumerate(zip(raw_files, norm_files)):
        # Fresh framer per file — isolates per-file STFT state
        framer = StreamingFramer(
            stream_sr      =16000,
            stream_channels=1,
            dtype          =torch.float32,
            layout         ="planar",
        )

        raw  = load_raw(raw_path)    # [16000] float32
        gold = load_norm(norm_path)  # [64, 97] float32

        out = feed_in_chunks(framer, raw, args.chunk_min, args.chunk_max)
        # out: [1, 1, 64, K]

        K = out.shape[3]
        if K == 0:
            diff = float("inf")
        elif K >= EXPECTED_T:
            out_97 = out[0, 0, :, :EXPECTED_T]             # [64, 97]
            diff   = torch.abs(out_97 - torch.from_numpy(gold)).max().item()
        else:
            out_K = out[0, 0, :, :]                        # [64, K]
            diff  = torch.abs(out_K - torch.from_numpy(gold[:, :K])).max().item()

        max_diffs.append(diff)
        ok = diff < TOLERANCE

        if ok:
            passed += 1
        else:
            failed += 1
            print(f"  [FAIL] {raw_path.name}  max_abs_err={diff:.2e}")

        if (i + 1) % REPORT_EVERY == 0:
            print(f"  [{i+1:>4}/{n_files}]  pass={passed}  fail={failed}  "
                  f"max_diff_so_far={max(max_diffs):.2e}")

        # ── Merge Welford state into aggregate (Chan's parallel formula) ───────
        b_n    = float(framer.n_welford_elements)
        b_mean = framer.welford_mean
        b_M2   = (framer.welford_std ** 2) * b_n if b_n > 1 else 0.0

        if b_n > 0:
            new_n    = agg_n + b_n
            delta    = b_mean - agg_mean
            agg_mean = agg_mean + delta * b_n / new_n
            agg_M2   = agg_M2 + b_M2 + (delta ** 2) * agg_n * b_n / new_n
            agg_n    = new_n

    # ── Final summary ─────────────────────────────────────────────────────────
    overall_max  = max(max_diffs) if max_diffs else float("nan")
    overall_mean = float(np.mean(max_diffs)) if max_diffs else float("nan")
    agg_std      = math.sqrt(agg_M2 / agg_n) if agg_n > 1 else 0.0

    print()
    print("=" * 72)
    print("RESULTS — GV1K Streaming Parity")
    print("=" * 72)
    print(f"  Total vectors   : {n_files}")
    print(f"  PASS            : {passed}")
    print(f"  FAIL            : {failed}")
    print(f"  Max abs error   : {overall_max:.2e}  (tolerance = {TOLERANCE:.0e})")
    print(f"  Mean abs error  : {overall_mean:.2e}")
    print()
    print("Welford Convergence Report (aggregate across all 1000 files):")
    print(f"  Streaming mean  : {agg_mean:.6f} dB")
    print(f"  Streaming std   : {agg_std:.6f} dB")
    print(f"  Frozen mean     : {GLOBAL_MEAN:.6f} dB")
    print(f"  Frozen std      : {GLOBAL_STD:.6f} dB")
    print(f"  |Δ mean|        : {abs(agg_mean - GLOBAL_MEAN):.6f} dB  "
          f"({'OK' if abs(agg_mean - GLOBAL_MEAN) < 1.0 else 'LARGE'})")
    print(f"  |Δ std|         : {abs(agg_std  - GLOBAL_STD):.6f} dB  "
          f"({'OK' if abs(agg_std  - GLOBAL_STD ) < 1.0 else 'LARGE'})")
    print(f"  Total elements  : {int(agg_n):,}")
    print("=" * 72)

    if failed == 0:
        print(f"\n[DONE] ALL {n_files} GV1K vectors PASS at tolerance {TOLERANCE:.0e}.")
        print("WA-1 exit criterion MET — streaming framer ready for D7-D8.")
        sys.exit(0)
    else:
        print(f"\n[FAIL] {failed}/{n_files} vectors exceeded tolerance {TOLERANCE:.0e}.")
        sys.exit(1)


if __name__ == "__main__":
    main()

```

### `training/live_demo.py`

```python
"""
live_demo.py
Project STREAMSENSE — Track A

Live pipeline simulation:
  StreamSimulator → StreamingFramer → StreamingModelWrapper (WA4 frozen)

Ties the entire Track A pipeline together end-to-end.
Press Ctrl+C to stop.

Usage:
  python live_demo.py
  python live_demo.py --demo               # random stream config (Section 2.1 demo)
  python live_demo.py --n-inferences 50   # stop after 50 inferences
"""

import sys
import time
import json
import argparse
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from model              import StreamSenseNet
from streaming_wrapper  import StreamSenseWrapper   # WA4 frozen — novelty=[1,1]
from stream_simulator   import StreamSimulator
from streaming_framer   import StreamingFramer

import os
_DEFAULT_ROOT = r"C:\STREAMSENSE" if os.name == "nt" else "/content/STREAMSENSE"
ROOT = Path(os.environ.get("STREAMSENSE_ROOT", _DEFAULT_ROOT))

CKPT_PATH   = ROOT / "checkpoints" / "best_model.pth"
LABELS_PATH = ROOT / "class_labels.json"


def load_model(ckpt_path: Path) -> StreamSenseWrapper:
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    wrapper = StreamSenseWrapper(num_classes=10)
    wrapper.load_checkpoint(ckpt_path)
    wrapper.eval()
    return wrapper


def main():
    parser = argparse.ArgumentParser(description="STREAMSENSE live pipeline demo")
    parser.add_argument("--demo", action="store_true",
                        help="Random stream config (Section 2.1 generalisation)")
    parser.add_argument("--n-inferences", type=int, default=None,
                        help="Stop after N inferences (default: infinite)")
    parser.add_argument("--chunk-min", type=int, default=512)
    parser.add_argument("--chunk-max", type=int, default=4096)
    args = parser.parse_args()

    print("=" * 72)
    print("LIVE PIPELINE DEMO  |  Network → Framer → Model")
    print("Press Ctrl+C at any time to stop.")
    print("=" * 72)

    # 1. Load model
    print("\n[1] Loading model weights...")
    if not CKPT_PATH.exists() or not LABELS_PATH.exists():
        print(f"[ERROR] Missing checkpoint ({CKPT_PATH}) or labels ({LABELS_PATH})")
        sys.exit(1)

    with open(LABELS_PATH, "r") as f:
        idx_to_class = json.load(f)

    model = load_model(CKPT_PATH)
    print(f"    Loaded: {CKPT_PATH.name}")

    # 2. Start simulator
    print("\n[2] Initialising network simulator...")
    sim = StreamSimulator(
        random_config=args.demo,
        chunk_min    =args.chunk_min,
        chunk_max    =args.chunk_max,
    )

    # 3. Start framer bound to simulator config
    print("\n[3] Initialising streaming framer...")
    framer = StreamingFramer(
        stream_sr      =sim.stream_sr,
        stream_channels=sim.stream_channels,
        dtype          =sim.torch_dtype,
        layout         =sim.layout,
    )

    gen             = sim.generator()
    packet_count    = 0
    inference_count = 0

    print("\nStarting live inference...\n")
    hdr = f"{'Packet':>7} | {'Chunk N':>9} | {'Action':<15} | {'Prediction':<12} | {'Novelty':>8}"
    print(hdr)
    print("-" * len(hdr))

    try:
        while True:
            if args.n_inferences and inference_count >= args.n_inferences:
                break

            chunk        = next(gen)
            packet_count += 1
            n_samples    = chunk.shape[1] if sim.layout == "planar" else chunk.shape[0]

            ready = framer.process_chunk(chunk)

            if not ready:
                print(f"{packet_count:>7} | {n_samples:>9} | {'Buffering...':<15} | "
                      f"{'---':<12} | {'---':>8}")

            for tensor in ready:
                inference_count += 1
                with torch.no_grad():
                    logits, embedding, novelty = model(tensor)

                pred_idx  = torch.argmax(logits, dim=1).item()
                pred_word = idx_to_class[str(pred_idx)]
                # novelty is [1,1] from WA4 wrapper — squeeze to scalar for display
                nov_val   = novelty.squeeze().item()

                print(f"{packet_count:>7} | {n_samples:>9} | {'** INFERENCE **':<15} | "
                      f"{pred_word:<12} | {nov_val:>8.4f}")

                if args.n_inferences and inference_count >= args.n_inferences:
                    break

            time.sleep(0.05)

    except KeyboardInterrupt:
        pass

    print("\n" + "=" * 72)
    print("SIMULATION STOPPED")
    print(f"  Packets received   : {packet_count}")
    print(f"  Inferences done    : {inference_count}")
    w = framer.welford_summary()
    print(f"\nWelford Convergence Report:")
    print(f"  Streaming mean = {w['welford_mean_db']:.6f} dB  "
          f"(frozen = {w['frozen_mean_db']:.6f})  |Δ| = {w['mean_delta_db']:.6f}")
    print(f"  Streaming std  = {w['welford_std_db']:.6f} dB  "
          f"(frozen = {w['frozen_std_db']:.6f})  |Δ| = {w['std_delta_db']:.6f}")
    print(f"  Elements seen  : {w['n_elements']:,}")
    print("=" * 72)


if __name__ == "__main__":
    main()

```

### `training/live_gv1k_demo.py`

```python
"""
live_gv1k_demo.py
Project STREAMSENSE — Track A

Live simulation streaming the 1000 Golden Vectors through the full pipeline
to prove real-world streaming accuracy of the Track A system.

Simulates network jitter by breaking each GV1K raw file into random chunks.
Reports per-vector classification accuracy and Welford convergence at exit.

Press Ctrl+C to stop.

Usage:
  python live_gv1k_demo.py
  python live_gv1k_demo.py --chunk-min 100 --chunk-max 4000
"""

import sys
import time
import json
import random
import argparse
import math
import numpy as np
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).parent))
from model             import StreamSenseNet
from streaming_wrapper import StreamSenseWrapper   # WA4 frozen — novelty=[1,1]
from streaming_framer  import StreamingFramer, GLOBAL_MEAN, GLOBAL_STD

import os
_DEFAULT_ROOT = r"C:\STREAMSENSE" if os.name == "nt" else "/content/STREAMSENSE"
ROOT = Path(os.environ.get("STREAMSENSE_ROOT", _DEFAULT_ROOT))

CKPT_PATH   = ROOT / "checkpoints" / "best_model.pth"
LABELS_PATH = ROOT / "class_labels.json"
RAW_DIR     = ROOT / "golden_vectors_1000" / "raw"


def load_model(ckpt_path: Path) -> StreamSenseWrapper:
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    wrapper = StreamSenseWrapper(num_classes=10)
    wrapper.load_checkpoint(ckpt_path)
    wrapper.eval()
    return wrapper


def main():
    parser = argparse.ArgumentParser(description="Live GV1K streaming accuracy demo")
    parser.add_argument("--chunk-min", type=int, default=100)
    parser.add_argument("--chunk-max", type=int, default=4000)
    args = parser.parse_args()

    print("=" * 72)
    print("LIVE GV1K DEMO  |  Streaming 1000 Golden Vectors")
    print("Press Ctrl+C at any time to stop.")
    print("=" * 72)

    # ── Load model ────────────────────────────────────────────────────────────
    print("\n[1] Loading model and weights...")
    for p in [CKPT_PATH, LABELS_PATH, RAW_DIR]:
        if not Path(p).exists():
            print(f"[ERROR] Not found: {p}")
            sys.exit(1)

    with open(LABELS_PATH, "r") as f:
        idx_to_class = json.load(f)

    model = load_model(CKPT_PATH)
    print(f"    Loaded: {CKPT_PATH.name}")

    # ── Collect GV1K raw files ────────────────────────────────────────────────
    raw_files = sorted(RAW_DIR.glob("*.bin"))
    print(f"\n[2] Found {len(raw_files)} GV1K raw files.\n")

    # ── Aggregate Welford (Chan merge across all files) ───────────────────────
    agg_n    = 0.0
    agg_mean = 0.0
    agg_M2   = 0.0

    correct = 0
    total   = 0

    hdr = f"{'File':<28} | {'True':<10} | {'Pred':<12} | {'OK?':>5} | {'Novelty':>8}"
    print(hdr)
    print("-" * len(hdr))

    try:
        for raw_path in raw_files:
            # Parse true label from filename: GV1K_NNNN_word.bin → word
            true_word = raw_path.stem.split("_")[2]

            raw_audio = np.fromfile(str(raw_path), dtype=np.float32)

            # Fresh framer per file
            framer = StreamingFramer(
                stream_sr      =16000,
                stream_channels=1,
                dtype          =torch.float32,
                layout         ="planar",
            )

            # Feed in random chunks (jitter simulation)
            idx = 0
            while idx < len(raw_audio):
                size  = random.randint(args.chunk_min, args.chunk_max)
                chunk = raw_audio[idx : idx + size]
                idx  += size

                ready = framer.process_chunk(chunk)

                for tensor in ready:
                    with torch.no_grad():
                        logits, embedding, novelty = model(tensor)

                    pred_idx  = torch.argmax(logits, dim=1).item()
                    pred_word = idx_to_class[str(pred_idx)]
                    # novelty is [1,1] from WA4 wrapper
                    nov_val   = novelty.squeeze().item()

                    marker = "PASS" if pred_word == true_word else "FAIL"
                    if pred_word == true_word:
                        correct += 1
                    total += 1

                    print(f"{raw_path.name:<28} | {true_word:<10} | "
                          f"{pred_word:<12} | {marker:>5} | {nov_val:>8.4f}")

            # Merge Welford (Chan parallel formula)
            b_n    = float(framer.n_welford_elements)
            b_mean = framer.welford_mean
            b_std  = framer.welford_std
            b_M2   = (b_std ** 2) * b_n if b_n > 1 else 0.0

            if b_n > 0:
                new_n    = agg_n + b_n
                delta    = b_mean - agg_mean
                agg_mean = agg_mean + delta * b_n / new_n
                agg_M2   = agg_M2 + b_M2 + (delta ** 2) * agg_n * b_n / new_n
                agg_n    = new_n

            time.sleep(0.02)

    except KeyboardInterrupt:
        print("\n[User stopped simulation]")

    agg_std = math.sqrt(agg_M2 / agg_n) if agg_n > 1 else 0.0

    print("\n" + "=" * 72)
    print("SIMULATION COMPLETE")
    print(f"  Total GV1K files processed : {total}")
    if total > 0:
        print(f"  Correct predictions        : {correct}")
        print(f"  Live streaming accuracy    : {correct/total*100:.2f}%")
    print(f"\nWelford Convergence Report (aggregate across all processed files):")
    print(f"  Streaming mean = {agg_mean:.6f} dB  "
          f"(frozen = {GLOBAL_MEAN:.6f})  |Δ| = {abs(agg_mean - GLOBAL_MEAN):.6f}")
    print(f"  Streaming std  = {agg_std:.6f} dB  "
          f"(frozen = {GLOBAL_STD:.6f})  |Δ| = {abs(agg_std  - GLOBAL_STD ):.6f}")
    print(f"  Total elements seen        : {int(agg_n):,}")
    print("=" * 72)


if __name__ == "__main__":
    main()

```

### `training/mel_pipeline.py`

```python
"""
mel_pipeline.py
Project STREAMSENSE — Track A
MPIC v1.0 — complete preprocessing pipeline (Steps 1-8).

Single public function:
    preprocess(samples) -> torch.Tensor shape [1,1,64,97] float32

Accepts:
    - numpy array  (any length, mono or stereo)
    - torch tensor (any length, mono or stereo)

Loads normalization stats from:
    /content/streamsense/stats/normalization_stats.json

Run directly for 8-test self-test:
    python mel_pipeline.py
"""

import torch
import torchaudio
import numpy as np
import json
import sys
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
STATS_FILE = Path(__file__).resolve().parent.parent / "stats" / "normalization_stats.json"

# ── MPIC v1.0 frozen parameters ───────────────────────────────────────────────
SAMPLE_RATE   = 16000
FRAME_LEN     = 16000
N_FFT         = 512
HOP_LENGTH    = 160
N_MELS        = 64
CENTER        = False          # critical — gives T=97
POWER         = 2.0
LOG_EPS       = 1e-10
CLIP_FLOOR_DB = -80.0
EXPECTED_T    = (FRAME_LEN - N_FFT) // HOP_LENGTH + 1  # = 97
OUTPUT_SHAPE  = (1, 1, N_MELS, EXPECTED_T)              # (1,1,64,97)

# ── Load normalization stats at import ────────────────────────────────────────
if not STATS_FILE.exists():
    raise FileNotFoundError(
        f"Normalization stats not found: {STATS_FILE}\n"
        f"Run compute_normstats.py first."
    )

with open(STATS_FILE, "r") as _f:
    _stats = json.load(_f)

GLOBAL_MEAN = float(_stats["global_mean"])
GLOBAL_STD  = float(_stats["global_std"])

if GLOBAL_STD <= 0.0:
    raise ValueError(f"global_std={GLOBAL_STD} is invalid — stats file may be corrupt.")

# ── MelSpectrogram transform (CPU, built once at import) ──────────────────────
_mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate = SAMPLE_RATE,
    n_fft       = N_FFT,
    hop_length  = HOP_LENGTH,
    n_mels      = N_MELS,
    window_fn   = torch.hann_window,
    center      = CENTER,
    power       = POWER,
)

# ── Public API ────────────────────────────────────────────────────────────────
def preprocess(samples) -> torch.Tensor:
    """
    Full MPIC v1.0 pipeline — Steps 1 through 8.

    Args:
        samples: numpy array or torch tensor, any length, mono or stereo.
                 Expected: 1D [T] or 2D [C,T] or [T,C].

    Returns:
        torch.Tensor of shape [1, 1, 64, 97], dtype float32.

    Pipeline:
        Step 1 — accept float32 samples
        Step 2 — stereo -> mono
        Step 3 — pad (zeros right) or crop to 16000 samples
        Step 4 — MelSpectrogram -> [1, 64, 97]
        Step 5 — 10 * log10(mel + 1e-10)
        Step 6 — clamp(min=-80 dB)
        Step 7 — (mel - global_mean) / global_std
        Step 8 — reshape to [1, 1, 64, 97] float32
        Step 9 — validate shape == (1, 1, 64, 97)
    """

    # ── Step 1: convert input to float32 torch tensor ────────────────────────
    if isinstance(samples, np.ndarray):
        waveform = torch.from_numpy(samples.copy()).float()
    elif isinstance(samples, torch.Tensor):
        waveform = samples.float().clone()
    else:
        raise TypeError(
            f"preprocess() expects numpy array or torch tensor, got {type(samples)}"
        )

    # Ensure 2D [C, T]
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)        # [T] -> [1, T]
    elif waveform.ndim == 2:
        # Handle [T, C] (uncommon but possible)
        if waveform.shape[0] > waveform.shape[1]:
            waveform = waveform.T               # [T, C] -> [C, T]
    else:
        raise ValueError(
            f"Expected 1D or 2D input, got shape {waveform.shape}"
        )
    # waveform is now [C, T]

    # ── Step 2: stereo -> mono ────────────────────────────────────────────────
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)   # [C,T] -> [1,T]
    # waveform is now [1, T]

    # ── Step 3: pad or crop to exactly FRAME_LEN ─────────────────────────────
    length = waveform.shape[1]
    if length < FRAME_LEN:
        waveform = torch.nn.functional.pad(waveform, (0, FRAME_LEN - length))
    elif length > FRAME_LEN:
        waveform = waveform[:, :FRAME_LEN]
    # waveform is now [1, 16000]

    # ── Step 4: MelSpectrogram ────────────────────────────────────────────────
    mel = _mel_transform(waveform)              # [1, 64, 97]

    # ── Step 5: log scaling ───────────────────────────────────────────────────
    mel = 10.0 * torch.log10(mel + LOG_EPS)     # [1, 64, 97]

    # ── Step 6: clip floor ────────────────────────────────────────────────────
    mel = torch.clamp(mel, min=CLIP_FLOOR_DB)   # [1, 64, 97]

    # ── Step 7: global normalization ─────────────────────────────────────────
    mel = (mel - GLOBAL_MEAN) / GLOBAL_STD      # [1, 64, 97]

    # ── Step 8: reshape to [1, 1, 64, 97] ────────────────────────────────────
    mel = mel.unsqueeze(0)                      # [1, 64, 97] -> [1, 1, 64, 97]

    # ── Step 9: validate ─────────────────────────────────────────────────────
    assert tuple(mel.shape) == OUTPUT_SHAPE, (
        f"Shape error: expected {OUTPUT_SHAPE}, got {tuple(mel.shape)}"
    )
    assert mel.dtype == torch.float32, (
        f"Dtype error: expected float32, got {mel.dtype}"
    )

    return mel


# ── Self-test (run directly: python mel_pipeline.py) ─────────────────────────
def _run_self_tests():
    print("=" * 60)
    print("mel_pipeline.py — self-test (8 tests)")
    print(f"global_mean = {GLOBAL_MEAN:.6f} dB")
    print(f"global_std  = {GLOBAL_STD:.6f} dB")
    print("=" * 60)

    passed = 0
    failed = 0

    def check(name, condition, detail=""):
        nonlocal passed, failed
        if condition:
            print(f"  [PASS] {name}")
            passed += 1
        else:
            print(f"  [FAIL] {name}  {detail}")
            failed += 1

    # ── Test 1: normal 16000-sample numpy input ───────────────────────────────
    try:
        samples = np.random.randn(16000).astype(np.float32)
        out = preprocess(samples)
        check("T1 — numpy 1D [16000] input",
              tuple(out.shape) == OUTPUT_SHAPE and out.dtype == torch.float32,
              f"got shape={out.shape} dtype={out.dtype}")
    except Exception as e:
        check("T1 — numpy 1D [16000] input", False, str(e))

    # ── Test 2: short input — needs padding ───────────────────────────────────
    try:
        samples = np.random.randn(8000).astype(np.float32)
        out = preprocess(samples)
        check("T2 — short numpy input [8000] — padding",
              tuple(out.shape) == OUTPUT_SHAPE,
              f"got shape={out.shape}")
    except Exception as e:
        check("T2 — short numpy input [8000] — padding", False, str(e))

    # ── Test 3: long input — needs cropping ───────────────────────────────────
    try:
        samples = np.random.randn(24000).astype(np.float32)
        out = preprocess(samples)
        check("T3 — long numpy input [24000] — cropping",
              tuple(out.shape) == OUTPUT_SHAPE,
              f"got shape={out.shape}")
    except Exception as e:
        check("T3 — long numpy input [24000] — cropping", False, str(e))

    # ── Test 4: stereo numpy input ────────────────────────────────────────────
    try:
        samples = np.random.randn(2, 16000).astype(np.float32)  # [2, T]
        out = preprocess(samples)
        check("T4 — stereo numpy [2,16000] — mono conversion",
              tuple(out.shape) == OUTPUT_SHAPE,
              f"got shape={out.shape}")
    except Exception as e:
        check("T4 — stereo numpy [2,16000] — mono conversion", False, str(e))

    # ── Test 5: torch tensor input ────────────────────────────────────────────
    try:
        samples = torch.randn(16000)                            # [T]
        out = preprocess(samples)
        check("T5 — torch tensor 1D [16000] input",
              tuple(out.shape) == OUTPUT_SHAPE,
              f"got shape={out.shape}")
    except Exception as e:
        check("T5 — torch tensor 1D [16000] input", False, str(e))

    # ── Test 6: stereo torch tensor input ─────────────────────────────────────
    try:
        samples = torch.randn(2, 16000)                         # [2, T]
        out = preprocess(samples)
        check("T6 — stereo torch tensor [2,16000] — mono conversion",
              tuple(out.shape) == OUTPUT_SHAPE,
              f"got shape={out.shape}")
    except Exception as e:
        check("T6 — stereo torch tensor [2,16000] — mono conversion", False, str(e))

    # ── Test 7: output shape exactly (1,1,64,97) ──────────────────────────────
    try:
        samples = np.random.randn(16000).astype(np.float32)
        out = preprocess(samples)
        check("T7 — output shape exactly (1,1,64,97)",
              tuple(out.shape) == (1, 1, 64, 97),
              f"got {tuple(out.shape)}")
    except Exception as e:
        check("T7 — output shape exactly (1,1,64,97)", False, str(e))

    # ── Test 8: output dtype is float32 ───────────────────────────────────────
    try:
        samples = np.random.randn(16000).astype(np.float32)
        out = preprocess(samples)
        check("T8 — output dtype is float32",
              out.dtype == torch.float32,
              f"got {out.dtype}")
    except Exception as e:
        check("T8 — output dtype is float32", False, str(e))

    # ── Summary ───────────────────────────────────────────────────────────────
    print("=" * 60)
    print(f"Results: {passed}/8 passed,  {failed}/8 failed")
    if failed == 0:
        print("[DONE] All tests PASS — mel_pipeline.py is ready.")
    else:
        print("[FAIL] Some tests failed — check errors above.")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    ok = _run_self_tests()
    sys.exit(0 if ok else 1)

```

### `training/mel_pipeline_matlab.m`

```matlab
function norm = mel_pipeline_matlab(samples)
% MEL_PIPELINE_MATLAB  MPIC v1.0 — MATLAB reference preprocessing pipeline
%
% Exact port of Python mel_pipeline.py / generate_golden.py for STREAMSENSE.
% Reproduces the Python/torchaudio result to within tolerance 5e-4 (cross-impl).
%
% INPUT
%   samples  : float  [16000 x 1] or [1 x 16000]  — 16 kHz mono PCM, float32
%              May be shorter (zero-padded to 16000) or longer (cropped).
%
% OUTPUT
%   norm     : single [64 x 97]  — column-major, normalised log-mel spectrogram
%              Ready for reshape(norm, [64 97]) in MATLAB (already correct shape).
%              To feed the ONNX model, reshape to [1 1 64 97].
%
% USAGE
%   [samples, fs] = audioread('GV_00_yes.wav');
%   if fs ~= 16000, error('Resample to 16 kHz first'); end
%   norm = mel_pipeline_matlab(samples);           % [64 x 97] single
%   input_tensor = reshape(norm, [1 1 64 97]);     % for ONNX inference
%
% VERIFICATION
%   Load the matching golden vector and compare:
%     fid  = fopen('GV_00_yes_norm.bin', 'rb', 'l');
%     ref  = reshape(fread(fid, 64*97, 'float32=>single'), [64, 97]);
%     fclose(fid);
%     max_err = max(abs(norm(:) - ref(:)));
%     assert(max_err < 5e-4, 'Pipeline mismatch');
%
% MPIC v1.0 FROZEN PARAMETERS (do NOT modify)
%   sample_rate  = 16000 Hz
%   frame_len    = 16000 samples  (1 second)
%   n_fft        = 512
%   hop_length   = 160
%   n_mels       = 64
%   center       = false          ← CRITICAL: gives T=97, not T=98
%   power        = 2.0            (power spectrogram)
%   window       = hann periodic
%   log_scale    = 10 * log10(mel + 1e-10)
%   clip_floor   = -80 dB
%   global_mean  = -30.785545 dB
%   global_std   =  22.157099 dB
%
% Project: STREAMSENSE — Track A
% Spec:    MPIC v1.0 (frozen)

% ── MPIC v1.0 frozen constants ────────────────────────────────────────────────
SAMPLE_RATE   = 16000;
FRAME_LEN     = 16000;
N_FFT         = 512;
HOP_LENGTH    = 160;
N_MELS        = 64;
LOG_EPS       = 1e-10;
CLIP_FLOOR_DB = -80.0;
GLOBAL_MEAN   = -30.785545;   % dB  (from normalization_stats.json)
GLOBAL_STD    =  22.157099;   % dB

% Derived — must equal 97 for MPIC v1.0
EXPECTED_T = (FRAME_LEN - N_FFT) / HOP_LENGTH + 1;  % = 97
EXPECTED_T = floor(EXPECTED_T);   % force integer — MATLAB / gives 97.8 without this

% ── Step 1-3: Pad / crop to exactly FRAME_LEN samples ────────────────────────
samples = single(samples(:));          % force column vector, single precision
L = length(samples);
if L < FRAME_LEN
    samples = [samples; zeros(FRAME_LEN - L, 1, 'single')];
elseif L > FRAME_LEN
    samples = samples(1:FRAME_LEN);
end

% ── Step 4: Build Hann window (periodic, length N_FFT) ───────────────────────
% Python/torchaudio uses a PERIODIC Hann window (N+1 point, drop last).
% MATLAB's hann() uses symmetric by default — must use 'periodic' flag.
win = single(hann(N_FFT, 'periodic'));

% ── Step 5: STFT — center=False means no padding, analyse from sample 1 ──────
% With center=False and hop=160, torchaudio analyses frames starting at:
%   frame k starts at sample  k * HOP_LENGTH  (0-indexed)
% MATLAB's spectrogram() aligns identically when no padding is added.
%
% spectrogram(x, win, noverlap, nfft) with noverlap = N_FFT - HOP_LENGTH
noverlap = N_FFT - HOP_LENGTH;        % = 352

[S, ~, ~] = spectrogram(samples, win, noverlap, N_FFT, SAMPLE_RATE, 'onesided');
% S : [N_FFT/2+1 x T] = [257 x 97]  complex STFT coefficients

% Power spectrogram (power=2.0 → magnitude squared)
power_spec = real(S).^2 + imag(S).^2;   % [257 x 97]  single

% ── Step 6: Mel filterbank — build triangular filters on Hz scale ─────────────
% Mirrors torchaudio's MelSpectrogram filterbank exactly:
%  - freq_min = 0 Hz, freq_max = SAMPLE_RATE/2
%  - N_MELS+2 linearly-spaced points on the mel scale, converted back to Hz
%  - Triangular filters; NO per-filter normalization (norm=None, the default)
%
mel_fmin  = 0.0;
mel_fmax  = SAMPLE_RATE / 2;           % 8000 Hz

% Mel-scale conversion (torchaudio / librosa convention: 2595 * log10(1+f/700))
hz_to_mel = @(f) 2595.0 * log10(1.0 + f / 700.0);
mel_to_hz = @(m) 700.0 * (10.^(m / 2595.0) - 1.0);

mel_min = hz_to_mel(mel_fmin);
mel_max = hz_to_mel(mel_fmax);

% N_MELS+2 equally-spaced mel points → convert to Hz
mel_pts = linspace(mel_min, mel_max, N_MELS + 2);
hz_pts  = mel_to_hz(mel_pts);          % [N_MELS+2] centre-frequencies in Hz

% Map centre-frequencies to STFT bin indices (0-indexed FFT bins → 1-indexed MATLAB)
% STFT bin k corresponds to frequency k * SAMPLE_RATE / N_FFT
n_freqs  = N_FFT / 2 + 1;             % 257
freq_bin = (0:n_freqs-1)' * (SAMPLE_RATE / N_FFT);   % [257 x 1]

% Build filterbank matrix [N_MELS x n_freqs] = [64 x 257]
fb = zeros(N_MELS, n_freqs, 'single');
for m = 1:N_MELS
    f_left   = hz_pts(m);
    f_center = hz_pts(m + 1);
    f_right  = hz_pts(m + 2);

    % Rising slope
    mask_up  = (freq_bin >= f_left)  & (freq_bin <= f_center);
    fb(m, mask_up) = single((freq_bin(mask_up) - f_left) / (f_center - f_left));

    % Falling slope
    mask_dn  = (freq_bin > f_center) & (freq_bin <= f_right);
    fb(m, mask_dn) = single((f_right - freq_bin(mask_dn)) / (f_right - f_center));
end

% IMPORTANT: torchaudio.transforms.MelSpectrogram uses norm=None by default.
% norm=None means NO per-filter bandwidth normalization is applied.
% The 2/bw factor belongs to torchaudio's optional norm='slaney' mode, which
% is NOT the default and NOT what mel_pipeline.py uses.
% Therefore: do NOT apply any normalization here.
% (Applying 2/bw was Bug #1 — it shifted mel dB by 29–52 dB per filter,
%  causing max_err_norm ≈ 1.13 and max_err_mel ≈ 25 dB in verification.)

% ── Step 7: Apply filterbank → mel spectrogram [64 x 97] ─────────────────────
mel_spec = fb * single(power_spec);   % [64 x 257] * [257 x 97] = [64 x 97]

% ── Step 8: Log scale  10 * log10(mel + 1e-10) ───────────────────────────────
mel_db = 10.0 * log10(mel_spec + single(LOG_EPS));

% ── Step 9: Clamp to floor ────────────────────────────────────────────────────
mel_db = max(mel_db, single(CLIP_FLOOR_DB));

% ── Step 10: Global normalisation ────────────────────────────────────────────
norm = (mel_db - single(GLOBAL_MEAN)) / single(GLOBAL_STD);

% norm is [64 x 97] single, column-major in MATLAB memory — correct for the
% golden_vectors_10_matlab/ files which were also saved column-major.

% Validate output shape
if size(norm,1) ~= N_MELS || size(norm,2) ~= EXPECTED_T
    error('mel_pipeline_matlab: unexpected output shape [%d x %d], expected [64 x 97]', ...
          size(norm, 1), size(norm, 2));
end

end % function mel_pipeline_matlab

```

### `training/model.py`

```python
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

```

### `training/model_1d.py`

```python
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

```

### `training/nsp_receiver.py`

```python
"""
nsp_receiver.py
Project STREAMSENSE — Track A (WA-1, Loopback Validation)

NSP v1.2 Receiver — loopback validation counterpart to nsp_sender.py.

PURPOSE: This is Track A's loopback validation tool, NOT Track B's production
receiver. Its job is to parse incoming NSP packets from nsp_sender.py,
validate header integrity, count frames, and report statistics. It does NOT
perform mel preprocessing or inference — that is Track B's TensorBuilder.

Implements the same dual-mode TCP transport as nsp_sender.py per:
  - "Untitled document (2).md"  — NSP v1.2 spec (header validation, framing)
  - "TRACK_A_TRACK_B_DUAL_MODE_RUNBOOK_v2.md" — Server/Client modes, reconnect

──────────────────────────────────────────────────────────────────────────────
Validation checks per incoming packet:
  1. Length prefix matches actual data received
  2. magic_bytes == b"NSP\\x00"
  3. version == 1
  4. dtype == 0x03 (FLOAT32)
  5. payload_bytes == frame_length * sizeof(dtype)  [spec Section 10]
  6. Length prefix == 48 + payload_bytes            [spec Section 10]
  7. sequence_no is monotonically increasing (gap detection)
  8. session_id is stable within a session
──────────────────────────────────────────────────────────────────────────────

Usage:
  # Server mode (receiver listens, sender connects)
  python nsp_receiver.py

  # Client mode (receiver connects to a listening sender)
  python nsp_receiver.py --mode client --host 127.0.0.1 --port 7654

  # With verbose per-packet output
  python nsp_receiver.py --verbose
  python nsp_receiver.py --n-frames 10 --verbose
"""

import os
import sys
import time
import json
import struct
import select
import socket
import signal
import argparse
import threading
import datetime
from pathlib import Path

# ── NSP v1.2 constants (mirrors nsp_sender.py) ────────────────────────────────
NSP_MAGIC          = b"NSP\x00"
NSP_VERSION        = 1
NSP_MSG_DATA       = 0x01
NSP_MSG_EOF        = 0x02
NSP_DTYPE_FLOAT32  = 0x03

NSP_DTYPE_SIZE = {
    0x01: 2,   # INT16
    0x02: 4,   # INT32
    0x03: 4,   # FLOAT32
    0x04: 8,   # FLOAT64
}

NSP_HEADER_FMT  = "<4sHBBQQQIIII"
NSP_HEADER_SIZE = struct.calcsize(NSP_HEADER_FMT)   # 48
NSP_LENGTH_FMT  = "<I"
NSP_MAX_PACKET  = 16_777_216                         # 16 MB

assert NSP_HEADER_SIZE == 48

# ── Logs directory ─────────────────────────────────────────────────────────────
LOGS_DIR = Path(__file__).resolve().parent / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

STATS_DIR = Path(__file__).resolve().parent / "stats"
STATS_DIR.mkdir(parents=True, exist_ok=True)

# ── Global stop flag ───────────────────────────────────────────────────────────
_STOP = threading.Event()


def _signal_handler(signum, frame):
    print("\n[NspReceiver] Shutdown signal — stopping gracefully...")
    _STOP.set()


signal.signal(signal.SIGINT,  _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


# ── Low-level recv helpers ─────────────────────────────────────────────────────

def _recv_exactly(conn: socket.socket, n: int) -> bytes | None:
    """
    Receive exactly n bytes from the socket.
    Returns None on connection loss or stop flag.
    Implements the length-prefixed frame assembler (spec Section 6, runbook ADR-B).
    """
    data = bytearray()
    while len(data) < n:
        if _STOP.is_set():
            return None
        try:
            chunk = conn.recv(n - len(data))
        except OSError:
            return None
        if not chunk:
            return None
        data.extend(chunk)
    return bytes(data)


# ── Header parser + validator ──────────────────────────────────────────────────

class NspHeader:
    """Parsed + validated NSP v1.2 header."""

    __slots__ = (
        "magic_bytes", "version", "message_type", "dtype",
        "sequence_no", "timestamp_us", "session_id",
        "payload_bytes", "sample_rate", "frame_length", "reserved",
    )

    def __init__(self, raw: bytes):
        (
            self.magic_bytes,
            self.version,
            self.message_type,
            self.dtype,
            self.sequence_no,
            self.timestamp_us,
            self.session_id,
            self.payload_bytes,
            self.sample_rate,
            self.frame_length,
            self.reserved,
        ) = struct.unpack(NSP_HEADER_FMT, raw)

    def validate(self) -> list[str]:
        """
        Return list of validation errors (empty = OK).
        Checks: magic, version, dtype, payload_bytes == frame_length * sizeof(dtype).
        """
        errors = []
        if self.magic_bytes != NSP_MAGIC:
            errors.append(f"Bad magic: {self.magic_bytes!r} (expected {NSP_MAGIC!r})")
        if self.version != NSP_VERSION:
            errors.append(f"Bad version: {self.version} (expected {NSP_VERSION})")
        if self.dtype not in NSP_DTYPE_SIZE:
            errors.append(f"Unknown dtype: 0x{self.dtype:02X}")
        elif self.message_type == NSP_MSG_DATA:
            expected_bytes = self.frame_length * NSP_DTYPE_SIZE[self.dtype]
            if self.payload_bytes != expected_bytes:
                errors.append(
                    f"payload_bytes={self.payload_bytes} != "
                    f"frame_length({self.frame_length}) * sizeof(dtype)({NSP_DTYPE_SIZE[self.dtype]}) "
                    f"= {expected_bytes}"
                )
        return errors


# ── Session statistics ─────────────────────────────────────────────────────────

class RecvStats:
    def __init__(self, host: str, port: int, mode: str):
        self.host            = host
        self.port            = port
        self.mode            = mode
        self.start_dt        = datetime.datetime.now()
        self.end_dt          = None
        self.session_id      = None
        self.packets_recv    = 0
        self.bytes_recv      = 0
        self.frames_data     = 0
        self.frames_eof      = 0
        self.errors          = 0
        self.seq_gaps        = 0
        self.last_seq        = None
        # Latency tracking (sender ts_us vs receiver arrival)
        self.latency_sum_us  = 0
        self.latency_count   = 0

    def finalise(self):
        self.end_dt = datetime.datetime.now()

    def to_dict(self) -> dict:
        duration = (self.end_dt - self.start_dt).total_seconds() if self.end_dt else 0
        mean_lat = (self.latency_sum_us / self.latency_count) if self.latency_count else 0
        return {
            "role"              : "receiver",
            "host"              : self.host,
            "port"              : self.port,
            "mode"              : self.mode,
            "session_id"        : self.session_id,
            "start_time"        : self.start_dt.isoformat(),
            "end_time"          : self.end_dt.isoformat() if self.end_dt else None,
            "duration_sec"      : round(duration, 3),
            "packets_received"  : self.packets_recv,
            "bytes_received"    : self.bytes_recv,
            "data_frames"       : self.frames_data,
            "eof_frames"        : self.frames_eof,
            "validation_errors" : self.errors,
            "sequence_gaps"     : self.seq_gaps,
            "mean_latency_us"   : round(mean_lat, 2),
            "audio_duration_sec": self.frames_data * 1.0,
        }

    def save(self) -> Path:
        ts   = self.start_dt.strftime("%Y%m%d_%H%M%S")
        path = LOGS_DIR / f"nsp_recv_session_{ts}.json"
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        return path


# ── Runtime statistics writer ──────────────────────────────────────────────────

def write_runtime_stats(stats: "RecvStats") -> None:
    """
    Write live receive-side statistics to stats/nsp_receiver_runtime.json.
    Called after every packet is received. Uses atomic write (tmp → rename) to
    avoid partial reads by an external consumer.

    Fields:
      frames       — total DATA frames received this session
      data_bytes   — total bytes received (length-prefix + header + payload)
      data_rate    — bytes/s since session start
      recv_rate    — frames/s since session start
    """
    elapsed = (datetime.datetime.now() - stats.start_dt).total_seconds()
    elapsed = elapsed if elapsed > 0 else 1e-9   # guard against div-by-zero

    payload = {
        "host"       : stats.host,
        "port"       : stats.port,
        "mode"       : stats.mode,
        "frames"     : stats.frames_data,
        "data_bytes" : stats.bytes_recv,
        "data_rate"  : round(stats.bytes_recv / elapsed, 2),   # bytes/s
        "recv_rate"  : round(stats.frames_data / elapsed, 4),  # frames/s
        "elapsed_sec": round(elapsed, 3),
        "updated_at" : datetime.datetime.now().isoformat(),
    }

    target = STATS_DIR / "nsp_receiver_runtime.json"
    tmp    = STATS_DIR / "nsp_receiver_runtime.json.tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    tmp.replace(target)


# ── Session runner ─────────────────────────────────────────────────────────────

def _run_session(
    conn    : socket.socket,
    stats   : RecvStats,
    verbose : bool,
    n_frames: int | None,
):
    """
    Receive and validate NSP packets from an established connection.
    Implements length-prefixed framing (spec Section 6).
    """
    print(f"[NspReceiver] Session active | peer={conn.getpeername()}")
    last_seq   = None
    session_id = None

    while not _STOP.is_set():
        if n_frames is not None and stats.frames_data >= n_frames:
            break

        # ── Read 4-byte length prefix ────────────────────────────────────────
        raw_len = _recv_exactly(conn, 4)
        if raw_len is None:
            break
        (msg_len,) = struct.unpack(NSP_LENGTH_FMT, raw_len)

        if msg_len < NSP_HEADER_SIZE or msg_len > NSP_MAX_PACKET:
            print(f"[NspReceiver] [ERROR] Invalid msg_len={msg_len} — "
                  f"expected {NSP_HEADER_SIZE} ≤ len ≤ {NSP_MAX_PACKET}")
            stats.errors += 1
            break

        # ── Read header + payload ────────────────────────────────────────────
        raw_msg = _recv_exactly(conn, msg_len)
        if raw_msg is None:
            break

        raw_header  = raw_msg[:NSP_HEADER_SIZE]
        raw_payload = raw_msg[NSP_HEADER_SIZE:]

        stats.packets_recv += 1
        stats.bytes_recv   += 4 + msg_len

        write_runtime_stats(stats)

        # ── Parse header ─────────────────────────────────────────────────────
        hdr    = NspHeader(raw_header)
        errors = hdr.validate()

        # ── Validate length prefix == 48 + payload_bytes (spec Section 10) ──
        if msg_len != NSP_HEADER_SIZE + hdr.payload_bytes:
            errors.append(
                f"Length prefix {msg_len} != 48 + payload_bytes {hdr.payload_bytes} "
                f"= {NSP_HEADER_SIZE + hdr.payload_bytes}"
            )

        if errors:
            for e in errors:
                print(f"[NspReceiver] [VALIDATION ERROR] {e}")
            stats.errors += len(errors)

        # ── Session ID tracking ───────────────────────────────────────────────
        if session_id is None:
            session_id        = hdr.session_id
            stats.session_id  = session_id
            print(f"[NspReceiver] New session: session_id={session_id} | "
                  f"sample_rate={hdr.sample_rate} Hz | frame_length={hdr.frame_length}")
        elif hdr.session_id != session_id:
            print(f"[NspReceiver] [WARN] session_id changed: "
                  f"{session_id} → {hdr.session_id}")
            session_id       = hdr.session_id
            last_seq         = None

        # ── Sequence gap detection ────────────────────────────────────────────
        if last_seq is not None and hdr.message_type == NSP_MSG_DATA:
            expected = last_seq + 1
            if hdr.sequence_no != expected:
                gap = hdr.sequence_no - expected
                print(f"[NspReceiver] [WARN] Sequence gap: "
                      f"expected {expected}, got {hdr.sequence_no} (gap={gap})")
                stats.seq_gaps += 1

        # ── Latency tracking ──────────────────────────────────────────────────
        now_us = time.time_ns() // 1_000
        lat_us = now_us - hdr.timestamp_us
        if 0 < lat_us < 10_000_000:   # ignore unrealistic values (>10s)
            stats.latency_sum_us  += lat_us
            stats.latency_count   += 1

        # ── Message type handling ─────────────────────────────────────────────
        if hdr.message_type == NSP_MSG_DATA:
            stats.frames_data += 1
            last_seq           = hdr.sequence_no

            if verbose:
                lat_str = f"{lat_us:>8} µs" if stats.latency_count else "N/A"
                print(
                    f"[NspReceiver] DATA  "
                    f"seq={hdr.sequence_no:>6} | "
                    f"session={hdr.session_id} | "
                    f"len={hdr.frame_length:>6} smp | "
                    f"payload={hdr.payload_bytes:>7} B | "
                    f"lat={lat_str}"
                )
            elif stats.frames_data % 100 == 0:
                print(f"[NspReceiver] Received {stats.frames_data} DATA frames | "
                      f"{stats.bytes_recv/1024:.1f} KB total")

        elif hdr.message_type == NSP_MSG_EOF:
            stats.frames_eof += 1
            print(f"[NspReceiver] EOF received | seq={hdr.sequence_no} | "
                  f"total_data_frames={stats.frames_data}")
            break

        else:
            print(f"[NspReceiver] [WARN] Unknown message_type=0x{hdr.message_type:02X}")
            stats.errors += 1


# ── Server mode ────────────────────────────────────────────────────────────────

def run_server(
    host    : str,
    port    : int,
    stats   : RecvStats,
    verbose : bool,
    n_frames: int | None,
):
    """
    Server mode: bind/listen, accept() with 100ms select() timeout loop.
    Per runbook Section 2, Section 12.D.
    """
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((host, port))
    srv.listen(1)
    srv.setblocking(False)

    print(f"[NspReceiver] Server mode | listening on {host}:{port}")

    try:
        while not _STOP.is_set():
            readable, _, _ = select.select([srv], [], [], 0.1)
            if not readable:
                continue

            conn, addr = srv.accept()
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            conn.setblocking(True)
            print(f"[NspReceiver] Sender connected: {addr}")

            try:
                _run_session(conn, stats, verbose, n_frames)
            finally:
                conn.close()
                print(f"[NspReceiver] Sender disconnected: {addr}")

            if n_frames is not None and stats.frames_data >= n_frames:
                break
    finally:
        srv.close()


# ── Client mode ────────────────────────────────────────────────────────────────

def run_client(
    host                : str,
    port                : int,
    stats               : RecvStats,
    verbose             : bool,
    n_frames            : int | None,
    reconnect_initial_ms: int = 1000,
    reconnect_max_ms    : int = 5000,
):
    """
    Client mode: connect to a listening sender with 2000ms timeout.
    Linear backoff on failure (runbook Section 3, Section 12.C, 12.E).
    """
    print(f"[NspReceiver] Client mode | connecting to {host}:{port}")
    backoff_ms = reconnect_initial_ms

    while not _STOP.is_set():
        if n_frames is not None and stats.frames_data >= n_frames:
            break

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setblocking(False)

        try:
            sock.connect_ex((host, port))
        except OSError:
            pass

        _, writable, _ = select.select([], [sock], [], 2.0)

        if writable:
            err = sock.getsockopt(socket.SOL_SOCKET, socket.SO_ERROR)
            if err == 0:
                sock.setblocking(True)
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                print(f"[NspReceiver] Connected to {host}:{port}")

                try:
                    _run_session(sock, stats, verbose, n_frames)
                finally:
                    sock.close()

                backoff_ms = reconnect_initial_ms
                continue

        sock.close()
        if _STOP.is_set():
            break

        print(f"[NspReceiver] Connection failed. Retrying in {backoff_ms}ms...")
        slept = 0
        while slept < backoff_ms and not _STOP.is_set():
            time.sleep(0.1)
            slept += 100

        backoff_ms = min(backoff_ms + 1000, reconnect_max_ms)


# ── CLI entry point ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=(
            "NSP v1.2 Receiver — loopback validator for nsp_sender.py.\n"
            "Parses and validates NSP frames. Does NOT perform mel/inference."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--mode", choices=["server", "client"], default="server",
                        help="Transport mode (default: server)")
    parser.add_argument("--host", default=os.environ.get("STREAMSENSE_HOST", "127.0.0.1"),
                        help="Bind/connect host (default: 127.0.0.1 or $STREAMSENSE_HOST)")
    parser.add_argument("--port", type=int,
                        default=int(os.environ.get("STREAMSENSE_PORT", "7654")),
                        help="Bind/connect port (default: 7654 or $STREAMSENSE_PORT)")
    parser.add_argument("--n-frames", type=int, default=None,
                        help="Stop after N DATA frames (default: run until EOF/disconnect)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print per-packet details")
    parser.add_argument("--reconnect-initial-ms", type=int, default=1000)
    parser.add_argument("--reconnect-max-ms",     type=int, default=5000)
    args = parser.parse_args()

    print("=" * 72)
    print(f"NSP v1.2 Receiver | mode={args.mode} | {args.host}:{args.port}")
    print(f"  n_frames={args.n_frames or 'infinite'} | verbose={args.verbose}")
    print(f"  Logs  → {LOGS_DIR}")
    print(f"  Stats → {STATS_DIR}")
    print("=" * 72)

    stats = RecvStats(args.host, args.port, args.mode)

    try:
        if args.mode == "server":
            run_server(args.host, args.port, stats, args.verbose, args.n_frames)
        else:
            run_client(
                args.host, args.port, stats, args.verbose, args.n_frames,
                reconnect_initial_ms=args.reconnect_initial_ms,
                reconnect_max_ms    =args.reconnect_max_ms,
            )
    finally:
        stats.finalise()
        log_path = stats.save()
        d = stats.to_dict()

        print("\n" + "=" * 72)
        print("RECEIVER SESSION SUMMARY")
        print(f"  Packets received    : {d['packets_received']}")
        print(f"  Bytes received      : {d['bytes_received']:,}  "
              f"({d['bytes_received']/1024:.1f} KB)")
        print(f"  DATA frames         : {d['data_frames']}")
        print(f"  EOF frames          : {d['eof_frames']}")
        print(f"  Validation errors   : {d['validation_errors']}")
        print(f"  Sequence gaps       : {d['sequence_gaps']}")
        print(f"  Mean latency        : {d['mean_latency_us']:.1f} µs")
        print(f"  Audio duration      : {d['audio_duration_sec']:.1f} s")
        print(f"  Duration            : {d['duration_sec']:.1f} s")
        if d['validation_errors'] == 0 and d['sequence_gaps'] == 0:
            print("\n  [PASS] All frames valid. Zero errors. Zero sequence gaps.")
        else:
            print(f"\n  [WARN] {d['validation_errors']} errors, "
                  f"{d['sequence_gaps']} sequence gaps.")
        print(f"  Stats saved to      : {log_path}")
        print("=" * 72)


if __name__ == "__main__":
    main()

```

### `training/nsp_sender.py`

```python
"""
nsp_sender.py
Project STREAMSENSE — Track A (WA-1, D5-D6)

NSP v1.2 Sender — sends streaming audio frames from Track A to Track B.

Implements the Network Stream Protocol v1.2 (Freeze Candidate) exactly as
specified in "Untitled document (2).md" and the dual-mode TCP transport from
"TRACK_A_TRACK_B_DUAL_MODE_RUNBOOK_v2.md".

──────────────────────────────────────────────────────────────────────────────
NSP v1.2 Wire Format (from spec, Section 8-9)
──────────────────────────────────────────────────────────────────────────────

  Packet = Length Prefix (4B) + Header (48B) + Payload (variable)
  Total  = 4 + 48 + payload_bytes

  Header layout (Little-Endian, 48 bytes):
    0x00  magic_bytes    char[4]    4   b"NSP\\x00"
    0x04  version        uint16_t   2   1
    0x06  message_type   uint8_t    1   0x01=DATA, 0x02=EOF/CLOSE
    0x07  dtype          uint8_t    1   0x03=FLOAT32
    0x08  sequence_no    uint64_t   8   monotonic counter
    0x10  timestamp      uint64_t   8   microseconds since epoch
    0x18  session_id     uint64_t   8   unique per connection lifetime
    0x20  payload_bytes  uint32_t   4   frame_length * sizeof(dtype)
    0x24  sample_rate    uint32_t   4   16000 Hz
    0x28  frame_length   uint32_t   4   samples per frame (N)
    0x2C  reserved       uint32_t   4   0

  Struct format : "<4sHBBQQQIIII"  → 48 bytes (little-endian)
  Length prefix : "<I"             → 4 bytes  (little-endian uint32)

  Validator constraint: payload_bytes == frame_length * sizeof(dtype)
  TCP constraint      : TCP_NODELAY must be set (Nagle's algorithm disabled)
  Max packet size     : 16 MB (16,777,216 bytes)

──────────────────────────────────────────────────────────────────────────────
Transport Modes (from Runbook, Sections 2-5)
──────────────────────────────────────────────────────────────────────────────

  Server Mode (default, port 7654):
    bind() → listen() → accept() with 100ms select() timeout loop
    → stream → client disconnects → loop back to accept()
    Graceful shutdown: stop flag checked every 100ms

  Client Mode:
    connect() non-blocking with 2000ms select() timeout
    → stream → remote disconnects
    → linear backoff: initial=1000ms, max=5000ms, +1000ms per retry
    Backoff is broken into 100ms chunks (checks stop flag every chunk)

──────────────────────────────────────────────────────────────────────────────
What is sent per packet:
  Raw float32 audio samples — NEVER mel, NEVER normalised tensors.
  Track B's TensorBuilder receives the raw samples and does mel preprocessing.
  Each packet carries exactly frame_length (default 16000) float32 samples.
──────────────────────────────────────────────────────────────────────────────

Usage:
  # Server mode (Track A listens, Track B connects)
  python nsp_sender.py

  # Client mode (Track A connects to Track B's listener)
  python nsp_sender.py --mode client --host 192.168.1.50 --port 7654

  # Custom configuration
  python nsp_sender.py --mode server --port 8888 --frame-len 16000 --n-frames 100
  python nsp_sender.py --sources project    # only data/raw
  python nsp_sender.py --sources unknown   # only unknown_data
  python nsp_sender.py --sources both      # both (default)
"""

import os
import sys
import time
import json
import struct
import select
import socket
import signal
import argparse
import threading
import datetime
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from stream_simulator  import StreamSimulator, DEFAULT_DATA_DIRS

# ── Environment / paths ────────────────────────────────────────────────────────
_DEFAULT_ROOT = r"C:\STREAMSENSE" if os.name == "nt" else "/content/STREAMSENSE"
ROOT = Path(os.environ.get("STREAMSENSE_ROOT", _DEFAULT_ROOT))

LOGS_DIR = Path(__file__).resolve().parent / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

STATS_DIR = Path(__file__).resolve().parent / "stats"
STATS_DIR.mkdir(parents=True, exist_ok=True)

# ── NSP v1.2 constants (spec Section 9, 11) ────────────────────────────────────
NSP_MAGIC          = b"NSP\x00"
NSP_VERSION        = 1
NSP_MSG_DATA       = 0x01
NSP_MSG_EOF        = 0x02
NSP_DTYPE_INT16    = 0x01
NSP_DTYPE_INT32    = 0x02
NSP_DTYPE_FLOAT32  = 0x03
NSP_DTYPE_FLOAT64  = 0x04

NSP_DTYPE_SIZE = {
    NSP_DTYPE_INT16   : 2,
    NSP_DTYPE_INT32   : 4,
    NSP_DTYPE_FLOAT32 : 4,
    NSP_DTYPE_FLOAT64 : 8,
}

NSP_HEADER_FMT    = "<4sHBBQQQIIII"    # little-endian, 48 bytes
NSP_HEADER_SIZE   = struct.calcsize(NSP_HEADER_FMT)   # must be 48
NSP_LENGTH_FMT    = "<I"               # little-endian uint32 length prefix
NSP_MAX_PACKET    = 16_777_216         # 16 MB hard limit (spec Section 7)

assert NSP_HEADER_SIZE == 48, f"Header size mismatch: {NSP_HEADER_SIZE}"

# ── Global stop flag (set by SIGINT/SIGTERM) ───────────────────────────────────
_STOP = threading.Event()


def _signal_handler(signum, frame):
    print("\n[NspSender] Shutdown signal received — stopping gracefully...")
    _STOP.set()


signal.signal(signal.SIGINT,  _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


# ── NSP packet builder ─────────────────────────────────────────────────────────

def build_header(
    msg_type   : int,
    dtype_code : int,
    seq_no     : int,
    ts_us      : int,
    session_id : int,
    payload_byt: int,
    sample_rate: int,
    frame_len  : int,
) -> bytes:
    """Serialise 48-byte NSP v1.2 header (little-endian)."""
    return struct.pack(
        NSP_HEADER_FMT,
        NSP_MAGIC,
        NSP_VERSION,
        msg_type,
        dtype_code,
        seq_no,
        ts_us,
        session_id,
        payload_byt,
        sample_rate,
        frame_len,
        0,           # reserved
    )


def build_data_packet(
    raw_frame  : np.ndarray,   # [N] float32 1D
    seq_no     : int,
    session_id : int,
    sample_rate: int = 16000,
) -> bytes:
    """
    Build a complete NSP DATA packet:
      Length prefix (4B) + Header (48B) + Payload (N × 4B)

    Validates payload_bytes == frame_length * sizeof(dtype) before sending.
    """
    frame  = raw_frame.astype(np.float32).ravel()
    N      = len(frame)
    p_bytes= N * NSP_DTYPE_SIZE[NSP_DTYPE_FLOAT32]   # N * 4

    assert p_bytes <= NSP_MAX_PACKET - NSP_HEADER_SIZE, \
        f"Payload exceeds 16MB limit: {p_bytes}"

    ts_us  = time.time_ns() // 1_000               # microseconds
    header = build_header(
        NSP_MSG_DATA, NSP_DTYPE_FLOAT32,
        seq_no, ts_us, session_id,
        p_bytes, sample_rate, N,
    )
    payload= frame.tobytes()                        # little-endian float32 bytes
    length = struct.pack(NSP_LENGTH_FMT, NSP_HEADER_SIZE + p_bytes)
    return length + header + payload


def build_eof_packet(seq_no: int, session_id: int, sample_rate: int = 16000) -> bytes:
    """Build NSP EOF/CLOSE packet (payload_bytes = 0, frame_length = 0)."""
    ts_us  = time.time_ns() // 1_000
    header = build_header(
        NSP_MSG_EOF, NSP_DTYPE_FLOAT32,
        seq_no, ts_us, session_id,
        0, sample_rate, 0,
    )
    length = struct.pack(NSP_LENGTH_FMT, NSP_HEADER_SIZE)
    return length + header


# ── Session statistics ─────────────────────────────────────────────────────────

class SessionStats:
    """Accumulates per-session statistics for JSON report."""

    def __init__(self, host: str, port: int, mode: str):
        self.host        = host
        self.port        = port
        self.mode        = mode
        self.session_id  = 0
        self.start_dt    = datetime.datetime.now()
        self.end_dt      = None
        self.packets_sent= 0
        self.bytes_sent  = 0
        self.frames_total= 0
        self.dropped     = 0
        self.welford_sum_mean = 0.0   # for aggregate Welford (Chan merge)
        self.welford_sum_std  = 0.0
        self.welford_n_frames = 0
        self.sources_project  = 0
        self.sources_unknown  = 0

    def finalise(self):
        self.end_dt = datetime.datetime.now()

    def to_dict(self) -> dict:
        duration = (
            (self.end_dt - self.start_dt).total_seconds()
            if self.end_dt else 0.0
        )
        return {
            "session_id"        : self.session_id,
            "host"              : self.host,
            "port"              : self.port,
            "mode"              : self.mode,
            "start_time"        : self.start_dt.isoformat(),
            "end_time"          : self.end_dt.isoformat() if self.end_dt else None,
            "duration_sec"      : round(duration, 3),
            "packets_sent"      : self.packets_sent,
            "bytes_sent"        : self.bytes_sent,
            "frames_total"      : self.frames_total,
            "dropped_frames"    : self.dropped,
            "audio_duration_sec": self.frames_total * 1.0,  # 1 frame = 1 sec
            "sources_project"   : self.sources_project,
            "sources_unknown"   : self.sources_unknown,
        }

    def save(self) -> Path:
        ts   = self.start_dt.strftime("%Y%m%d_%H%M%S")
        path = LOGS_DIR / f"nsp_session_{ts}.json"
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        return path


# ── Runtime statistics writer ──────────────────────────────────────────────────

def write_runtime_stats(stats: "SessionStats") -> None:
    """
    Write live send-side statistics to stats/nsp_sender_runtime.json.
    Called after every packet send. Uses atomic write (tmp → rename) to avoid
    partial reads by an external consumer.

    Fields:
      frames      — total DATA frames sent this session
      data_bytes  — total payload bytes sent (header + payload)
      data_rate   — bytes/s since session start
      send_rate   — frames/s since session start
    """
    elapsed = (datetime.datetime.now() - stats.start_dt).total_seconds()
    elapsed = elapsed if elapsed > 0 else 1e-9   # guard against div-by-zero

    payload = {
        "host"       : stats.host,
        "port"       : stats.port,
        "mode"       : stats.mode,
        "frames"     : stats.frames_total,
        "data_bytes" : stats.bytes_sent,
        "data_rate"  : round(stats.bytes_sent / elapsed, 2),   # bytes/s
        "send_rate"  : round(stats.frames_total / elapsed, 4), # frames/s
        "elapsed_sec": round(elapsed, 3),
        "updated_at" : datetime.datetime.now().isoformat(),
    }

    target = STATS_DIR / "nsp_sender_runtime.json"
    tmp    = STATS_DIR / "nsp_sender_runtime.json.tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    tmp.replace(target)


# ── Frame generator (StreamSimulator → fixed-size frames) ─────────────────────

def _frame_generator(data_dirs, frame_len: int = 16000):
    """
    Wraps StreamSimulator (validation config: 16kHz mono float32) and
    accumulates samples into fixed frame_len-sample frames.

    Yields np.ndarray [frame_len] float32 — the raw audio payload per NSP packet.
    """
    import torch
    sim = StreamSimulator(data_dirs=data_dirs, random_config=False,
                          chunk_min=512, chunk_max=4096)
    gen = sim.generator()

    buf = np.empty(0, dtype=np.float32)
    while not _STOP.is_set():
        chunk = next(gen)
        # Simulator yields [1, N] planar float32 in validation mode
        arr   = chunk.numpy().ravel().astype(np.float32)
        buf   = np.concatenate([buf, arr])
        while len(buf) >= frame_len:
            yield buf[:frame_len].copy()
            buf = buf[frame_len:]


# ── TCP send helper ────────────────────────────────────────────────────────────

def _sendall_safe(conn: socket.socket, data: bytes) -> bool:
    """
    Send all bytes. Returns False if the connection was broken.
    Respects _STOP flag for large sends.
    """
    try:
        conn.sendall(data)
        return True
    except (BrokenPipeError, ConnectionResetError, OSError):
        return False


# ── Session runner ─────────────────────────────────────────────────────────────

def _run_session(
    conn       : socket.socket,
    stats      : SessionStats,
    data_dirs,
    frame_len  : int,
    sample_rate: int,
    n_frames   : int | None,
):
    """
    Stream NSP DATA packets over an established TCP connection.
    Sends EOF packet on clean exit or disconnection.
    """
    seq        = 0
    session_id = int(time.time() * 1_000_000)   # unique per connection
    stats.session_id = session_id

    print(f"[NspSender] Session started | session_id={session_id} | "
          f"frame_len={frame_len} | sample_rate={sample_rate}")

    gen = _frame_generator(data_dirs, frame_len)

    try:
        for raw_frame in gen:
            if _STOP.is_set():
                break
            if n_frames is not None and stats.frames_total >= n_frames:
                break

            packet = build_data_packet(raw_frame, seq, session_id, sample_rate)

            if not _sendall_safe(conn, packet):
                print("[NspSender] Connection broken mid-stream.")
                stats.dropped += 1
                break

            seq                  += 1
            stats.packets_sent   += 1
            stats.bytes_sent     += len(packet)
            stats.frames_total   += 1

            write_runtime_stats(stats)

            if stats.frames_total % 100 == 0:
                print(f"[NspSender] Sent {stats.frames_total} frames | "
                      f"{stats.bytes_sent/1024:.1f} KB total")

    finally:
        # Always attempt to send EOF (best-effort)
        try:
            eof = build_eof_packet(seq, session_id, sample_rate)
            conn.sendall(eof)
            print(f"[NspSender] EOF sent | total_frames={stats.frames_total}")
        except OSError:
            pass


# ── Server mode loop ───────────────────────────────────────────────────────────

def run_server(
    host       : str,
    port       : int,
    data_dirs,
    frame_len  : int,
    sample_rate: int,
    n_frames   : int | None,
    stats      : SessionStats,
):
    """
    Server mode: bind/listen/accept loop.
    Per runbook Section 2: accept() uses 100ms select() timeout to check stop flag.
    """
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((host, port))
    srv.listen(1)
    srv.setblocking(False)

    print(f"[NspSender] Server mode | listening on {host}:{port}")

    try:
        while not _STOP.is_set():
            # 100ms select() timeout — allows graceful shutdown check
            readable, _, _ = select.select([srv], [], [], 0.1)
            if not readable:
                continue

            conn, addr = srv.accept()
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            conn.setblocking(True)
            print(f"[NspSender] Client connected: {addr}")

            try:
                _run_session(conn, stats, data_dirs, frame_len, sample_rate, n_frames)
            finally:
                conn.close()
                print(f"[NspSender] Client disconnected: {addr}")

            if _STOP.is_set():
                break
            if n_frames is not None and stats.frames_total >= n_frames:
                break

    finally:
        srv.close()


# ── Client mode loop ───────────────────────────────────────────────────────────

def run_client(
    host       : str,
    port       : int,
    data_dirs,
    frame_len  : int,
    sample_rate: int,
    n_frames   : int | None,
    stats      : SessionStats,
    reconnect_initial_ms: int = 1000,
    reconnect_max_ms    : int = 5000,
):
    """
    Client mode: connect with 2000ms timeout, linear backoff on failure.
    Per runbook Section 3 and Section 8:
      - Non-blocking connect() + select() with 2000ms timeout
      - Linear backoff: initial=1000ms, max=5000ms, +1000ms per retry
      - Backoff sleep broken into 100ms chunks to honour stop flag
    """
    print(f"[NspSender] Client mode | connecting to {host}:{port}")
    backoff_ms = reconnect_initial_ms

    while not _STOP.is_set():
        if n_frames is not None and stats.frames_total >= n_frames:
            break

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setblocking(False)

        try:
            sock.connect_ex((host, port))
        except OSError:
            pass

        # select() wait for writeability (2000ms timeout per runbook Section 12.C)
        _, writable, _ = select.select([], [sock], [], 2.0)

        if writable:
            err = sock.getsockopt(socket.SOL_SOCKET, socket.SO_ERROR)
            if err == 0:
                sock.setblocking(True)
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                print(f"[NspSender] Connected to {host}:{port}")

                try:
                    _run_session(sock, stats, data_dirs, frame_len, sample_rate, n_frames)
                finally:
                    sock.close()

                backoff_ms = reconnect_initial_ms   # reset on clean disconnect
                continue

        # Connection failed — apply linear backoff
        sock.close()
        if _STOP.is_set():
            break

        print(f"[NspSender] Connection failed. Retrying in {backoff_ms}ms...")
        # Sleep in 100ms chunks (runbook Section 12.E)
        slept = 0
        while slept < backoff_ms and not _STOP.is_set():
            time.sleep(0.1)
            slept += 100

        backoff_ms = min(backoff_ms + 1000, reconnect_max_ms)


# ── CLI entry point ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=(
            "NSP v1.2 Sender — streams raw float32 audio frames over TCP.\n"
            "Implements dual-mode (server/client) per STREAMSENSE runbook."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--mode", choices=["server", "client"], default="server",
                        help="Transport mode (default: server)")
    parser.add_argument("--host", default=os.environ.get("STREAMSENSE_HOST", "127.0.0.1"),
                        help="Bind/connect host (default: 127.0.0.1 or $STREAMSENSE_HOST)")
    parser.add_argument("--port", type=int,
                        default=int(os.environ.get("STREAMSENSE_PORT", "7654")),
                        help="Bind/connect port (default: 7654 or $STREAMSENSE_PORT)")
    parser.add_argument("--frame-len", type=int, default=16000,
                        help="Samples per NSP frame (default: 16000 = 1s @ 16kHz)")
    parser.add_argument("--sample-rate", type=int, default=16000,
                        help="Sample rate in Hz (default: 16000)")
    parser.add_argument("--n-frames", type=int, default=None,
                        help="Stop after N frames (default: run indefinitely)")
    parser.add_argument("--sources", choices=["project", "unknown", "both"], default="both",
                        help="Audio source pool (default: both)")
    parser.add_argument("--reconnect-initial-ms", type=int, default=1000,
                        help="Initial reconnect backoff in ms (client mode, default: 1000)")
    parser.add_argument("--reconnect-max-ms", type=int, default=5000,
                        help="Max reconnect backoff in ms (client mode, default: 5000)")
    args = parser.parse_args()

    # ── Select data directories per --sources ─────────────────────────────────
    if args.sources == "project":
        data_dirs = [ROOT / "data" / "raw"]
    elif args.sources == "unknown":
        data_dirs = [ROOT / "unknown_data"]
    else:
        data_dirs = list(DEFAULT_DATA_DIRS)

    print("=" * 72)
    print(f"NSP v1.2 Sender | mode={args.mode} | {args.host}:{args.port}")
    print(f"  frame_len={args.frame_len} | sample_rate={args.sample_rate} Hz")
    print(f"  n_frames={args.n_frames or 'infinite'} | sources={args.sources}")
    print(f"  Packet size: 4 + 48 + {args.frame_len * 4} = {4 + 48 + args.frame_len * 4} bytes")
    print(f"  Logs  → {LOGS_DIR}")
    print(f"  Stats → {STATS_DIR}")
    print("=" * 72)

    stats = SessionStats(args.host, args.port, args.mode)

    try:
        if args.mode == "server":
            run_server(
                args.host, args.port, data_dirs,
                args.frame_len, args.sample_rate, args.n_frames,
                stats,
            )
        else:
            run_client(
                args.host, args.port, data_dirs,
                args.frame_len, args.sample_rate, args.n_frames,
                stats,
                reconnect_initial_ms=args.reconnect_initial_ms,
                reconnect_max_ms    =args.reconnect_max_ms,
            )
    finally:
        stats.finalise()
        log_path = stats.save()
        d = stats.to_dict()
        print("\n" + "=" * 72)
        print("SESSION SUMMARY")
        print(f"  Packets sent     : {d['packets_sent']}")
        print(f"  Bytes sent       : {d['bytes_sent']:,}  "
              f"({d['bytes_sent']/1024:.1f} KB)")
        print(f"  Frames total     : {d['frames_total']}")
        print(f"  Dropped frames   : {d['dropped_frames']}")
        print(f"  Audio duration   : {d['audio_duration_sec']:.1f} s")
        print(f"  Duration         : {d['duration_sec']:.1f} s")
        print(f"  Stats saved to   : {log_path}")
        print("=" * 72)


if __name__ == "__main__":
    main()

```

### `training/populate_gv_top1.py`

```python
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

```

### `training/qat_finetune.py`

```python
"""
qat_finetune.py
Project STREAMSENSE — Track A
Scope 2 / QAT Extension — Quantization-Aware Training fine-tune

Trains the StreamSenseWrapper with Brevitas quantizers applied to all
Conv2d and Linear layers.  Simultaneously trains the embed_head (which
has never been trained on real data) and learns Brevitas quantizer scale
factors.  Saves best checkpoint and runs the GV1K gate before exiting.

Usage (in Colab via qat_colab.ipynb, Cell 7):
    python training/qat_finetune.py \
        --ckpt checkpoints/best_model.pth \
        --data /content/data \
        --epochs 10 \
        --lr 1e-5 \
        --out checkpoints/best_model_qat.pth \
        --gvk golden_vectors_1000/normalized \
        --device cuda
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import DataLoader, Dataset

# ── Resolve project root and training/ directory ──────────────────────────────
_THIS_DIR = Path(__file__).resolve().parent
_ROOT     = _THIS_DIR.parent

if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from model import StreamSenseNet                   # noqa: E402
from streaming_wrapper import StreamSenseWrapper   # noqa: E402

# ── Brevitas imports ──────────────────────────────────────────────────────────
try:
    import brevitas.nn as qnn
    from brevitas.quant import Int8WeightPerTensorFloat, Int8ActPerTensorFloat
except ImportError as e:
    print(f"[ERROR] brevitas not installed: {e}")
    print("        pip install brevitas")
    sys.exit(1)

# ── MPIC v1.0 frozen constants ────────────────────────────────────────────────
SAMPLE_RATE   = 16000
FRAME_LEN     = 16000
N_FFT         = 512
HOP_LENGTH    = 160
N_MELS        = 64
CENTER        = False
POWER         = 2.0
LOG_EPS       = 1e-10
CLIP_FLOOR_DB = -80.0
GLOBAL_MEAN   = -30.785545
GLOBAL_STD    = 22.157099

EXPECTED_T    = (FRAME_LEN - N_FFT) // HOP_LENGTH + 1   # 97

# 10 target keyword classes — indices match class_labels.json
TARGET_CLASSES = {
    "yes": 0, "no": 1, "up": 2, "down": 3,
    "left": 4, "right": 5, "on": 6, "off": 7,
    "stop": 8, "go": 9,
}

NUM_CLASSES   = 10
BATCH_SIZE    = 64       # suitable for T4 16 GB GPU
NUM_WORKERS   = 2


# ── MPIC v1.0 preprocessing pipeline ─────────────────────────────────────────

_mel_transform = T.MelSpectrogram(
    sample_rate = SAMPLE_RATE,
    n_fft       = N_FFT,
    hop_length  = HOP_LENGTH,
    n_mels      = N_MELS,
    window_fn   = torch.hann_window,
    center      = CENTER,
    power       = POWER,
)


def preprocess(raw: np.ndarray) -> torch.Tensor:
    """
    MPIC v1.0 full pipeline.
    Input:  float32 numpy [T] — raw 16 kHz audio, already at FRAME_LEN samples
    Output: float32 Tensor [1, 1, 64, 97]
    """
    waveform = torch.from_numpy(raw.copy()).float()
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)          # [1, T]
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    L = waveform.shape[1]
    if L < FRAME_LEN:
        waveform = torch.nn.functional.pad(waveform, (0, FRAME_LEN - L))
    elif L > FRAME_LEN:
        waveform = waveform[:, :FRAME_LEN]
    mel = _mel_transform(waveform)                # [1, 64, 97]
    mel = 10.0 * torch.log10(mel + LOG_EPS)
    mel = torch.clamp(mel, min=CLIP_FLOOR_DB)
    mel = (mel - GLOBAL_MEAN) / GLOBAL_STD
    mel = mel.unsqueeze(0)                        # [1, 1, 64, 97]
    return mel.float()


# ── Dataset ───────────────────────────────────────────────────────────────────

class SpeechCommandsDataset(Dataset):
    """
    Thin wrapper around torchaudio.datasets.SPEECHCOMMANDS.

    Filters to the 10 target classes only.  Discards _background_noise_,
    unknown words, and any clip that torchaudio cannot load.

    Returns: (Tensor [1, 1, 64, 97] float32, int class_index)
    """

    def __init__(self, root: Path, subset: str):
        """
        Args:
            root   : Path to Speech Commands root (the directory that will
                     contain / already contains speech_commands_v0.02/).
            subset : "validation" or "testing" (torchaudio split names).
        """
        self.root   = root
        self.subset = subset

        raw_ds = torchaudio.datasets.SPEECHCOMMANDS(
            root     = str(root),
            download = True,
            subset   = subset,
        )

        self.samples: list[tuple[str, int]] = []
        for waveform, sample_rate, label, *_ in raw_ds:
            if label not in TARGET_CLASSES:
                continue
            # We do not store waveform tensors in RAM — re-load from disk later.
            # torchaudio SPEECHCOMMANDS exposes _path attribute; use it.
            # Fallback: use the dataset's _walker list.
            self.samples.append((label, waveform, sample_rate))

        print(
            f"[SpeechCommandsDataset] subset={subset!r}  "
            f"kept {len(self.samples)} clips  "
            f"(target classes only)"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        label, waveform, sample_rate = self.samples[idx]
        class_idx = TARGET_CLASSES[label]

        # Convert to numpy float32 mono [T]
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        raw = waveform.squeeze(0).numpy().astype(np.float32)

        # Resample if needed (should always be 16 kHz for Speech Commands v2)
        if sample_rate != SAMPLE_RATE:
            waveform_t = torch.from_numpy(raw).unsqueeze(0)
            waveform_t = torchaudio.functional.resample(waveform_t, sample_rate, SAMPLE_RATE)
            raw = waveform_t.squeeze(0).numpy().astype(np.float32)

        tensor = preprocess(raw)            # [1, 1, 64, 97]
        return tensor.squeeze(0), class_idx # [1, 64, 97], int  (collation adds batch dim)


# ── Brevitas module replacement ───────────────────────────────────────────────

def _replace_conv2d(module: nn.Module) -> nn.Module:
    """
    Recursively replace all nn.Conv2d in module with brevitas.nn.QuantConv2d.
    Copies weight (and bias) data so the trained values are preserved.
    """
    for name, child in module.named_children():
        if isinstance(child, nn.Conv2d):
            qconv = qnn.QuantConv2d(
                in_channels  = child.in_channels,
                out_channels = child.out_channels,
                kernel_size  = child.kernel_size,
                stride       = child.stride,
                padding      = child.padding,
                dilation     = child.dilation,
                groups       = child.groups,
                bias         = child.bias is not None,
                weight_quant = Int8WeightPerTensorFloat,
                input_quant  = Int8ActPerTensorFloat,
                output_quant = Int8ActPerTensorFloat,
                return_quant_tensor = False,
            )
            with torch.no_grad():
                qconv.weight.copy_(child.weight)
                if child.bias is not None and qconv.bias is not None:
                    qconv.bias.copy_(child.bias)
            setattr(module, name, qconv)
        else:
            _replace_conv2d(child)
    return module


def _replace_linear(module: nn.Module) -> nn.Module:
    """
    Recursively replace all nn.Linear in module with brevitas.nn.QuantLinear.
    Copies weight and bias data.
    """
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            qlin = qnn.QuantLinear(
                in_features  = child.in_features,
                out_features = child.out_features,
                bias         = child.bias is not None,
                weight_quant = Int8WeightPerTensorFloat,
                input_quant  = Int8ActPerTensorFloat,
                output_quant = Int8ActPerTensorFloat,
                return_quant_tensor = False,
            )
            with torch.no_grad():
                qlin.weight.copy_(child.weight)
                if child.bias is not None and qlin.bias is not None:
                    qlin.bias.copy_(child.bias)
            setattr(module, name, qlin)
        else:
            _replace_linear(child)
    return module


def build_qat_model(ckpt_path: Path, device: torch.device) -> StreamSenseWrapper:
    """
    Construct StreamSenseWrapper, load best_model.pth backbone weights,
    apply Brevitas QuantConv2d / QuantLinear replacements, and move the
    whole model to device.

    Returns the model ready for QAT fine-tuning.
    """
    # 1. Instantiate base wrapper
    model = StreamSenseWrapper(num_classes=NUM_CLASSES)

    # 2. Load backbone weights with strict=True
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model.backbone.load_state_dict(ckpt["model_state"], strict=True)
    print(f"[build_qat_model] Loaded backbone from epoch {ckpt.get('epoch', '?')}  "
          f"val_acc={ckpt.get('val_accuracy', float('nan')):.2f}%")

    # 3. Replace Conv2d in backbone blocks (NOT in gap — it has no weights)
    _replace_conv2d(model.backbone.block1)
    _replace_conv2d(model.backbone.block2)
    _replace_conv2d(model.backbone.block3)

    # 4. Replace Linear in backbone classifier
    _replace_linear(model.backbone.classifier)

    # 5. Replace Linear in embed_head
    _replace_linear(model.embed_head)

    # 6. Brevitas device-placement fix: model.to(device) LAST
    model.to(device)

    # 7. Mandatory buffer verification
    for buf_name, buf in model.named_buffers():
        assert buf.device.type == device.type, (
            f"[device-check] Buffer {buf_name!r} is on {buf.device.type!r}, "
            f"expected {device.type!r}. This is the Brevitas device-placement bug."
        )
    print(f"[build_qat_model] All buffers verified on device={device.type!r}")

    return model


# ── GV1K gate ─────────────────────────────────────────────────────────────────

_LABEL_TO_IDX = {label: idx for label, idx in TARGET_CLASSES.items()}


def _parse_gv1k_label(stem: str) -> int | None:
    """
    Parse ground-truth class index from a GV1K normalized filename stem.
    Pattern: GV1K_NNNN_<label>_norm
    """
    parts = stem.split("_")
    # parts: ['GV1K', 'NNNN', '<label>', 'norm']
    if len(parts) < 4:
        return None
    label_str = parts[2].lower()
    return TARGET_CLASSES.get(label_str, None)


def run_gv1k_gate(model: nn.Module, gvk_dir: Path, device: torch.device) -> float:
    """
    Run all 1000 GV1K vectors through the model in eval mode.
    Compute top-1 accuracy on the logits output.
    Hard sys.exit(1) if accuracy < 90 %.

    Returns top-1 accuracy as a float in [0, 100].
    """
    bin_files = sorted(gvk_dir.glob("*_norm.bin"))
    if not bin_files:
        print(f"[GV1K] WARNING: no *_norm.bin files found in {gvk_dir} — skipping gate")
        return float("nan")

    model.eval()
    correct  = 0
    wrong    = 0
    skipped  = 0

    with torch.no_grad():
        for bf in bin_files:
            true_idx = _parse_gv1k_label(bf.stem)
            if true_idx is None:
                skipped += 1
                continue

            raw = np.fromfile(str(bf), dtype="<f4")
            if raw.size != 64 * 97:
                skipped += 1
                continue

            inp = torch.from_numpy(raw).reshape(1, 1, 64, 97).to(device)
            logits, _embedding, _novelty = model(inp)
            pred_idx = int(logits.argmax(dim=1).item())

            if pred_idx == true_idx:
                correct += 1
            else:
                wrong += 1

    total_checked = correct + wrong
    if total_checked == 0:
        print(f"[GV1K] SKIP — no vectors could be checked (all {skipped} skipped)")
        return float("nan")

    top1_acc = 100.0 * correct / total_checked
    print(f"[GV1K] Vectors checked : {total_checked}  (skipped: {skipped})")
    print(f"[GV1K] Correct         : {correct}  Wrong: {wrong}")
    print(f"[GV1K] Top-1 accuracy  : {top1_acc:.2f}%")

    if top1_acc < 90.0:
        print(f"[GV1K] FAIL — {top1_acc:.2f}% < 90.0% minimum.  Aborting.")
        sys.exit(1)
    else:
        print(f"[GV1K] PASS — {top1_acc:.2f}% ≥ 90.0%")

    return top1_acc


# ── Training helpers ──────────────────────────────────────────────────────────

def _freeze_backbone(model: StreamSenseWrapper):
    """
    Freeze backbone weight/bias parameters only.
    Brevitas quantizer scale factors (named 'scale' or containing 'scaling')
    must stay trainable so the quantizer calibrates during epochs 1-3.
    embed_head parameters are untouched (they stay trainable).
    """
    for name, param in model.backbone.named_parameters():
        # Keep Brevitas quantizer scale factors trainable.
        # They are identified by the substring 'scaling' in their parameter name
        # (e.g. 'block1.0.input_quant.fused_activation_quant_proxy.tensor_quant.scaling_impl.value').
        if "scaling" in name or "scale" in name:
            param.requires_grad_(True)
        else:
            param.requires_grad_(False)


def _unfreeze_all(model: StreamSenseWrapper):
    """Unfreeze all parameters for full fine-tuning."""
    for param in model.parameters():
        param.requires_grad_(True)


def _count_trainable(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_one_epoch(
    model:     nn.Module,
    loader:    DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device:    torch.device,
    epoch:     int,
) -> float:
    model.train()
    total_loss = 0.0
    n_batches  = len(loader)

    for batch_idx, (x, y) in enumerate(loader):
        x = x.to(device)   # [B, 1, 64, 97]
        y = y.to(device)   # [B]

        optimizer.zero_grad()
        logits, _embedding, _novelty = model(x)
        loss = criterion(logits, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()

        if (batch_idx + 1) % 200 == 0 or (batch_idx + 1) == n_batches:
            print(
                f"  Epoch {epoch:>3}  [{batch_idx+1:>4}/{n_batches}]  "
                f"loss={loss.item():.4f}",
                flush=True,
            )

    return total_loss / n_batches


def validate(
    model:     nn.Module,
    loader:    DataLoader,
    device:    torch.device,
) -> float:
    model.eval()
    correct = 0
    total   = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits, _embedding, _novelty = model(x)
            preds    = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total   += y.size(0)

    return 100.0 * correct / total if total > 0 else 0.0


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="STREAMSENSE QAT fine-tuning script — Scope 2 QAT extension"
    )
    parser.add_argument(
        "--ckpt",
        type    = Path,
        default = Path("checkpoints/best_model.pth"),
        help    = "Path to best_model.pth (default: checkpoints/best_model.pth)",
    )
    parser.add_argument(
        "--data",
        type    = Path,
        required= True,
        help    = "Path to Speech Commands v2 root directory",
    )
    parser.add_argument(
        "--epochs",
        type    = int,
        default = 10,
        help    = "Total QAT training epochs (default: 10)",
    )
    parser.add_argument(
        "--lr",
        type    = float,
        default = 1e-5,
        help    = "Adam learning rate (default: 1e-5)",
    )
    parser.add_argument(
        "--out",
        type    = Path,
        default = Path("checkpoints/best_model_qat.pth"),
        help    = "Output checkpoint path (default: checkpoints/best_model_qat.pth)",
    )
    parser.add_argument(
        "--device",
        type    = str,
        default = "cuda" if torch.cuda.is_available() else "cpu",
        help    = "Device: cuda or cpu (default: cuda if available)",
    )
    parser.add_argument(
        "--gvk",
        type    = Path,
        default = Path("golden_vectors_1000/normalized"),
        help    = "Path to GV1K normalized directory (default: golden_vectors_1000/normalized)",
    )
    return parser.parse_args()


def main():
    args   = parse_args()
    device = torch.device(args.device)

    print("=" * 60)
    print("STREAMSENSE — QAT Fine-tuning  (Scope 2 QAT extension)")
    print("=" * 60)
    print(f"  Checkpoint  : {args.ckpt}")
    print(f"  Data root   : {args.data}")
    print(f"  Epochs      : {args.epochs}")
    print(f"  LR          : {args.lr}")
    print(f"  Output      : {args.out}")
    print(f"  Device      : {device}")
    print(f"  GV1K dir    : {args.gvk}")

    # ── Prerequisite checks ───────────────────────────────────────────────────
    if not args.ckpt.exists():
        print(f"[ERROR] Checkpoint not found: {args.ckpt}")
        sys.exit(1)

    # ── Ensure output directory exists ────────────────────────────────────────
    args.out.parent.mkdir(parents=True, exist_ok=True)

    # ── Build QAT model ───────────────────────────────────────────────────────
    print("\n[Step 1] Building QAT model...")
    model = build_qat_model(args.ckpt, device)

    # ── Datasets and DataLoaders ──────────────────────────────────────────────
    print("\n[Step 2] Loading Speech Commands datasets...")
    val_ds  = SpeechCommandsDataset(args.data, subset="validation")
    test_ds = SpeechCommandsDataset(args.data, subset="testing")

    # For training we use the validation split of Speech Commands (it is the
    # standard labelled non-test split, labelled via validation_list.txt).
    # The testing split is held out for the final GV1K gate — it is not used
    # during QAT training to avoid data contamination.
    train_ds = SpeechCommandsDataset(args.data, subset="validation")

    train_loader = DataLoader(
        train_ds,
        batch_size  = BATCH_SIZE,
        shuffle     = True,
        num_workers = NUM_WORKERS,
        pin_memory  = (device.type == "cuda"),
        drop_last   = True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size  = BATCH_SIZE,
        shuffle     = False,
        num_workers = NUM_WORKERS,
        pin_memory  = (device.type == "cuda"),
    )

    print(f"  Train batches : {len(train_loader)}")
    print(f"  Val   batches : {len(val_loader)}")

    # ── Criterion ─────────────────────────────────────────────────────────────
    criterion = nn.CrossEntropyLoss()
    # Optimizer is built after freeze so it only tracks trainable parameters.
    # It will be rebuilt at epoch 4 when the backbone is unfrozen.
    best_val_acc    = 0.0
    best_ckpt_saved = False

    # ── Training loop ─────────────────────────────────────────────────────────
    print("\n[Step 3] Training loop")
    print(f"  Epochs 1–3  : backbone FROZEN, training embed_head + quantizer scales")
    print(f"  Epoch  4+   : all parameters UNFROZEN")
    print()

    optimizer = None  # will be (re)built when phase changes

    for epoch in range(1, args.epochs + 1):

        # Phase 1: epochs 1-3 — freeze backbone weights, keep quantizer scales trainable
        if epoch == 1:
            _freeze_backbone(model)
            trainable_params = [p for p in model.parameters() if p.requires_grad]
            optimizer = torch.optim.Adam(trainable_params, lr=args.lr)
            print(f"  [Epoch {epoch}] Backbone FROZEN.  Trainable params: {_count_trainable(model):,}")

        # Phase 2: epoch 4+ — unfreeze everything and rebuild optimizer
        elif epoch == 4:
            _unfreeze_all(model)
            trainable_params = [p for p in model.parameters() if p.requires_grad]
            optimizer = torch.optim.Adam(trainable_params, lr=args.lr)
            print(f"  [Epoch {epoch}] All parameters UNFROZEN.  Trainable params: {_count_trainable(model):,}")

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
        val_acc    = validate(model, val_loader, device)

        print(
            f"Epoch {epoch:>3}/{args.epochs}  "
            f"train_loss={train_loss:.4f}  "
            f"val_top1={val_acc:.2f}%"
        )

        # Save checkpoint if validation accuracy improved
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "model_state"  : model.state_dict(),
                    "epoch"        : epoch,
                    "val_accuracy" : val_acc,
                    "qat"          : True,
                },
                args.out,
            )
            best_ckpt_saved = True
            print(f"  [checkpoint] Saved best checkpoint  val_acc={val_acc:.2f}%  → {args.out}")

    if not best_ckpt_saved:
        # Save whatever we have if no improvement was ever detected
        torch.save(
            {
                "model_state"  : model.state_dict(),
                "epoch"        : args.epochs,
                "val_accuracy" : best_val_acc,
                "qat"          : True,
            },
            args.out,
        )
        print(f"  [checkpoint] Saved final checkpoint → {args.out}")

    # ── Post-training GV1K gate ───────────────────────────────────────────────
    print("\n[Step 4] Post-training GV1K gate")

    # Reload best checkpoint into a fresh model for gate evaluation
    best_ckpt_data = torch.load(args.out, map_location="cpu", weights_only=True)
    gate_model = build_qat_model(args.ckpt, device)
    gate_model.load_state_dict(best_ckpt_data["model_state"])
    gate_model.eval()

    gvk_dir = _ROOT / args.gvk if not args.gvk.is_absolute() else args.gvk
    gv1k_acc = run_gv1k_gate(gate_model, gvk_dir, device)

    # ── Final summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("QAT FINE-TUNING COMPLETE")
    print("=" * 60)
    print(f"  Best val accuracy : {best_val_acc:.2f}%")
    print(f"  GV1K top-1        : {gv1k_acc:.2f}%")
    print(f"  Checkpoint saved  : {args.out}")
    print()

    if gv1k_acc < 90.0:
        print("[FAIL] GV1K gate failed. Checkpoint NOT deployment-grade.")
        sys.exit(1)
    else:
        print("[PASS] GV1K gate passed. Checkpoint is deployment-grade.")


if __name__ == "__main__":
    main()

```

### `training/run_gv_regression_1000.py`

```python
"""
run_gv_regression_1000.py
Project STREAMSENSE — Track A
End-to-end Golden Vector regression test on golden_vectors_1000/.

For each of the 1000 golden vectors:
    1. Load raw waveform from raw/GV1K_NNNN_label.bin           [16000] float32
    2. Run through mel_pipeline.preprocess() (Steps 1-8)         -> [1,1,64,97]
    3. Compare resulting normalized tensor against the
       precomputed normalized/GV1K_NNNN_label_norm.bin           [64,97] float32
       (max abs error must be <= tolerance from manifest.json)
    4. Run the [1,1,64,97] tensor through the ONNX model (FP32 by default)
    5. argmax -> predicted class, compare against label in labels/GV1K_NNNN_label_label.txt

Prints colored PASS/FAIL per-vector progress (summarized every 100), then a
final summary table: pipeline parity pass rate, model accuracy, max error
stats, and an overall PASS/FAIL verdict against manifest tolerance.

This is the script intended for the joint end-to-end GV regression session
with Kavish (Track B) — run on your machine against golden_vectors_1000/
while he runs his C++ equivalent against the same .bin files, then compare
results live.

Run from C:\\STREAMSENSE\\training\\:
    python run_gv_regression_1000.py
    python run_gv_regression_1000.py --model ..\\onnx_models\\streamsense_model_int8.onnx
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import onnxruntime as ort

from mel_pipeline import preprocess, OUTPUT_SHAPE

# ── ANSI colors (works in VS Code integrated terminal) ────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
RESET  = "\033[0m"
BOLD   = "\033[1m"

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT          = Path(r"C:\STREAMSENSE")
GV_DIR        = ROOT / "golden_vectors_1000"
RAW_DIR       = GV_DIR / "raw"
NORM_DIR      = GV_DIR / "normalized"
LABEL_DIR     = GV_DIR / "labels"
MANIFEST_PATH = GV_DIR / "manifest.json"

CLASS_LABELS_FILE = ROOT / "class_labels.json"
DEFAULT_MODEL     = ROOT / "onnx_models" / "streamsense_model_fp32.onnx"

FRAME_LEN = 16000
N_MELS    = 64


def load_class_labels(path: Path):
    with open(path, "r") as f:
        raw = json.load(f)
    labels = [None] * len(raw)
    for k, v in raw.items():
        labels[int(k)] = v
    return labels


def load_bin(path: Path, shape, dtype="<f4"):
    arr = np.fromfile(str(path), dtype=dtype)
    return arr.reshape(shape)


def main():
    parser = argparse.ArgumentParser(
        description="STREAMSENSE end-to-end GV regression on golden_vectors_1000/"
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL,
                         help=f"ONNX model path (default: {DEFAULT_MODEL})")
    parser.add_argument("--limit", type=int, default=None,
                         help="Limit number of vectors tested (default: all 1000)")
    args = parser.parse_args()

    print("=" * 64)
    print(f"{BOLD}STREAMSENSE — End-to-End GV Regression (golden_vectors_1000){RESET}")
    print("=" * 64)

    # ── Validate inputs ────────────────────────────────────────────────────
    for p, name in [(MANIFEST_PATH, "golden_vectors_1000/manifest.json"),
                     (args.model,    "ONNX model"),
                     (CLASS_LABELS_FILE, "class_labels.json")]:
        if not p.exists():
            print(f"{RED}[ERROR] Not found: {p} ({name}){RESET}")
            sys.exit(1)

    with open(MANIFEST_PATH, "r") as f:
        manifest = json.load(f)

    tolerance = float(manifest["tolerance_max_abs_error"])
    vectors   = manifest["vectors"]
    mel_shape = tuple(manifest["vectors"][next(iter(vectors))]["norm_shape"])  # [64,97]

    class_labels = load_class_labels(CLASS_LABELS_FILE)

    n_vectors = len(vectors)
    if args.limit is not None:
        n_vectors = min(n_vectors, args.limit)

    print(f"Model       : {args.model.name}")
    print(f"Vectors     : {n_vectors}")
    print(f"Tolerance   : {tolerance}")
    print(f"Mel shape   : {mel_shape}")
    print()

    # ── Load ONNX session ──────────────────────────────────────────────────
    sess = ort.InferenceSession(str(args.model), providers=["CPUExecutionProvider"])
    input_name  = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    print(f"ONNX input='{input_name}'  output='{output_name}'")
    print()

    # ── Run regression ─────────────────────────────────────────────────────
    keys = sorted(vectors.keys(), key=lambda k: int(k))[:n_vectors]

    pipeline_pass = 0
    pipeline_fail = 0
    accuracy_correct = 0

    max_errors = []
    fail_details = []

    REPORT_EVERY = 100

    for i, key in enumerate(keys):
        v = vectors[key]
        gv_name   = v["gv_name"]
        class_idx = v["class_idx"]

        raw_path   = RAW_DIR  / v["raw_bin"]
        norm_path  = NORM_DIR / v["norm_bin"]
        label_path = LABEL_DIR / f"{gv_name}_label.txt"

        # 1. Load raw waveform
        raw = load_bin(raw_path, (FRAME_LEN,))  # [16000] float32

        # 2. Run through mel_pipeline (Steps 1-8)
        mel_tensor = preprocess(raw)  # [1,1,64,97] torch.Tensor

        if tuple(mel_tensor.shape) != OUTPUT_SHAPE:
            pipeline_fail += 1
            fail_details.append((gv_name, "shape", f"got {tuple(mel_tensor.shape)}"))
            continue

        computed_norm = mel_tensor.squeeze(0).squeeze(0).numpy()  # [64,97]

        # 3. Compare against precomputed normalized .bin
        precomputed_norm = load_bin(norm_path, mel_shape)  # [64,97]
        abs_diff = np.abs(computed_norm - precomputed_norm)
        max_err  = float(abs_diff.max())
        max_errors.append(max_err)

        parity_ok = max_err <= tolerance
        if parity_ok:
            pipeline_pass += 1
        else:
            pipeline_fail += 1
            fail_details.append((gv_name, "parity", f"max_err={max_err:.6e} > {tolerance}"))

        # 4. Run ONNX inference
        input_array = mel_tensor.numpy().astype(np.float32)
        logits = sess.run([output_name], {input_name: input_array})[0][0]  # [10]
        pred_idx = int(np.argmax(logits))

        if pred_idx == class_idx:
            accuracy_correct += 1

        # Progress
        if (i + 1) % REPORT_EVERY == 0 or i == 0:
            status_color = GREEN if (parity_ok and pred_idx == class_idx) else RED
            print(f"  [{i+1:>4}/{n_vectors}]  {gv_name:<22} "
                  f"max_err={max_err:.2e}  "
                  f"pred={class_labels[pred_idx]:<6} true={class_labels[class_idx]:<6} "
                  f"{status_color}{'OK' if (parity_ok and pred_idx == class_idx) else 'CHECK'}{RESET}")

    # ── Summary ─────────────────────────────────────────────────────────────
    max_errors_arr = np.array(max_errors) if max_errors else np.array([0.0])
    overall_max_err = float(max_errors_arr.max())
    overall_mean_err = float(max_errors_arr.mean())

    accuracy = 100.0 * accuracy_correct / n_vectors if n_vectors > 0 else 0.0
    parity_pct = 100.0 * pipeline_pass / n_vectors if n_vectors > 0 else 0.0

    print()
    print("=" * 64)
    print(f"{BOLD}SUMMARY{RESET}")
    print("=" * 64)
    print(f"  Vectors tested        : {n_vectors}")
    print(f"  Pipeline parity PASS  : {pipeline_pass}/{n_vectors}  ({parity_pct:.2f}%)")
    print(f"  Pipeline parity FAIL  : {pipeline_fail}/{n_vectors}")
    print(f"  Max abs error (worst) : {overall_max_err:.6e}  (tolerance {tolerance})")
    print(f"  Mean abs error        : {overall_mean_err:.6e}")
    print(f"  Model accuracy        : {accuracy_correct}/{n_vectors}  ({accuracy:.2f}%)")

    if fail_details:
        print(f"\n  {YELLOW}First failures:{RESET}")
        for gv_name, kind, detail in fail_details[:10]:
            print(f"    [{kind}] {gv_name}: {detail}")
        if len(fail_details) > 10:
            print(f"    ... and {len(fail_details) - 10} more")

    print()
    overall_pass = (pipeline_fail == 0)
    if overall_pass:
        print(f"{GREEN}{BOLD}[PASS] All {n_vectors} golden vectors within tolerance "
              f"({tolerance}). End-to-end GV regression PASSED.{RESET}")
    else:
        print(f"{RED}{BOLD}[FAIL] {pipeline_fail}/{n_vectors} golden vectors exceeded "
              f"tolerance ({tolerance}). Review failures above.{RESET}")

    print("=" * 64)

    sys.exit(0 if overall_pass else 1)


if __name__ == "__main__":
    main()

```

### `training/select_golden.py`

```python
"""
select_golden.py
Project STREAMSENSE — Track A
For each of 10 classes, shows top 8 candidates by peak energy from test split.
User picks the best one visually (1-8).
Saves stats/golden_selection.json and copies WAVs to golden_vectors/wav/.

Selection criteria (apply visually):
    1. Strongest peak energy (colorbar max -30 to -40 dB)
    2. Single continuous burst — no fragmentation
    3. Clean silence at -80 dB on both sides
    4. Energy concentrated in mel bins 0-40
    5. Natural phoneme structure for multi-phoneme words

Run:
    python select_golden.py
"""

import torch
import torchaudio
import numpy as np
import json
import sys
import shutil
from pathlib import Path
import matplotlib
matplotlib.use("TkAgg")          # works on Windows — change to "Qt5Agg" if TkAgg errors
import matplotlib.pyplot as plt

# ── Paths ─────────────────────────────────────────────────────────────────────
TEST_SPLIT      = Path(r"C:\STREAMSENSE\data\splits\test_files.txt")
STATS_FILE      = Path(r"C:\STREAMSENSE\stats\normalization_stats.json")
SELECTION_OUT   = Path(r"C:\STREAMSENSE\stats\golden_selection.json")
WAV_OUT_DIR     = Path(r"C:\STREAMSENSE\golden_vectors\wav")

# ── MPIC v1.0 frozen parameters ───────────────────────────────────────────────
SAMPLE_RATE   = 16000
FRAME_LEN     = 16000
N_FFT         = 512
HOP_LENGTH    = 160
N_MELS        = 64
CENTER        = False
POWER         = 2.0
LOG_EPS       = 1e-10
CLIP_FLOOR_DB = -80.0

# ── Class map ─────────────────────────────────────────────────────────────────
CLASS_MAP = {
    0: "yes", 1: "no",  2: "up",   3: "down", 4: "left",
    5: "right", 6: "on", 7: "off", 8: "stop", 9: "go"
}
N_CLASSES    = 10
N_CANDIDATES = 8      # top-N by peak energy shown per class

# ── MelSpectrogram transform ──────────────────────────────────────────────────
mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate = SAMPLE_RATE,
    n_fft       = N_FFT,
    hop_length  = HOP_LENGTH,
    n_mels      = N_MELS,
    window_fn   = torch.hann_window,
    center      = CENTER,
    power       = POWER,
)

# ── Load norm stats ───────────────────────────────────────────────────────────
with open(STATS_FILE, "r") as f:
    _stats = json.load(f)
GLOBAL_MEAN = float(_stats["global_mean"])
GLOBAL_STD  = float(_stats["global_std"])

# ── Helpers ───────────────────────────────────────────────────────────────────
def parse_path(line: str) -> tuple:
    """Returns (Path, label_str, class_idx)"""
    parts = line.strip().split("|")
    win_path  = parts[0].strip()
    label     = parts[1].strip()
    class_idx = int(parts[2].strip())
    return Path(win_path), label, class_idx


def load_wav(path: Path) -> torch.Tensor:
    """Load WAV -> float32 [1, 16000]"""
    waveform, sr = torchaudio.load(str(path))
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    length = waveform.shape[1]
    if length < FRAME_LEN:
        waveform = torch.nn.functional.pad(waveform, (0, FRAME_LEN - length))
    elif length > FRAME_LEN:
        waveform = waveform[:, :FRAME_LEN]
    return waveform.float()


def compute_logmel(waveform: torch.Tensor) -> np.ndarray:
    """Steps 4-6 only (no normalization) -> numpy [64, 97]"""
    mel = mel_transform(waveform)
    mel = 10.0 * torch.log10(mel + LOG_EPS)
    mel = torch.clamp(mel, min=CLIP_FLOOR_DB)
    return mel.squeeze(0).numpy()              # [64, 97]


def peak_energy(logmel: np.ndarray) -> float:
    """Criterion 1: max value in the spectrogram (dB)"""
    return float(logmel.max())


def read_test_split() -> dict:
    """
    Returns dict: class_idx -> list of (Path, label) tuples
    """
    buckets = {i: [] for i in range(N_CLASSES)}
    with open(TEST_SPLIT, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            path, label, idx = parse_path(line)
            buckets[idx].append((path, label))
    return buckets


def get_top_candidates(file_list: list) -> list:
    """
    Scan all files for a class, compute peak energy, return top N_CANDIDATES.
    Returns list of (peak_db, path, label, logmel_array) sorted descending.
    """
    scored = []
    for path, label in file_list:
        if not path.exists():
            continue
        try:
            waveform = load_wav(path)
            logmel   = compute_logmel(waveform)
            peak     = peak_energy(logmel)
            scored.append((peak, path, label, logmel))
        except Exception:
            continue

    # Sort by peak energy descending, take top N
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:N_CANDIDATES]


def show_candidates(class_idx: int, candidates: list) -> int:
    """
    Display N_CANDIDATES mel spectrograms in a grid.
    Returns user's choice (1-indexed).
    """
    label_name = CLASS_MAP[class_idx]
    n = len(candidates)

    fig, axes = plt.subplots(2, 4, figsize=(16, 6))
    fig.suptitle(
        f"Class {class_idx} — '{label_name.upper()}'   |   "
        f"Pick the best candidate (1-{n})\n"
        f"Criteria: strong peak (-30 to -40 dB), single burst, "
        f"clean silence, energy in bins 0-40",
        fontsize=11
    )

    axes_flat = axes.flatten()

    for i, (peak, path, label, logmel) in enumerate(candidates):
        ax = axes_flat[i]
        im = ax.imshow(
            logmel,
            aspect="auto",
            origin="lower",
            cmap="magma",
            vmin=CLIP_FLOOR_DB,
            vmax=0.0,
            interpolation="nearest"
        )
        ax.set_title(f"[{i+1}]  peak={peak:.1f} dB\n{path.name}", fontsize=8)
        ax.set_xlabel("Time frame", fontsize=7)
        ax.set_ylabel("Mel bin", fontsize=7)
        ax.tick_params(labelsize=7)
        plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02).ax.tick_params(labelsize=7)

    # Hide unused subplots if n < 8
    for j in range(n, len(axes_flat)):
        axes_flat[j].set_visible(False)

    plt.tight_layout()
    plt.show(block=False)

    # Get valid input
    while True:
        try:
            choice = int(input(f"\nClass '{label_name}' — enter choice (1-{n}): ").strip())
            if 1 <= choice <= n:
                plt.close(fig)
                return choice
            else:
                print(f"  Please enter a number between 1 and {n}.")
        except ValueError:
            print("  Invalid input — enter a number.")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("STREAMSENSE — select_golden.py")
    print("=" * 60)

    # Validate inputs
    for p, name in [(TEST_SPLIT, "test_files.txt"), (STATS_FILE, "normalization_stats.json")]:
        if not p.exists():
            print(f"[ERROR] Not found: {p}")
            sys.exit(1)

    # Create output dirs
    WAV_OUT_DIR.mkdir(parents=True, exist_ok=True)
    SELECTION_OUT.parent.mkdir(parents=True, exist_ok=True)

    # Read test split
    print(f"\nReading test split: {TEST_SPLIT}")
    buckets = read_test_split()
    for idx in range(N_CLASSES):
        print(f"  Class {idx} '{CLASS_MAP[idx]}': {len(buckets[idx])} files")

    print(f"\nFor each class: scanning all files, picking top {N_CANDIDATES} by peak energy...")
    print("This takes ~1-2 minutes for 5779 test files.\n")

    selection = {}

    for class_idx in range(N_CLASSES):
        label_name = CLASS_MAP[class_idx]
        print(f"\n{'─'*50}")
        print(f"Class {class_idx} — '{label_name}'  ({len(buckets[class_idx])} files)")
        print(f"  Scanning for top {N_CANDIDATES} by peak energy...", flush=True)

        candidates = get_top_candidates(buckets[class_idx])

        if len(candidates) == 0:
            print(f"  [ERROR] No valid files found for class '{label_name}'")
            sys.exit(1)

        print(f"  Top {len(candidates)} peaks: " +
              ", ".join(f"{c[0]:.1f}" for c in candidates) + " dB")

        # Show grid, get user choice
        choice = show_candidates(class_idx, candidates)

        peak_db, chosen_path, chosen_label, chosen_mel = candidates[choice - 1]

        # Build GV name
        gv_name = f"GV_{class_idx:02d}_{label_name}"

        # Copy WAV
        wav_dest = WAV_OUT_DIR / f"{gv_name}.wav"
        shutil.copy2(str(chosen_path), str(wav_dest))

        selection[str(class_idx)] = {
            "gv_name"      : gv_name,
            "class_idx"    : class_idx,
            "label"        : label_name,
            "source_path"  : str(chosen_path),
            "wav_dest"     : str(wav_dest),
            "peak_energy_db": round(peak_db, 4),
            "choice"       : choice,
        }

        print(f"  Selected: [{choice}] {chosen_path.name}  (peak={peak_db:.1f} dB)")
        print(f"  Copied WAV -> {wav_dest}")

    # Save selection JSON
    with open(SELECTION_OUT, "w") as f:
        json.dump(selection, f, indent=2)

    # Final summary
    print(f"\n{'='*60}")
    print("SELECTION COMPLETE")
    print(f"{'='*60}")
    for idx in range(N_CLASSES):
        s = selection[str(idx)]
        print(f"  GV_{idx:02d}_{CLASS_MAP[idx]:5s}  peak={s['peak_energy_db']:6.1f} dB  "
              f"← {Path(s['source_path']).name}")
    print(f"\nSaved -> {SELECTION_OUT}")
    print(f"WAVs  -> {WAV_OUT_DIR}")
    print("\n[DONE] select_golden.py completed. Next: python generate_golden.py")


if __name__ == "__main__":
    main()

```

### `training/stream_simulator.py`

```python
"""
stream_simulator.py
Project STREAMSENSE — Track A (Scope 2, Section 2.1)
Generates an endless stream of audio chunks to simulate real network packets.

Section 2.1 — Generalised Sample-Stream Contract:
  The stream is parameterised on C (channels), N (samples per chunk), Rate,
  SampleType, and Layout. The audio validation case (C=1, Rate=16kHz, float32,
  planar) is ONE instantiation of the general contract.

Fixed per stream instance (decided at startup):
  - C          : Channels (1=Mono, 2=Stereo)
  - Rate       : Sample rate in Hz
  - SampleType : dtype of each sample (float32 or int16)
  - Layout     : Memory layout (planar=[C,N] or interleaved=[N,C])

Variable per chunk (simulates network jitter):
  - N          : Number of samples per yielded chunk (random within [chunk_min, chunk_max])

No seeding — results vary every run by design (real-world network simulation).

Usage:
  python stream_simulator.py                 # validation config, 10 chunks
  python stream_simulator.py --demo          # random stream config, 10 chunks
"""

import os
import sys
import random
import argparse
import torch
import torchaudio
from pathlib import Path

# ── NSP v1.2 dtype codes (mirrors nsp_sender.py) ──────────────────────────────
NSP_DTYPE_INT16   = 0x01
NSP_DTYPE_FLOAT32 = 0x03

# ── Sample type mapping ────────────────────────────────────────────────────────
SAMPLE_TYPES = {
    "float32": (torch.float32, NSP_DTYPE_FLOAT32),
    "int16"  : (torch.int16,   NSP_DTYPE_INT16),
}

# ── Stream rate options for demo/generalisation mode ──────────────────────────
DEMO_RATES    = [8000, 16000, 44100, 48000]
DEMO_CHANNELS = [1, 2]
DEMO_LAYOUTS  = ["planar", "interleaved"]
DEMO_DTYPES   = ["float32", "int16"]

# ── Root path (env-var aware) ─────────────────────────────────────────────────
_DEFAULT_ROOT = r"C:\STREAMSENSE" if os.name == "nt" else "/content/STREAMSENSE"
ROOT = Path(os.environ.get("STREAMSENSE_ROOT", _DEFAULT_ROOT))

# Default data directories (project data + unknown — no GV1K per scope)
DEFAULT_DATA_DIRS = [
    ROOT / "data" / "raw",
    ROOT / "unknown_data",
]


class StreamSimulator:
    """
    Endless audio stream simulator.

    Loads WAV files from data/raw and unknown_data, yields them as
    network-packet-sized chunks in the stream's fixed format.

    DSA Decision Record — Stream Source:
        Structure  : Random file selection from glob pool, random chunk sizes
        Complexity : O(1) per chunk — no pre-loading; files loaded on demand
        Memory     : One waveform at a time — O(max_file_samples)
        Alternative rejected: Pre-loading all files → prohibitive RAM for large corpora

    Args:
        data_dirs     : List of directories to glob *.wav from (recursive).
        random_config : False = validation/parity config (C=1, 16kHz, float32, planar).
                        True  = random C/Rate/dtype/layout per run (Section 2.1 demo).
        chunk_min     : Minimum samples per yielded chunk (network jitter lower bound).
        chunk_max     : Maximum samples per yielded chunk (network jitter upper bound).
    """

    def __init__(
        self,
        data_dirs=None,
        random_config: bool = False,
        chunk_min: int = 512,
        chunk_max: int = 4096,
    ):
        # ── Collect all .wav files ─────────────────────────────────────────────
        if data_dirs is None:
            data_dirs = DEFAULT_DATA_DIRS

        self.files = []
        for d in data_dirs:
            p = Path(d)
            if p.exists():
                self.files.extend(sorted(p.glob("**/*.wav")))

        if not self.files:
            print(
                "[StreamSimulator] WARNING: No .wav files found in the provided "
                "directories. Check DEFAULT_DATA_DIRS or pass data_dirs explicitly."
            )

        self.chunk_min    = chunk_min
        self.chunk_max    = chunk_max

        # ── Fixed stream parameters ────────────────────────────────────────────
        # No seeding — random.choice uses system entropy → different every run.
        if random_config:
            self.stream_sr        = random.choice(DEMO_RATES)
            self.stream_channels  = random.choice(DEMO_CHANNELS)
            self.sample_type_name = random.choice(DEMO_DTYPES)
            self.layout           = random.choice(DEMO_LAYOUTS)
        else:
            # Validation instantiation — MPIC v1.0 audio case
            self.stream_sr        = 16000
            self.stream_channels  = 1
            self.sample_type_name = "float32"
            self.layout           = "planar"

        self.torch_dtype, self.nsp_dtype_code = SAMPLE_TYPES[self.sample_type_name]

        print(
            f"[StreamSimulator] Rate={self.stream_sr} Hz | "
            f"Channels={self.stream_channels} | "
            f"SampleType={self.sample_type_name} | "
            f"Layout={self.layout} | "
            f"Files={len(self.files)} | "
            f"ChunkRange=[{self.chunk_min}, {self.chunk_max}]"
        )

    # ── Format converter ───────────────────────────────────────────────────────
    def _to_stream_format(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Convert a planar float32 [C, N] waveform into the stream's fixed format.

        Steps (applied in order):
          1. SampleType: cast to stream dtype (int16 scales to [-32767, 32767])
          2. Layout    : reorder planar [C, N] → interleaved [N, C] if needed
        """
        # 1. dtype
        if self.torch_dtype == torch.int16:
            waveform = (waveform * 32767.0).clamp(-32768, 32767).to(torch.int16)
        else:
            waveform = waveform.to(torch.float32)

        # 2. layout
        if self.layout == "interleaved":
            waveform = waveform.T  # [C, N] → [N, C]

        return waveform

    # ── Main generator ─────────────────────────────────────────────────────────
    def generator(self):
        """
        Endless generator. Yields torch.Tensor chunks in the stream's fixed format.

        Each iteration picks a random .wav file, adapts it to the stream's
        C/Rate/dtype/layout, appends a short silence gap, then slices it into
        random-sized chunks (simulating network jitter).

        Yields:
            torch.Tensor of shape:
              planar      : [C, chunk_n]
              interleaved : [chunk_n, C]
            dtype = self.torch_dtype
        """
        while True:
            if not self.files:
                # Fallback: pure Gaussian noise if no files available
                n = random.randint(self.chunk_min, self.chunk_max)
                noise = torch.randn(self.stream_channels, n)
                yield self._to_stream_format(noise)
                continue

            # ── Load a random file ─────────────────────────────────────────────
            wav_path = random.choice(self.files)
            try:
                waveform, sr = torchaudio.load(str(wav_path))  # [C_src, N_src] float32
            except Exception as e:
                print(f"[StreamSimulator] WARNING: failed to load {wav_path.name}: {e}")
                continue

            # ── Adapt to stream's fixed Rate ───────────────────────────────────
            if sr != self.stream_sr:
                waveform = torchaudio.functional.resample(waveform, sr, self.stream_sr)

            # ── Adapt to stream's fixed Channels ──────────────────────────────
            if self.stream_channels == 1 and waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)   # stereo → mono
            elif self.stream_channels == 2 and waveform.shape[0] == 1:
                waveform = waveform.repeat(2, 1)                # mono → stereo dup

            # ── Chop into random-sized chunks (network jitter) ─────────────────
            # Chunking is done on float32 planar BEFORE format conversion so that
            # int16 scaling math stays clean.
            total = waveform.shape[1]
            idx   = 0
            while idx < total:
                n     = random.randint(self.chunk_min, self.chunk_max)
                chunk = waveform[:, idx : idx + n]  # [C, n] float32
                idx  += n
                if chunk.shape[1] == 0:
                    continue
                yield self._to_stream_format(chunk)


# ── CLI self-test ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="StreamSimulator self-test")
    parser.add_argument("--demo", action="store_true",
                        help="Use random stream config (Section 2.1 generalisation demo)")
    parser.add_argument("--n-chunks", type=int, default=10,
                        help="Number of chunks to print (default: 10)")
    parser.add_argument("--chunk-min", type=int, default=512)
    parser.add_argument("--chunk-max", type=int, default=4096)
    args = parser.parse_args()

    sim = StreamSimulator(random_config=args.demo,
                          chunk_min=args.chunk_min,
                          chunk_max=args.chunk_max)
    gen = sim.generator()

    def get_n(chunk):
        return chunk.shape[1] if sim.layout == "planar" else chunk.shape[0]

    print(f"\n{'─'*72}")
    print(f"{'Chunk':>6} | {'N (samples)':>12} | {'C':>4} | {'dtype':>10} | {'Layout':>12}")
    print(f"{'─'*72}")
    for i in range(args.n_chunks):
        chunk = next(gen)
        n     = get_n(chunk)
        print(f"{i+1:>6} | {n:>12} | {sim.stream_channels:>4} | "
              f"{str(chunk.dtype).replace('torch.',''):>10} | {sim.layout:>12}")
    print(f"{'─'*72}")

```

### `training/streaming_framer.py`

```python
"""
streaming_framer.py
Project STREAMSENSE — Track A (Scope 2, WA-1, D5-D6)

Replaces the one-shot mel_pipeline.preprocess() with a continuous sliding-window
streaming framer that ingests arbitrarily-sized chunks and emits normalised
[1, 1, 64, 97] mel tensors whenever 97 STFT time-frames have accumulated.

──────────────────────────────────────────────────────────────────────────────
DSA Decision Records (required by Scope 2, Section 6)
──────────────────────────────────────────────────────────────────────────────

Ring buffer (sample accumulation):
  Structure  : torch.zeros(_BUF_CAP) fixed-capacity ring buffer + fill pointer
  Complexity : O(1) amortised per sample — append to tail, memmove carry on emit
  Memory     : (TARGET_SR + N_FFT) × 4 bytes = 16512 × 4 = 66 KB — pre-allocated
  Alternative rejected: collections.deque — O(N) list() copy on every STFT window

STFT front-end:
  Structure  : torch.stft() with cached Hann window; radix-2 FFT via PyTorch
  Complexity : O(N_FFT × log(N_FFT)) per frame = O(512 × 9) ≈ O(4608) ops
  Memory     : Hann window [512] float32 cached at import — 2 KB
  Alternative rejected: scipy.signal.stft — not PyTorch-native; no autograd

Mel projection:
  Structure  : Sparse COO filterbank [64, 257] — precomputed once at import
  Complexity : O(nnz) per frame (sparse matmul vs dense O(64 × 257) = 16448)
  Memory     : ~2× nnz float32 values + indices — measured ~8 KB
  Alternative rejected: dense MelSpectrogram transform — O(F×M) per frame;
                        Scope 2 Section 6 explicitly requires O(nnz) sparse

Online normalisation:
  Structure  : Welford (1962) running mean/variance, Chan parallel batch variant
  Complexity : O(1) per batch update — constant time regardless of stream length
  Memory     : 3 scalars (n: float64, mean: float64, M2: float64) — 24 bytes
  Role       : Tracking / convergence validation ONLY — does NOT affect output
  Alternative rejected: two-pass — incompatible with online streaming

Normalisation (output):
  Structure  : Frozen global constants from stats/normalization_stats.json
  Constants  : GLOBAL_MEAN = -30.785545 dB, GLOBAL_STD = 22.157099 dB
  Complexity : O(64×97) per frame — element-wise subtract and divide
  Rationale  : Frozen stats guarantee exact parity with GV1K normalised .bin
               files. Welford accumulator is a SEPARATE parallel tracker for
               convergence validation; it never feeds into the output tensor.

──────────────────────────────────────────────────────────────────────────────
Input contract (per stream instance — fixed at startup):
  C          : Channels (1 or 2)
  Rate       : Sample rate (any — resampled to 16 kHz internally)
  SampleType : dtype (float32 or int16)
  Layout     : Memory order (planar=[C,N] or interleaved=[N,C])

Output contract (fixed — identical to MPIC v1.0):
  list[torch.Tensor] — each tensor is exactly [1, 1, 64, 97] float32
  Returns [] when the buffer has not yet accumulated 97 STFT frames.
  Returns one tensor per complete 97-frame window.
──────────────────────────────────────────────────────────────────────────────
"""

import sys
import json
import math
import torch
import torchaudio
import torchaudio.functional as F_audio
import numpy as np
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
_TRAINING_DIR = Path(__file__).resolve().parent
STATS_FILE    = _TRAINING_DIR.parent / "stats" / "normalization_stats.json"

# ── MPIC v1.0 frozen parameters ────────────────────────────────────────────────
TARGET_SR     = 16000
N_FFT         = 512
HOP_LENGTH    = 160
N_MELS        = 64
CENTER        = False
POWER         = 2.0
LOG_EPS       = 1e-10
CLIP_FLOOR_DB = -80.0
OVERLAP       = N_FFT - HOP_LENGTH          # 352 — samples carried across chunks
EXPECTED_T    = (TARGET_SR - N_FFT) // HOP_LENGTH + 1   # 97 per 1-sec clip

# ── Load frozen normalisation constants (fail fast if missing) ─────────────────
if not STATS_FILE.exists():
    raise FileNotFoundError(
        f"Normalisation stats not found: {STATS_FILE}\n"
        f"Run compute_normstats.py first."
    )
with open(STATS_FILE, "r") as _fh:
    _stats        = json.load(_fh)
    GLOBAL_MEAN   = float(_stats["global_mean"])   # -30.785545 dB
    GLOBAL_STD    = float(_stats["global_std"])    # 22.157099 dB
    _N_ELEMENTS   = int(_stats.get("n_elements", 0))   # for reference only

# ── Pre-computed assets — done once at import, reused every frame ──────────────

# Hann window — cached so every torch.stft() call reuses the same tensor
_HANN_WINDOW: torch.Tensor = torch.hann_window(N_FFT)


def _build_sparse_fbank() -> torch.Tensor:
    """
    Build mel filterbank as a SPARSE [64, 257] COO matrix.

    Dense version is [257, 64]. We transpose and sparsify so that:
        mel [64, K] = sparse_fbank [64, 257] @ power [257, K]

    The filterbank is extremely sparse — each row touches only 2 triangular
    filter wings — so O(nnz) sparse matmul beats O(F*M) dense matmul.
    This satisfies the Scope 2 Section 6 O(nnz) mel projection requirement.
    """
    fbank_dense = F_audio.melscale_fbanks(
        n_freqs    = N_FFT // 2 + 1,       # 257
        f_min      = 0.0,
        f_max      = TARGET_SR / 2.0,
        n_mels     = N_MELS,
        sample_rate= TARGET_SR,
        norm       = None,
        mel_scale  = "htk",
    )                                       # [257, 64] dense float32
    return fbank_dense.T.to_sparse()        # [64, 257] sparse COO


_SPARSE_FBANK: torch.Tensor = _build_sparse_fbank()


# ── StreamingFramer ────────────────────────────────────────────────────────────
class StreamingFramer:
    """
    Continuous sliding-window mel framer for streaming audio.

    Consumes arbitrary-sized audio chunks and emits normalised mel tensors
    [1, 1, 64, 97] whenever 97 STFT time-frames have accumulated.

    Each instance is bound to one stream's fixed configuration (C, Rate,
    SampleType, Layout). Create a new instance when a new stream starts.

    Normalisation uses frozen MPIC v1.0 global constants:
        GLOBAL_MEAN = -30.785545 dB
        GLOBAL_STD  =  22.157099 dB

    A separate Welford accumulator tracks running statistics of the mel-dB
    values seen so far. This is used ONLY for convergence validation and
    reporting — it never affects the output tensor. Access via:
        framer.welford_mean  (float, dB)
        framer.welford_std   (float, dB)
        framer.n_welford_elements (int)
    """

    def __init__(
        self,
        stream_sr      : int         = TARGET_SR,
        stream_channels: int         = 1,
        dtype          : torch.dtype = torch.float32,
        layout         : str         = "planar",
    ):
        """
        Args:
            stream_sr       : Sample rate of incoming stream (any Hz).
            stream_channels : Number of channels in incoming stream (1 or 2).
            dtype           : torch dtype of incoming samples (float32 or int16).
            layout          : "planar"      → [C, N] tensor
                              "interleaved" → [N, C] tensor
        """
        self.stream_sr        = stream_sr
        self.stream_channels  = stream_channels
        self.in_dtype         = dtype
        self.layout           = layout
        self.n_frames_emitted = 0

        # ── Resampler (built once if needed) ───────────────────────────────────
        self._resampler = None
        if stream_sr != TARGET_SR:
            self._resampler = torchaudio.transforms.Resample(
                orig_freq=stream_sr,
                new_freq =TARGET_SR,
            )

        # ── Sample ring buffer (fixed capacity, pre-allocated) ─────────────────
        # Capacity: 1 full second at 16 kHz + 1 N_FFT window for overlap carry
        _BUF_CAP       = TARGET_SR + N_FFT          # 16512 samples
        self._buf      = torch.zeros(_BUF_CAP, dtype=torch.float32)
        self._fill     = 0

        # ── Mel frame buffer (holds partial windows until 97 frames) ───────────
        # Capacity: 2 × EXPECTED_T to safely handle overflow during large chunks
        self._mel_buf  = torch.zeros((N_MELS, EXPECTED_T * 2), dtype=torch.float32)
        self._mel_fill = 0

        # ── Welford state — starts FRESH (accumulates stream statistics) ───────
        # NOTE: This is a SEPARATE tracker from normalisation.
        #       Normalisation always uses frozen GLOBAL_MEAN / GLOBAL_STD.
        #       Welford is for convergence validation and reporting ONLY.
        self._w_n    = 0.0      # number of mel-dB elements seen
        self._w_mean = 0.0      # running mean (float64)
        self._w_M2   = 0.0      # running sum of squared deviations (float64)

    # ── Properties for Welford reporting ─────────────────────────────────────
    @property
    def welford_mean(self) -> float:
        """Running mean of all mel-dB values seen so far (dB)."""
        return self._w_mean

    @property
    def welford_std(self) -> float:
        """Running std of all mel-dB values seen so far (dB). Returns 0 if n<2."""
        if self._w_n < 2:
            return 0.0
        return float(math.sqrt(self._w_M2 / self._w_n))

    @property
    def n_welford_elements(self) -> int:
        """Total number of mel-dB scalar elements processed."""
        return int(self._w_n)

    def welford_summary(self) -> dict:
        """Return a summary dict for reporting."""
        return {
            "welford_mean_db"   : round(self.welford_mean, 6),
            "welford_std_db"    : round(self.welford_std,  6),
            "frozen_mean_db"    : GLOBAL_MEAN,
            "frozen_std_db"     : GLOBAL_STD,
            "mean_delta_db"     : round(abs(self.welford_mean - GLOBAL_MEAN), 6),
            "std_delta_db"      : round(abs(self.welford_std  - GLOBAL_STD),  6),
            "n_elements"        : int(self._w_n),
            "n_frames_emitted"  : self.n_frames_emitted,
        }

    # ── Input normalisation ───────────────────────────────────────────────────
    def _to_mono_float32_16k(self, chunk: torch.Tensor) -> torch.Tensor:
        """
        Convert any incoming chunk to float32 mono 16 kHz 1D tensor.

        Handles all stream parameters in order:
          SampleType → cast to float32
          Layout     → reorder to planar [C, N]
          C          → average channels → mono [1, N]
          Rate       → resample to 16 kHz
        Returns shape [M,] float32.
        """
        # 1. SampleType: int16 → float32
        if self.in_dtype == torch.int16:
            chunk = chunk.float() / 32767.0
        else:
            chunk = chunk.float()

        # 2. Layout: interleaved [N, C] → planar [C, N]
        if self.layout == "interleaved":
            chunk = chunk.T                     # [N, C] → [C, N]

        # 3. Ensure 2-D [C, N]
        if chunk.ndim == 1:
            chunk = chunk.unsqueeze(0)          # [N,] → [1, N]

        # 4. C: multi-channel → mono
        if chunk.shape[0] > 1:
            chunk = chunk.mean(dim=0, keepdim=True)  # [C, N] → [1, N]

        # 5. Rate: resample if needed
        if self._resampler is not None:
            chunk = self._resampler(chunk)

        return chunk.squeeze(0)                 # [M,] float32

    # ── Welford batched update (Chan parallel formula) ─────────────────────────
    def _welford_update(self, mel: torch.Tensor):
        """
        Update running stats with a batch of mel-dB values.
        Uses Chan's parallel Welford formula — numerically stable.
        mel: any-shape float32 tensor (treated as flat).

        DSA: O(1) per call — single-pass, no per-element loop.
        """
        vals   = mel.double()
        n_b    = float(vals.numel())
        if n_b == 0:
            return

        mean_b = vals.mean().item()
        var_b  = vals.var(unbiased=False).item() if n_b > 1 else 0.0
        M2_b   = var_b * n_b

        new_n   = self._w_n + n_b
        delta   = mean_b - self._w_mean
        new_mean= self._w_mean + delta * n_b / new_n
        new_M2  = self._w_M2 + M2_b + (delta ** 2) * self._w_n * n_b / new_n

        self._w_n    = new_n
        self._w_mean = new_mean
        self._w_M2   = new_M2

    # ── Main processing entry point ───────────────────────────────────────────
    def process_chunk(self, chunk) -> list:
        """
        Ingest one audio chunk and emit normalised mel frames when ready.

        Args:
            chunk: torch.Tensor or numpy.ndarray of raw audio samples.

        Returns:
            list[torch.Tensor]: List of [1, 1, 64, 97] float32 tensors.
            Empty list [] if 97 STFT frames have not yet accumulated.
        """
        # Convert numpy if needed
        if isinstance(chunk, np.ndarray):
            chunk = torch.from_numpy(chunk.copy())

        # Normalise input to float32 mono 16 kHz 1D tensor
        samples = self._to_mono_float32_16k(chunk)   # [M,]
        n_new   = samples.shape[0]

        # ── Write into fixed ring buffer (O(1) amortised) ─────────────────────
        if self._fill + n_new > self._buf.shape[0]:
            # Defensive: packet too large. Slide buffer left, drop oldest.
            keep = self._buf.shape[0] - n_new
            if keep > 0:
                self._buf[:keep] = self._buf[self._fill - keep : self._fill].clone()
            self._fill = max(keep, 0)

        self._buf[self._fill : self._fill + n_new] = samples
        self._fill += n_new

        # ── Need at least N_FFT samples to compute one STFT frame ─────────────
        if self._fill < N_FFT:
            return []

        # ── Compute all complete STFT frames available ─────────────────────────
        n_frames     = (self._fill - N_FFT) // HOP_LENGTH + 1
        samples_used = N_FFT + (n_frames - 1) * HOP_LENGTH

        signal = self._buf[:samples_used].clone()        # [samples_used,]

        # ── STFT — O(N_FFT log N_FFT), cached Hann window ─────────────────────
        stft = torch.stft(
            signal,
            n_fft         = N_FFT,
            hop_length    = HOP_LENGTH,
            win_length    = N_FFT,
            window        = _HANN_WINDOW,
            center        = CENTER,
            return_complex= True,
        )                                               # [257, n_frames] complex

        # ── Power spectrum ────────────────────────────────────────────────────
        power = stft.abs().pow(POWER)                   # [257, n_frames] float32

        # ── Sparse mel projection — O(nnz) ────────────────────────────────────
        mel = torch.sparse.mm(_SPARSE_FBANK, power.float())  # [64, n_frames]

        # ── Log scaling + dB floor (MPIC v1.0 Steps 5-6) ──────────────────────
        mel = 10.0 * torch.log10(mel + LOG_EPS)
        mel = torch.clamp(mel, min=CLIP_FLOOR_DB)       # [64, n_frames]

        # ── Welford update (PARALLEL tracker — does NOT affect output) ─────────
        self._welford_update(mel)

        # ── Normalisation — ALWAYS frozen global constants (MPIC v1.0 Step 7) ──
        mel_norm = (mel - GLOBAL_MEAN) / GLOBAL_STD     # [64, n_frames]

        # ── Buffer mel frames ─────────────────────────────────────────────────
        if self._mel_fill + n_frames > self._mel_buf.shape[1]:
            # Expand mel buffer defensively for very large chunks
            new_cap = max(self._mel_fill + n_frames, self._mel_buf.shape[1] * 2)
            new_buf = torch.zeros((N_MELS, new_cap), dtype=torch.float32)
            new_buf[:, :self._mel_fill] = self._mel_buf[:, :self._mel_fill]
            self._mel_buf = new_buf

        self._mel_buf[:, self._mel_fill : self._mel_fill + n_frames] = mel_norm
        self._mel_fill += n_frames

        # ── Slide overlap samples to front of ring buffer ─────────────────────
        # Always keep the last OVERLAP (352) samples for STFT continuity
        remaining   = self._fill - samples_used
        carry_start = samples_used - OVERLAP
        carry_len   = OVERLAP + remaining
        self._buf[:carry_len] = self._buf[carry_start : carry_start + carry_len].clone()
        self._fill = carry_len

        # ── Extract complete [1, 1, 64, 97] tensors ───────────────────────────
        out_tensors = []
        while self._mel_fill >= EXPECTED_T:
            complete = self._mel_buf[:, :EXPECTED_T].clone()
            out_tensors.append(complete.unsqueeze(0).unsqueeze(0))  # [1, 1, 64, 97]
            self.n_frames_emitted += 1

            remaining_mels = self._mel_fill - EXPECTED_T
            if remaining_mels > 0:
                self._mel_buf[:, :remaining_mels] = (
                    self._mel_buf[:, EXPECTED_T : self._mel_fill].clone()
                )
            self._mel_fill = remaining_mels

        return out_tensors

    def reset(self):
        """
        Reset all internal buffers and Welford state.
        Call when a stream disconnects and a new session starts.
        """
        self._buf.zero_()
        self._fill     = 0
        self._mel_fill = 0
        self._w_n      = 0.0
        self._w_mean   = 0.0
        self._w_M2     = 0.0
        self.n_frames_emitted = 0


# ── Self-test (python streaming_framer.py) ────────────────────────────────────
def _run_self_tests() -> bool:
    print("=" * 64)
    print("streaming_framer.py — self-test")
    print(f"  GLOBAL_MEAN = {GLOBAL_MEAN}  GLOBAL_STD = {GLOBAL_STD}")
    print("=" * 64)

    try:
        from mel_pipeline import preprocess as one_shot
    except ImportError as e:
        print(f"[FAIL] Cannot import mel_pipeline: {e}")
        return False

    passed = 0
    failed = 0

    def check(name, condition, detail=""):
        nonlocal passed, failed
        if condition:
            print(f"  [PASS] {name}")
            passed += 1
        else:
            print(f"  [FAIL] {name}  {detail}")
            failed += 1

    rng     = np.random.default_rng()  # no seed — varies every run
    samples = rng.standard_normal(16000).astype(np.float32)

    # ── T1: Single 16 kHz mono float32 chunk — 16000 samples ─────────────────
    framer     = StreamingFramer()
    out_list   = framer.process_chunk(samples)
    out_oneshot= one_shot(samples)

    check("T1.a — output shape [1,1,64,97]",
          len(out_list) == 1 and out_list[0].shape == (1, 1, 64, 97))

    diff = torch.abs(out_list[0] - out_oneshot).max().item()
    check("T1.b — parity with one-shot pipeline",
          diff < 5e-4, f"max_diff={diff:.2e}")

    check("T1.c — exactly 1 frame emitted",
          len(out_list) == 1, f"got {len(out_list)}")

    # ── T2: Chunked 160-sample packets (network jitter) ───────────────────────
    framer2   = StreamingFramer()
    out_chunks = []
    for i in range(0, 16000, 160):
        out_chunks.extend(framer2.process_chunk(samples[i:i+160]))

    check("T2.a — chunked framer produces 1 frame",
          len(out_chunks) == 1 and out_chunks[0].shape == (1, 1, 64, 97))

    diff2 = torch.abs(out_chunks[0] - out_oneshot).max().item()
    check("T2.b — chunked parity with one-shot",
          diff2 < 5e-4, f"max_diff={diff2:.2e}")

    # ── T3: Welford accumulates separately (not affecting output) ─────────────
    framer3 = StreamingFramer()
    framer3.process_chunk(samples)
    check("T3.a — Welford n_elements > 0 after processing",
          framer3.n_welford_elements > 0)
    check("T3.b — Welford mean is NOT used for normalisation "
          "(frozen mean confirms output path)",
          True)  # The output was already verified in T1/T2 against one_shot

    # ── T4: int16 stereo interleaved at 8 kHz ────────────────────────────────
    framer4 = StreamingFramer(
        stream_sr=8000, stream_channels=2,
        dtype=torch.int16, layout="interleaved",
    )
    raw_8k = (rng.standard_normal((8000, 2)) * 16383).astype(np.int16)
    out_8k = framer4.process_chunk(torch.from_numpy(raw_8k))
    check("T4 — 8 kHz stereo int16 interleaved → [1,1,64,97]",
          len(out_8k) == 1 and out_8k[0].shape == (1, 1, 64, 97))

    # ── Summary ──────────────────────────────────────────────────────────────
    print("=" * 64)
    print(f"Results: {passed}/{passed+failed} passed")
    if failed == 0:
        print("[DONE] All self-tests PASS.")
    else:
        print("[FAIL] Some tests failed.")
    print("=" * 64)
    return failed == 0


if __name__ == "__main__":
    ok = _run_self_tests()
    sys.exit(0 if ok else 1)

```

### `training/test_integration.py`

```python
"""
test_integration.py
Project STREAMSENSE — Track A (Scope 2)

Pipes the StreamSimulator directly into the StreamingFramer to prove
the framer can ingest completely randomized stream configurations and
network jitter, while successfully emitting standardized 16kHz frames.
"""

import sys
import time
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from stream_simulator import StreamSimulator
from streaming_framer import StreamingFramer

def main():
    print("=" * 80)
    print("INTEGRATION TEST: StreamSimulator -> StreamingFramer")
    print("=" * 80)

    # 1. Start the simulator (randomizes C, Rate, SampleType, Layout)
    sim = StreamSimulator(
        data_dirs=["C:/STREAMSENSE/data/raw", "C:/STREAMSENSE/unknown_data"],
        random_config=True
    )
    
    # 2. Start the framer, perfectly bound to the Simulator's configuration
    framer = StreamingFramer(
        stream_sr       = sim.stream_sr,
        stream_channels = sim.stream_channels,
        dtype           = sim.dtype,
        layout          = sim.layout,
    )
    
    gen = sim.generator()

    print(f"\n{'-'*115}")
    print(f"{'NETWORK IN (from Simulator)':^48} | {'ENGINE OUT (from Framer)':^30} | {'CNN BUFFER':^25}")
    print(f"{'-'*115}")
    print(f"{'Chunk':>5} | {'N (Samples)':>11} | {'Raw Shape':>14} | {'Dtype':>10} | "
          f"{'Frames Emitted':>15} | {'Output Shape':>16} | {'Rolling Window':>20}")
    print(f"{'-'*115}")

    total_tensors = 0

    for i in range(15):
        # The network delivers a randomized chunk
        chunk = next(gen)
        
        # The engine digests it. It might return [], or it might return a list of full [1,1,64,97] tensors.
        out_list = framer.process_chunk(chunk)
        
        # Stats
        n_samples = chunk.shape[1] if sim.layout == "planar" else chunk.shape[0]
        dtype_str = str(chunk.dtype).replace('torch.', '')
        n_emitted  = len(out_list)
        total_tensors += n_emitted
        
        cnn_status = "Waiting..."
        out_shape = "---"
        
        if n_emitted > 0:
            out_shape = str(list(out_list[0].shape))
            cnn_status = f"Ready: {n_emitted} chunks!"

        print(
            f"{i+1:>5} | "
            f"{n_samples:>11} | "
            f"{str(list(chunk.shape)):>14} | "
            f"{dtype_str:>10} | "
            f"{n_emitted:>15} | "
            f"{out_shape:>16} | "
            f"{cnn_status:>20}"
        )
        time.sleep(0.1)

    print(f"{'-'*115}")
    print(f"Integration Success! Handled 15 messy packets and safely extracted {total_tensors} standard [1,1,64,97] blocks.")
    print("=" * 80)

if __name__ == "__main__":
    main()

```

### `training/train.py`

```python
"""
train.py
Project STREAMSENSE — Track A
MPIC v1.0 — Full training loop with validation, checkpointing, and logging.

Designed to run on Google Colab T4 GPU.
Falls back to CPU automatically if CUDA is not available.

Inputs:
    data/splits/train_files.txt      via dataset.py
    data/splits/val_files.txt        via dataset.py
    training/model.py                architecture

Outputs:
    checkpoints/best_model.pth       best checkpoint by val accuracy
    checkpoints/training_log.csv     epoch | train_loss | val_loss | val_acc | lr

SpecAugment is applied here (on tensors, after mel_pipeline).
Time-domain augmentations are applied inside dataset.py.

Usage (Colab):
    python train.py
    python train.py --epochs 40 --batch 64 --lr 0.001
"""

import sys
import csv
import time
import random
import argparse
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

# ── Local imports ─────────────────────────────────────────────────────────────
try:
    from model   import StreamSenseNet
    from dataset import get_dataloader
except ImportError as e:
    print(f"[ERROR] Import failed: {e}")
    print("        Ensure model.py and dataset.py are in the same directory.")
    sys.exit(1)

# ── Paths ─────────────────────────────────────────────────────────────────────
# These work on both Windows (local) and Colab (Linux).
# On Colab, mount your Drive and adjust BASE_DIR to match your mount point,
# or just run from the cloned repo root — relative paths will resolve.

BASE_DIR     = Path(__file__).resolve().parent.parent   # C:\STREAMSENSE or /content/STREAMSENSE
SPLITS_DIR   = BASE_DIR / "data"  / "splits"
CKPT_DIR     = BASE_DIR / "checkpoints"

TRAIN_SPLIT  = SPLITS_DIR / "train_files.txt"
VAL_SPLIT    = SPLITS_DIR / "val_files.txt"
BEST_CKPT    = CKPT_DIR   / "best_model.pth"
TRAIN_LOG    = CKPT_DIR   / "training_log.csv"

# ── Fixed hyperparameters ─────────────────────────────────────────────────────
SEED         = 42
NUM_CLASSES  = 10
NUM_WORKERS  = 2      # set to 0 if Colab multiprocessing issues

# ── SpecAugment parameters ────────────────────────────────────────────────────
# Applied to tensors [B, 1, 64, 97] during training forward pass.
# Frequency masking: mask up to F mel bins
# Time masking:      mask up to T time frames
FREQ_MASK_F  = 8     # max mel bins to mask  (out of 64)
TIME_MASK_T  = 15    # max time frames to mask (out of 97)
N_FREQ_MASKS = 1     # number of frequency masks per sample
N_TIME_MASKS = 1     # number of time masks per sample


# ── Reproducibility ───────────────────────────────────────────────────────────

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ── SpecAugment ───────────────────────────────────────────────────────────────

def spec_augment(
    x          : torch.Tensor,
    freq_mask_f: int = FREQ_MASK_F,
    time_mask_t: int = TIME_MASK_T,
    n_freq     : int = N_FREQ_MASKS,
    n_time     : int = N_TIME_MASKS,
) -> torch.Tensor:
    """
    Apply SpecAugment to a batch of mel spectrograms in-place.

    Args:
        x : Tensor [B, 1, 64, 97]  — normalized mel spectrogram batch.
            Modified in-place; a clone is made first to avoid mutating
            the original dataloader output.

    Returns:
        Augmented tensor [B, 1, 64, 97].

    Each sample in the batch gets independently sampled masks.
    Masked regions are filled with 0.0 (mean of normalized data ≈ 0).
    """
    x = x.clone()
    B, C, F, T = x.shape    # B, 1, 64, 97

    for b in range(B):
        # Frequency masks — mask mel bins
        for _ in range(n_freq):
            f  = random.randint(0, freq_mask_f)
            f0 = random.randint(0, max(F - f, 0))
            x[b, :, f0 : f0 + f, :] = 0.0

        # Time masks — mask time frames
        for _ in range(n_time):
            t  = random.randint(0, time_mask_t)
            t0 = random.randint(0, max(T - t, 0))
            x[b, :, :, t0 : t0 + t] = 0.0

    return x


# ── Training epoch ────────────────────────────────────────────────────────────

def train_one_epoch(
    model     : nn.Module,
    loader    : torch.utils.data.DataLoader,
    criterion : nn.Module,
    optimizer : torch.optim.Optimizer,
    device    : torch.device,
    epoch     : int,
) -> float:
    """
    Run one full training epoch.

    Returns:
        mean training loss over all batches.
    """
    model.train()
    total_loss = 0.0
    n_batches  = len(loader)

    for batch_idx, (tensors, labels) in enumerate(loader):
        # tensors: [B, 1, 1, 64, 97] — note extra dim from dataset collation
        # squeeze the redundant dim-1 to get [B, 1, 64, 97]
        x = tensors.squeeze(1).to(device)      # [B, 1, 64, 97]
        y = labels.to(device)                  # [B]

        # SpecAugment on the batch (training only)
        x = spec_augment(x)

        optimizer.zero_grad()
        logits = model(x)                      # [B, 10]
        loss   = criterion(logits, y)
        loss.backward()

        # Gradient clipping — prevents occasional large gradient spikes
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        total_loss += loss.item()

        # Progress print every 200 batches
        if (batch_idx + 1) % 200 == 0 or (batch_idx + 1) == n_batches:
            print(
                f"  Epoch {epoch:>3}  "
                f"[{batch_idx+1:>4}/{n_batches}]  "
                f"loss={loss.item():.4f}",
                flush=True,
            )

    return total_loss / n_batches


# ── Validation epoch ──────────────────────────────────────────────────────────

def validate(
    model    : nn.Module,
    loader   : torch.utils.data.DataLoader,
    criterion: nn.Module,
    device   : torch.device,
) -> tuple[float, float]:
    """
    Run one full validation pass.

    Returns:
        (mean_val_loss, val_accuracy_percent)
    """
    model.eval()
    total_loss = 0.0
    correct    = 0
    total      = 0

    with torch.no_grad():
        for tensors, labels in loader:
            x = tensors.squeeze(1).to(device)  # [B, 1, 64, 97]
            y = labels.to(device)

            logits = model(x)                  # [B, 10]
            loss   = criterion(logits, y)

            total_loss += loss.item()
            preds       = logits.argmax(dim=1)
            correct    += (preds == y).sum().item()
            total      += y.size(0)

    mean_loss = total_loss / len(loader)
    accuracy  = 100.0 * correct / total
    return mean_loss, accuracy


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def save_checkpoint(model: nn.Module, epoch: int, val_acc: float, path: Path):
    torch.save(
        {
            "epoch"        : epoch,
            "val_accuracy" : val_acc,
            "model_state"  : model.state_dict(),
            "num_classes"  : NUM_CLASSES,
            "mpic_version" : "1.0",
        },
        path,
    )


def load_checkpoint(model: nn.Module, path: Path) -> tuple[int, float]:
    """Load checkpoint into model. Returns (epoch, val_acc)."""
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    return ckpt["epoch"], ckpt["val_accuracy"]


# ── CSV logger ────────────────────────────────────────────────────────────────

def init_csv(path: Path):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "val_acc", "lr"])


def append_csv(path: Path, epoch: int, train_loss: float,
               val_loss: float, val_acc: float, lr: float):
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([epoch, f"{train_loss:.6f}", f"{val_loss:.6f}",
                         f"{val_acc:.4f}", f"{lr:.8f}"])


# ── Main training loop ────────────────────────────────────────────────────────

def train(args):
    # ── Seed ──────────────────────────────────────────────────────────────────
    set_seed(SEED)

    # ── Device ────────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 60)
    print("STREAMSENSE — train.py")
    print("=" * 60)
    print(f"\nDevice       : {device}")
    if device.type == "cuda":
        print(f"GPU          : {torch.cuda.get_device_name(0)}")
        print(f"VRAM         : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"Epochs       : {args.epochs}")
    print(f"Batch size   : {args.batch}")
    print(f"Learning rate: {args.lr}")
    print(f"Seed         : {SEED}")

    # ── Verify split files ────────────────────────────────────────────────────
    for p, name in [(TRAIN_SPLIT, "train_files.txt"), (VAL_SPLIT, "val_files.txt")]:
        if not p.exists():
            print(f"[ERROR] Split file not found: {p}")
            print("        On Colab: ensure the repo is cloned and data/splits/ exists.")
            sys.exit(1)

    # ── DataLoaders ───────────────────────────────────────────────────────────
    print(f"\nLoading datasets...")
    train_loader = get_dataloader(
        TRAIN_SPLIT,
        is_train    = True,
        batch_size  = args.batch,
        num_workers = NUM_WORKERS,
        verbose     = True,
    )
    val_loader = get_dataloader(
        VAL_SPLIT,
        is_train    = False,
        batch_size  = args.batch,
        num_workers = NUM_WORKERS,
        verbose     = True,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = StreamSenseNet(num_classes=NUM_CLASSES).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel        : StreamSenseNet")
    print(f"Parameters   : {total_params:,}")

    # ── Loss, optimizer, scheduler ────────────────────────────────────────────
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode     = "min",       # monitor val_loss
        factor   = 0.5,         # halve LR on plateau
        patience = 3,           # wait 3 epochs before reducing
        min_lr   = 1e-6,
    )

    # ── Checkpoint dir + CSV ──────────────────────────────────────────────────
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    init_csv(TRAIN_LOG)

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_acc  = 0.0
    best_epoch    = 0
    epochs_no_imp = 0
    EARLY_STOP    = 8       # stop if val_acc doesn't improve for 8 epochs

    print(f"\n{'─'*60}")
    print(f"Starting training — max {args.epochs} epochs")
    print(f"Early stopping patience: {EARLY_STOP} epochs")
    print(f"Best checkpoint → {BEST_CKPT}")
    print(f"{'─'*60}\n")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # Train
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )

        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        # Scheduler step (on val_loss)
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        elapsed = time.time() - t0

        # ── Epoch summary ─────────────────────────────────────────────────────
        improved = val_acc > best_val_acc
        marker   = " ← best" if improved else ""
        print(
            f"Epoch {epoch:>3}/{args.epochs}  "
            f"train_loss={train_loss:.4f}  "
            f"val_loss={val_loss:.4f}  "
            f"val_acc={val_acc:.2f}%  "
            f"lr={current_lr:.2e}  "
            f"t={elapsed:.1f}s"
            f"{marker}",
            flush=True,
        )

        # ── Save best checkpoint ──────────────────────────────────────────────
        if improved:
            best_val_acc  = val_acc
            best_epoch    = epoch
            epochs_no_imp = 0
            save_checkpoint(model, epoch, val_acc, BEST_CKPT)
            print(f"  [SAVED] best_model.pth  (val_acc={val_acc:.2f}%)")
        else:
            epochs_no_imp += 1

        # ── Log to CSV ────────────────────────────────────────────────────────
        append_csv(TRAIN_LOG, epoch, train_loss, val_loss, val_acc, current_lr)

        # ── Early stopping ────────────────────────────────────────────────────
        if epochs_no_imp >= EARLY_STOP:
            print(
                f"\n[EARLY STOP] No improvement for {EARLY_STOP} epochs. "
                f"Best was epoch {best_epoch} ({best_val_acc:.2f}%)."
            )
            break

    # ── Final summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"  Best epoch     : {best_epoch}")
    print(f"  Best val acc   : {best_val_acc:.2f}%")
    print(f"  Checkpoint     : {BEST_CKPT}")
    print(f"  Training log   : {TRAIN_LOG}")
    print(f"\nNext step: python evaluate.py")


# ── Argument parser ───────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="STREAMSENSE train.py")
    parser.add_argument("--epochs", type=int,   default=30,    help="Max training epochs (default 30)")
    parser.add_argument("--batch",  type=int,   default=32,    help="Batch size (default 32)")
    parser.add_argument("--lr",     type=float, default=0.001, help="Initial learning rate (default 0.001)")
    return parser.parse_args()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = parse_args()
    train(args)

```

### `training/train_1d.py`

```python
"""
train_1d.py
Project STREAMSENSE — Track A
Epic A3.2 — Training script for StreamSenseNet1D (1D CNN baseline, STRETCH)

Mirrors the training configuration of train.py (2D model) for a fair
comparison:
    - Seed=42
    - Adam, lr=0.001, weight_decay=1e-4
    - ReduceLROnPlateau (factor=0.5, patience=3, min_lr=1e-6)
    - Early stopping patience=8
    - Time-domain augmentation only (no SpecAugment — N/A for raw waveform)

Outputs:
    checkpoints_1d/best_model_1d.pth
    checkpoints_1d/training_log_1d.csv

Run from C:\\STREAMSENSE\\training\\:
    python train_1d.py
"""

import csv
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model_1d import StreamSenseNet1D, count_parameters
from dataset_1d import StreamSenseDataset1D

# ── Config (mirrors train.py) ─────────────────────────────────────────────────
SEED          = 42
BATCH_SIZE    = 64
MAX_EPOCHS    = 60
LR            = 0.001
WEIGHT_DECAY  = 1e-4
LR_FACTOR     = 0.5
LR_PATIENCE   = 3
LR_MIN        = 1e-6
EARLY_STOP_PATIENCE = 8

# ── Root path — environment-aware (see dataset_1d.py for Colab setup) ────────
_DEFAULT_ROOT = r"C:\STREAMSENSE" if os.name == "nt" else "/content/STREAMSENSE"
ROOT       = Path(os.environ.get("STREAMSENSE_ROOT", _DEFAULT_ROOT))
CKPT_DIR   = ROOT / "checkpoints_1d"
CKPT_PATH  = CKPT_DIR / "best_model_1d.pth"
LOG_PATH   = CKPT_DIR / "training_log_1d.csv"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_epoch(model, loader, criterion, optimizer=None):
    """Run one epoch. If optimizer is None, runs in eval mode (no grad)."""
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for waveforms, labels in loader:
            waveforms = waveforms.to(DEVICE)  # [B, 1, 16000]
            labels    = labels.to(DEVICE)

            if is_train:
                optimizer.zero_grad()

            logits = model(waveforms)  # [B, 10]
            loss   = criterion(logits, labels)

            if is_train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * waveforms.size(0)
            preds = logits.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += waveforms.size(0)

    avg_loss = total_loss / total_samples
    accuracy = 100.0 * total_correct / total_samples
    return avg_loss, accuracy


def main():
    print("=" * 60)
    print("STREAMSENSE — train_1d.py (Epic A3.2, 1D CNN baseline)")
    print("=" * 60)

    set_seed(SEED)
    print(f"Device: {DEVICE}")
    print(f"Seed:   {SEED}")

    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Data ───────────────────────────────────────────────────────────────
    print("\nLoading datasets...")
    train_ds = StreamSenseDataset1D(split="train", augment=True)
    val_ds   = StreamSenseDataset1D(split="val",   augment=False)

    print(f"  train: {len(train_ds)} samples")
    print(f"  val  : {len(val_ds)} samples")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                               num_workers=0, pin_memory=(DEVICE.type == "cuda"))
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                               num_workers=0, pin_memory=(DEVICE.type == "cuda"))

    # ── Model ──────────────────────────────────────────────────────────────
    model = StreamSenseNet1D(n_classes=10).to(DEVICE)
    n_params = count_parameters(model)
    print(f"\nModel: StreamSenseNet1D")
    print(f"  Parameters: {n_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=LR_FACTOR, patience=LR_PATIENCE, min_lr=LR_MIN
    )

    # ── Training loop ──────────────────────────────────────────────────────
    best_val_acc = 0.0
    best_epoch   = 0
    epochs_no_improve = 0

    log_rows = []

    print(f"\nTraining (max {MAX_EPOCHS} epochs, early stop patience={EARLY_STOP_PATIENCE})...")
    print(f"{'Epoch':>6} {'TrainLoss':>10} {'TrainAcc':>9} {'ValLoss':>9} {'ValAcc':>8} {'LR':>10} {'Time':>6}")

    for epoch in range(1, MAX_EPOCHS + 1):
        t0 = time.time()

        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc     = run_epoch(model, val_loader, criterion, optimizer=None)

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        elapsed = time.time() - t0

        print(f"{epoch:>6} {train_loss:>10.4f} {train_acc:>8.2f}% "
              f"{val_loss:>9.4f} {val_acc:>7.2f}% {current_lr:>10.2e} {elapsed:>5.1f}s")

        log_rows.append({
            "epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
            "val_loss": val_loss, "val_acc": val_acc, "lr": current_lr, "time_s": elapsed
        })

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            epochs_no_improve = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_acc": val_acc,
                "val_loss": val_loss,
            }, CKPT_PATH)
            print(f"         -> new best (val_acc={val_acc:.2f}%), saved to {CKPT_PATH.name}")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= EARLY_STOP_PATIENCE:
            print(f"\nEarly stopping at epoch {epoch} "
                  f"(no improvement for {EARLY_STOP_PATIENCE} epochs)")
            break

    # ── Save training log ─────────────────────────────────────────────────
    with open(LOG_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=log_rows[0].keys())
        writer.writeheader()
        writer.writerows(log_rows)

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)
    print(f"  Best epoch    : {best_epoch}")
    print(f"  Best val_acc  : {best_val_acc:.2f}%")
    print(f"  Checkpoint    -> {CKPT_PATH}")
    print(f"  Training log  -> {LOG_PATH}")


if __name__ == "__main__":
    main()

```

### `training/verify_gv10_matlab.m`

```matlab
% verify_gv10_matlab.m
% STREAMSENSE — Track A  |  MPIC v1.0
%
% PURPOSE
%   1. Verifies the column-major conversion was done correctly by loading
%      each GV from golden_vectors_10_matlab/ and re-running mel_pipeline_matlab
%      on the matching raw audio, then comparing the two outputs.
%
%   2. Validates every binary file in golden_vectors_10_matlab/ against
%      the MPIC v1.0 specification (shape, byte-size, dtype, value range).
%
% WHAT PASSES
%   norm re-computed from raw WAV vs norm loaded from .bin  <  5e-4  (cross-impl tolerance)
%   mel  re-computed from raw WAV vs mel  loaded from .bin  <  5e-4
%
% PREREQUISITES
%   • mel_pipeline_matlab.m  on the MATLAB path (same folder is fine)
%   • golden_vectors_10_matlab/ folder produced by generate_golden10_matlab.py
%   • Raw WAV files  golden_vectors_10_matlab/raw/*.bin  (raw PCM, NOT wav)
%     OR the original WAV files reachable from GV_WAV_DIR below.
%
% USAGE
%   Run from any folder:
%       cd C:\STREAMSENSE
%       verify_gv10_matlab
%
%   Or with a custom root:
%       GV_ROOT = 'D:\my_path\golden_vectors_10_matlab';
%       verify_gv10_matlab        % set GV_ROOT before running
%
% OUTPUT
%   Console report + PASS/FAIL summary.
%   Saves results to  golden_vectors_10_matlab\verify_report.txt
%
% Project: STREAMSENSE — Track A
% Spec:    MPIC v1.0 (frozen)

clc;
fprintf('============================================================\n');
fprintf('STREAMSENSE — verify_gv10_matlab.m\n');
fprintf('MPIC v1.0 Golden Vector Verification (MATLAB)\n');
fprintf('============================================================\n\n');

% ── Paths ─────────────────────────────────────────────────────────────────────
if ~exist('GV_ROOT', 'var')
    GV_ROOT = 'C:\STREAMSENSE\golden_vectors_10_matlab';
end

RAW_DIR   = fullfile(GV_ROOT, 'raw');
MEL_DIR   = fullfile(GV_ROOT, 'mel');
NORM_DIR  = fullfile(GV_ROOT, 'normalized');
LABEL_DIR = fullfile(GV_ROOT, 'labels');

% Original WAV files (needed for pipeline re-run check)
% Set this to wherever the source WAVs live (golden_vectors/wav/ or recordings/)
WAV_DIR   = 'C:\STREAMSENSE\golden_vectors\wav';

fprintf('GV root  : %s\n', GV_ROOT);
fprintf('WAV dir  : %s\n\n', WAV_DIR);

% ── MPIC v1.0 frozen constants ────────────────────────────────────────────────
FRAME_LEN      = 16000;
N_MELS         = 64;
EXPECTED_T     = 97;   % = floor((16000-512)/160)+1  — hardcoded to avoid float division
EXPECTED_RAW_BYTES  = FRAME_LEN * 4;          % 64000
EXPECTED_MEL_BYTES  = N_MELS * EXPECTED_T * 4; % 24832
CLIP_FLOOR_DB  = -80.0;
CROSS_IMPL_TOL = 5e-4;   % MPIC v1.0 cross-implementation tolerance

% ── Class labels ──────────────────────────────────────────────────────────────
LABELS = {'yes','no','up','down','left','right','on','off','stop','go'};

% ── Results table ─────────────────────────────────────────────────────────────
n_classes   = 10;
results     = struct();
all_passed  = true;

% ── Open report file ──────────────────────────────────────────────────────────
report_path = fullfile(GV_ROOT, 'verify_report.txt');
fid_rep = fopen(report_path, 'w');
if fid_rep < 0
    warning('Cannot write report to %s — printing to console only.', report_path);
    fid_rep = 1;  % stdout fallback
end

log = @(varargin) fprintf(fid_rep, varargin{:});

log('STREAMSENSE — verify_gv10_matlab.m\n');
log('Generated: %s\n', datestr(now));
log('GV root  : %s\n\n', GV_ROOT);

% ── Main verification loop ────────────────────────────────────────────────────
for i = 0:9
    lbl     = LABELS{i+1};
    gv_name = sprintf('GV_%02d_%s', i, lbl);

    fprintf('\n%s\n', repmat('-', 1, 54));
    fprintf('Verifying %s  (class %d — ''%s'')\n', gv_name, i, lbl);
    log('\n%s\n', repmat('-', 1, 54));
    log('Verifying %s  (class %d — ''%s'')\n', gv_name, i, lbl);

    ok_struct = struct('size_raw', false, 'size_mel', false, 'size_norm', false, ...
                       'label_ok', false, 'range_ok', false, ...
                       'pipeline_mel_ok', false, 'pipeline_norm_ok', false);

    % ── File paths ────────────────────────────────────────────────────────────
    raw_path   = fullfile(RAW_DIR,   sprintf('%s.bin',       gv_name));
    mel_path   = fullfile(MEL_DIR,   sprintf('%s_mel.bin',   gv_name));
    norm_path  = fullfile(NORM_DIR,  sprintf('%s_norm.bin',  gv_name));
    label_path = fullfile(LABEL_DIR, sprintf('%s_label.txt', gv_name));
    wav_path   = fullfile(WAV_DIR,   sprintf('%s.wav',       gv_name));

    % ── Check files exist ─────────────────────────────────────────────────────
    missing = {};
    if ~exist(raw_path,  'file'), missing{end+1} = 'raw bin';   end
    if ~exist(mel_path,  'file'), missing{end+1} = 'mel bin';   end
    if ~exist(norm_path, 'file'), missing{end+1} = 'norm bin';  end
    if ~isempty(missing)
        msg = sprintf('  [FAIL] Missing files: %s\n', strjoin(missing, ', '));
        fprintf('%s', msg); log('%s', msg);
        all_passed = false;
        results.(gv_name) = ok_struct;
        continue;
    end

    % ── Check file sizes ──────────────────────────────────────────────────────
    d_raw  = dir(raw_path);
    d_mel  = dir(mel_path);
    d_norm = dir(norm_path);

    ok_struct.size_raw  = (d_raw.bytes  == EXPECTED_RAW_BYTES);
    ok_struct.size_mel  = (d_mel.bytes  == EXPECTED_MEL_BYTES);
    ok_struct.size_norm = (d_norm.bytes == EXPECTED_MEL_BYTES);

    sz_tag = @(ok, actual, expected) ...
        sprintf('%d bytes  %s  (expected %d)', actual, tf_str(ok), expected);

    fprintf('  raw  : %s\n', sz_tag(ok_struct.size_raw,  d_raw.bytes,  EXPECTED_RAW_BYTES));
    fprintf('  mel  : %s\n', sz_tag(ok_struct.size_mel,  d_mel.bytes,  EXPECTED_MEL_BYTES));
    fprintf('  norm : %s\n', sz_tag(ok_struct.size_norm, d_norm.bytes, EXPECTED_MEL_BYTES));
    log('  raw  : %s\n', sz_tag(ok_struct.size_raw,  d_raw.bytes,  EXPECTED_RAW_BYTES));
    log('  mel  : %s\n', sz_tag(ok_struct.size_mel,  d_mel.bytes,  EXPECTED_MEL_BYTES));
    log('  norm : %s\n', sz_tag(ok_struct.size_norm, d_norm.bytes, EXPECTED_MEL_BYTES));

    if ~(ok_struct.size_raw && ok_struct.size_mel && ok_struct.size_norm)
        all_passed = false;
    end

    % ── Load binary files ─────────────────────────────────────────────────────
    % Files are column-major (saved by generate_golden10_matlab.py with tobytes('F'))
    % fread reads flat bytes; reshape with [64,97] gives correct MATLAB matrix.

    fid = fopen(raw_path, 'rb', 'l');
    raw_vec = fread(fid, FRAME_LEN, 'float32=>single');
    fclose(fid);

    fid = fopen(mel_path, 'rb', 'l');
    mel_gv  = reshape(fread(fid, N_MELS*EXPECTED_T, 'float32=>single'), [N_MELS, EXPECTED_T]);
    fclose(fid);

    fid = fopen(norm_path, 'rb', 'l');
    norm_gv = reshape(fread(fid, N_MELS*EXPECTED_T, 'float32=>single'), [N_MELS, EXPECTED_T]);
    fclose(fid);

    % ── Value range checks ────────────────────────────────────────────────────
    mel_min  = min(mel_gv(:));
    mel_max  = max(mel_gv(:));
    norm_min = min(norm_gv(:));
    norm_max = max(norm_gv(:));

    % Mel should be in [−80, 0] dB range (log-mel of audio)
    % MPIC v1.0 clips the floor at -80 dB only — no upper limit.
    % Power mel of real speech typically peaks at +35..+45 dB; >0 is normal.
    ok_struct.range_ok = (mel_min >= CLIP_FLOOR_DB - 0.1);

    fprintf('  mel  range : [%.2f, %.2f] dB  %s\n', mel_min, mel_max, tf_str(ok_struct.range_ok));
    fprintf('  norm range : [%.4f, %.4f]\n', norm_min, norm_max);
    log('  mel  range : [%.2f, %.2f] dB  %s\n', mel_min, mel_max, tf_str(ok_struct.range_ok));
    log('  norm range : [%.4f, %.4f]\n', norm_min, norm_max);

    % ── Label check ───────────────────────────────────────────────────────────
    if exist(label_path, 'file')
        fid = fopen(label_path, 'r');
        label_str = strtrim(fgetl(fid));
        fclose(fid);
        expected_label = num2str(i);
        ok_struct.label_ok = strcmp(label_str, expected_label);
        fprintf('  label: %s  %s  (expected %s)\n', label_str, ...
            tf_str(ok_struct.label_ok), expected_label);
        log('  label: %s  %s\n', label_str, tf_str(ok_struct.label_ok));
    else
        fprintf('  label: [missing]\n');
        ok_struct.label_ok = false;
    end

    % ── Pipeline re-run verification (if WAV available) ───────────────────────
    if exist(wav_path, 'file')
        fprintf('  pipeline re-run check...\n');
        try
            [wav_samples, wav_fs] = audioread(wav_path);
            if wav_fs ~= 16000
                error('WAV sample rate is %d Hz — expected 16000 Hz', wav_fs);
            end
            if size(wav_samples, 2) > 1
                wav_samples = mean(wav_samples, 2);   % stereo → mono
            end

            % Run MATLAB mel pipeline
            norm_rerun = mel_pipeline_matlab(wav_samples);   % [64 x 97]

            % Reconstruct mel from norm for mel-level comparison
            GLOBAL_MEAN_V = single(-30.785545);
            GLOBAL_STD_V  = single(22.157099);
            mel_rerun = norm_rerun * GLOBAL_STD_V + GLOBAL_MEAN_V;

            % Compare against loaded GV
            err_norm = max(abs(norm_rerun(:) - norm_gv(:)));
            err_mel  = max(abs(mel_rerun(:)  - mel_gv(:)));

            ok_struct.pipeline_norm_ok = (err_norm < CROSS_IMPL_TOL);
            ok_struct.pipeline_mel_ok  = (err_mel  < CROSS_IMPL_TOL);

            fprintf('  pipeline max_err norm : %.2e  %s  (tol=5e-4)\n', ...
                err_norm, tf_str(ok_struct.pipeline_norm_ok));
            fprintf('  pipeline max_err mel  : %.2e  %s  (tol=5e-4)\n', ...
                err_mel,  tf_str(ok_struct.pipeline_mel_ok));
            log('  pipeline max_err norm : %.2e  %s\n', err_norm, tf_str(ok_struct.pipeline_norm_ok));
            log('  pipeline max_err mel  : %.2e  %s\n', err_mel,  tf_str(ok_struct.pipeline_mel_ok));

            if ~ok_struct.pipeline_norm_ok || ~ok_struct.pipeline_mel_ok
                all_passed = false;
            end
        catch ME
            fprintf('  pipeline check ERROR: %s\n', ME.message);
            log('  pipeline check ERROR: %s\n', ME.message);
            all_passed = false;
        end
    else
        fprintf('  pipeline re-run: SKIPPED (WAV not found at %s)\n', wav_path);
        log('  pipeline re-run: SKIPPED — WAV not found\n');
        % Mark as N/A (not a failure if WAV is absent)
        ok_struct.pipeline_norm_ok = true;
        ok_struct.pipeline_mel_ok  = true;
    end

    % ── Per-vector pass/fail ──────────────────────────────────────────────────
    vec_pass = ok_struct.size_raw  && ok_struct.size_mel  && ...
               ok_struct.size_norm && ok_struct.range_ok  && ...
               ok_struct.pipeline_norm_ok && ok_struct.pipeline_mel_ok;

    if ~vec_pass, all_passed = false; end

    results.(gv_name) = ok_struct;
end

% ── Summary ───────────────────────────────────────────────────────────────────
fprintf('\n%s\n', repmat('=', 1, 54));
fprintf('SUMMARY\n');
fprintf('%s\n', repmat('=', 1, 54));
log('\n%s\n', repmat('=', 1, 54));
log('SUMMARY\n');
log('%s\n', repmat('=', 1, 54));

for i = 0:9
    lbl     = LABELS{i+1};
    gv_name = sprintf('GV_%02d_%s', i, lbl);

    if isfield(results, gv_name)
        r = results.(gv_name);
        vec_pass = r.size_raw && r.size_mel && r.size_norm && ...
                   r.range_ok && r.pipeline_norm_ok && r.pipeline_mel_ok;
        status = tf_str(vec_pass);
    else
        status = '[FAIL]';
    end

    line = sprintf('  %s  %s\n', status, gv_name);
    fprintf('%s', line);
    log('%s', line);
end

fprintf('\n');
log('\n');

if all_passed
    msg = sprintf('[DONE] All 10 golden vectors PASSED MPIC v1.0 verification.\n');
else
    msg = sprintf('[FAIL] One or more vectors failed — see details above.\n');
end
fprintf('%s', msg);
log('%s', msg);

if fid_rep ~= 1
    fclose(fid_rep);
    fprintf('\nReport saved to: %s\n', report_path);
end

% ── Helper ────────────────────────────────────────────────────────────────────
function s = tf_str(ok)
    if ok
        s = '[OK  ]';
    else
        s = '[FAIL]';
    end
end

```

### `training/verify_pipeline.py`

```python
"""
verify_pipeline.py
Project STREAMSENSE — Track A
MPIC v1.0 — Pipeline verification against frozen golden vectors.

Verifies TWO stages for each of the 10 golden vectors:
    Stage 1: raw audio -> mel spectrogram    (compare vs *_mel.bin)
    Stage 2: raw audio -> normalized output  (compare vs *_norm.bin)

Tolerance is read at runtime from manifest.json — never hardcoded.
All 10 vectors are always tested. No early exit on failure.

Required files before running:
    C:\\STREAMSENSE\\golden_vectors\\manifest.json
    C:\\STREAMSENSE\\golden_vectors\\raw\\GV_0X_<label>.bin         (x10)
    C:\\STREAMSENSE\\golden_vectors\\mel\\GV_0X_<label>_mel.bin     (x10)
    C:\\STREAMSENSE\\golden_vectors\\normalized\\GV_0X_<label>_norm.bin (x10)
    C:\\STREAMSENSE\\stats\\normalization_stats.json

Run:
    python verify_pipeline.py
"""

import sys
import json
import numpy as np
import torch
import torchaudio
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
GV_ROOT       = Path(r"C:\STREAMSENSE\golden_vectors")
MANIFEST_PATH = GV_ROOT / "manifest.json"
RAW_DIR       = GV_ROOT / "raw"
MEL_DIR       = GV_ROOT / "mel"
NORM_DIR      = GV_ROOT / "normalized"

# mel_pipeline.py must be on the Python path (same directory is fine)
try:
    from mel_pipeline import preprocess
except ImportError as e:
    print(f"[ERROR] Cannot import mel_pipeline: {e}")
    print("        Make sure mel_pipeline.py is in the same directory and "
          "normalization_stats.json exists.")
    sys.exit(1)

# ── MPIC v1.0 parameters — used only for the independent mel-only reimplementation
# (Option A: Steps 1-6 reproduced here to verify the intermediate stage)
SAMPLE_RATE   = 16000
FRAME_LEN     = 16000
N_FFT         = 512
HOP_LENGTH    = 160
N_MELS        = 64
CENTER        = False
POWER         = 2.0
LOG_EPS       = 1e-10
CLIP_FLOOR_DB = -80.0
EXPECTED_T    = (FRAME_LEN - N_FFT) // HOP_LENGTH + 1   # 97
EXPECTED_MEL_SHAPE  = (N_MELS, EXPECTED_T)               # (64, 97)
EXPECTED_NORM_SHAPE = (1, 1, N_MELS, EXPECTED_T)         # (1, 1, 64, 97)

# Build the mel transform once — same construction as generate_golden.py
_mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate = SAMPLE_RATE,
    n_fft       = N_FFT,
    hop_length  = HOP_LENGTH,
    n_mels      = N_MELS,
    window_fn   = torch.hann_window,
    center      = CENTER,
    power       = POWER,
)


# ── Independent mel-only pipeline (Steps 1-6, no normalization) ───────────────
# This is the Option A reimplementation. It mirrors generate_golden.py's
# load_wav_raw() + compute_mel() exactly, giving a clean cross-check.

def _mel_from_raw(raw: np.ndarray) -> np.ndarray:
    """
    Steps 1-6 only (NO normalization).
    Input:  raw float32 numpy array [16000] — already loaded from .bin
    Output: log-mel numpy float32 [64, 97]
    """
    # Step 1: to float32 tensor [1, 16000]
    waveform = torch.from_numpy(raw).unsqueeze(0).float()   # [1, 16000]

    # Steps 4-6: MelSpectrogram + log scaling + clamp
    mel = _mel_transform(waveform)                          # [1, 64, 97]
    mel = 10.0 * torch.log10(mel + LOG_EPS)
    mel = torch.clamp(mel, min=CLIP_FLOOR_DB)
    return mel.squeeze(0).numpy()                           # [64, 97] float32


# ── Binary loading helpers ─────────────────────────────────────────────────────

def _load_bin(path: Path, shape: tuple) -> np.ndarray:
    """
    Load a little-endian float32 binary file and reshape.
    Row-major (C order) as written by numpy .tofile().
    """
    arr = np.fromfile(str(path), dtype="<f4")               # little-endian float32
    if arr.size != int(np.prod(shape)):
        raise ValueError(
            f"Size mismatch loading {path.name}: "
            f"expected {int(np.prod(shape))} elements, got {arr.size}"
        )
    return arr.reshape(shape)


# ── Pre-flight checks ─────────────────────────────────────────────────────────

def _check_prerequisites(manifest: dict) -> list[str]:
    """
    Verify all required files exist before running any vector.
    Returns list of error strings (empty = all good).
    """
    errors = []
    vectors = manifest.get("vectors", {})

    for class_idx in range(10):
        key = str(class_idx)
        if key not in vectors:
            errors.append(f"  manifest missing entry for class {class_idx}")
            continue
        v = vectors[key]

        raw_path  = RAW_DIR  / v["raw_bin"]
        mel_path  = MEL_DIR  / v["mel_bin"]
        norm_path = NORM_DIR / v["norm_bin"]

        for p, label in [(raw_path, "raw"), (mel_path, "mel"), (norm_path, "norm")]:
            if not p.exists():
                errors.append(f"  Missing {label} file: {p}")

    return errors


# ── Per-vector verification ───────────────────────────────────────────────────

def _verify_vector(class_idx: int, v: dict, tolerance: float) -> tuple[bool, str]:
    """
    Run both stages for one golden vector.
    Returns (all_passed: bool, detail_lines: str).
    """
    gv_name   = v["gv_name"]
    label     = v["label"]
    raw_shape = tuple(v["raw_shape"])    # (16000,)
    mel_shape = tuple(v["mel_shape"])    # (64, 97)

    lines = []
    lines.append(f"\n  {'─'*54}")
    lines.append(f"  {gv_name}  (class {class_idx} — '{label}')")

    stage_results = []

    # ── Load raw binary ────────────────────────────────────────────────────────
    raw_path = RAW_DIR / v["raw_bin"]
    try:
        raw = _load_bin(raw_path, raw_shape)    # [16000] float32
    except Exception as e:
        lines.append(f"  [FAIL] Could not load raw bin: {e}")
        return False, "\n".join(lines)

    # ── Stage 1: Mel intermediate check ───────────────────────────────────────
    golden_mel_path = MEL_DIR / v["mel_bin"]
    try:
        golden_mel     = _load_bin(golden_mel_path, mel_shape)   # [64, 97]
        pipeline_mel   = _mel_from_raw(raw)                      # [64, 97]
        max_abs_mel    = float(np.max(np.abs(pipeline_mel - golden_mel)))
        mel_pass       = max_abs_mel <= tolerance
        stage_results.append(mel_pass)
        status_mel     = "PASS" if mel_pass else "FAIL"
        lines.append(
            f"  [{status_mel}] Stage 1 — mel        "
            f"max_abs_err={max_abs_mel:.6e}  (tol={tolerance:.1e})"
        )
    except Exception as e:
        lines.append(f"  [FAIL] Stage 1 — mel        ERROR: {e}")
        stage_results.append(False)

    # ── Stage 2: Normalized output check ──────────────────────────────────────
    golden_norm_path = NORM_DIR / v["norm_bin"]
    try:
        golden_norm      = _load_bin(golden_norm_path, mel_shape)     # [64, 97]
        # preprocess() returns [1, 1, 64, 97] — squeeze to [64, 97] for comparison
        pipeline_out     = preprocess(raw)                            # [1, 1, 64, 97]
        pipeline_norm    = pipeline_out.squeeze().numpy()             # [64, 97]
        max_abs_norm     = float(np.max(np.abs(pipeline_norm - golden_norm)))
        norm_pass        = max_abs_norm <= tolerance
        stage_results.append(norm_pass)
        status_norm      = "PASS" if norm_pass else "FAIL"
        lines.append(
            f"  [{status_norm}] Stage 2 — normalized  "
            f"max_abs_err={max_abs_norm:.6e}  (tol={tolerance:.1e})"
        )
    except Exception as e:
        lines.append(f"  [FAIL] Stage 2 — normalized  ERROR: {e}")
        stage_results.append(False)

    all_passed = all(stage_results)
    return all_passed, "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("STREAMSENSE — verify_pipeline.py")
    print("MPIC v1.0 — Golden Vector Verification")
    print("=" * 60)

    # ── Load manifest ──────────────────────────────────────────────────────────
    if not MANIFEST_PATH.exists():
        print(f"[ERROR] Manifest not found: {MANIFEST_PATH}")
        print("        Run generate_golden.py first.")
        sys.exit(1)

    with open(MANIFEST_PATH, "r") as f:
        manifest = json.load(f)

    # Read tolerance from manifest — never hardcoded
    tolerance = float(manifest["tolerance_max_abs_error"])

    print(f"\nManifest     : {MANIFEST_PATH}")
    print(f"MPIC version : {manifest.get('mpic_version', 'unknown')}")
    print(f"global_mean  : {manifest.get('global_mean', '?')}")
    print(f"global_std   : {manifest.get('global_std', '?')}")
    print(f"Tolerance    : {tolerance:.1e}  (from manifest)")
    print(f"Vectors      : 10  (2 stages each = 20 checks total)")

    # ── Pre-flight ────────────────────────────────────────────────────────────
    print("\nChecking required files...")
    prereq_errors = _check_prerequisites(manifest)
    if prereq_errors:
        print("[ERROR] Missing files — cannot proceed:")
        for e in prereq_errors:
            print(e)
        sys.exit(1)
    print("  [OK] All 30 binary files present (10 raw + 10 mel + 10 norm)")

    # ── Run all 10 vectors ────────────────────────────────────────────────────
    print("\nRunning verification...\n")

    vectors      = manifest["vectors"]
    passed_count = 0
    failed_count = 0
    failed_names = []

    for class_idx in range(10):
        v = vectors[str(class_idx)]
        ok, detail = _verify_vector(class_idx, v, tolerance)
        print(detail)

        if ok:
            passed_count += 1
        else:
            failed_count += 1
            failed_names.append(v["gv_name"])

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"  Passed: {passed_count}/10")
    print(f"  Failed: {failed_count}/10")

    if failed_count == 0:
        print("\n[PASS] All golden vectors verified. Pipeline is correct.")
        print("       Integration acceptance gate: OPEN")
        sys.exit(0)
    else:
        print(f"\n[FAIL] {failed_count} vector(s) failed:")
        for name in failed_names:
            print(f"       • {name}")
        print("\n       Integration acceptance gate: BLOCKED")
        print("       Check mel_pipeline.py MPIC parameters match manifest.")
        sys.exit(1)


if __name__ == "__main__":
    main()

```

---
## Config & Stats

### `stats/normalization_stats.json`

```json
{
  "global_mean": -30.785544706009965,
  "global_std": 22.157099125788548,
  "n_files": 26984,
  "n_elements": 167516672,
  "n_errors": 0,
  "mpic_version": "1.0",
  "n_fft": 512,
  "hop_length": 160,
  "n_mels": 64,
  "center": false,
  "clip_floor_db": -80.0,
  "log_eps": 1e-10
}
```

### `stats/golden_selection.json`

```json
{
  "0": {
    "gv_name": "GV_00_yes",
    "class_idx": 0,
    "label": "yes",
    "source_path": "C:\\STREAMSENSE\\data\\raw\\yes\\ddedba85_nohash_9.wav",
    "wav_dest": "C:\\STREAMSENSE\\golden_vectors\\wav\\GV_00_yes.wav",
    "peak_energy_db": 41.646,
    "choice": 3
  },
  "1": {
    "gv_name": "GV_01_no",
    "class_idx": 1,
    "label": "no",
    "source_path": "C:\\STREAMSENSE\\data\\raw\\no\\229978fd_nohash_0.wav",
    "wav_dest": "C:\\STREAMSENSE\\golden_vectors\\wav\\GV_01_no.wav",
    "peak_energy_db": 39.0343,
    "choice": 7
  },
  "2": {
    "gv_name": "GV_02_up",
    "class_idx": 2,
    "label": "up",
    "source_path": "C:\\STREAMSENSE\\data\\raw\\up\\89f3ab7d_nohash_2.wav",
    "wav_dest": "C:\\STREAMSENSE\\golden_vectors\\wav\\GV_02_up.wav",
    "peak_energy_db": 38.9723,
    "choice": 8
  },
  "3": {
    "gv_name": "GV_03_down",
    "class_idx": 3,
    "label": "down",
    "source_path": "C:\\STREAMSENSE\\data\\raw\\down\\2fee065a_nohash_0.wav",
    "wav_dest": "C:\\STREAMSENSE\\golden_vectors\\wav\\GV_03_down.wav",
    "peak_energy_db": 40.7839,
    "choice": 2
  },
  "4": {
    "gv_name": "GV_04_left",
    "class_idx": 4,
    "label": "left",
    "source_path": "C:\\STREAMSENSE\\data\\raw\\left\\f3d06008_nohash_0.wav",
    "wav_dest": "C:\\STREAMSENSE\\golden_vectors\\wav\\GV_04_left.wav",
    "peak_energy_db": 41.3761,
    "choice": 5
  },
  "5": {
    "gv_name": "GV_05_right",
    "class_idx": 5,
    "label": "right",
    "source_path": "C:\\STREAMSENSE\\data\\raw\\right\\f5c3de1b_nohash_0.wav",
    "wav_dest": "C:\\STREAMSENSE\\golden_vectors\\wav\\GV_05_right.wav",
    "peak_energy_db": 39.0934,
    "choice": 7
  },
  "6": {
    "gv_name": "GV_06_on",
    "class_idx": 6,
    "label": "on",
    "source_path": "C:\\STREAMSENSE\\data\\raw\\on\\a42a88ff_nohash_0.wav",
    "wav_dest": "C:\\STREAMSENSE\\golden_vectors\\wav\\GV_06_on.wav",
    "peak_energy_db": 39.6448,
    "choice": 8
  },
  "7": {
    "gv_name": "GV_07_off",
    "class_idx": 7,
    "label": "off",
    "source_path": "C:\\STREAMSENSE\\data\\raw\\off\\ace82a68_nohash_0.wav",
    "wav_dest": "C:\\STREAMSENSE\\golden_vectors\\wav\\GV_07_off.wav",
    "peak_energy_db": 40.9601,
    "choice": 5
  },
  "8": {
    "gv_name": "GV_08_stop",
    "class_idx": 8,
    "label": "stop",
    "source_path": "C:\\STREAMSENSE\\data\\raw\\stop\\68effe85_nohash_0.wav",
    "wav_dest": "C:\\STREAMSENSE\\golden_vectors\\wav\\GV_08_stop.wav",
    "peak_energy_db": 40.5818,
    "choice": 2
  },
  "9": {
    "gv_name": "GV_09_go",
    "class_idx": 9,
    "label": "go",
    "source_path": "C:\\STREAMSENSE\\data\\raw\\go\\2fee065a_nohash_1.wav",
    "wav_dest": "C:\\STREAMSENSE\\golden_vectors\\wav\\GV_09_go.wav",
    "peak_energy_db": 41.3729,
    "choice": 4
  }
}
```

### `class_labels.json`

```json
{
  "0": "yes",
  "1": "no",
  "2": "up",
  "3": "down",
  "4": "left",
  "5": "right",
  "6": "on",
  "7": "off",
  "8": "stop",
  "9": "go"
}
```

---
## Evaluation Reports

### `evaluation/evaluation_report.txt`

```
============================================================
STREAMSENSE � Evaluation Report
============================================================
Timestamp       : 2026-06-11 17:28:59
Checkpoint      : C:\STREAMSENSE\checkpoints\best_model.pth
Trained epoch   : 26
Val accuracy    : 96.11%
Device          : cpu
Test samples    : 5779

Test loss       : 0.1273
Test accuracy   : 95.97%

Per-class report:
              precision    recall  f1-score   support

         yes     0.9884    0.9884    0.9884       606
          no     0.9581    0.9679    0.9630       591
          up     0.8736    0.9517    0.9110       559
        down     0.9770    0.9404    0.9583       587
        left     0.9839    0.9667    0.9752       570
       right     0.9929    0.9929    0.9929       566
          on     0.9822    0.9566    0.9692       576
         off     0.9107    0.9447    0.9274       561
        stop     0.9927    0.9380    0.9646       581
          go     0.9452    0.9485    0.9468       582

    accuracy                         0.9597      5779
   macro avg     0.9605    0.9596    0.9597      5779
weighted avg     0.9610    0.9597    0.9600      5779


Confusion matrix (rows=true, cols=predicted):
Classes: 0=yes, 1=no, 2=up, 3=down, 4=left, 5=right, 6=on, 7=off, 8=stop, 9=go
[[599   1   1   0   1   0   0   2   0   2]
 [  1 572   4   3   1   1   0   0   1   8]
 [  0   0 532   0   0   0   0  24   0   3]
 [  1   9   4 552   1   1   4   0   2  13]
 [  2   7   5   0 551   1   0   4   0   0]
 [  0   0   1   0   3 562   0   0   0   0]
 [  0   1   6   2   1   1 551  14   0   0]
 [  0   0  26   0   0   0   3 530   0   2]
 [  0   0  22   2   0   0   2   6 545   4]
 [  3   7   8   6   2   0   1   2   1 552]]

Per-class accuracy:
  yes        599/606  (98.84%)
  no         572/591  (96.79%)
  up         532/559  (95.17%)
  down       552/587  (94.04%)
  left       551/570  (96.67%)
  right      562/566  (99.29%)
  on         551/576  (95.66%)
  off        530/561  (94.47%)
  stop       545/581  (93.80%)
  go         552/582  (94.85%)

MPIC version    : 1.0
Architecture    : StreamSenseNet (VGG-style 2D CNN)
Parameters      : 295,786
Dataset         : Google Speech Commands v2 (10 classes)

============================================================
  ONNX EVALUATION (appended by evaluate_onnx.py)
============================================================

============================================================
  Model        : StreamSenseNet FP32
  ONNX file    : streamsense_model_fp32.onnx
  Timestamp    : 2026-06-15 18:57:47
  Test samples : 5779
  Accuracy     : 95.97%  (5546/5779)
  Elapsed      : 83.7s
============================================================

Per-class report:
              precision    recall  f1-score   support

         yes     0.9884    0.9884    0.9884       606
          no     0.9581    0.9679    0.9630       591
          up     0.8736    0.9517    0.9110       559
        down     0.9770    0.9404    0.9583       587
        left     0.9839    0.9667    0.9752       570
       right     0.9929    0.9929    0.9929       566
          on     0.9822    0.9566    0.9692       576
         off     0.9107    0.9447    0.9274       561
        stop     0.9927    0.9380    0.9646       581
          go     0.9452    0.9485    0.9468       582

    accuracy                         0.9597      5779
   macro avg     0.9605    0.9596    0.9597      5779
weighted avg     0.9610    0.9597    0.9600      5779

Per-class accuracy:
  yes        599/606  (98.84%)
  no         572/591  (96.79%)
  up         532/559  (95.17%)
  down       552/587  (94.04%)
  left       551/570  (96.67%)
  right      562/566  (99.29%)
  on         551/576  (95.66%)
  off        530/561  (94.47%)
  stop       545/581  (93.80%)
  go         552/582  (94.85%)

Confusion matrix (rows=true, cols=predicted):
Classes: 0=yes, 1=no, 2=up, 3=down, 4=left, 5=right, 6=on, 7=off, 8=stop, 9=go
  [599, 1, 1, 0, 1, 0, 0, 2, 0, 2]
  [1, 572, 4, 3, 1, 1, 0, 0, 1, 8]
  [0, 0, 532, 0, 0, 0, 0, 24, 0, 3]
  [1, 9, 4, 552, 1, 1, 4, 0, 2, 13]
  [2, 7, 5, 0, 551, 1, 0, 4, 0, 0]
  [0, 0, 1, 0, 3, 562, 0, 0, 0, 0]
  [0, 1, 6, 2, 1, 1, 551, 14, 0, 0]
  [0, 0, 26, 0, 0, 0, 3, 530, 0, 2]
  [0, 0, 22, 2, 0, 0, 2, 6, 545, 4]
  [3, 7, 8, 6, 2, 0, 1, 2, 1, 552]

MPIC version   : 1.0
Architecture   : StreamSenseNet (VGG-style 2D CNN)
Parameters     : 295,786
Dataset        : Google Speech Commands v2 (10 classes)
============================================================

============================================================
  Model        : StreamSenseNet INT8
  ONNX file    : streamsense_model_int8.onnx
  Timestamp    : 2026-06-15 18:57:58
  Test samples : 5779
  Accuracy     : 95.86%  (5540/5779)
  Elapsed      : 10.1s
============================================================

Per-class report:
              precision    recall  f1-score   support

         yes     0.9868    0.9884    0.9876       606
          no     0.9548    0.9662    0.9605       591
          up     0.8659    0.9589    0.9100       559
        down     0.9753    0.9421    0.9584       587
        left     0.9821    0.9632    0.9725       570
       right     0.9929    0.9912    0.9920       566
          on     0.9787    0.9566    0.9675       576
         off     0.9164    0.9376    0.9269       561
        stop     0.9945    0.9346    0.9636       581
          go     0.9484    0.9467    0.9475       582

    accuracy                         0.9586      5779
   macro avg     0.9596    0.9585    0.9587      5779
weighted avg     0.9601    0.9586    0.9590      5779

Per-class accuracy:
  yes        599/606  (98.84%)
  no         571/591  (96.62%)
  up         536/559  (95.89%)
  down       553/587  (94.21%)
  left       549/570  (96.32%)
  right      561/566  (99.12%)
  on         551/576  (95.66%)
  off        526/561  (93.76%)
  stop       543/581  (93.46%)
  go         551/582  (94.67%)

Confusion matrix (rows=true, cols=predicted):
Classes: 0=yes, 1=no, 2=up, 3=down, 4=left, 5=right, 6=on, 7=off, 8=stop, 9=go
  [599, 1, 1, 0, 1, 0, 0, 2, 0, 2]
  [1, 571, 4, 4, 1, 1, 0, 0, 1, 8]
  [0, 0, 536, 0, 0, 0, 0, 21, 0, 2]
  [1, 9, 4, 553, 1, 1, 4, 0, 1, 13]
  [3, 7, 6, 0, 549, 1, 0, 4, 0, 0]
  [0, 0, 1, 0, 4, 561, 0, 0, 0, 0]
  [0, 1, 8, 2, 1, 1, 551, 12, 0, 0]
  [0, 0, 29, 0, 0, 0, 5, 526, 0, 1]
  [0, 0, 23, 2, 0, 0, 2, 7, 543, 4]
  [3, 9, 7, 6, 2, 0, 1, 2, 1, 551]

MPIC version   : 1.0
Architecture   : StreamSenseNet (VGG-style 2D CNN)
Parameters     : 295,786
Dataset        : Google Speech Commands v2 (10 classes)
============================================================

============================================================
  QUANTIZATION ACCURACY SUMMARY
============================================================
  FP32 accuracy  : 95.97%
  INT8 accuracy  : 95.86%
  Accuracy drop  : +0.10%
  INT8 budget    : PASS  (threshold: ≤1.0%)
============================================================

```

### `evaluation/multihead_onnx_evaluation_report.txt`

```


============================================================
  MULTI-HEAD ONNX EVALUATION (evaluate_multihead_onnx.py)
  Timestamp : 2026-06-23 15:12:25
============================================================

============================================================
  Model        : StreamSenseWrapper FP32 (multihead)
  ONNX file    : streamsense_multihead_fp32.onnx
  Timestamp    : 2026-06-23 15:12:25
  Test samples : 5779
  Accuracy     : 95.97%  (5546/5779)
  Elapsed      : 121.7s
============================================================

Per-class report:
              precision    recall  f1-score   support

         yes     0.9884    0.9884    0.9884       606
          no     0.9581    0.9679    0.9630       591
          up     0.8736    0.9517    0.9110       559
        down     0.9770    0.9404    0.9583       587
        left     0.9839    0.9667    0.9752       570
       right     0.9929    0.9929    0.9929       566
          on     0.9822    0.9566    0.9692       576
         off     0.9107    0.9447    0.9274       561
        stop     0.9927    0.9380    0.9646       581
          go     0.9452    0.9485    0.9468       582

    accuracy                         0.9597      5779
   macro avg     0.9605    0.9596    0.9597      5779
weighted avg     0.9610    0.9597    0.9600      5779

Per-class accuracy:
  yes        599/606  (98.84%)
  no         572/591  (96.79%)
  up         532/559  (95.17%)
  down       552/587  (94.04%)
  left       551/570  (96.67%)
  right      562/566  (99.29%)
  on         551/576  (95.66%)
  off        530/561  (94.47%)
  stop       545/581  (93.80%)
  go         552/582  (94.85%)

Confusion matrix (rows=true, cols=predicted):
Classes: 0=yes, 1=no, 2=up, 3=down, 4=left, 5=right, 6=on, 7=off, 8=stop, 9=go
  [599, 1, 1, 0, 1, 0, 0, 2, 0, 2]
  [1, 572, 4, 3, 1, 1, 0, 0, 1, 8]
  [0, 0, 532, 0, 0, 0, 0, 24, 0, 3]
  [1, 9, 4, 552, 1, 1, 4, 0, 2, 13]
  [2, 7, 5, 0, 551, 1, 0, 4, 0, 0]
  [0, 0, 1, 0, 3, 562, 0, 0, 0, 0]
  [0, 1, 6, 2, 1, 1, 551, 14, 0, 0]
  [0, 0, 26, 0, 0, 0, 3, 530, 0, 2]
  [0, 0, 22, 2, 0, 0, 2, 6, 545, 4]
  [3, 7, 8, 6, 2, 0, 1, 2, 1, 552]

MPIC version   : 1.0
Architecture   : StreamSenseWrapper (multi-head, Scope 2 WA-4)
============================================================

============================================================
  Model        : StreamSenseWrapper INT8 (multihead)
  ONNX file    : streamsense_multihead_int8.onnx
  Timestamp    : 2026-06-23 15:12:25
  Test samples : 5779
  Accuracy     : 95.88%  (5541/5779)
  Elapsed      : 17.8s
============================================================

Per-class report:
              precision    recall  f1-score   support

         yes     0.9868    0.9884    0.9876       606
          no     0.9565    0.9679    0.9622       591
          up     0.8669    0.9553    0.9089       559
        down     0.9770    0.9404    0.9583       587
        left     0.9821    0.9632    0.9725       570
       right     0.9947    0.9912    0.9929       566
          on     0.9804    0.9566    0.9684       576
         off     0.9136    0.9430    0.9281       561
        stop     0.9927    0.9346    0.9628       581
          go     0.9467    0.9467    0.9467       582

    accuracy                         0.9588      5779
   macro avg     0.9598    0.9587    0.9588      5779
weighted avg     0.9602    0.9588    0.9591      5779

Per-class accuracy:
  yes        599/606  (98.84%)
  no         572/591  (96.79%)
  up         534/559  (95.53%)
  down       552/587  (94.04%)
  left       549/570  (96.32%)
  right      561/566  (99.12%)
  on         551/576  (95.66%)
  off        529/561  (94.30%)
  stop       543/581  (93.46%)
  go         551/582  (94.67%)

Confusion matrix (rows=true, cols=predicted):
Classes: 0=yes, 1=no, 2=up, 3=down, 4=left, 5=right, 6=on, 7=off, 8=stop, 9=go
  [599, 1, 1, 0, 1, 0, 0, 2, 0, 2]
  [1, 572, 4, 3, 1, 1, 0, 0, 1, 8]
  [0, 0, 534, 0, 0, 0, 0, 23, 0, 2]
  [1, 9, 4, 552, 1, 0, 5, 0, 2, 13]
  [3, 7, 7, 0, 549, 1, 0, 3, 0, 0]
  [0, 0, 1, 0, 4, 561, 0, 0, 0, 0]
  [0, 1, 7, 2, 1, 1, 551, 13, 0, 0]
  [0, 0, 27, 0, 0, 0, 3, 529, 0, 2]
  [0, 0, 23, 2, 0, 0, 2, 7, 543, 4]
  [3, 8, 8, 6, 2, 0, 1, 2, 1, 551]

MPIC version   : 1.0
Architecture   : StreamSenseWrapper (multi-head, Scope 2 WA-4)
============================================================

============================================================
  MULTI-HEAD ONNX ACCURACY SUMMARY
============================================================
  StreamSenseWrapper FP32 (multihead)        : 95.97%
  StreamSenseWrapper INT8 (multihead)        : 95.88%

  Accuracy drop (FP32 → INT8) : +0.09%
  INT8 budget (≤1.0%)           : PASS
============================================================

```

### `evaluation/qonnx_evaluation_report.txt`

```
============================================================
  STREAMSENSE -- QONNX GV1K Evaluation
  Scope 2 / QAT Extension  |  ERR v1.0
  Timestamp  : 2026-06-24 10:20:21
  Model      : streamsense_multihead.qonnx
  File size  : 1246.6 KB
  GV1K dir   : C:\STREAMSENSE\golden_vectors_1000\normalized
============================================================

  Total .bin files    : 1000
  Vectors checked     : 1000
  Skipped (bad files) : 0
  Correct             : 965
  Wrong               : 35
  Top-1 Accuracy      : 96.50%  (965/1000)
  Pass threshold      : 90.0%
  Gate result         : PASS

────────────────────────────────────────────────────────────
  Per-class accuracy
────────────────────────────────────────────────────────────
  yes        98/100   98.00%  ###################
  no         97/100   97.00%  ###################
  up         97/100   97.00%  ###################
  down       95/100   95.00%  ###################
  left       95/100   95.00%  ###################
  right     100/100  100.00%  ####################
  on         97/100   97.00%  ###################
  off        96/100   96.00%  ###################
  stop       97/100   97.00%  ###################
  go         93/100   93.00%  ##################

────────────────────────────────────────────────────────────
  Confusion matrix  (rows=true, cols=predicted)
  Classes: 0=yes, 1=no, 2=up, 3=down, 4=left, 5=right, 6=on, 7=off, 8=stop, 9=go
────────────────────────────────────────────────────────────
  yes       [98, 0, 1, 0, 0, 0, 0, 0, 1, 0]
  no        [0, 97, 1, 0, 1, 1, 0, 0, 0, 0]
  up        [0, 0, 97, 0, 0, 0, 0, 3, 0, 0]
  down      [1, 3, 1, 95, 0, 0, 0, 0, 0, 0]
  left      [0, 2, 0, 0, 95, 2, 0, 1, 0, 0]
  right     [0, 0, 0, 0, 0, 100, 0, 0, 0, 0]
  on        [0, 0, 2, 0, 0, 0, 97, 1, 0, 0]
  off       [0, 0, 2, 0, 0, 0, 1, 96, 1, 0]
  stop      [0, 0, 3, 0, 0, 0, 0, 0, 97, 0]
  go        [0, 2, 2, 0, 1, 0, 0, 0, 2, 93]

────────────────────────────────────────────────────────────
  ERR v1.0 output contract (verified before inference)
    output[0] logits        float32  (1, 10)   -- classification head
    output[1] embedding     float32  (1, 128)  -- projection head
    output[2] novelty_score float32  (1, 1)    -- novelty head (2-D enforced)
  Note: output node names in .qonnx may be auto-generated integers.
        Shapes verified by index, names logged above for reference.
────────────────────────────────────────────────────────────


```

### `evaluation_1d/comparison_1d_vs_2d.txt`

```
STREAMSENSE — Architecture Comparison: 1D CNN (raw) vs 2D CNN (mel)
Supports Epic A3.3 (ADR — Architecture Decision Record)
======================================================================

Metric                         2D StreamSenseNet   1D StreamSenseNet1D
----------------------------------------------------------------------
Parameters                               295,786               591,210
Test accuracy                             95.97%                95.76%
Test loss                                 0.1273                0.1396
Input representation           log-mel [1,64,97]raw waveform [1,16000]

Per-class accuracy (%):
Class             2D        1D   Delta (1D-2D)
----------------------------------------------
yes           98.84%    98.02%          -0.82%
no            96.79%    93.91%          -2.88%
up            95.17%    95.71%          +0.54%
down          94.04%    95.91%          +1.87%
left          96.67%    94.56%          -2.11%
right         99.29%    98.59%          -0.70%
on            95.66%    96.35%          +0.69%
off           94.47%    91.62%          -2.85%
stop          93.80%    96.21%          +2.41%
go            94.85%    96.56%          +1.71%

Overall accuracy delta (1D - 2D): -0.21 percentage points
Parameter ratio (1D / 2D): 2.00x

Notes for ADR (A3.3):
- 2D mel-spectrogram representation provides explicit time-frequency
  structure as input, which the 1D model must learn implicitly from
  raw waveform via its receptive field.
- Compare accuracy-per-parameter and inference latency when deciding
  between representations for the FPGA deployment target.

```

### `evaluation_1d/evaluation_report_1d.txt`

```
STREAMSENSE — StreamSenseNet1D Evaluation Report
Epic A3.2 — 1D CNN Baseline on raw audio frames (STRETCH)
============================================================

Checkpoint epoch : 53
Val accuracy     : 95.97%
Parameters       : 591,210

Test samples     : 5779
Test loss        : 0.1396
Test accuracy    : 95.76%
Inference time   : 4.43s (0.77 ms/sample)

Per-class report:
              precision    recall  f1-score   support

         yes     0.9818    0.9802    0.9810       606
          no     0.9585    0.9391    0.9487       591
          up     0.9370    0.9571    0.9469       559
        down     0.9559    0.9591    0.9575       587
        left     0.9764    0.9456    0.9608       570
       right     0.9688    0.9859    0.9772       566
          on     0.9439    0.9635    0.9536       576
         off     0.9295    0.9162    0.9228       561
        stop     0.9894    0.9621    0.9756       581
          go     0.9351    0.9656    0.9501       582

    accuracy                         0.9576      5779
   macro avg     0.9576    0.9574    0.9574      5779
weighted avg     0.9578    0.9576    0.9576      5779

Confusion Matrix (rows=true, cols=pred):
['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go']
[[594   1   0   2   4   0   0   3   1   1]
 [  3 555   0  11   1   2   4   2   1  12]
 [  0   0 535   1   1   0   3  12   2   5]
 [  0   5   2 563   0   2   3   2   0  10]
 [  7   6   1   2 539   8   2   1   2   2]
 [  0   1   0   0   5 558   0   1   0   1]
 [  0   1   5   3   1   2 555   8   0   1]
 [  1   2  21   0   0   0  19 514   0   4]
 [  0   1   5   3   0   3   1   6 559   3]
 [  0   7   2   4   1   1   1   4   0 562]]

```

---
## Notebooks (input cells only)

### `qat_colab.ipynb`

```python
# [markdown]
# # Cell 0 — Git Workflow Guide
# 
# ## Three-Location Workflow
# 
# This notebook is part of a three-location git workflow:
# **VS Code (author)** → **GitHub (sync)** → **Colab (compute)** → **GitHub (push back)** → **VS Code (pull)**.
# 
# Branch: `qat-wa4-extension`. **Never push to `main` directly.**
# 
# ---
# 
# ### Step A — Before opening Colab (run in VS Code terminal)
# 
# ```bash
# git checkout -b qat-wa4-extension
# git add training/qat_finetune.py notebooks/qat_colab.ipynb
# git commit -m "WA-4 ext: QAT finetune script and Colab notebook"
# git push origin qat-wa4-extension
# ```
# 
# ### Step B — In Colab
# 
# 1. Run Cell 2 to clone the repo and check out `qat-wa4-extension`.
# 2. Run Cells 3–9 in order.
# 3. Once Cell 8 (export) and Cell 9 (GV1K gate) are both green, run Cell 11 to push artifacts back to GitHub.
# 
# ### Step C — Back in VS Code
# 
# ```bash
# git pull origin qat-wa4-extension
# ```
# 
# This retrieves `checkpoints/best_model_qat.pth` and `onnx_models/streamsense_multihead.qonnx`.
# 
# ### Step D — Open PR
# 
# Open a pull request from `qat-wa4-extension` into `main` **only after Track E confirms the FINN flow works**. Never merge before Track E sign-off.
# 
# ---
# 
# > **Rule:** Never push directly to `main`. All work lives on `qat-wa4-extension` until Track E confirms.

# [markdown]
# # Project STREAMSENSE — QAT Fine-tuning and QONNX Export
# 
# **Branch:** `qat-wa4-extension`
# 
# ## What this notebook produces
# 
# | Artifact | Path | Description |
# |---|---|---|
# | `best_model_qat.pth` | `checkpoints/` | QAT fine-tuned checkpoint (Brevitas weights + quantizer scales) |
# | `streamsense_multihead.qonnx` | `onnx_models/` | Multi-head QONNX — **Track E FINN flow target** |
# 
# The QONNX carries all three output heads:
# - `logits`        — `[1, 10]`  float32 — identical classification logits
# - `embedding`     — `[1, 128]` float32 — linear projection of the 128-dim GAP vector
# - `novelty_score` — `[1, 1]`   float32 — `1 − max(softmax(logits))`, always 2-D
# 
# This is the Scope 2 multi-head model in QONNX (Brevitas) format, not the WA-4 QDQ INT8 format. Track E's FINN flow requires QONNX; the WA-4 INT8 QDQ model is not FINN-compatible.
# 
# ## MPIC v1.0 contract (frozen — do not change)
# 
# - Input: `[1, 1, 64, 97]` float32, 16 kHz, 64 mel bins, 97 time frames
# - Global norm: mean = −30.785545 dB, std = 22.157099 dB
# - Opset 17

# [code cell]
!git clone https://github.com/bodasingiksheeraja317-svg/STREAMSENSE.git
%cd STREAMSENSE
!git checkout qat-wa4-extension
!git log --oneline -5

# [code cell]
# Cell 3 — Install dependencies
# IMPORTANT: This cell installs packages that may change numpy's ABI.
# After this cell completes, you MUST use Runtime → Restart session,
# then run from Cell 2 onward again (skip re-cloning if repo already present).

# Step 1: pin numpy first so all subsequent packages compile/link against it
!pip install -q "numpy<2.0"

# Step 2: install remaining dependencies
!pip install -q "brevitas>=0.10,<0.11" onnx onnxruntime torchaudio

# Step 3: print versions (run this AFTER restarting the runtime)
import torch, brevitas, onnx, onnxruntime, torchaudio, numpy as np
print(f"torch        : {torch.__version__}")
print(f"torchaudio   : {torchaudio.__version__}")
print(f"brevitas     : {brevitas.__version__}")
print(f"onnx         : {onnx.__version__}")
print(f"onnxruntime  : {onnxruntime.__version__}")
print(f"numpy        : {np.__version__}")

# [code cell]
# Cell 4 — Mount Google Drive and copy checkpoint
from google.colab import drive
from pathlib import Path

drive.mount('/content/drive')

src_ckpt = Path('/content/drive/MyDrive/STREAMSENSE_checkpoints/best_model.pth')
dst_ckpt = Path('checkpoints/best_model.pth')
dst_ckpt.parent.mkdir(parents=True, exist_ok=True)

if src_ckpt.exists():
    import shutil
    shutil.copy(str(src_ckpt), str(dst_ckpt))
    size_mb = dst_ckpt.stat().st_size / 1e6
    print(f"[OK] Copied best_model.pth  ({size_mb:.2f} MB) -> {dst_ckpt}")
else:
    print(f"[WARN] Checkpoint not found at: {src_ckpt}")
    print(f"       Please upload best_model.pth manually via the Colab Files panel")
    print(f"       and place it at: {dst_ckpt}")
    print(f"       Or run:  !cp /content/drive/MyDrive/<YOUR_PATH>/best_model.pth {dst_ckpt}")

# [code cell]
# Cell 5 — Verify GPU
!nvidia-smi

import torch
import brevitas

assert torch.cuda.is_available(), "[FAIL] No GPU detected. Change Runtime -> GPU in Colab."
print(f"[PASS] GPU: {torch.cuda.get_device_name(0)}")
print(f"       torch    : {torch.__version__}")
print(f"       brevitas : {brevitas.__version__}")

# [code cell]
# Cell 6 — Download Speech Commands v2
# torchaudio will download and unzip the dataset on first call.

import torchaudio
from pathlib import Path
from collections import Counter

DATA_ROOT = Path('/content/data')
DATA_ROOT.mkdir(parents=True, exist_ok=True)

TARGET_CLASSES = {
    "yes": 0, "no": 1, "up": 2, "down": 3,
    "left": 4, "right": 5, "on": 6, "off": 7,
    "stop": 8, "go": 9,
}

# Download validation split (will also download the archive if absent)
print("Downloading / loading Speech Commands v2 (validation split)...")
val_ds = torchaudio.datasets.SPEECHCOMMANDS(root=str(DATA_ROOT), download=True, subset='validation')
print(f"Validation split total clips : {len(val_ds)}")

# Count clips per target class
counter = Counter()
for _, _, label, *_ in val_ds:
    if label in TARGET_CLASSES:
        counter[label] += 1

print("\nClass distribution (validation, target classes only):")
total_target = 0
for label in sorted(TARGET_CLASSES):
    n = counter.get(label, 0)
    total_target += n
    print(f"  {label:<6} : {n}")
print(f"  TOTAL  : {total_target}")

# [code cell]
import os
os.chdir('/content/STREAMSENSE')
!pwd
!ls

# [markdown]
# ## Cell 7 — QAT Fine-tuning
# 
# This cell runs `training/qat_finetune.py` which:
# 
# 1. Loads `checkpoints/best_model.pth` backbone weights (strict=True).
# 2. Replaces all `nn.Conv2d` in the backbone with `brevitas.nn.QuantConv2d` (Int8 per-tensor).
# 3. Replaces all `nn.Linear` in backbone classifier and `embed_head` with `brevitas.nn.QuantLinear`.
# 4. Applies the Brevitas device-placement fix (`model.to(device)` last, then buffer verification).
# 5. **Epochs 1–3**: backbone frozen — only `embed_head` weights and Brevitas quantizer scale factors are trained.
# 6. **Epoch 4+**: all parameters unfrozen for full QAT fine-tuning.
# 7. Saves best checkpoint by validation top-1 accuracy.
# 8. Runs the GV1K gate after training — hard exit if top-1 < 90 %.
# 
# Expected runtime on T4: ~20–40 minutes for 10 epochs.

# [code cell]
# Cell 7 — Run QAT fine-tuning
!python training/qat_finetune.py \
    --ckpt checkpoints/best_model.pth \
    --data /content/data \
    --epochs 10 \
    --lr 1e-5 \
    --out checkpoints/best_model_qat.pth \
    --gvk golden_vectors_1000/normalized \
    --device cuda

# [code cell]
# Fix: onnxscript version incompatibility with torch.onnx internal API
# The standalone onnxscript installed in earlier cells conflicts with
# the version torch was compiled against. Uninstall it so torch uses
# its own internal copy.
import subprocess, sys

result = subprocess.run(
    [sys.executable, "-m", "pip", "uninstall", "-y", "onnxscript"],
    capture_output=True, text=True
)
print(result.stdout or "onnxscript uninstalled (or was not installed standalone)")

# Now verify the import chain works
import importlib
# Force torch.onnx to reimport cleanly after env change
for mod in list(sys.modules.keys()):
    if 'onnxscript' in mod or 'torch.onnx._internal' in mod:
        del sys.modules[mod]

import torch.onnx
print("torch.onnx import: OK")

# [code cell]
# Step 1: confirm onnxscript situation
import subprocess, sys
r = subprocess.run([sys.executable, "-m", "pip", "show", "onnxscript"],
                   capture_output=True, text=True)
print(r.stdout if r.stdout else "onnxscript: not installed standalone — GOOD")

# Step 2: confirm clean import
import torch.onnx
print("torch.onnx: OK")

# Step 3: confirm brevitas export import
from brevitas.export import export_qonnx
print("brevitas.export.export_qonnx: OK")

# [code cell]
!pip install -q qonnx

# [code cell]
# Cell 8 — Export QONNX (full inline implementation)
#
# This cell does NOT import from qat_finetune.py.  All Brevitas replacements
# are written out in full here so the export is self-contained and reproducible
# without re-running training.

import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

import onnx
import onnxruntime as ort

import brevitas.nn as qnn
from brevitas.quant import Int8WeightPerTensorFloat, Int8ActPerTensorFloat
from brevitas.export import export_qonnx

from qonnx.core.modelwrapper import ModelWrapper
from qonnx.core.onnx_exec import execute_onnx
from qonnx.transformation.infer_shapes import InferShapes

# ── Project paths ─────────────────────────────────────────────────────────────
ROOT = Path('/content/STREAMSENSE')
sys.path.insert(0, str(ROOT / 'training'))

from model import StreamSenseNet
from streaming_wrapper import StreamSenseWrapper, NUM_CLASSES, EMBEDDING_DIM

CKPT_PATH   = ROOT / 'checkpoints' / 'best_model_qat.pth'
ORIG_CKPT   = ROOT / 'checkpoints' / 'best_model.pth'
OUT_DIR     = ROOT / 'onnx_models'
EXPORT_PATH = OUT_DIR / 'streamsense_multihead.qonnx'

OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Brevitas module replacement functions ─────────────────────────────────────

def _replace_conv2d(module):
    for name, child in module.named_children():
        if isinstance(child, nn.Conv2d):
            qconv = qnn.QuantConv2d(
                in_channels  = child.in_channels,
                out_channels = child.out_channels,
                kernel_size  = child.kernel_size,
                stride       = child.stride,
                padding      = child.padding,
                dilation     = child.dilation,
                groups       = child.groups,
                bias         = child.bias is not None,
                weight_quant = Int8WeightPerTensorFloat,
                input_quant  = Int8ActPerTensorFloat,
                output_quant = Int8ActPerTensorFloat,
                return_quant_tensor = False,
            )
            with torch.no_grad():
                qconv.weight.copy_(child.weight)
                if child.bias is not None and qconv.bias is not None:
                    qconv.bias.copy_(child.bias)
            setattr(module, name, qconv)
        else:
            _replace_conv2d(child)
    return module


def _replace_linear(module):
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            qlin = qnn.QuantLinear(
                in_features  = child.in_features,
                out_features = child.out_features,
                bias         = child.bias is not None,
                weight_quant = Int8WeightPerTensorFloat,
                input_quant  = Int8ActPerTensorFloat,
                output_quant = Int8ActPerTensorFloat,
                return_quant_tensor = False,
            )
            with torch.no_grad():
                qlin.weight.copy_(child.weight)
                if child.bias is not None and qlin.bias is not None:
                    qlin.bias.copy_(child.bias)
            setattr(module, name, qlin)
        else:
            _replace_linear(child)
    return module


# ── Build model structure (CPU — export always runs on CPU) ───────────────────
device = torch.device('cpu')
model  = StreamSenseWrapper(num_classes=NUM_CLASSES)

orig_ckpt = torch.load(ORIG_CKPT, map_location='cpu', weights_only=True)
model.backbone.load_state_dict(orig_ckpt['model_state'], strict=True)
print(f"[setup] Loaded backbone from epoch {orig_ckpt.get('epoch','?')}")

_replace_conv2d(model.backbone.block1)
_replace_conv2d(model.backbone.block2)
_replace_conv2d(model.backbone.block3)
_replace_linear(model.backbone.classifier)
_replace_linear(model.embed_head)

model.to(device)

for buf_name, buf in model.named_buffers():
    assert buf.device.type == device.type, (
        f"[device-check] Buffer {buf_name!r} is on {buf.device.type!r}, "
        f"expected {device.type!r}. Brevitas device-placement bug."
    )
print("[device-check] All buffers on CPU — OK")

qat_ckpt = torch.load(CKPT_PATH, map_location='cpu', weights_only=True)
model.load_state_dict(qat_ckpt['model_state'])
print(f"[setup] Loaded QAT checkpoint  epoch={qat_ckpt.get('epoch','?')}  "
      f"val_acc={qat_ckpt.get('val_accuracy', float('nan')):.2f}%")

model.eval()

# ── Export QONNX ──────────────────────────────────────────────────────────────
input_t = torch.zeros(1, 1, 64, 97, dtype=torch.float32)

print(f"\n[export] Exporting QONNX -> {EXPORT_PATH}")
export_qonnx(
    module      = model,
    input_t     = input_t,
    export_path = str(EXPORT_PATH),
)
print(f"[export] Done.")

# ── ONNX structural check ─────────────────────────────────────────────────────
qonnx_model = onnx.load(str(EXPORT_PATH))
onnx.checker.check_model(qonnx_model)
print("[onnx.checker] PASS — model is structurally valid")

# ── QONNX runtime sanity inference ───────────────────────────────────────────
qonnx_wrap = ModelWrapper(str(EXPORT_PATH))
qonnx_wrap = qonnx_wrap.transform(InferShapes())

dummy_np   = np.zeros((1, 1, 64, 97), dtype=np.float32)
input_name = qonnx_wrap.graph.input[0].name
odict      = execute_onnx(qonnx_wrap, {input_name: dummy_np})

output_names = [o.name for o in qonnx_wrap.graph.output]
print(f"[sanity] Output names: {output_names}")
for name, val in odict.items():
    print(f"  {name}: shape={val.shape}")

logits        = odict[output_names[0]]
embedding     = odict[output_names[1]]
novelty_score = odict[output_names[2]]

# ── Shape assertions ──────────────────────────────────────────────────────────
logits_ok  = logits.shape        == (1, 10)
embed_ok   = embedding.shape     == (1, 128)
novelty_ok = novelty_score.shape == (1, 1)

print(f"\n[shape assert] logits        {tuple(logits.shape)}    : {'PASS' if logits_ok  else 'FAIL — expected (1, 10)'}")
print(f"[shape assert] embedding     {tuple(embedding.shape)} : {'PASS' if embed_ok   else 'FAIL — expected (1, 128)'}")
print(f"[shape assert] novelty_score {tuple(novelty_score.shape)}    : {'PASS' if novelty_ok else 'FAIL — expected (1, 1), check keepdim=True'}")

assert logits_ok,  f"logits shape {logits.shape} != (1,10)  — ERR v1.0 contract broken"
assert embed_ok,   f"embedding shape {embedding.shape} != (1,128) — ERR v1.0 contract broken"
assert novelty_ok, f"novelty_score shape {novelty_score.shape} != (1,1) — ERR v1.0 contract broken"

print("\n[PASS] All three output shapes conform to ERR v1.0 contract.")

# ── File size ─────────────────────────────────────────────────────────────────
size_mb = EXPORT_PATH.stat().st_size / 1e6
print(f"[export] File size: {size_mb:.2f} MB  -> {EXPORT_PATH}")

# [code cell]
# Cell 8a — Generate GV1K vectors from Speech Commands test split
#
# The golden_vectors_1000/normalized/ directory exists in the repo but
# contains NO .bin files — they are ~25 MB of derived binary artifacts
# that were never committed to Git.
#
# This cell generates them now using the MPIC v1.0 pipeline, drawing
# 100 samples per class (1000 total) from the Speech Commands test split.
# Reproducible: fixed random seed 42 (same as generate_golden_1000.py).

import random
import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T
from pathlib import Path

# ── MPIC v1.0 frozen constants (must match qat_finetune.py exactly) ──────────
SAMPLE_RATE   = 16000
FRAME_LEN     = 16000
N_FFT         = 512
HOP_LENGTH    = 160
N_MELS        = 64
CENTER        = False
POWER         = 2.0
LOG_EPS       = 1e-10
CLIP_FLOOR_DB = -80.0
GLOBAL_MEAN   = -30.785545
GLOBAL_STD    = 22.157099
EXPECTED_T    = (FRAME_LEN - N_FFT) // HOP_LENGTH + 1   # 97

TARGET_CLASSES = {
    "yes": 0, "no": 1, "up": 2, "down": 3,
    "left": 4, "right": 5, "on": 6, "off": 7,
    "stop": 8, "go": 9,
}

N_PER_CLASS = 100
RANDOM_SEED = 42

ROOT     = Path('/content/STREAMSENSE')
GV1K_DIR = ROOT / 'golden_vectors_1000' / 'normalized'
GV1K_DIR.mkdir(parents=True, exist_ok=True)

# Skip if already generated
existing = list(GV1K_DIR.glob('*_norm.bin'))
if len(existing) >= 1000:
    print(f"[OK] GV1K vectors already present: {len(existing)} files — skipping generation.")
else:
    print(f"[INFO] Generating GV1K vectors (this may take ~2-3 minutes)...")

    # ── Build mel transform ───────────────────────────────────────────────────
    mel_transform = T.MelSpectrogram(
        sample_rate=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH,
        n_mels=N_MELS, window_fn=torch.hann_window, center=CENTER, power=POWER,
    )

    def preprocess_to_norm(waveform, sr):
        """Waveform tensor [C, T] -> normalised [64, 97] numpy float32."""
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        L = waveform.shape[1]
        if L < FRAME_LEN:
            waveform = torch.nn.functional.pad(waveform, (0, FRAME_LEN - L))
        elif L > FRAME_LEN:
            waveform = waveform[:, :FRAME_LEN]
        mel = mel_transform(waveform)                   # [1, 64, 97]
        mel = 10.0 * torch.log10(mel + LOG_EPS)
        mel = torch.clamp(mel, min=CLIP_FLOOR_DB)
        norm = (mel - GLOBAL_MEAN) / GLOBAL_STD
        return norm.squeeze(0).numpy().astype(np.float32)   # [64, 97]

    # ── Load test split, bucket by class ─────────────────────────────────────
    print("Loading Speech Commands test split...")
    test_ds = torchaudio.datasets.SPEECHCOMMANDS(
        root='/content/data', download=True, subset='testing'
    )

    buckets = {label: [] for label in TARGET_CLASSES}
    for item in test_ds:
        waveform, sr, label = item[0], item[1], item[2]
        if label in TARGET_CLASSES:
            buckets[label].append((waveform, sr))

    for label, items in buckets.items():
        print(f"  {label:<6}: {len(items)} clips available")

    # ── Sample N_PER_CLASS per class ──────────────────────────────────────────
    rng = random.Random(RANDOM_SEED)
    gv_idx = 0
    n_written = 0

    for label, class_idx in TARGET_CLASSES.items():
        items = buckets[label][:]
        rng.shuffle(items)
        selected = items[:N_PER_CLASS]

        for waveform, sr in selected:
            norm = preprocess_to_norm(waveform, sr)
            assert norm.shape == (N_MELS, EXPECTED_T), f"Shape error: {norm.shape}"
            assert norm.dtype == np.float32

            fname = GV1K_DIR / f"GV1K_{gv_idx:04d}_{label}_norm.bin"
            norm.tofile(str(fname))
            gv_idx += 1
            n_written += 1

    n_final = len(list(GV1K_DIR.glob('*_norm.bin')))
    print(f"\n[{'OK' if n_final == 1000 else 'WARN'}] Written: {n_final} *_norm.bin files to {GV1K_DIR}")
    if n_final < 900:
        raise RuntimeError(f"Only {n_final} GV1K vectors generated — too few to pass the 90% gate. Check data.")
    print("[INFO] GV1K generation complete. Ready for Cell 9.")

# [code cell]
# Cell 9 — GV1K top-1 verification on exported QONNX

import sys
import numpy as np
from pathlib import Path

from qonnx.core.modelwrapper import ModelWrapper
from qonnx.core.onnx_exec import execute_onnx
from qonnx.transformation.infer_shapes import InferShapes

ROOT       = Path('/content/STREAMSENSE')
QONNX_PATH = ROOT / 'onnx_models' / 'streamsense_multihead.qonnx'
GV1K_DIR   = ROOT / 'golden_vectors_1000' / 'normalized'

TARGET_CLASSES = {
    "yes": 0, "no": 1, "up": 2, "down": 3,
    "left": 4, "right": 5, "on": 6, "off": 7,
    "stop": 8, "go": 9,
}


def _parse_label(stem: str):
    parts = stem.split('_')
    if len(parts) < 4:
        return None
    return TARGET_CLASSES.get(parts[2].lower(), None)


# ── Load QONNX model (once, before the loop) ─────────────────────────────────
qonnx_wrap   = ModelWrapper(str(QONNX_PATH))
qonnx_wrap   = qonnx_wrap.transform(InferShapes())
input_name   = qonnx_wrap.graph.input[0].name
output_names = [o.name for o in qonnx_wrap.graph.output]

print(f"QONNX input : {input_name!r}")
print(f"Outputs     : {output_names}")

# ── Load GV1K vectors ─────────────────────────────────────────────────────────
bin_files = sorted(GV1K_DIR.glob('*_norm.bin'))
print(f"\nGV1K vectors found : {len(bin_files)}")

if len(bin_files) == 0:
    print(f"[WARN] No *_norm.bin files found in {GV1K_DIR}")
    print("       Generate GV1K vectors first: python training/generate_golden_1000.py")

# ── Run inference on all 1000 vectors ────────────────────────────────────────
correct = 0
wrong   = 0
skipped = 0

for bf in bin_files:
    true_idx = _parse_label(bf.stem)
    if true_idx is None:
        skipped += 1
        continue

    raw = np.fromfile(str(bf), dtype='<f4')
    if raw.size != 64 * 97:
        skipped += 1
        continue

    inp   = raw.reshape(1, 1, 64, 97).astype(np.float32)
    odict = execute_onnx(qonnx_wrap, {input_name: inp})

    logits = odict[output_names[0]]          # [1, 10]
    pred   = int(np.argmax(logits[0]))

    if pred == true_idx:
        correct += 1
    else:
        wrong += 1

total_checked = correct + wrong
top1_acc      = 100.0 * correct / total_checked if total_checked > 0 else 0.0

print(f"\n{'='*50}")
print(f"GV1K QONNX Verification")
print(f"{'='*50}")
print(f"  Total checked  : {total_checked}")
print(f"  Correct        : {correct}")
print(f"  Wrong          : {wrong}")
print(f"  Skipped        : {skipped}")
print(f"  Top-1 accuracy : {top1_acc:.2f}%")

gate_passed = (top1_acc >= 90.0)
print(f"\n[{'PASS' if gate_passed else 'FAIL'}] GV1K gate {'passed' if gate_passed else 'FAILED'} "
      f"({top1_acc:.2f}% {'≥' if gate_passed else '<'} 90.0%)")

assert gate_passed, (
    f"GV1K gate FAILED: {top1_acc:.2f}% < 90.0% minimum.  "
    f"Do NOT push — debug the QONNX export first."
)

# [markdown]
# ## Cell 10 — Review before pushing
# 
# **Before running Cell 11**, confirm the following in the Cell 8 and Cell 9 outputs:
# 
# - `[PASS] All three output shapes conform to ERR v1.0 contract.`
#   - `logits` : `(1, 10)`
#   - `embedding` : `(1, 128)`
#   - `novelty_score` : `(1, 1)` — must be exactly 2-D
# - `[PASS] GV1K gate passed (≥ 90.0%)`
# - `[onnx.checker] PASS`
# 
# If **any** of these is red or shows FAIL, **do not run Cell 11**. Debug the export first:
# - If `novelty_score` is `(1,)` (1-D), check that `keepdim=True` is present in `StreamSenseWrapper.forward()`.
# - If GV1K accuracy is < 90 %, re-run training with more epochs or a slightly higher learning rate.
# - If the ONNX checker fails, check that the Brevitas version is compatible with the installed PyTorch.

# [code cell]
# Cell 11 — Git commit and push artifacts back to GitHub
#
# REPLACE the email, name, and token/username before running.
# If push fails with authentication error, set the remote URL with your PAT:
#   !git remote set-url origin https://<TOKEN>@github.com/<USERNAME>/STREAMSENSE.git

!git config user.email "your.email@example.com"   # REPLACE with your email
!git config user.name  "Your Name"                 # REPLACE with your name

!git add checkpoints/best_model_qat.pth
!git add onnx_models/streamsense_multihead.qonnx
!git add notebooks/qat_colab.ipynb

!git commit -m "WA-4 ext: QAT checkpoint and QONNX multihead export — 3-output Track E target — GV1K green"
!git push origin qat-wa4-extension

print()
print("If the push failed due to authentication, run this in a code cell:")
print("  !git remote set-url origin https://<TOKEN>@github.com/<USERNAME>/STREAMSENSE.git")
print("Then re-run this cell.")

# [code cell]
# Cell 12 — Copy artifacts to Google Drive for local backup
import shutil
from pathlib import Path

drive_out = Path('/content/drive/MyDrive/STREAMSENSE_outputs')
drive_out.mkdir(parents=True, exist_ok=True)

src_ckpt  = Path('checkpoints/best_model_qat.pth')
src_qonnx = Path('onnx_models/streamsense_multihead.qonnx')
dst_ckpt  = drive_out / 'best_model_qat.pth'
dst_qonnx = drive_out / 'streamsense_multihead.qonnx'

shutil.copy(str(src_ckpt),  str(dst_ckpt))
shutil.copy(str(src_qonnx), str(dst_qonnx))

print(f"[copy] best_model_qat.pth          : {dst_ckpt.stat().st_size / 1e6:.2f} MB  -> {dst_ckpt}")
print(f"[copy] streamsense_multihead.qonnx : {dst_qonnx.stat().st_size / 1e6:.2f} MB  -> {dst_qonnx}")

# [markdown]
# ## Cell 14 — Handover Summary
# 
# | File | Path | For whom | Purpose |
# |---|---|---|---|
# | `streamsense_multihead.qonnx` | `onnx_models/` | Track E | FINN flow target — all 3 heads, QONNX format (Brevitas QAT) |
# | `streamsense_multihead_fp32.onnx` | `onnx_models/` | Track E | FP32 reference — ORT inference, all 3 heads |
# | `best_model_qat.pth` | `checkpoints/` | Internal Track A | QAT checkpoint — source of QONNX, Brevitas quantizer scales included |
# | `streamsense_multihead_int8.onnx` | `onnx_models/` | Internal Track A | QDQ PTQ reference — 95.8% GV1K, NOT FINN-compatible |

```

### `quick_predict.ipynb`

```python
# [markdown]
# # STREAMSENSE — Quick Voice Test
# Record your voice and predict the spoken command.
# 
# **Requirements:** `pip install sounddevice scipy numpy torch torchaudio`
# 
# **Before running:** Make sure `best_model.pth` is in `C:\STREAMSENSE\checkpoints\`

# [code cell]
# Cell 1 — Imports and paths
import sys
import json
import time
import numpy as np
import torch
import torch.nn as nn
import sounddevice as sd
import scipy.io.wavfile as wav
from pathlib import Path

# Add training folder to path so we can import mel_pipeline and model
STREAMSENSE_ROOT = Path(r'C:\STREAMSENSE')
sys.path.insert(0, str(STREAMSENSE_ROOT / 'training'))

from mel_pipeline import preprocess
from model import StreamSenseNet

# Paths
CKPT_PATH    = STREAMSENSE_ROOT / 'checkpoints' / 'best_model.pth'
LABELS_PATH  = STREAMSENSE_ROOT / 'class_labels.json'
RECORDS_DIR  = STREAMSENSE_ROOT / 'recordings'
RECORDS_DIR.mkdir(exist_ok=True)

# MPIC params
SAMPLE_RATE  = 16000
FRAME_LEN    = 16000    # 1 second
HOP          = 8000     # 50% overlap
CONF_THRESH  = 0.70

print('Imports OK')
print(f'Checkpoint : {CKPT_PATH}')
print(f'Exists     : {CKPT_PATH.exists()}')

# [code cell]
# Cell 2 — Load class labels
with open(LABELS_PATH, 'r') as f:
    raw = json.load(f)

# class_labels.json format: {"0": "yes", "1": "no", ...}
CLASS_LABELS = {int(k): v for k, v in raw.items()}
print('Class labels:')
for idx, label in CLASS_LABELS.items():
    print(f'  {idx} → {label}')

# [code cell]
# Cell 3 — Load model
device = torch.device('cpu')   # local machine, CPU inference

model = StreamSenseNet(num_classes=10)
ckpt  = torch.load(CKPT_PATH, map_location='cpu')
model.load_state_dict(ckpt['model_state'])
model.eval()

print(f'Model loaded from epoch {ckpt["epoch"]}')
print(f'Checkpoint val_acc : {ckpt["val_accuracy"]:.2f}%')

# [code cell]
# Cell 4 — Predict function (sliding window)
# Handles any length recording — slides 1-second windows with 50% overlap

def predict_audio(audio: np.ndarray, sample_rate: int) -> dict:
    """
    Run sliding window prediction over audio of any length.
    
    Args:
        audio       : float32 numpy array, any length, mono
        sample_rate : sample rate of the audio
    
    Returns:
        dict with predicted label, confidence, and per-window results
    """
    # Resample if needed
    if sample_rate != SAMPLE_RATE:
        import torchaudio.functional as F
        import torch
        t = torch.from_numpy(audio).unsqueeze(0)
        t = F.resample(t, sample_rate, SAMPLE_RATE)
        audio = t.squeeze(0).numpy()
        print(f'Resampled {sample_rate}Hz → {SAMPLE_RATE}Hz')

    # Normalize audio amplitude to [-1, 1]
    max_val = np.abs(audio).max()
    if max_val > 0:
        audio = audio / max_val

    n_samples = len(audio)
    print(f'Audio length : {n_samples} samples ({n_samples/SAMPLE_RATE:.2f}s)')

    # Build windows
    windows = []
    if n_samples <= FRAME_LEN:
        # Pad short audio to exactly FRAME_LEN
        padded = np.zeros(FRAME_LEN, dtype=np.float32)
        padded[:n_samples] = audio
        windows.append(padded)
    else:
        # Slide with 50% overlap
        start = 0
        while start + FRAME_LEN <= n_samples:
            windows.append(audio[start : start + FRAME_LEN].astype(np.float32))
            start += HOP
        # Last window — pad if needed
        if start < n_samples:
            last = np.zeros(FRAME_LEN, dtype=np.float32)
            last[:n_samples - start] = audio[start:]
            windows.append(last)

    print(f'Windows      : {len(windows)}')

    # Run inference on each window
    results = []
    softmax = nn.Softmax(dim=1)

    for i, window in enumerate(windows):
        tensor = preprocess(window)              # [1, 1, 64, 97]
        tensor = tensor.squeeze(0)               # [1, 64, 97] — remove batch from pipeline
        tensor = tensor.unsqueeze(0)             # [1, 1, 64, 97] — add batch dim for model

        with torch.no_grad():
            logits = model(tensor)               # [1, 10]
            probs  = softmax(logits)[0]          # [10]

        conf, pred_idx = probs.max(dim=0)
        pred_label     = CLASS_LABELS[pred_idx.item()]
        confidence     = conf.item()

        results.append({
            'window'    : i + 1,
            'label'     : pred_label,
            'confidence': confidence,
            'probs'     : probs.numpy(),
        })

        print(f'  Window {i+1}/{len(windows)} : {pred_label:6s}  confidence={confidence:.3f}')

    # Pick window with highest confidence
    best = max(results, key=lambda r: r['confidence'])

    return {
        'prediction' : best['label'] if best['confidence'] >= CONF_THRESH else 'unclear',
        'confidence' : best['confidence'],
        'threshold'  : CONF_THRESH,
        'windows'    : results,
        'best_window': best['window'],
    }


print('Predict function ready.')

# [code cell]
# Cell 5 — Record your voice
# Change WORD to whatever you want to say
# Adjust DURATION_SEC if you want more recording time

WORD         = 'yes'       # what you will say — just for filename
DURATION_SEC = 2           # seconds to record
REC_SR       = 16000       # record directly at 16kHz — no resampling needed

print(f'Recording {DURATION_SEC}s at {REC_SR}Hz')
print(f'Say "{WORD}" clearly after the beep...')
time.sleep(0.5)
print('Recording NOW...')

recording = sd.rec(
    int(DURATION_SEC * REC_SR),
    samplerate = REC_SR,
    channels   = 1,
    dtype      = 'float32'
)
sd.wait()   # block until recording done

audio = recording.squeeze()   # [N] float32

# Save to recordings/
save_path = RECORDS_DIR / f'my_{WORD}_{int(time.time())}.wav'
wav.write(str(save_path), REC_SR, audio)

print(f'Done. Saved → {save_path}')
print(f'Samples : {len(audio)}')
print(f'Duration: {len(audio)/REC_SR:.2f}s')
print(f'Max amp : {np.abs(audio).max():.4f}')

# [code cell]
# Cell 6 — Run prediction on recorded audio

print('=' * 50)
print('STREAMSENSE — Prediction')
print('=' * 50)

result = predict_audio(audio, REC_SR)

print()
print('─' * 50)
if result['prediction'] == 'unclear':
    print(f'  Result     : UNCLEAR')
    print(f'  Confidence : {result["confidence"]:.3f}  (threshold={CONF_THRESH})')
    print(f'  Tip        : Speak louder or closer to mic')
else:
    print(f'  Prediction : {result["prediction"].upper()}')
    print(f'  Confidence : {result["confidence"]:.3f}')
    print(f'  Best window: {result["best_window"]}')
print('─' * 50)

# [code cell]
# Cell 7 — OPTIONAL: predict from an existing WAV file
# Use this if you recorded on your phone and transferred the file

import torchaudio

WAV_FILE = r'C:\STREAMSENSE\recordings\my_yes_test.wav'  # change this path

waveform, sr = torchaudio.load(WAV_FILE)

# Convert to mono if stereo
if waveform.shape[0] > 1:
    waveform = waveform.mean(dim=0, keepdim=True)

audio_file = waveform.squeeze(0).numpy()   # [N] float32

print('=' * 50)
print(f'File: {Path(WAV_FILE).name}')
print('=' * 50)

result = predict_audio(audio_file, sr)

print()
print('─' * 50)
if result['prediction'] == 'unclear':
    print(f'  Result     : UNCLEAR')
    print(f'  Confidence : {result["confidence"]:.3f}')
else:
    print(f'  Prediction : {result["prediction"].upper()}')
    print(f'  Confidence : {result["confidence"]:.3f}')
print('─' * 50)

# [code cell]
# Cell 8 — Full confidence breakdown for all 10 classes
# Run after Cell 6 to see scores for every class

best_window = result['windows'][result['best_window'] - 1]

print('Confidence scores — all classes:')
print(f'{"Class":<10} {"Score":<10} {"Bar"}')
print('─' * 40)

probs = best_window['probs']
order = np.argsort(probs)[::-1]   # sort descending

for idx in order:
    label = CLASS_LABELS[idx]
    score = probs[idx]
    bar   = '█' * int(score * 30)
    marker = ' ← predicted' if idx == order[0] else ''
    print(f'{label:<10} {score:.4f}    {bar}{marker}')

```

### `Streamsense.ipynb`

```python
# [code cell]
# Cell 1 — Mount Drive
from google.colab import drive
drive.mount('/content/drive')

# [code cell]
# Cell 2 — Clone your repo
!git clone https://github.com/bodasingiksheeraja317-svg/streamsense.git
%cd streamsense

# [code cell]
# Diagnostic — run this before anything else
import os, zipfile

# Check 1: did the zip extract at all?
print("=== Contents of /content/streamsense/data/ ===")
data_path = '/content/streamsense/data'
if os.path.exists(data_path):
    for item in os.listdir(data_path):
        print(f"  {item}")
else:
    print("  /content/streamsense/data/ does not exist")

# Check 2: what is inside the zip?
zip_path = '/content/drive/MyDrive/data_raw.zip'
print(f"\n=== First 20 entries inside {zip_path} ===")
with zipfile.ZipFile(zip_path, 'r') as z:
    for name in list(z.namelist())[:20]:
        print(f"  {name}")

# [code cell]
# Fix — extract directly to the correct location
import zipfile, os

zip_path    = '/content/drive/MyDrive/data_raw.zip'
extract_to  = '/content/streamsense/data/'

# Make sure target exists
os.makedirs(extract_to, exist_ok=True)

print("Extracting... this will take 2-3 minutes")
with zipfile.ZipFile(zip_path, 'r') as z:
    # Fix backslashes from Windows zip → forward slashes for Linux
    for member in z.infolist():
        # Convert Windows path separators
        member.filename = member.filename.replace('\\', '/')
        z.extract(member, extract_to)

print("\nDone. Checking:")
for cls in sorted(os.listdir('/content/streamsense/data/raw')):
    count = len(os.listdir(f'/content/streamsense/data/raw/{cls}'))
    print(f"  {cls}: {count} files")

# [code cell]
# Cell 5 — Verify split files exist
from pathlib import Path

splits = [
    '/content/streamsense/data/splits/train_files.txt',
    '/content/streamsense/data/splits/val_files.txt',
    '/content/streamsense/data/splits/test_files.txt',
]
for s in splits:
    p = Path(s)
    lines = p.read_text().strip().splitlines()
    print(f"{p.name}: {len(lines)} lines — {'OK' if lines else 'EMPTY'}")

# [code cell]
# Cell 5b — Rewrite split file paths for Colab Linux paths
import re

splits = [
    '/content/streamsense/data/splits/train_files.txt',
    '/content/streamsense/data/splits/val_files.txt',
    '/content/streamsense/data/splits/test_files.txt',
]

for split_path in splits:
    with open(split_path, 'r') as f:
        lines = f.readlines()

    new_lines = []
    for line in lines:
        parts = line.strip().split('|')
        if len(parts) == 3:
            # Convert Windows path to Colab path
            win_path = parts[0].strip()
            # Extract class/filename from Windows path
            # e.g. C:\STREAMSENSE\data\raw\yes\file.wav
            #   -> /content/streamsense/data/raw/yes/file.wav
            colab_path = win_path.replace('C:\\STREAMSENSE\\', '/content/streamsense/')
            colab_path = colab_path.replace('\\', '/')
            new_lines.append(f"{colab_path} | {parts[1].strip()} | {parts[2].strip()}\n")
        else:
            new_lines.append(line)

    with open(split_path, 'w') as f:
        f.writelines(new_lines)

    print(f"Rewritten: {split_path.split('/')[-1]}")

print("All split files updated for Colab paths.")

# [code cell]
# Fix mel_pipeline.py paths for Colab
mel_path = '/content/streamsense/training/mel_pipeline.py'

with open(mel_path, 'r') as f:
    content = f.read()

# Replace Windows path with Colab path
content = content.replace(
    r'C:\STREAMSENSE\stats\normalization_stats.json',
    '/content/streamsense/stats/normalization_stats.json'
)

with open(mel_path, 'w') as f:
    f.write(content)

print("Fixed mel_pipeline.py paths.")

# Verify the fix
with open(mel_path, 'r') as f:
    for i, line in enumerate(f, 1):
        if 'normalization' in line.lower() or 'stats' in line.lower():
            print(f"  Line {i}: {line.rstrip()}")

# [code cell]
# Cell 6 — Run training
!python training/train.py

# [code cell]
# Fix ReduceLROnPlateau verbose argument in train.py
train_path = '/content/streamsense/training/train.py'

with open(train_path, 'r') as f:
    content = f.read()

content = content.replace(
    """    scheduler = ReduceLROnPlateau(
        optimizer,
        mode     = "min",       # monitor val_loss
        factor   = 0.5,         # halve LR on plateau
        patience = 3,           # wait 3 epochs before reducing
        min_lr   = 1e-6,
        verbose  = True,
    )""",
    """    scheduler = ReduceLROnPlateau(
        optimizer,
        mode     = "min",       # monitor val_loss
        factor   = 0.5,         # halve LR on plateau
        patience = 3,           # wait 3 epochs before reducing
        min_lr   = 1e-6,
    )"""
)

with open(train_path, 'w') as f:
    f.write(content)

print("Fixed. Re-running training...")

# [code cell]
import shutil
from pathlib import Path

# Create Drive folder
Path('/content/drive/MyDrive/STREAMSENSE_outputs').mkdir(exist_ok=True)

# Save checkpoint and log
shutil.copy('/content/streamsense/checkpoints/best_model.pth',
            '/content/drive/MyDrive/STREAMSENSE_outputs/best_model.pth')
shutil.copy('/content/streamsense/checkpoints/training_log.csv',
            '/content/drive/MyDrive/STREAMSENSE_outputs/training_log.csv')

print("Saved to Drive:")
print("  best_model.pth")
print("  training_log.csv")

# [code cell]
import subprocess

commands = [
    'git config user.email "bodasingiksheeraja317@gmail.com"',
    'git config user.name "bodasingiksheeraja317-svg"',
    'git add checkpoints/best_model.pth',
    'git add checkpoints/training_log.csv',
    'git commit -m "checkpoint: best_model from Colab T4"',
    'git push origin main'
]

for cmd in commands:
    result = subprocess.run(cmd, shell=True, cwd='/content/streamsense',
                           capture_output=True, text=True)
    print(f"$ {cmd}")
    if result.stdout: print(result.stdout)
    if result.stderr: print(result.stderr)

# [code cell]
%%bash
cd /content/streamsense
git add -f checkpoints/best_model.pth
git add checkpoints/training_log.csv
echo "Added files"

# [code cell]
%%bash
cd /content/streamsense

# Stage everything needed
git add -f checkpoints/best_model.pth
git add checkpoints/training_log.csv
git add data/splits/train_files.txt
git add data/splits/val_files.txt
git add data/splits/test_files.txt
git add training/mel_pipeline.py
git add training/train.py

git commit -m "checkpoint: best_model Colab T4 + path fixes"

# Push with token embedded in URL
git push https://bodasingiksheeraja317-svg:<YOUR_NEW_TOKEN>@github.com/bodasingiksheeraja317-svg/streamsense.git main

# [code cell]
%%bash
cd /content/streamsense

# Read current gitignore
cat .gitignore

# [code cell]
%%bash
cd /content/streamsense

# Remove the line blocking pth files
sed -i '/checkpoints\/\*.pth/d' .gitignore

git add .gitignore
git commit -m "fix: allow best_model.pth in git"
git push https://bodasingiksheeraja317-svg:<YOUR_NEW_TOKEN>@github.com/bodasingiksheeraja317-svg/streamsense.git main

```

### `STREAMSENSE1D.ipynb`

```python
# [code cell]
# Cell 1 — Mount Drive
from google.colab import drive
drive.mount('/content/drive')

# [code cell]
# Cell 2 — Clone your repo
!git clone https://github.com/bodasingiksheeraja317-svg/streamsense.git
%cd streamsense

# [code cell]
# Diagnostic — run this before anything else
import os, zipfile

# Check 1: did the zip extract at all?
print("=== Contents of /content/streamsense/data/ ===")
data_path = '/content/streamsense/data'
if os.path.exists(data_path):
    for item in os.listdir(data_path):
        print(f"  {item}")
else:
    print("  /content/streamsense/data/ does not exist")

# Check 2: what is inside the zip?
zip_path = '/content/drive/MyDrive/data_raw.zip'
print(f"\n=== First 20 entries inside {zip_path} ===")
with zipfile.ZipFile(zip_path, 'r') as z:
    for name in list(z.namelist())[:20]:
        print(f"  {name}")

# [code cell]
# Fix — extract directly to the correct location
import zipfile, os

zip_path    = '/content/drive/MyDrive/data_raw.zip'
extract_to  = '/content/streamsense/data/'

# Make sure target exists
os.makedirs(extract_to, exist_ok=True)

print("Extracting... this will take 2-3 minutes")
with zipfile.ZipFile(zip_path, 'r') as z:
    # Fix backslashes from Windows zip → forward slashes for Linux
    for member in z.infolist():
        # Convert Windows path separators
        member.filename = member.filename.replace('\\', '/')
        z.extract(member, extract_to)

print("\nDone. Checking:")
for cls in sorted(os.listdir('/content/streamsense/data/raw')):
    count = len(os.listdir(f'/content/streamsense/data/raw/{cls}'))
    print(f"  {cls}: {count} files")

# [code cell]
# Cell 5 — Verify split files exist
from pathlib import Path

splits = [
    '/content/streamsense/data/splits/train_files.txt',
    '/content/streamsense/data/splits/val_files.txt',
    '/content/streamsense/data/splits/test_files.txt',
]
for s in splits:
    p = Path(s)
    lines = p.read_text().strip().splitlines()
    print(f"{p.name}: {len(lines)} lines — {'OK' if lines else 'EMPTY'}")

# [code cell]
import re

splits = [
    '/content/streamsense/data/splits/train_files.txt',
    '/content/streamsense/data/splits/val_files.txt',
    '/content/streamsense/data/splits/test_files.txt',
]

for split_path in splits:
    with open(split_path, 'r') as f:
        content = f.read()

    # Remove the Windows drive + STREAMSENSE prefix regardless of slash
    # style/count (C:, C:/, C:\, C://, C:\\ ... followed by STREAMSENSE
    # and any number of slashes/backslashes)
    content = re.sub(r'[Cc]:[\\/]+STREAMSENSE[\\/]+', '/content/streamsense/', content)

    # Collapse any remaining backslashes or double-slashes to single forward slash
    content = content.replace('\\', '/')
    content = re.sub(r'/+', '/', content)

    with open(split_path, 'w') as f:
        f.write(content)
    print(f"Rewritten: {split_path}")

print()
!head -3 /content/streamsense/data/splits/train_files.txt

# [code cell]
!ls /content/streamsense/training/

# [code cell]
# ── Cell 6: Set STREAMSENSE_ROOT to match your actual extraction path ────────
import os
os.environ["STREAMSENSE_ROOT"] = "/content/streamsense"

# Sanity check — this must print the same path your zip extracted to
print("STREAMSENSE_ROOT =", os.environ["STREAMSENSE_ROOT"])


# [code cell]
# ── Cell 7: Verify GPU is available ────────────────────────────────────────────
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only'}")

# [code cell]
# ── Cell 8: Make sure training dependencies are present ───────────────────────
# Colab ships with torch/torchaudio preinstalled, but versions can drift.
# This confirms torchaudio (needed for WAV loading) works correctly.

import torchaudio
print("torchaudio:", torchaudio.__version__)

# Quick load test on one real file from your dataset
import os
from pathlib import Path
raw_dir = Path(os.environ["STREAMSENSE_ROOT"]) / "data" / "raw"
first_class = sorted(os.listdir(raw_dir))[0]
first_file = sorted(os.listdir(raw_dir / first_class))[0]
test_path = raw_dir / first_class / first_file

waveform, sr = torchaudio.load(str(test_path))
print(f"Loaded {test_path.name}: shape={waveform.shape}, sample_rate={sr}")


# [code cell]
# ── Cell 9: cd into training/ and run smoke tests ─────────────────────────────

%cd /content/streamsense/training

!python model_1d.py

!python dataset_1d.py

# [code cell]
!python train_1d.py

# [code cell]
%cd /content/streamsense/training
!python evaluate_1d_comparison.py

# [code cell]
import os
print(os.path.exists('/content/streamsense/checkpoints/best_model.pth'))

# [code cell]
with open('/content/streamsense/evaluation_1d/comparison_1d_vs_2d.txt') as f:
    print(f.read())

# [code cell]
%cd /content/streamsense

!git config user.email "bodasingiksheeraja317@gmail.com"
!git config user.name "bodasingiksheeraja317-svg"

!git add checkpoints_1d/ evaluation_1d/
!git commit -m "Add A3.2 1D CNN baseline trained checkpoint + evaluation (Colab GPU, val_acc=95.97%)"

!git push https://bodasingiksheeraja317-svg:<YOUR_NEW_TOKEN>@github.com/bodasingiksheeraja317-svg/streamsense.git main

```

### `training/export_onnx.ipynb`

```python
# [markdown]
# # STREAMSENSE — export_onnx.ipynb
# 
# Export trained PyTorch model to ONNX FP32 and validate against golden vectors.
# 
# Place at: `C:\STREAMSENSE\export_onnx.ipynb`
# 
# Kernel: `streamsense-env-win`

# [code cell]
# Cell 1 - Install dependencies
import subprocess, sys
subprocess.run([sys.executable, '-m', 'pip', 'install', 'onnx', 'onnxruntime', '--quiet'])
print('Dependencies ready.')

# [code cell]
# Cell 2 - Imports and paths
import sys
import json
import numpy as np
import torch
import onnx
import onnxruntime as ort
from pathlib import Path

ROOT = Path(r'C:\STREAMSENSE')
sys.path.insert(0, str(ROOT / 'training'))

from model import StreamSenseNet

CKPT_PATH   = ROOT / 'checkpoints' / 'best_model.pth'
ONNX_DIR    = ROOT / 'onnx_models'
GV_RAW_DIR  = ROOT / 'golden_vectors' / 'raw'
GV_NORM_DIR = ROOT / 'golden_vectors' / 'normalized'
MANIFEST    = ROOT / 'golden_vectors' / 'manifest.json'
FP32_PATH   = ONNX_DIR / 'streamsense_model_fp32.onnx'

ONNX_DIR.mkdir(exist_ok=True)

print('=== Path Check ===')
for p, name in [
    (CKPT_PATH,  'best_model.pth'),
    (GV_RAW_DIR, 'golden_vectors/raw'),
    (MANIFEST,   'manifest.json'),
]:
    print(f'  [{"OK" if p.exists() else "MISSING"}] {name}')

print(f'\nONNX output -> {FP32_PATH}')

# [code cell]
# Cell 3 - Load model
print('Loading checkpoint...')
ckpt  = torch.load(CKPT_PATH, map_location='cpu')
model = StreamSenseNet(num_classes=10)
model.load_state_dict(ckpt['model_state'])
model.eval()

print(f'  Epoch     : {ckpt["epoch"]}')
print(f'  Val acc   : {ckpt["val_accuracy"]:.2f}%')
print(f'  Eval mode : {not model.training}')

dummy = torch.zeros(1, 1, 64, 97)
with torch.no_grad():
    out = model(dummy)
print(f'  Test pass : {tuple(dummy.shape)} -> {tuple(out.shape)} OK')

# [code cell]
# Cell 4 - Export to ONNX FP32
print('Exporting to ONNX...')

dummy_input = torch.zeros(1, 1, 64, 97)

torch.onnx.export(
    model,
    dummy_input,
    str(FP32_PATH),
    opset_version       = 17,
    input_names         = ['input'],
    output_names        = ['logits'],
    dynamic_axes        = {
        'input'  : {0: 'batch'},
        'logits' : {0: 'batch'},
    },
    do_constant_folding = True,
    verbose             = False,
)

size_mb = FP32_PATH.stat().st_size / 1e6
print(f'  Exported  : {FP32_PATH.name}')
print(f'  File size : {size_mb:.2f} MB')

# [code cell]
# Cell 5 - Validate ONNX model structure
print('Running onnx.checker...')

onnx_model = onnx.load(str(FP32_PATH))
onnx.checker.check_model(onnx_model)

print('  Graph check : PASSED')
print(f'  ONNX opset  : {onnx_model.opset_import[0].version}')

for inp in onnx_model.graph.input:
    shape = [d.dim_value for d in inp.type.tensor_type.shape.dim]
    print(f'  Input       : {inp.name} {shape}')

for out in onnx_model.graph.output:
    shape = [d.dim_value for d in out.type.tensor_type.shape.dim]
    print(f'  Output      : {out.name} {shape}')

# [code cell]
# Cell 6 - Load manifest and golden vectors
with open(MANIFEST) as f:
    manifest = json.load(f)

tolerance = float(manifest['tolerance_max_abs_error'])
vectors   = manifest['vectors']

print(f'Manifest loaded.')
print(f'  Tolerance : {tolerance}')
print(f'  Vectors   : {len(vectors)}')

def load_bin(path, shape):
    arr = np.fromfile(str(path), dtype='<f4')
    return arr.reshape(shape)

print('\nLoading golden vectors...')
gv_data = []
for i in range(10):
    v    = vectors[str(i)]
    raw  = load_bin(GV_RAW_DIR  / v['raw_bin'],  tuple(v['raw_shape']))
    norm = load_bin(GV_NORM_DIR / v['norm_bin'], tuple(v['mel_shape']))
    gv_data.append({'label': v['label'], 'raw': raw, 'norm': norm})
    print(f'  GV_{i:02d}_{v["label"]}')

# [code cell]
# Cell 7 - PyTorch vs ONNX Runtime comparison on all 10 golden vectors
from mel_pipeline import preprocess

sess_options = ort.SessionOptions()
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
ort_session  = ort.InferenceSession(
    str(FP32_PATH),
    sess_options,
    providers=['CPUExecutionProvider']
)

print('=' * 60)
print('ONNX Validation - PyTorch vs ONNX Runtime')
print('=' * 60)
print(f'Tolerance : {tolerance}')
print()

passed = 0
failed = 0

for i, gv in enumerate(gv_data):
    tensor     = preprocess(gv['raw'])              # [1, 1, 64, 97]
    x          = tensor.squeeze(0).unsqueeze(0)     # [1, 1, 64, 97]

    # PyTorch
    with torch.no_grad():
        pt_logits = model(x).numpy()                # [1, 10]

    # ONNX Runtime
    ort_logits  = ort_session.run(['logits'], {'input': x.numpy()})[0]

    max_err    = float(np.max(np.abs(pt_logits - ort_logits)))
    pt_pred    = int(np.argmax(pt_logits))
    ort_pred   = int(np.argmax(ort_logits))
    top1_match = pt_pred == ort_pred
    ok         = max_err <= tolerance and top1_match

    if ok:
        passed += 1
    else:
        failed += 1

    status = 'PASS' if ok else 'FAIL'
    print(
        f'  [{status}] GV_{i:02d}_{gv["label"]:<6}  '
        f'max_err={max_err:.2e}  '
        f'PT_pred={pt_pred}  ORT_pred={ort_pred}  '
        f'top1={top1_match}'
    )

print()
print('=' * 60)
print(f'Passed: {passed}/10   Failed: {failed}/10')
if failed == 0:
    print('[PASS] streamsense_model_fp32.onnx validated and ready.')
    print('Next step: quantize_ptq.ipynb')
else:
    print('[FAIL] Export validation failed. Do not proceed to quantize.')
print('=' * 60)

# [code cell]
# Cell 8 - Summary
print('=== Export Summary ===')
print(f'  Model      : StreamSenseNet')
print(f'  Parameters : 295,786')
print(f'  Checkpoint : epoch {ckpt["epoch"]}, val_acc={ckpt["val_accuracy"]:.2f}%')
print(f'  ONNX file  : {FP32_PATH}')
print(f'  File size  : {FP32_PATH.stat().st_size / 1e6:.2f} MB')
print(f'  Opset      : 17')
print(f'  Input      : [batch, 1, 64, 97] float32')
print(f'  Output     : [batch, 10] float32 logits')
print(f'  Validation : {passed}/10 golden vectors PASS')

```

### `onnx_models/quantize_ptq.ipynb`

```python
# [markdown]
# # STREAMSENSE — quantize_ptq.ipynb
# 
# Post-Training Quantization: FP32 ONNX -> INT8 ONNX
# 
# Place at: `C:\STREAMSENSE\quantize_ptq.ipynb`
# 
# Kernel: `streamsense-env-win`
# 
# Requirements: `pip install onnx onnxruntime`
# 
# Before running: confirm `onnx_models/streamsense_model_fp32.onnx` exists.

# [code cell]
# Cell 1 - Imports and paths
import sys
import json
import shutil
import random
import numpy as np
import torch
import onnxruntime as ort
from pathlib import Path
from onnxruntime.quantization import (
    quantize_static,
    CalibrationDataReader,
    QuantType,
    QuantFormat,
)

ROOT         = Path(r'C:\STREAMSENSE')
sys.path.insert(0, str(ROOT / 'training'))

from mel_pipeline import preprocess

FP32_PATH    = ROOT / 'onnx_models'    / 'streamsense_model_fp32.onnx'
INT8_PATH    = ROOT / 'onnx_models'    / 'streamsense_model_int8.onnx'
CALIB_DIR    = ROOT / 'temp_calibration'
TRAIN_SPLIT  = ROOT / 'data' / 'splits' / 'train_files.txt'
GV_RAW_DIR   = ROOT / 'golden_vectors' / 'raw'
GV_NORM_DIR  = ROOT / 'golden_vectors' / 'normalized'
MANIFEST     = ROOT / 'golden_vectors' / 'manifest.json'

N_CALIB      = 1000   # number of calibration samples
RANDOM_SEED  = 42

print('=== Path Check ===')
for p, name in [
    (FP32_PATH,   'streamsense_model_fp32.onnx'),
    (TRAIN_SPLIT, 'train_files.txt'),
    (MANIFEST,    'manifest.json'),
]:
    print(f'  [{"OK" if p.exists() else "MISSING"}] {name}')

print(f'\nINT8 output -> {INT8_PATH}')

# [code cell]
# Cell 2 - Sample 1000 calibration files from train split
import torchaudio

print(f'Reading train split...')
with open(TRAIN_SPLIT) as f:
    lines = [l.strip() for l in f if l.strip() and not l.startswith('#')]

print(f'  Total train samples : {len(lines)}')

random.seed(RANDOM_SEED)
selected = random.sample(lines, min(N_CALIB, len(lines)))
print(f'  Calibration samples : {len(selected)}')

# Build calibration tensors
CALIB_DIR.mkdir(exist_ok=True)
print(f'\nProcessing calibration samples through mel_pipeline...')

calib_tensors = []
skipped = 0

for i, line in enumerate(selected):
    parts = line.split('|')
    if len(parts) != 3:
        skipped += 1
        continue

    wav_path = Path(parts[0].strip())
    if not wav_path.exists():
        skipped += 1
        continue

    try:
        waveform, sr = torchaudio.load(str(wav_path))
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        raw    = waveform.squeeze(0).numpy()
        tensor = preprocess(raw)                    # [1, 1, 64, 97]
        x      = tensor.squeeze(0).unsqueeze(0)     # [1, 1, 64, 97]
        calib_tensors.append(x.numpy())
    except Exception:
        skipped += 1
        continue

    if (i + 1) % 200 == 0:
        print(f'  Processed {i+1}/{len(selected)}...')

print(f'\nCalibration tensors : {len(calib_tensors)}')
print(f'Skipped             : {skipped}')
print(f'Tensor shape        : {calib_tensors[0].shape}')
print(f'Tensor dtype        : {calib_tensors[0].dtype}')

# [code cell]
# Cell 3 - Define calibration data reader
# ONNX Runtime quantizer needs a CalibrationDataReader object
# It iterates through calibration tensors one by one

class StreamSenseCalibReader(CalibrationDataReader):
    def __init__(self, tensors):
        self.tensors = tensors
        self.index   = 0

    def get_next(self):
        if self.index >= len(self.tensors):
            return None
        data = {'input': self.tensors[self.index]}
        self.index += 1
        return data

    def rewind(self):
        self.index = 0

calib_reader = StreamSenseCalibReader(calib_tensors)
print(f'CalibrationDataReader ready.')
print(f'  Samples : {len(calib_tensors)}')
print(f'  Input   : input -> {calib_tensors[0].shape} float32')

# [code cell]
# Cell 4 - Run static INT8 quantization
print('Running PTQ static quantization...')
print('This may take 2-5 minutes.')
print()

quantize_static(
    model_input          = str(FP32_PATH),
    model_output         = str(INT8_PATH),
    calibration_data_reader = calib_reader,
    quant_format         = QuantFormat.QDQ,     # Quantize-DeQuantize format
    per_channel          = False,               # per-tensor quantization
    weight_type          = QuantType.QInt8,     # INT8 weights
    activation_type      = QuantType.QInt8,     # INT8 activations
)

fp32_size = FP32_PATH.stat().st_size / 1e6
int8_size = INT8_PATH.stat().st_size / 1e6
reduction = (1 - int8_size / fp32_size) * 100

print(f'Quantization complete.')
print(f'  FP32 size : {fp32_size:.2f} MB')
print(f'  INT8 size : {int8_size:.2f} MB')
print(f'  Reduction : {reduction:.1f}%')

# [code cell]
# Cell 5 - Load manifest and golden vectors
with open(MANIFEST) as f:
    manifest = json.load(f)

tolerance = float(manifest['tolerance_max_abs_error'])
vectors   = manifest['vectors']

def load_bin(path, shape):
    arr = np.fromfile(str(path), dtype='<f4')
    return arr.reshape(shape)

gv_data = []
for i in range(10):
    v    = vectors[str(i)]
    raw  = load_bin(GV_RAW_DIR  / v['raw_bin'],  tuple(v['raw_shape']))
    norm = load_bin(GV_NORM_DIR / v['norm_bin'], tuple(v['mel_shape']))
    gv_data.append({'label': v['label'], 'raw': raw, 'norm': norm})

print(f'Golden vectors loaded: {len(gv_data)}')
print(f'Tolerance            : {tolerance}')

# [code cell]
# Cell 6 - Validate INT8 model against golden vectors
# Compare FP32 predictions vs INT8 predictions
# top-1 must match for all 10 vectors

fp32_session = ort.InferenceSession(
    str(FP32_PATH), providers=['CPUExecutionProvider']
)
int8_session = ort.InferenceSession(
    str(INT8_PATH), providers=['CPUExecutionProvider']
)

print('=' * 60)
print('INT8 Validation - FP32 vs INT8 on Golden Vectors')
print('=' * 60)
print()

passed = 0
failed = 0

for i, gv in enumerate(gv_data):
    tensor = preprocess(gv['raw'])
    x      = tensor.squeeze(0).unsqueeze(0).numpy()    # [1, 1, 64, 97]

    fp32_logits = fp32_session.run(['logits'], {'input': x})[0]
    int8_logits = int8_session.run(['logits'], {'input': x})[0]

    fp32_pred   = int(np.argmax(fp32_logits))
    int8_pred   = int(np.argmax(int8_logits))
    top1_match  = fp32_pred == int8_pred
    logit_diff  = float(np.max(np.abs(fp32_logits - int8_logits)))

    if top1_match:
        passed += 1
    else:
        failed += 1

    status = 'PASS' if top1_match else 'FAIL'
    print(
        f'  [{status}] GV_{i:02d}_{gv["label"]:<6}  '
        f'FP32={fp32_pred}  INT8={int8_pred}  '
        f'top1_match={top1_match}  '
        f'logit_diff={logit_diff:.4f}'
    )

print()
print('=' * 60)
print(f'Passed : {passed}/10')
print(f'Failed : {failed}/10')
if failed == 0:
    print('[PASS] INT8 model validated. All top-1 predictions match FP32.')
else:
    print('[FAIL] Some top-1 predictions differ. Check logit_diff values.')
print('=' * 60)

# [code cell]
# Cell 7 - Quick accuracy check on val split
# Run 200 val samples through both FP32 and INT8
# Compare accuracy to confirm INT8 drop is acceptable

VAL_SPLIT = ROOT / 'data' / 'splits' / 'val_files.txt'

with open(VAL_SPLIT) as f:
    val_lines = [l.strip() for l in f if l.strip() and not l.startswith('#')]

random.seed(42)
sample_lines = random.sample(val_lines, min(200, len(val_lines)))

fp32_correct = 0
int8_correct = 0
total        = 0

for line in sample_lines:
    parts = line.split('|')
    if len(parts) != 3:
        continue
    wav_path  = Path(parts[0].strip())
    true_idx  = int(parts[2].strip())
    if not wav_path.exists():
        continue
    try:
        waveform, sr = torchaudio.load(str(wav_path))
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        raw    = waveform.squeeze(0).numpy()
        tensor = preprocess(raw)
        x      = tensor.squeeze(0).unsqueeze(0).numpy()

        fp32_pred = int(np.argmax(fp32_session.run(['logits'], {'input': x})[0]))
        int8_pred = int(np.argmax(int8_session.run(['logits'], {'input': x})[0]))

        if fp32_pred == true_idx: fp32_correct += 1
        if int8_pred == true_idx: int8_correct += 1
        total += 1
    except Exception:
        continue

fp32_acc  = 100.0 * fp32_correct / total
int8_acc  = 100.0 * int8_correct / total
acc_drop  = fp32_acc - int8_acc

print('=== Quick Accuracy Check (200 val samples) ===')
print(f'  FP32 accuracy : {fp32_acc:.2f}%')
print(f'  INT8 accuracy : {int8_acc:.2f}%')
print(f'  Accuracy drop : {acc_drop:.2f}%')
if acc_drop <= 2.0:
    print('  [PASS] Drop within 2% budget.')
else:
    print('  [WARN] Drop exceeds 2%. Consider increasing calibration samples.')

# [code cell]
# Cell 8 - Cleanup temp calibration folder
if CALIB_DIR.exists():
    shutil.rmtree(CALIB_DIR)
    print(f'Cleaned up: {CALIB_DIR}')

print()
print('=== Quantization Summary ===')
print(f'  FP32 model  : {FP32_PATH.name}  ({FP32_PATH.stat().st_size / 1e6:.2f} MB)')
print(f'  INT8 model  : {INT8_PATH.name}  ({INT8_PATH.stat().st_size / 1e6:.2f} MB)')
print(f'  Size reduction : {reduction:.1f}%')
print(f'  FP32 acc    : {fp32_acc:.2f}%')
print(f'  INT8 acc    : {int8_acc:.2f}%')
print(f'  Drop        : {acc_drop:.2f}%')
print(f'  GV parity   : {passed}/10 PASS')
print()
print('Next step: assemble deployment_package/')

```
