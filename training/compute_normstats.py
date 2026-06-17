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
