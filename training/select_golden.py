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
