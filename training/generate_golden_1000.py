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
