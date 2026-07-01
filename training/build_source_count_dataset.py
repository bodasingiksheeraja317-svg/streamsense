"""
build_source_count_dataset.py
Project STREAMSENSE — WA-3 Source Counting

Builds a synthetic "how many people are talking" dataset from the existing
GSC wav files by summing waveforms from N distinct speakers.

Only reads from data/speaker_splits/speaker_train.csv — the speaker-disjoint
TRAIN split already produced by build_speaker_dataset.py. The original
val/test speakers are never touched here, so they stay reserved for the
keyword-spotting / speaker-ID tasks.

speaker_train.csv columns: filepath, speaker_id, class_label
(speaker_id is the existing stable integer ID assigned in
build_speaker_dataset.py — we reuse it directly instead of re-deriving the
hash from the filename, since it already encodes the correct disjoint split.)

For N in [1..8]:
    - draw N DISTINCT speaker_ids (random.sample -> no repeats)
    - pick 1 random wav per chosen speaker
    - sum the waveforms, peak-normalise
    - save as data/source_count_splits/clips/n{N}/clip_XXXX.npy
    - label = N - 1  (so N=1 -> 0 ... N=8 -> 7)

Writes:
    data/source_count_splits/source_count_train.csv
    data/source_count_splits/source_count_val.csv
    Columns: filepath, label

Run:
    python training/build_source_count_dataset.py
    python training/build_source_count_dataset.py --root /content/STREAMSENSE --seed 42 --clips_per_class 1000
"""

import argparse
import csv
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.io import wavfile

# ── Constants ────────────────────────────────────────────────────────────────
N_VALUES = list(range(1, 9))       # 1..8 simultaneous speakers
SAMPLE_RATE = 16000
FRAME_LEN = 16000
TRAIN_FRAC = 0.80


# ── Helpers ──────────────────────────────────────────────────────────────────
def load_wav_float32(path: str) -> np.ndarray:
    """
    Read a GSC wav file and return float32 samples in [-1, 1], length exactly
    FRAME_LEN (pad with zeros / crop as a safety guard — GSC clips are
    already 16000 samples at 16kHz).
    """
    sr, raw = wavfile.read(path)
    if sr != SAMPLE_RATE:
        raise ValueError(f"Unexpected sample rate {sr} in {path} (expected {SAMPLE_RATE}).")

    if raw.dtype == np.int16:
        data = raw.astype(np.float32) / 32768.0
    else:
        data = raw.astype(np.float32)

    if data.ndim > 1:                       # guard against stray stereo files
        data = data.mean(axis=1)

    if len(data) < FRAME_LEN:
        data = np.pad(data, (0, FRAME_LEN - len(data)))
    elif len(data) > FRAME_LEN:
        data = data[:FRAME_LEN]

    return data.astype(np.float32)


def remap_to_local_raw(filepath: str, raw_dir: Path) -> str:
    """
    speaker_train.csv may have been built on a different machine/OS (e.g. a
    Windows path like 'C:\\STREAMSENSE\\data\\raw\\down\\xxx.wav'), so the
    stored path won't resolve on this machine (e.g. Colab). We only need the
    last two path components -- <class>/<filename> -- and re-root them under
    this run's actual data/raw/ directory.
    """
    parts = filepath.replace("\\", "/").split("/")
    class_name, filename = parts[-2], parts[-1]
    return str(raw_dir / class_name / filename)


def load_speaker_map(csv_path: Path, raw_dir: Path) -> dict:
    """
    Returns { speaker_id (int) -> [filepath, ...] }, built from
    speaker_train.csv, with every path re-rooted to this machine's
    data/raw/ directory (see remap_to_local_raw). Only speakers with >=1
    file are kept (the upstream build_speaker_dataset.py already filtered
    out speakers with <2 utterances, so in practice every speaker here
    qualifies).
    """
    speaker_map = defaultdict(list)
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            local_path = remap_to_local_raw(row["filepath"], raw_dir)
            speaker_map[int(row["speaker_id"])].append(local_path)
    return {sid: files for sid, files in speaker_map.items() if len(files) >= 1}


def mix_clip(rng: random.Random, speaker_ids: list, speaker_map: dict, n: int) -> np.ndarray:
    """Pick n distinct speakers, 1 random wav each, sum + peak-normalise."""
    chosen = rng.sample(speaker_ids, n)
    waves = []
    for sid in chosen:
        wav_path = rng.choice(speaker_map[sid])
        waves.append(load_wav_float32(wav_path))

    mixed = np.sum(waves, axis=0).astype(np.float32)
    peak = np.abs(mixed).max()
    mixed = mixed / (peak + 1e-8)
    return mixed


def write_csv(out_path: Path, rows: list) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filepath", "label"])
        for filepath, label in rows:
            writer.writerow([filepath, label])


# ── Main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="Build synthetic source-counting dataset from GSC.")
    parser.add_argument("--root", type=str, default=None, help="Project root (default: parent of this script's directory)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--clips_per_class", type=int, default=1000)
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    root = Path(args.root).resolve() if args.root else script_dir.parent

    speaker_train_csv = root / "data" / "speaker_splits" / "speaker_train.csv"
    raw_dir = root / "data" / "raw"
    out_root = root / "data" / "source_count_splits"
    clips_root = out_root / "clips"

    if not speaker_train_csv.exists():
        raise FileNotFoundError(
            f"Missing {speaker_train_csv}. Run training/build_speaker_dataset.py first."
        )

    print(f"[INFO] Project root      : {root}")
    print(f"[INFO] Speaker train CSV : {speaker_train_csv}")
    print(f"[INFO] Output dir        : {out_root}")
    print(f"[INFO] Seed              : {args.seed}")
    print(f"[INFO] Clips per class   : {args.clips_per_class}")
    print()

    speaker_map = load_speaker_map(speaker_train_csv, raw_dir)
    speaker_ids = list(speaker_map.keys())
    print(f"[INFO] Eligible train speakers: {len(speaker_ids)}")
    if len(speaker_ids) < max(N_VALUES):
        raise RuntimeError(
            f"Only {len(speaker_ids)} speakers available in speaker_train.csv, "
            f"need at least {max(N_VALUES)} distinct speakers for N=8 mixes."
        )

    # Fail fast with a clear message if paths still don't resolve, instead of
    # dying partway through building 8000 clips.
    sample_path = next(iter(speaker_map.values()))[0]
    if not Path(sample_path).exists():
        raise FileNotFoundError(
            f"Remapped path does not exist: {sample_path}\n"
            f"Check that {raw_dir} contains <class>/<file>.wav (i.e. data/raw/ was extracted)."
        )

    rng = random.Random(args.seed)
    records = []
    t0 = time.time()

    for n in N_VALUES:
        n_dir = clips_root / f"n{n}"
        n_dir.mkdir(parents=True, exist_ok=True)
        label = n - 1

        for clip_idx in range(args.clips_per_class):
            mixed = mix_clip(rng, speaker_ids, speaker_map, n)
            clip_path = n_dir / f"clip_{clip_idx:04d}.npy"
            np.save(clip_path, mixed)
            records.append((str(clip_path), label))

        print(f"  N={n}: {args.clips_per_class} clips  -> {n_dir}")

    # ── Shuffle + split ────────────────────────────────────────────────────
    rng.shuffle(records)
    n_total = len(records)
    n_train = int(n_total * TRAIN_FRAC)
    train_records = records[:n_train]
    val_records = records[n_train:]

    write_csv(out_root / "source_count_train.csv", train_records)
    write_csv(out_root / "source_count_val.csv", val_records)

    elapsed = time.time() - t0
    print()
    print("=" * 52)
    print("  SOURCE COUNT DATASET SUMMARY")
    print("=" * 52)
    for n in N_VALUES:
        print(f"  N={n}: {args.clips_per_class} clips")
    print(f"  Total clips : {n_total}")
    print(f"  Train       : {len(train_records)}")
    print(f"  Val         : {len(val_records)}")
    print(f"  Elapsed     : {elapsed:.1f}s")
    print()
    print("Sample path check:")
    sample_path, sample_label = train_records[0]
    print(f"  {sample_path}  (label={sample_label})  exists={Path(sample_path).exists()}")
    print("=" * 52)
    print()
    print(f"Wrote: {out_root / 'source_count_train.csv'}")
    print(f"Wrote: {out_root / 'source_count_val.csv'}")


if __name__ == "__main__":
    main()
