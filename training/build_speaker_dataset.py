"""
build_speaker_dataset.py
========================
OSL-IPL-2026-INT-002  |  Track A  |  WA-3 Dataset Prep

Scans data/raw/<class>/ for all 10 command classes, extracts the
speaker hash from the GSC v2 filename convention:

    <speaker_hash>_nohash_<utterance_idx>.wav

Builds a speaker-disjoint 80/10/10 train/val/test split and writes:

    data/speaker_splits/speaker_train.csv
    data/speaker_splits/speaker_val.csv
    data/speaker_splits/speaker_test.csv

Each CSV has columns:
    filepath | speaker_id | class_label

speaker_id is a stable integer (0-indexed, sorted by hex hash string)
so IDs are reproducible across machines as long as the dataset is the
same version of GSC v2.

Run once:  python training/build_speaker_dataset.py
           (or from repo root: python training/build_speaker_dataset.py)

No frozen artifacts (MPIC, GV1K) are touched.
"""

import os
import csv
import random
import argparse
from pathlib import Path
from collections import defaultdict

# ── Constants ────────────────────────────────────────────────────────────────

# Must match class_labels.json ordering used in StreamSenseNet
CLASS_NAMES = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]

TRAIN_FRAC = 0.80
VAL_FRAC   = 0.10
TEST_FRAC  = 0.10
MIN_UTTERANCES = 2       # speakers with fewer files are excluded from training
RANDOM_SEED    = 42


# ── Helpers ──────────────────────────────────────────────────────────────────

def extract_speaker_hash(stem: str) -> str | None:
    """
    'ddedba85_nohash_9'  ->  'ddedba85'
    Returns None if the filename does not follow the GSC convention.
    """
    parts = stem.split("_nohash_")
    if len(parts) != 2:
        return None
    return parts[0]


def scan_dataset(raw_dir: Path) -> dict[str, list[tuple[str, int]]]:
    """
    Walk raw_dir/<class>/ for each class in CLASS_NAMES.
    Returns:
        speaker_map: { speaker_hash -> [(abs_filepath, class_idx), ...] }
    """
    speaker_map: dict[str, list[tuple[str, int]]] = defaultdict(list)
    missing_classes = []

    for class_idx, class_name in enumerate(CLASS_NAMES):
        class_dir = raw_dir / class_name
        if not class_dir.is_dir():
            missing_classes.append(class_name)
            continue

        wav_files = sorted(class_dir.glob("*.wav"))
        for wav_path in wav_files:
            speaker_hash = extract_speaker_hash(wav_path.stem)
            if speaker_hash is None:
                # non-standard filename — skip silently
                continue
            speaker_map[speaker_hash].append((str(wav_path), class_idx))

    if missing_classes:
        print(f"[WARN] Missing class directories: {missing_classes}")

    return speaker_map


def assign_integer_ids(speaker_map: dict) -> dict[str, int]:
    """
    Assign stable integer IDs by sorting speaker hashes lexicographically.
    Returns:  { speaker_hash -> integer_id }
    """
    sorted_hashes = sorted(speaker_map.keys())
    return {h: idx for idx, h in enumerate(sorted_hashes)}


def speaker_level_split(
    speaker_hashes: list[str],
    rng: random.Random,
) -> tuple[list[str], list[str], list[str]]:
    """
    Shuffles speaker list and splits 80/10/10 by speaker count.
    Returns (train_hashes, val_hashes, test_hashes).
    """
    hashes = list(speaker_hashes)
    rng.shuffle(hashes)
    n = len(hashes)
    n_train = int(n * TRAIN_FRAC)
    n_val   = int(n * VAL_FRAC)
    train = hashes[:n_train]
    val   = hashes[n_train : n_train + n_val]
    test  = hashes[n_train + n_val :]
    return train, val, test


def write_csv(
    out_path: Path,
    rows: list[tuple[str, int, int]],
) -> None:
    """
    Writes CSV with header:  filepath,speaker_id,class_label
    rows: [(filepath, speaker_id, class_label), ...]
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filepath", "speaker_id", "class_label"])
        for filepath, speaker_id, class_label in rows:
            writer.writerow([filepath, speaker_id, class_label])


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Build speaker-disjoint split manifests.")
    parser.add_argument(
        "--root",
        type=str,
        default=None,
        help="Project root (default: parent of this script's directory)",
    )
    args = parser.parse_args()

    # Resolve paths robustly whether run from repo root or training/
    script_dir = Path(__file__).resolve().parent
    if args.root:
        root = Path(args.root).resolve()
    else:
        # script lives in training/; project root is one level up
        root = script_dir.parent

    raw_dir  = root / "data" / "raw"
    out_dir  = root / "data" / "speaker_splits"

    print(f"[INFO] Project root : {root}")
    print(f"[INFO] Raw data dir : {raw_dir}")
    print(f"[INFO] Output dir   : {out_dir}")
    print()

    # ── 1. Scan ───────────────────────────────────────────────────────────────
    print("Scanning dataset …")
    speaker_map = scan_dataset(raw_dir)
    print(f"  Total unique speaker hashes found : {len(speaker_map)}")
    total_files = sum(len(v) for v in speaker_map.values())
    print(f"  Total WAV files indexed           : {total_files}")

    # ── 2. Filter ─────────────────────────────────────────────────────────────
    eligible = {
        h: files
        for h, files in speaker_map.items()
        if len(files) >= MIN_UTTERANCES
    }
    excluded = len(speaker_map) - len(eligible)
    print(f"  Speakers with < {MIN_UTTERANCES} utterances (excluded) : {excluded}")
    print(f"  Eligible speakers                 : {len(eligible)}")

    if len(eligible) < 10:
        raise RuntimeError(
            f"Only {len(eligible)} eligible speakers found — dataset may not be set up correctly."
        )

    # ── 3. Assign integer IDs ─────────────────────────────────────────────────
    hash_to_id = assign_integer_ids(eligible)
    n_speakers = len(hash_to_id)

    # ── 4. Speaker-level split ────────────────────────────────────────────────
    rng = random.Random(RANDOM_SEED)
    train_hashes, val_hashes, test_hashes = speaker_level_split(
        list(eligible.keys()), rng
    )

    # ── 5. Build row lists ────────────────────────────────────────────────────
    def hashes_to_rows(hashes):
        rows = []
        for h in hashes:
            sid = hash_to_id[h]
            for filepath, class_idx in eligible[h]:
                rows.append((filepath, sid, class_idx))
        return rows

    train_rows = hashes_to_rows(train_hashes)
    val_rows   = hashes_to_rows(val_hashes)
    test_rows  = hashes_to_rows(test_hashes)

    # ── 6. Write CSVs ─────────────────────────────────────────────────────────
    write_csv(out_dir / "speaker_train.csv", train_rows)
    write_csv(out_dir / "speaker_val.csv",   val_rows)
    write_csv(out_dir / "speaker_test.csv",  test_rows)

    # ── 7. Summary ────────────────────────────────────────────────────────────
    utterances_per_speaker = [len(v) for v in eligible.values()]
    utterances_per_speaker.sort()

    print()
    print("=" * 52)
    print("  SPEAKER DATASET SUMMARY")
    print("=" * 52)
    print(f"  Total eligible speakers : {n_speakers}")
    print(f"  Train speakers          : {len(train_hashes)}")
    print(f"  Val   speakers          : {len(val_hashes)}")
    print(f"  Test  speakers          : {len(test_hashes)}")
    print()
    print(f"  Train rows (utterances) : {len(train_rows)}")
    print(f"  Val   rows              : {len(val_rows)}")
    print(f"  Test  rows              : {len(test_rows)}")
    print()
    print(f"  Utterances/speaker — min  : {utterances_per_speaker[0]}")
    print(f"  Utterances/speaker — max  : {utterances_per_speaker[-1]}")
    print(f"  Utterances/speaker — mean : {sum(utterances_per_speaker)/len(utterances_per_speaker):.1f}")
    p50 = utterances_per_speaker[len(utterances_per_speaker) // 2]
    print(f"  Utterances/speaker — p50  : {p50}")
    print("=" * 52)
    print()
    print("Output files:")
    print(f"  {out_dir / 'speaker_train.csv'}")
    print(f"  {out_dir / 'speaker_val.csv'}")
    print(f"  {out_dir / 'speaker_test.csv'}")
    print()
    print("Done. Commit data/speaker_splits/ to the repo before training.")


if __name__ == "__main__":
    main()
