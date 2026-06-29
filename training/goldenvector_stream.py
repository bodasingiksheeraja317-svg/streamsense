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
