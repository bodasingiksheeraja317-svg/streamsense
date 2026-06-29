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
