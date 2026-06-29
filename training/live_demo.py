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
