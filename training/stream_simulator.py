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
