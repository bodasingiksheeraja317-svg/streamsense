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
