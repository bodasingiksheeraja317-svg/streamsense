"""
mel_pipeline.py
Project STREAMSENSE — Track A
MPIC v1.0 — complete preprocessing pipeline (Steps 1-8).

Single public function:
    preprocess(samples) -> torch.Tensor shape [1,1,64,97] float32

Accepts:
    - numpy array  (any length, mono or stereo)
    - torch tensor (any length, mono or stereo)

Loads normalization stats from:
    /content/streamsense/stats/normalization_stats.json

Run directly for 8-test self-test:
    python mel_pipeline.py
"""

import torch
import torchaudio
import numpy as np
import json
import sys
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
STATS_FILE = Path(__file__).resolve().parent.parent / "stats" / "normalization_stats.json"

# ── MPIC v1.0 frozen parameters ───────────────────────────────────────────────
SAMPLE_RATE   = 16000
FRAME_LEN     = 16000
N_FFT         = 512
HOP_LENGTH    = 160
N_MELS        = 64
CENTER        = False          # critical — gives T=97
POWER         = 2.0
LOG_EPS       = 1e-10
CLIP_FLOOR_DB = -80.0
EXPECTED_T    = (FRAME_LEN - N_FFT) // HOP_LENGTH + 1  # = 97
OUTPUT_SHAPE  = (1, 1, N_MELS, EXPECTED_T)              # (1,1,64,97)

# ── Load normalization stats at import ────────────────────────────────────────
if not STATS_FILE.exists():
    raise FileNotFoundError(
        f"Normalization stats not found: {STATS_FILE}\n"
        f"Run compute_normstats.py first."
    )

with open(STATS_FILE, "r") as _f:
    _stats = json.load(_f)

GLOBAL_MEAN = float(_stats["global_mean"])
GLOBAL_STD  = float(_stats["global_std"])

if GLOBAL_STD <= 0.0:
    raise ValueError(f"global_std={GLOBAL_STD} is invalid — stats file may be corrupt.")

# ── MelSpectrogram transform (CPU, built once at import) ──────────────────────
_mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate = SAMPLE_RATE,
    n_fft       = N_FFT,
    hop_length  = HOP_LENGTH,
    n_mels      = N_MELS,
    window_fn   = torch.hann_window,
    center      = CENTER,
    power       = POWER,
)

# ── Public API ────────────────────────────────────────────────────────────────
def preprocess(samples) -> torch.Tensor:
    """
    Full MPIC v1.0 pipeline — Steps 1 through 8.

    Args:
        samples: numpy array or torch tensor, any length, mono or stereo.
                 Expected: 1D [T] or 2D [C,T] or [T,C].

    Returns:
        torch.Tensor of shape [1, 1, 64, 97], dtype float32.

    Pipeline:
        Step 1 — accept float32 samples
        Step 2 — stereo -> mono
        Step 3 — pad (zeros right) or crop to 16000 samples
        Step 4 — MelSpectrogram -> [1, 64, 97]
        Step 5 — 10 * log10(mel + 1e-10)
        Step 6 — clamp(min=-80 dB)
        Step 7 — (mel - global_mean) / global_std
        Step 8 — reshape to [1, 1, 64, 97] float32
        Step 9 — validate shape == (1, 1, 64, 97)
    """

    # ── Step 1: convert input to float32 torch tensor ────────────────────────
    if isinstance(samples, np.ndarray):
        waveform = torch.from_numpy(samples.copy()).float()
    elif isinstance(samples, torch.Tensor):
        waveform = samples.float().clone()
    else:
        raise TypeError(
            f"preprocess() expects numpy array or torch tensor, got {type(samples)}"
        )

    # Ensure 2D [C, T]
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)        # [T] -> [1, T]
    elif waveform.ndim == 2:
        # Handle [T, C] (uncommon but possible)
        if waveform.shape[0] > waveform.shape[1]:
            waveform = waveform.T               # [T, C] -> [C, T]
    else:
        raise ValueError(
            f"Expected 1D or 2D input, got shape {waveform.shape}"
        )
    # waveform is now [C, T]

    # ── Step 2: stereo -> mono ────────────────────────────────────────────────
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)   # [C,T] -> [1,T]
    # waveform is now [1, T]

    # ── Step 3: pad or crop to exactly FRAME_LEN ─────────────────────────────
    length = waveform.shape[1]
    if length < FRAME_LEN:
        waveform = torch.nn.functional.pad(waveform, (0, FRAME_LEN - length))
    elif length > FRAME_LEN:
        waveform = waveform[:, :FRAME_LEN]
    # waveform is now [1, 16000]

    # ── Step 4: MelSpectrogram ────────────────────────────────────────────────
    mel = _mel_transform(waveform)              # [1, 64, 97]

    # ── Step 5: log scaling ───────────────────────────────────────────────────
    mel = 10.0 * torch.log10(mel + LOG_EPS)     # [1, 64, 97]

    # ── Step 6: clip floor ────────────────────────────────────────────────────
    mel = torch.clamp(mel, min=CLIP_FLOOR_DB)   # [1, 64, 97]

    # ── Step 7: global normalization ─────────────────────────────────────────
    mel = (mel - GLOBAL_MEAN) / GLOBAL_STD      # [1, 64, 97]

    # ── Step 8: reshape to [1, 1, 64, 97] ────────────────────────────────────
    mel = mel.unsqueeze(0)                      # [1, 64, 97] -> [1, 1, 64, 97]

    # ── Step 9: validate ─────────────────────────────────────────────────────
    assert tuple(mel.shape) == OUTPUT_SHAPE, (
        f"Shape error: expected {OUTPUT_SHAPE}, got {tuple(mel.shape)}"
    )
    assert mel.dtype == torch.float32, (
        f"Dtype error: expected float32, got {mel.dtype}"
    )

    return mel


# ── Self-test (run directly: python mel_pipeline.py) ─────────────────────────
def _run_self_tests():
    print("=" * 60)
    print("mel_pipeline.py — self-test (8 tests)")
    print(f"global_mean = {GLOBAL_MEAN:.6f} dB")
    print(f"global_std  = {GLOBAL_STD:.6f} dB")
    print("=" * 60)

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

    # ── Test 1: normal 16000-sample numpy input ───────────────────────────────
    try:
        samples = np.random.randn(16000).astype(np.float32)
        out = preprocess(samples)
        check("T1 — numpy 1D [16000] input",
              tuple(out.shape) == OUTPUT_SHAPE and out.dtype == torch.float32,
              f"got shape={out.shape} dtype={out.dtype}")
    except Exception as e:
        check("T1 — numpy 1D [16000] input", False, str(e))

    # ── Test 2: short input — needs padding ───────────────────────────────────
    try:
        samples = np.random.randn(8000).astype(np.float32)
        out = preprocess(samples)
        check("T2 — short numpy input [8000] — padding",
              tuple(out.shape) == OUTPUT_SHAPE,
              f"got shape={out.shape}")
    except Exception as e:
        check("T2 — short numpy input [8000] — padding", False, str(e))

    # ── Test 3: long input — needs cropping ───────────────────────────────────
    try:
        samples = np.random.randn(24000).astype(np.float32)
        out = preprocess(samples)
        check("T3 — long numpy input [24000] — cropping",
              tuple(out.shape) == OUTPUT_SHAPE,
              f"got shape={out.shape}")
    except Exception as e:
        check("T3 — long numpy input [24000] — cropping", False, str(e))

    # ── Test 4: stereo numpy input ────────────────────────────────────────────
    try:
        samples = np.random.randn(2, 16000).astype(np.float32)  # [2, T]
        out = preprocess(samples)
        check("T4 — stereo numpy [2,16000] — mono conversion",
              tuple(out.shape) == OUTPUT_SHAPE,
              f"got shape={out.shape}")
    except Exception as e:
        check("T4 — stereo numpy [2,16000] — mono conversion", False, str(e))

    # ── Test 5: torch tensor input ────────────────────────────────────────────
    try:
        samples = torch.randn(16000)                            # [T]
        out = preprocess(samples)
        check("T5 — torch tensor 1D [16000] input",
              tuple(out.shape) == OUTPUT_SHAPE,
              f"got shape={out.shape}")
    except Exception as e:
        check("T5 — torch tensor 1D [16000] input", False, str(e))

    # ── Test 6: stereo torch tensor input ─────────────────────────────────────
    try:
        samples = torch.randn(2, 16000)                         # [2, T]
        out = preprocess(samples)
        check("T6 — stereo torch tensor [2,16000] — mono conversion",
              tuple(out.shape) == OUTPUT_SHAPE,
              f"got shape={out.shape}")
    except Exception as e:
        check("T6 — stereo torch tensor [2,16000] — mono conversion", False, str(e))

    # ── Test 7: output shape exactly (1,1,64,97) ──────────────────────────────
    try:
        samples = np.random.randn(16000).astype(np.float32)
        out = preprocess(samples)
        check("T7 — output shape exactly (1,1,64,97)",
              tuple(out.shape) == (1, 1, 64, 97),
              f"got {tuple(out.shape)}")
    except Exception as e:
        check("T7 — output shape exactly (1,1,64,97)", False, str(e))

    # ── Test 8: output dtype is float32 ───────────────────────────────────────
    try:
        samples = np.random.randn(16000).astype(np.float32)
        out = preprocess(samples)
        check("T8 — output dtype is float32",
              out.dtype == torch.float32,
              f"got {out.dtype}")
    except Exception as e:
        check("T8 — output dtype is float32", False, str(e))

    # ── Summary ───────────────────────────────────────────────────────────────
    print("=" * 60)
    print(f"Results: {passed}/8 passed,  {failed}/8 failed")
    if failed == 0:
        print("[DONE] All tests PASS — mel_pipeline.py is ready.")
    else:
        print("[FAIL] Some tests failed — check errors above.")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    ok = _run_self_tests()
    sys.exit(0 if ok else 1)
