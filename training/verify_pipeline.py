"""
verify_pipeline.py
Project STREAMSENSE — Track A
MPIC v1.0 — Pipeline verification against frozen golden vectors.

Verifies TWO stages for each of the 10 golden vectors:
    Stage 1: raw audio -> mel spectrogram    (compare vs *_mel.bin)
    Stage 2: raw audio -> normalized output  (compare vs *_norm.bin)

Tolerance is read at runtime from manifest.json — never hardcoded.
All 10 vectors are always tested. No early exit on failure.

Required files before running:
    C:\\STREAMSENSE\\golden_vectors\\manifest.json
    C:\\STREAMSENSE\\golden_vectors\\raw\\GV_0X_<label>.bin         (x10)
    C:\\STREAMSENSE\\golden_vectors\\mel\\GV_0X_<label>_mel.bin     (x10)
    C:\\STREAMSENSE\\golden_vectors\\normalized\\GV_0X_<label>_norm.bin (x10)
    C:\\STREAMSENSE\\stats\\normalization_stats.json

Run:
    python verify_pipeline.py
"""

import sys
import json
import numpy as np
import torch
import torchaudio
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
GV_ROOT       = Path(r"C:\STREAMSENSE\golden_vectors")
MANIFEST_PATH = GV_ROOT / "manifest.json"
RAW_DIR       = GV_ROOT / "raw"
MEL_DIR       = GV_ROOT / "mel"
NORM_DIR      = GV_ROOT / "normalized"

# mel_pipeline.py must be on the Python path (same directory is fine)
try:
    from mel_pipeline import preprocess
except ImportError as e:
    print(f"[ERROR] Cannot import mel_pipeline: {e}")
    print("        Make sure mel_pipeline.py is in the same directory and "
          "normalization_stats.json exists.")
    sys.exit(1)

# ── MPIC v1.0 parameters — used only for the independent mel-only reimplementation
# (Option A: Steps 1-6 reproduced here to verify the intermediate stage)
SAMPLE_RATE   = 16000
FRAME_LEN     = 16000
N_FFT         = 512
HOP_LENGTH    = 160
N_MELS        = 64
CENTER        = False
POWER         = 2.0
LOG_EPS       = 1e-10
CLIP_FLOOR_DB = -80.0
EXPECTED_T    = (FRAME_LEN - N_FFT) // HOP_LENGTH + 1   # 97
EXPECTED_MEL_SHAPE  = (N_MELS, EXPECTED_T)               # (64, 97)
EXPECTED_NORM_SHAPE = (1, 1, N_MELS, EXPECTED_T)         # (1, 1, 64, 97)

# Build the mel transform once — same construction as generate_golden.py
_mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate = SAMPLE_RATE,
    n_fft       = N_FFT,
    hop_length  = HOP_LENGTH,
    n_mels      = N_MELS,
    window_fn   = torch.hann_window,
    center      = CENTER,
    power       = POWER,
)


# ── Independent mel-only pipeline (Steps 1-6, no normalization) ───────────────
# This is the Option A reimplementation. It mirrors generate_golden.py's
# load_wav_raw() + compute_mel() exactly, giving a clean cross-check.

def _mel_from_raw(raw: np.ndarray) -> np.ndarray:
    """
    Steps 1-6 only (NO normalization).
    Input:  raw float32 numpy array [16000] — already loaded from .bin
    Output: log-mel numpy float32 [64, 97]
    """
    # Step 1: to float32 tensor [1, 16000]
    waveform = torch.from_numpy(raw).unsqueeze(0).float()   # [1, 16000]

    # Steps 4-6: MelSpectrogram + log scaling + clamp
    mel = _mel_transform(waveform)                          # [1, 64, 97]
    mel = 10.0 * torch.log10(mel + LOG_EPS)
    mel = torch.clamp(mel, min=CLIP_FLOOR_DB)
    return mel.squeeze(0).numpy()                           # [64, 97] float32


# ── Binary loading helpers ─────────────────────────────────────────────────────

def _load_bin(path: Path, shape: tuple) -> np.ndarray:
    """
    Load a little-endian float32 binary file and reshape.
    Row-major (C order) as written by numpy .tofile().
    """
    arr = np.fromfile(str(path), dtype="<f4")               # little-endian float32
    if arr.size != int(np.prod(shape)):
        raise ValueError(
            f"Size mismatch loading {path.name}: "
            f"expected {int(np.prod(shape))} elements, got {arr.size}"
        )
    return arr.reshape(shape)


# ── Pre-flight checks ─────────────────────────────────────────────────────────

def _check_prerequisites(manifest: dict) -> list[str]:
    """
    Verify all required files exist before running any vector.
    Returns list of error strings (empty = all good).
    """
    errors = []
    vectors = manifest.get("vectors", {})

    for class_idx in range(10):
        key = str(class_idx)
        if key not in vectors:
            errors.append(f"  manifest missing entry for class {class_idx}")
            continue
        v = vectors[key]

        raw_path  = RAW_DIR  / v["raw_bin"]
        mel_path  = MEL_DIR  / v["mel_bin"]
        norm_path = NORM_DIR / v["norm_bin"]

        for p, label in [(raw_path, "raw"), (mel_path, "mel"), (norm_path, "norm")]:
            if not p.exists():
                errors.append(f"  Missing {label} file: {p}")

    return errors


# ── Per-vector verification ───────────────────────────────────────────────────

def _verify_vector(class_idx: int, v: dict, tolerance: float) -> tuple[bool, str]:
    """
    Run both stages for one golden vector.
    Returns (all_passed: bool, detail_lines: str).
    """
    gv_name   = v["gv_name"]
    label     = v["label"]
    raw_shape = tuple(v["raw_shape"])    # (16000,)
    mel_shape = tuple(v["mel_shape"])    # (64, 97)

    lines = []
    lines.append(f"\n  {'─'*54}")
    lines.append(f"  {gv_name}  (class {class_idx} — '{label}')")

    stage_results = []

    # ── Load raw binary ────────────────────────────────────────────────────────
    raw_path = RAW_DIR / v["raw_bin"]
    try:
        raw = _load_bin(raw_path, raw_shape)    # [16000] float32
    except Exception as e:
        lines.append(f"  [FAIL] Could not load raw bin: {e}")
        return False, "\n".join(lines)

    # ── Stage 1: Mel intermediate check ───────────────────────────────────────
    golden_mel_path = MEL_DIR / v["mel_bin"]
    try:
        golden_mel     = _load_bin(golden_mel_path, mel_shape)   # [64, 97]
        pipeline_mel   = _mel_from_raw(raw)                      # [64, 97]
        max_abs_mel    = float(np.max(np.abs(pipeline_mel - golden_mel)))
        mel_pass       = max_abs_mel <= tolerance
        stage_results.append(mel_pass)
        status_mel     = "PASS" if mel_pass else "FAIL"
        lines.append(
            f"  [{status_mel}] Stage 1 — mel        "
            f"max_abs_err={max_abs_mel:.6e}  (tol={tolerance:.1e})"
        )
    except Exception as e:
        lines.append(f"  [FAIL] Stage 1 — mel        ERROR: {e}")
        stage_results.append(False)

    # ── Stage 2: Normalized output check ──────────────────────────────────────
    golden_norm_path = NORM_DIR / v["norm_bin"]
    try:
        golden_norm      = _load_bin(golden_norm_path, mel_shape)     # [64, 97]
        # preprocess() returns [1, 1, 64, 97] — squeeze to [64, 97] for comparison
        pipeline_out     = preprocess(raw)                            # [1, 1, 64, 97]
        pipeline_norm    = pipeline_out.squeeze().numpy()             # [64, 97]
        max_abs_norm     = float(np.max(np.abs(pipeline_norm - golden_norm)))
        norm_pass        = max_abs_norm <= tolerance
        stage_results.append(norm_pass)
        status_norm      = "PASS" if norm_pass else "FAIL"
        lines.append(
            f"  [{status_norm}] Stage 2 — normalized  "
            f"max_abs_err={max_abs_norm:.6e}  (tol={tolerance:.1e})"
        )
    except Exception as e:
        lines.append(f"  [FAIL] Stage 2 — normalized  ERROR: {e}")
        stage_results.append(False)

    all_passed = all(stage_results)
    return all_passed, "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("STREAMSENSE — verify_pipeline.py")
    print("MPIC v1.0 — Golden Vector Verification")
    print("=" * 60)

    # ── Load manifest ──────────────────────────────────────────────────────────
    if not MANIFEST_PATH.exists():
        print(f"[ERROR] Manifest not found: {MANIFEST_PATH}")
        print("        Run generate_golden.py first.")
        sys.exit(1)

    with open(MANIFEST_PATH, "r") as f:
        manifest = json.load(f)

    # Read tolerance from manifest — never hardcoded
    tolerance = float(manifest["tolerance_max_abs_error"])

    print(f"\nManifest     : {MANIFEST_PATH}")
    print(f"MPIC version : {manifest.get('mpic_version', 'unknown')}")
    print(f"global_mean  : {manifest.get('global_mean', '?')}")
    print(f"global_std   : {manifest.get('global_std', '?')}")
    print(f"Tolerance    : {tolerance:.1e}  (from manifest)")
    print(f"Vectors      : 10  (2 stages each = 20 checks total)")

    # ── Pre-flight ────────────────────────────────────────────────────────────
    print("\nChecking required files...")
    prereq_errors = _check_prerequisites(manifest)
    if prereq_errors:
        print("[ERROR] Missing files — cannot proceed:")
        for e in prereq_errors:
            print(e)
        sys.exit(1)
    print("  [OK] All 30 binary files present (10 raw + 10 mel + 10 norm)")

    # ── Run all 10 vectors ────────────────────────────────────────────────────
    print("\nRunning verification...\n")

    vectors      = manifest["vectors"]
    passed_count = 0
    failed_count = 0
    failed_names = []

    for class_idx in range(10):
        v = vectors[str(class_idx)]
        ok, detail = _verify_vector(class_idx, v, tolerance)
        print(detail)

        if ok:
            passed_count += 1
        else:
            failed_count += 1
            failed_names.append(v["gv_name"])

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"  Passed: {passed_count}/10")
    print(f"  Failed: {failed_count}/10")

    if failed_count == 0:
        print("\n[PASS] All golden vectors verified. Pipeline is correct.")
        print("       Integration acceptance gate: OPEN")
        sys.exit(0)
    else:
        print(f"\n[FAIL] {failed_count} vector(s) failed:")
        for name in failed_names:
            print(f"       • {name}")
        print("\n       Integration acceptance gate: BLOCKED")
        print("       Check mel_pipeline.py MPIC parameters match manifest.")
        sys.exit(1)


if __name__ == "__main__":
    main()
