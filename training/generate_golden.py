"""
generate_golden.py
Project STREAMSENSE — Track A
Reads golden_selection.json, generates all binary artifacts for all 10 vectors.

Outputs per class (in golden_vectors/):
    raw/        GV_0X_label.bin         [16000] float32 = 64000 bytes
    mel/        GV_0X_label_mel.bin     [64,97] float32 = 24896 bytes
    normalized/ GV_0X_label_norm.bin    [64,97] float32 = 24896 bytes
    labels/     GV_0X_label_label.txt   class index
    manifest.json

Run:
    python generate_golden.py
"""

import torch
import torchaudio
import numpy as np
import json
import sys
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
SELECTION_JSON  = Path(r"C:\STREAMSENSE\stats\golden_selection.json")
STATS_FILE      = Path(r"C:\STREAMSENSE\stats\normalization_stats.json")
GV_ROOT         = Path(r"C:\STREAMSENSE\golden_vectors")

RAW_DIR         = GV_ROOT / "raw"
MEL_DIR         = GV_ROOT / "mel"
NORM_DIR        = GV_ROOT / "normalized"
LABEL_DIR       = GV_ROOT / "labels"
MANIFEST_PATH   = GV_ROOT / "manifest.json"

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
EXPECTED_T    = (FRAME_LEN - N_FFT) // HOP_LENGTH + 1   # 97

# ── Expected binary sizes ─────────────────────────────────────────────────────
RAW_BYTES   = FRAME_LEN * 4                              # 64000
MEL_BYTES   = N_MELS * EXPECTED_T * 4                   # 24896

# ── Load normalization stats ──────────────────────────────────────────────────
with open(STATS_FILE, "r") as f:
    _stats = json.load(f)
GLOBAL_MEAN = float(_stats["global_mean"])
GLOBAL_STD  = float(_stats["global_std"])

# ── MelSpectrogram transform ──────────────────────────────────────────────────
mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate = SAMPLE_RATE,
    n_fft       = N_FFT,
    hop_length  = HOP_LENGTH,
    n_mels      = N_MELS,
    window_fn   = torch.hann_window,
    center      = CENTER,
    power       = POWER,
)

# ── Helpers ───────────────────────────────────────────────────────────────────
def load_wav_raw(path: Path) -> np.ndarray:
    """
    Steps 1-3: load WAV, mono, pad/crop to FRAME_LEN.
    Returns numpy float32 [16000] — this is the raw GV binary content.
    """
    waveform, sr = torchaudio.load(str(path))
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    length = waveform.shape[1]
    if length < FRAME_LEN:
        waveform = torch.nn.functional.pad(waveform, (0, FRAME_LEN - length))
    elif length > FRAME_LEN:
        waveform = waveform[:, :FRAME_LEN]
    return waveform.squeeze(0).float().numpy()           # [16000] float32


def compute_mel(raw: np.ndarray) -> np.ndarray:
    """
    Steps 4-6: MelSpectrogram + log + clamp.
    Returns numpy float32 [64, 97] row-major — mel GV binary content.
    """
    waveform = torch.from_numpy(raw).unsqueeze(0)        # [1, 16000]
    mel = mel_transform(waveform)                        # [1, 64, 97]
    mel = 10.0 * torch.log10(mel + LOG_EPS)
    mel = torch.clamp(mel, min=CLIP_FLOOR_DB)
    return mel.squeeze(0).numpy()                        # [64, 97] float32


def compute_norm(mel: np.ndarray) -> np.ndarray:
    """
    Step 7: global normalization.
    Returns numpy float32 [64, 97] — normalized GV binary content.
    """
    return ((mel - GLOBAL_MEAN) / GLOBAL_STD).astype(np.float32)


def validate_size(path: Path, expected_bytes: int, name: str) -> bool:
    actual = path.stat().st_size
    if actual != expected_bytes:
        print(f"  [FAIL] {name}: expected {expected_bytes} bytes, got {actual}")
        return False
    return True


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("STREAMSENSE — generate_golden.py")
    print("=" * 60)
    print(f"global_mean = {GLOBAL_MEAN:.6f} dB")
    print(f"global_std  = {GLOBAL_STD:.6f} dB")
    print(f"Expected T  = {EXPECTED_T}")
    print(f"RAW_BYTES   = {RAW_BYTES}  ({FRAME_LEN} x 4)")
    print(f"MEL_BYTES   = {MEL_BYTES}  ({N_MELS} x {EXPECTED_T} x 4)")

    # Validate inputs
    for p, name in [(SELECTION_JSON, "golden_selection.json"),
                    (STATS_FILE,     "normalization_stats.json")]:
        if not p.exists():
            print(f"[ERROR] Not found: {p}")
            sys.exit(1)

    # Create output dirs
    for d in [RAW_DIR, MEL_DIR, NORM_DIR, LABEL_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    # Load selection
    with open(SELECTION_JSON, "r") as f:
        selection = json.load(f)

    manifest = {}
    all_passed = True

    for class_idx in range(10):
        s         = selection[str(class_idx)]
        gv_name   = s["gv_name"]               # e.g. GV_00_yes
        label     = s["label"]
        src_path  = Path(s["source_path"])

        print(f"\n{'─'*50}")
        print(f"Generating {gv_name}  (class {class_idx} — '{label}')")
        print(f"  Source: {src_path.name}")

        if not src_path.exists():
            print(f"  [ERROR] Source WAV not found: {src_path}")
            all_passed = False
            continue

        # ── Generate arrays ───────────────────────────────────────────────────
        raw  = load_wav_raw(src_path)           # [16000] float32
        mel  = compute_mel(raw)                 # [64, 97] float32
        norm = compute_norm(mel)                # [64, 97] float32

        # Validate shapes
        assert raw.shape  == (FRAME_LEN,),           f"raw shape error: {raw.shape}"
        assert mel.shape  == (N_MELS, EXPECTED_T),   f"mel shape error: {mel.shape}"
        assert norm.shape == (N_MELS, EXPECTED_T),   f"norm shape error: {norm.shape}"
        assert raw.dtype  == np.float32
        assert mel.dtype  == np.float32
        assert norm.dtype == np.float32

        # ── Write binary files ────────────────────────────────────────────────
        raw_path  = RAW_DIR  / f"{gv_name}.bin"
        mel_path  = MEL_DIR  / f"{gv_name}_mel.bin"
        norm_path = NORM_DIR / f"{gv_name}_norm.bin"
        lbl_path  = LABEL_DIR / f"{gv_name}_label.txt"

        # numpy .tofile() writes row-major (C order) little-endian float32
        raw.tofile(str(raw_path))
        mel.tofile(str(mel_path))
        norm.tofile(str(norm_path))

        with open(lbl_path, "w") as f:
            f.write(str(class_idx))

        # ── Validate file sizes ───────────────────────────────────────────────
        ok_raw  = validate_size(raw_path,  RAW_BYTES, f"{gv_name}.bin")
        ok_mel  = validate_size(mel_path,  MEL_BYTES, f"{gv_name}_mel.bin")
        ok_norm = validate_size(norm_path, MEL_BYTES, f"{gv_name}_norm.bin")

        passed = ok_raw and ok_mel and ok_norm
        if not passed:
            all_passed = False

        print(f"  raw  → {raw_path.name}   {raw_path.stat().st_size} bytes  {'OK' if ok_raw else 'FAIL'}")
        print(f"  mel  → {mel_path.name}  {mel_path.stat().st_size} bytes  {'OK' if ok_mel else 'FAIL'}")
        print(f"  norm → {norm_path.name}  {norm_path.stat().st_size} bytes  {'OK' if ok_norm else 'FAIL'}")
        print(f"  lbl  → {lbl_path.name}")

        # Quick stats for manifest
        manifest[str(class_idx)] = {
            "gv_name"              : gv_name,
            "class_idx"            : class_idx,
            "label"                : label,
            "source_file"          : src_path.name,
            "raw_bin"              : str(raw_path.name),
            "mel_bin"              : str(mel_path.name),
            "norm_bin"             : str(norm_path.name),
            "raw_bytes"            : int(raw_path.stat().st_size),
            "mel_bytes"            : int(mel_path.stat().st_size),
            "norm_bytes"           : int(norm_path.stat().st_size),
            "raw_shape"            : [FRAME_LEN],
            "mel_shape"            : [N_MELS, EXPECTED_T],
            "norm_shape"           : [N_MELS, EXPECTED_T],
            "dtype"                : "float32",
            "endianness"           : "little",
            "layout"               : "row-major C order",
            "mel_peak_db"          : round(float(mel.max()), 4),
            "mel_min_db"           : round(float(mel.min()), 4),
            "norm_mean"            : round(float(norm.mean()), 6),
            "norm_std"             : round(float(norm.std()), 6),
            "expected_top1_index"  : None,   # filled after training
            "size_validated"       : passed,
        }

    # ── Write manifest ────────────────────────────────────────────────────────
    manifest_out = {
        "mpic_version"   : "1.0",
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
        "tolerance_max_abs_error" : 1e-4,
        "vectors"        : manifest,
    }

    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest_out, f, indent=2)

    # ── Final summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for idx in range(10):
        if str(idx) in manifest:
            m = manifest[str(idx)]
            status = "PASS" if m["size_validated"] else "FAIL"
            print(f"  [{status}] {m['gv_name']:20s}  "
                  f"mel_peak={m['mel_peak_db']:6.1f} dB  "
                  f"norm_mean={m['norm_mean']:+.4f}")

    print(f"\nManifest -> {MANIFEST_PATH}")

    if all_passed:
        print("\n[DONE] All 10 golden vectors generated and validated.")
        print("Next: python verify_pipeline.py")
    else:
        print("\n[FAIL] Some vectors failed — check errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
