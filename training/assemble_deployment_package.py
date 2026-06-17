"""
assemble_deployment_package.py
Project STREAMSENSE — Track A
Phase 3 — Deployment Package Assembly

Collects every artifact a downstream consumer (Kavish / Track B, or FPGA
team in Phase 4) needs into a single self-contained folder:

    C:\\STREAMSENSE\\deployment_package\\
    ├── models\\
    │   ├── streamsense_model_fp32.onnx
    │   └── streamsense_model_int8.onnx
    ├── preprocessing\\
    │   └── mel_pipeline.py
    ├── config\\
    │   ├── class_labels.json
    │   ├── normalization_stats.json
    │   └── mpic_v1.0.json              (machine-readable MPIC summary)
    ├── golden_vectors\\
    │   ├── raw\\, mel\\, normalized\\, manifest.json
    ├── protocol\\
    │   ├── nsp_protocol.py
    │   ├── nsp_sender.py
    │   ├── nsp_receiver.py
    │   └── nsp_node.py
    ├── evaluation\\
    │   ├── evaluation_report.txt
    │   └── confusion_matrix.png
    ├── README.md
    └── model_card.md                   (placeholder — filled separately)

Run from: C:\\STREAMSENSE\\training\\
Usage:
    python assemble_deployment_package.py
"""

import json
import shutil
import sys
from pathlib import Path

import torch
import torchaudio

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT      = Path(r'C:\STREAMSENSE')
DEPLOY    = ROOT / 'deployment_package'

SRC = {
    'fp32_onnx'   : ROOT / 'onnx_models' / 'streamsense_model_fp32.onnx',
    'int8_onnx'   : ROOT / 'onnx_models' / 'streamsense_model_int8.onnx',
    'mel_pipeline': ROOT / 'training'    / 'mel_pipeline.py',
    'class_labels': ROOT / 'class_labels.json',
    'norm_stats'  : ROOT / 'stats'       / 'normalization_stats.json',
    'gv_manifest' : ROOT / 'golden_vectors' / 'manifest.json',
    'gv_raw'      : ROOT / 'golden_vectors' / 'raw',
    'gv_mel'      : ROOT / 'golden_vectors' / 'mel',
    'gv_norm'     : ROOT / 'golden_vectors' / 'normalized',
    'eval_report' : ROOT / 'evaluation'  / 'evaluation_report.txt',
    'confusion'   : ROOT / 'evaluation'  / 'confusion_matrix.png',
    'nsp_protocol': ROOT / 'training'    / 'nsp_protocol.py',
    'nsp_sender'  : ROOT / 'training'    / 'nsp_sender.py',
    'nsp_receiver': ROOT / 'training'    / 'nsp_receiver.py',
    'nsp_node'    : ROOT / 'training'    / 'nsp_node.py',
}

DST = {
    'models'      : DEPLOY / 'models',
    'preprocessing': DEPLOY / 'preprocessing',
    'config'      : DEPLOY / 'config',
    'golden'      : DEPLOY / 'golden_vectors',
    'protocol'    : DEPLOY / 'protocol',
    'evaluation'  : DEPLOY / 'evaluation',
}

# MPIC v1.0 frozen parameters (mirrors mel_pipeline.py constants)
MPIC_V1_0 = {
    "version": "1.0",
    "sample_rate": 16000,
    "frame_len": 16000,
    "n_fft": 512,
    "hop_length": 160,
    "n_mels": 64,
    "window": "hann_periodic",
    "center": False,
    "power": 2.0,
    "log_scale": "10*log10(mel + 1e-10)",
    "clip_floor_db": -80.0,
    "global_mean": -30.785545,
    "global_std": 22.157099,
    "mel_shape": [64, 97],
    "input_tensor_shape": [1, 1, 64, 97],
    "input_tensor_name": "input",
    "output_tensor_name": "logits",
    "output_shape": [1, 10],
    "output_dtype": "float32",
    "output_scale_zero_point": "N/A (QDQ INT8 model has float32 I/O, identical to FP32)",
    "model_format": "ONNX (opset per export_onnx.ipynb); QDQ INT8 via quantize_ptq.ipynb",
    "tolerance": {
        "same_implementation": 1e-4,
        "cross_implementation": 5e-4,
        "manifest_value": 0.0005,
    },
}


def check_sources():
    """Verify every required source file/dir exists before copying anything."""
    print("=== Source check ===")
    missing = []
    for name, path in SRC.items():
        exists = path.exists()
        status = "OK" if exists else "MISSING"
        print(f"  [{status}] {name:<14} {path}")
        if not exists:
            missing.append(name)
    return missing


def make_dirs():
    for d in DST.values():
        d.mkdir(parents=True, exist_ok=True)


def copy_models():
    print("\n=== Copying models ===")
    for key in ('fp32_onnx', 'int8_onnx'):
        dst = DST['models'] / SRC[key].name
        shutil.copy2(SRC[key], dst)
        size_mb = dst.stat().st_size / 1e6
        print(f"  {SRC[key].name}  ({size_mb:.2f} MB) -> {dst}")


def copy_preprocessing():
    print("\n=== Copying preprocessing ===")
    dst = DST['preprocessing'] / SRC['mel_pipeline'].name
    shutil.copy2(SRC['mel_pipeline'], dst)
    print(f"  mel_pipeline.py -> {dst}")


def copy_config():
    print("\n=== Copying config ===")
    for key, fname in (('class_labels', 'class_labels.json'),
                        ('norm_stats', 'normalization_stats.json')):
        dst = DST['config'] / fname
        shutil.copy2(SRC[key], dst)
        print(f"  {fname} -> {dst}")

    mpic_path = DST['config'] / 'mpic_v1.0.json'
    with open(mpic_path, 'w', encoding='utf-8') as f:
        json.dump(MPIC_V1_0, f, indent=2)
    print(f"  mpic_v1.0.json (generated) -> {mpic_path}")


def copy_golden_vectors():
    print("\n=== Copying golden vectors ===")
    for key, sub in (('gv_raw', 'raw'), ('gv_mel', 'mel'), ('gv_norm', 'normalized')):
        src_dir = SRC[key]
        dst_dir = DST['golden'] / sub
        if dst_dir.exists():
            shutil.rmtree(dst_dir)
        shutil.copytree(src_dir, dst_dir)
        n_files = len(list(dst_dir.glob('*.bin')))
        print(f"  {sub}/  ({n_files} files) -> {dst_dir}")

    dst_manifest = DST['golden'] / 'manifest.json'
    shutil.copy2(SRC['gv_manifest'], dst_manifest)
    print(f"  manifest.json -> {dst_manifest}")


def copy_protocol():
    print("\n=== Copying NSP protocol scripts ===")
    for key in ('nsp_protocol', 'nsp_sender', 'nsp_receiver', 'nsp_node'):
        dst = DST['protocol'] / SRC[key].name
        shutil.copy2(SRC[key], dst)
        print(f"  {SRC[key].name} -> {dst}")


def copy_evaluation():
    print("\n=== Copying evaluation artifacts ===")
    for key in ('eval_report', 'confusion'):
        src = SRC[key]
        if not src.exists():
            print(f"  [SKIP] {key} not found: {src}")
            continue
        dst = DST['evaluation'] / src.name
        shutil.copy2(src, dst)
        print(f"  {src.name} -> {dst}")


def write_readme():
    print("\n=== Writing README.md ===")
    readme = f"""# STREAMSENSE Deployment Package

Self-contained deliverable for Track B (Kavish) and the Phase 4 FPGA team.
Generated by `assemble_deployment_package.py`.

## Contents

| Folder            | Contents                                                          |
|-------------------|--------------------------------------------------------------------|
| `models/`         | `streamsense_model_fp32.onnx`, `streamsense_model_int8.onnx`      |
| `preprocessing/`  | `mel_pipeline.py` (MPIC v1.0 reference impl)                       |
| `config/`         | `class_labels.json`, `normalization_stats.json`, `mpic_v1.0.json` |
| `golden_vectors/` | `raw/`, `mel/`, `normalized/`, `manifest.json` — 10/10 PASS       |
| `protocol/`       | NSP v1.2 sender/receiver/dual-node scripts                        |
| `evaluation/`     | `evaluation_report.txt`, `confusion_matrix.png`                  |

## Model I/O contract (both FP32 and INT8)

- Input  tensor: `"input"`,  shape `[1, 1, 64, 97]`, dtype `float32`
- Output tensor: `"logits"`, shape `[1, 10]`, dtype `float32`
- Class order: see `config/class_labels.json` (indices 0-9 = yes, no, up, down, left, right, on, off, stop, go)
- INT8 model is QDQ-quantized internally — I/O tensors are float32, identical
  shapes/names to FP32. No output scale/zero-point handling required.

## Preprocessing contract (MPIC v1.0)

See `config/mpic_v1.0.json` for the full machine-readable spec, or
`preprocessing/mel_pipeline.py` for the reference implementation
(`preprocess(samples) -> torch.Tensor [1,1,64,97]`).

## Validation status

- Golden vectors: 10/10 PASS (FP32 and INT8), tolerance = 0.0005 (manifest)
- Test accuracy: 95.97% (FP32) — see `evaluation/evaluation_report.txt`
- INT8 vs FP32: 0% top-1 accuracy drop on 200 val samples

## NSP v1.2 (network streaming)

`protocol/` contains the sender/receiver/dual-role scripts implementing
NSP v1.2 framing (4-byte LE length prefix + 48-byte header + 64000-byte
float32 payload). See inline docstrings in each script for usage.

## Versioning

- MPIC: v1.0 (frozen)
- NSP: v1.2 (frozen)
- Model checkpoint: epoch 26, val_acc=96.11%, test_acc=95.97%
"""
    readme_path = DEPLOY / 'README.md'
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(readme)
    print(f"  README.md -> {readme_path}")


def write_manifest():
    print("\n=== Writing deployment_package/manifest.json ===")
    manifest = {
        "package_name": "streamsense_deployment_package",
        "mpic_version": "1.0",
        "nsp_version": "1.2",
        "model_checkpoint": {
            "epoch": 26,
            "val_acc": 96.11,
            "test_acc": 95.97,
        },
        "contents": {
            "models": ["streamsense_model_fp32.onnx", "streamsense_model_int8.onnx"],
            "preprocessing": ["mel_pipeline.py"],
            "config": ["class_labels.json", "normalization_stats.json", "mpic_v1.0.json"],
            "golden_vectors": ["raw/", "mel/", "normalized/", "manifest.json"],
            "protocol": ["nsp_protocol.py", "nsp_sender.py", "nsp_receiver.py", "nsp_node.py"],
            "evaluation": ["evaluation_report.txt", "confusion_matrix.png"],
        },
    }
    path = DEPLOY / 'manifest.json'
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2)
    print(f"  manifest.json -> {path}")


def main():
    print("=" * 60)
    print("STREAMSENSE — Deployment Package Assembly")
    print("=" * 60)

    missing = check_sources()
    if missing:
        print(f"\n[ABORT] {len(missing)} required source(s) missing: {missing}")
        print("Fix paths/files above before assembling the package.")
        sys.exit(1)

    make_dirs()
    copy_models()
    copy_preprocessing()
    copy_config()
    copy_golden_vectors()
    copy_protocol()
    copy_evaluation()
    write_readme()
    write_manifest()

    print("\n" + "=" * 60)
    print(f"[DONE] Deployment package assembled at: {DEPLOY}")
    print("=" * 60)


if __name__ == "__main__":
    main()
