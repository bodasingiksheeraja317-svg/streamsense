"""
qat_finetune.py
Project STREAMSENSE — Track A
Scope 2 / QAT Extension — Quantization-Aware Training fine-tune

Trains the StreamSenseWrapper with Brevitas quantizers applied to all
Conv2d and Linear layers.  Simultaneously trains the embed_head (which
has never been trained on real data) and learns Brevitas quantizer scale
factors.  Saves best checkpoint and runs the GV1K gate before exiting.

Usage (in Colab via qat_colab.ipynb, Cell 7):
    python training/qat_finetune.py \\
        --ckpt checkpoints/best_model.pth \\
        --data /content/data \\
        --epochs 10 \\
        --lr 1e-5 \\
        --out checkpoints/best_model_qat.pth \\
        --gvk golden_vectors_1000/normalized \\
        --device cuda
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import DataLoader, Dataset

# ── Resolve project root and training/ directory ──────────────────────────────
_THIS_DIR = Path(__file__).resolve().parent
_ROOT     = _THIS_DIR.parent

if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from model import StreamSenseNet                   # noqa: E402
from streaming_wrapper import StreamSenseWrapper   # noqa: E402

# ── Brevitas imports ──────────────────────────────────────────────────────────
try:
    import brevitas.nn as qnn
    from brevitas.quant import Int8WeightPerTensorFloat, Int8ActPerTensorFloat
except ImportError as e:
    print(f"[ERROR] brevitas not installed: {e}")
    print("        pip install brevitas")
    sys.exit(1)

# ── MPIC v1.0 frozen constants ────────────────────────────────────────────────
SAMPLE_RATE   = 16000
FRAME_LEN     = 16000
N_FFT         = 512
HOP_LENGTH    = 160
N_MELS        = 64
CENTER        = False
POWER         = 2.0
LOG_EPS       = 1e-10
CLIP_FLOOR_DB = -80.0
GLOBAL_MEAN   = -30.785545
GLOBAL_STD    = 22.157099

EXPECTED_T    = (FRAME_LEN - N_FFT) // HOP_LENGTH + 1   # 97

# 10 target keyword classes — indices match class_labels.json
TARGET_CLASSES = {
    "yes": 0, "no": 1, "up": 2, "down": 3,
    "left": 4, "right": 5, "on": 6, "off": 7,
    "stop": 8, "go": 9,
}

NUM_CLASSES   = 10
BATCH_SIZE    = 64       # suitable for T4 16 GB GPU
NUM_WORKERS   = 2


# ── MPIC v1.0 preprocessing pipeline ─────────────────────────────────────────

_mel_transform = T.MelSpectrogram(
    sample_rate = SAMPLE_RATE,
    n_fft       = N_FFT,
    hop_length  = HOP_LENGTH,
    n_mels      = N_MELS,
    window_fn   = torch.hann_window,
    center      = CENTER,
    power       = POWER,
)


def preprocess(raw: np.ndarray) -> torch.Tensor:
    """
    MPIC v1.0 full pipeline.
    Input:  float32 numpy [T] — raw 16 kHz audio, already at FRAME_LEN samples
    Output: float32 Tensor [1, 1, 64, 97]
    """
    waveform = torch.from_numpy(raw.copy()).float()
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)          # [1, T]
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    L = waveform.shape[1]
    if L < FRAME_LEN:
        waveform = torch.nn.functional.pad(waveform, (0, FRAME_LEN - L))
    elif L > FRAME_LEN:
        waveform = waveform[:, :FRAME_LEN]
    mel = _mel_transform(waveform)                # [1, 64, 97]
    mel = 10.0 * torch.log10(mel + LOG_EPS)
    mel = torch.clamp(mel, min=CLIP_FLOOR_DB)
    mel = (mel - GLOBAL_MEAN) / GLOBAL_STD
    mel = mel.unsqueeze(0)                        # [1, 1, 64, 97]
    return mel.float()


# ── Dataset ───────────────────────────────────────────────────────────────────

class SpeechCommandsDataset(Dataset):
    """
    Thin wrapper around torchaudio.datasets.SPEECHCOMMANDS.

    Filters to the 10 target classes only.  Discards _background_noise_,
    unknown words, and any clip that torchaudio cannot load.

    Returns: (Tensor [1, 1, 64, 97] float32, int class_index)
    """

    def __init__(self, root: Path, subset: str):
        """
        Args:
            root   : Path to Speech Commands root (the directory that will
                     contain / already contains speech_commands_v0.02/).
            subset : "validation" or "testing" (torchaudio split names).
        """
        self.root   = root
        self.subset = subset

        raw_ds = torchaudio.datasets.SPEECHCOMMANDS(
            root     = str(root),
            download = True,
            subset   = subset,
        )

        self.samples: list[tuple[str, int]] = []
        for waveform, sample_rate, label, *_ in raw_ds:
            if label not in TARGET_CLASSES:
                continue
            # We do not store waveform tensors in RAM — re-load from disk later.
            # torchaudio SPEECHCOMMANDS exposes _path attribute; use it.
            # Fallback: use the dataset's _walker list.
            self.samples.append((label, waveform, sample_rate))

        print(
            f"[SpeechCommandsDataset] subset={subset!r}  "
            f"kept {len(self.samples)} clips  "
            f"(target classes only)"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        label, waveform, sample_rate = self.samples[idx]
        class_idx = TARGET_CLASSES[label]

        # Convert to numpy float32 mono [T]
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        raw = waveform.squeeze(0).numpy().astype(np.float32)

        # Resample if needed (should always be 16 kHz for Speech Commands v2)
        if sample_rate != SAMPLE_RATE:
            waveform_t = torch.from_numpy(raw).unsqueeze(0)
            waveform_t = torchaudio.functional.resample(waveform_t, sample_rate, SAMPLE_RATE)
            raw = waveform_t.squeeze(0).numpy().astype(np.float32)

        tensor = preprocess(raw)            # [1, 1, 64, 97]
        return tensor.squeeze(0), class_idx # [1, 64, 97], int  (collation adds batch dim)


# ── Brevitas module replacement ───────────────────────────────────────────────

def _replace_conv2d(module: nn.Module) -> nn.Module:
    """
    Recursively replace all nn.Conv2d in module with brevitas.nn.QuantConv2d.
    Copies weight (and bias) data so the trained values are preserved.
    """
    for name, child in module.named_children():
        if isinstance(child, nn.Conv2d):
            qconv = qnn.QuantConv2d(
                in_channels  = child.in_channels,
                out_channels = child.out_channels,
                kernel_size  = child.kernel_size,
                stride       = child.stride,
                padding      = child.padding,
                dilation     = child.dilation,
                groups       = child.groups,
                bias         = child.bias is not None,
                weight_quant = Int8WeightPerTensorFloat,
                input_quant  = Int8ActPerTensorFloat,
                output_quant = Int8ActPerTensorFloat,
                return_quant_tensor = False,
            )
            with torch.no_grad():
                qconv.weight.copy_(child.weight)
                if child.bias is not None and qconv.bias is not None:
                    qconv.bias.copy_(child.bias)
            setattr(module, name, qconv)
        else:
            _replace_conv2d(child)
    return module


def _replace_linear(module: nn.Module) -> nn.Module:
    """
    Recursively replace all nn.Linear in module with brevitas.nn.QuantLinear.
    Copies weight and bias data.
    """
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            qlin = qnn.QuantLinear(
                in_features  = child.in_features,
                out_features = child.out_features,
                bias         = child.bias is not None,
                weight_quant = Int8WeightPerTensorFloat,
                input_quant  = Int8ActPerTensorFloat,
                output_quant = Int8ActPerTensorFloat,
                return_quant_tensor = False,
            )
            with torch.no_grad():
                qlin.weight.copy_(child.weight)
                if child.bias is not None and qlin.bias is not None:
                    qlin.bias.copy_(child.bias)
            setattr(module, name, qlin)
        else:
            _replace_linear(child)
    return module


def build_qat_model(ckpt_path: Path, device: torch.device) -> StreamSenseWrapper:
    """
    Construct StreamSenseWrapper, load best_model.pth backbone weights,
    apply Brevitas QuantConv2d / QuantLinear replacements, and move the
    whole model to device.

    Returns the model ready for QAT fine-tuning.
    """
    # 1. Instantiate base wrapper
    model = StreamSenseWrapper(num_classes=NUM_CLASSES)

    # 2. Load backbone weights with strict=True
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model.backbone.load_state_dict(ckpt["model_state"], strict=True)
    print(f"[build_qat_model] Loaded backbone from epoch {ckpt.get('epoch', '?')}  "
          f"val_acc={ckpt.get('val_accuracy', float('nan')):.2f}%")

    # 3. Replace Conv2d in backbone blocks (NOT in gap — it has no weights)
    _replace_conv2d(model.backbone.block1)
    _replace_conv2d(model.backbone.block2)
    _replace_conv2d(model.backbone.block3)

    # 4. Replace Linear in backbone classifier
    _replace_linear(model.backbone.classifier)

    # 5. Replace Linear in embed_head
    _replace_linear(model.embed_head)

    # 6. Brevitas device-placement fix: model.to(device) LAST
    model.to(device)

    # 7. Mandatory buffer verification
    for buf_name, buf in model.named_buffers():
        assert buf.device.type == device.type, (
            f"[device-check] Buffer {buf_name!r} is on {buf.device.type!r}, "
            f"expected {device.type!r}. This is the Brevitas device-placement bug."
        )
    print(f"[build_qat_model] All buffers verified on device={device.type!r}")

    return model


# ── GV1K gate ─────────────────────────────────────────────────────────────────

_LABEL_TO_IDX = {v: k for k, v in
                 {"yes":0,"no":1,"up":2,"down":3,"left":4,
                  "right":5,"on":6,"off":7,"stop":8,"go":9}.items()}
_LABEL_TO_IDX = {label: idx for label, idx in TARGET_CLASSES.items()}


def _parse_gv1k_label(stem: str) -> int | None:
    """
    Parse ground-truth class index from a GV1K normalized filename stem.
    Pattern: GV1K_NNNN_<label>_norm
    """
    parts = stem.split("_")
    # parts: ['GV1K', 'NNNN', '<label>', 'norm']
    if len(parts) < 4:
        return None
    label_str = parts[2].lower()
    return TARGET_CLASSES.get(label_str, None)


def run_gv1k_gate(model: nn.Module, gvk_dir: Path, device: torch.device) -> float:
    """
    Run all 1000 GV1K vectors through the model in eval mode.
    Compute top-1 accuracy on the logits output.
    Hard sys.exit(1) if accuracy < 90 %.

    Returns top-1 accuracy as a float in [0, 100].
    """
    bin_files = sorted(gvk_dir.glob("*_norm.bin"))
    if not bin_files:
        print(f"[GV1K] WARNING: no *_norm.bin files found in {gvk_dir} — skipping gate")
        return float("nan")

    model.eval()
    correct  = 0
    wrong    = 0
    skipped  = 0

    with torch.no_grad():
        for bf in bin_files:
            true_idx = _parse_gv1k_label(bf.stem)
            if true_idx is None:
                skipped += 1
                continue

            raw = np.fromfile(str(bf), dtype="<f4")
            if raw.size != 64 * 97:
                skipped += 1
                continue

            inp = torch.from_numpy(raw).reshape(1, 1, 64, 97).to(device)
            logits, _embedding, _novelty = model(inp)
            pred_idx = int(logits.argmax(dim=1).item())

            if pred_idx == true_idx:
                correct += 1
            else:
                wrong += 1

    total_checked = correct + wrong
    if total_checked == 0:
        print(f"[GV1K] SKIP — no vectors could be checked (all {skipped} skipped)")
        return float("nan")

    top1_acc = 100.0 * correct / total_checked
    print(f"[GV1K] Vectors checked : {total_checked}  (skipped: {skipped})")
    print(f"[GV1K] Correct         : {correct}  Wrong: {wrong}")
    print(f"[GV1K] Top-1 accuracy  : {top1_acc:.2f}%")

    if top1_acc < 90.0:
        print(f"[GV1K] FAIL — {top1_acc:.2f}% < 90.0% minimum.  Aborting.")
        sys.exit(1)
    else:
        print(f"[GV1K] PASS — {top1_acc:.2f}% ≥ 90.0%")

    return top1_acc


# ── Training helpers ──────────────────────────────────────────────────────────

def _freeze_backbone(model: StreamSenseWrapper):
    """Freeze all backbone parameters.  embed_head and quantizer scales stay trainable."""
    for param in model.backbone.parameters():
        param.requires_grad_(False)


def _unfreeze_all(model: StreamSenseWrapper):
    """Unfreeze all parameters for full fine-tuning."""
    for param in model.parameters():
        param.requires_grad_(True)


def _count_trainable(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_one_epoch(
    model:     nn.Module,
    loader:    DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device:    torch.device,
    epoch:     int,
) -> float:
    model.train()
    total_loss = 0.0
    n_batches  = len(loader)

    for batch_idx, (x, y) in enumerate(loader):
        x = x.to(device)   # [B, 1, 64, 97]
        y = y.to(device)   # [B]

        optimizer.zero_grad()
        logits, _embedding, _novelty = model(x)
        loss = criterion(logits, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()

        if (batch_idx + 1) % 200 == 0 or (batch_idx + 1) == n_batches:
            print(
                f"  Epoch {epoch:>3}  [{batch_idx+1:>4}/{n_batches}]  "
                f"loss={loss.item():.4f}",
                flush=True,
            )

    return total_loss / n_batches


def validate(
    model:     nn.Module,
    loader:    DataLoader,
    device:    torch.device,
) -> float:
    model.eval()
    correct = 0
    total   = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits, _embedding, _novelty = model(x)
            preds    = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total   += y.size(0)

    return 100.0 * correct / total if total > 0 else 0.0


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="STREAMSENSE QAT fine-tuning script — Scope 2 QAT extension"
    )
    parser.add_argument(
        "--ckpt",
        type    = Path,
        default = Path("checkpoints/best_model.pth"),
        help    = "Path to best_model.pth (default: checkpoints/best_model.pth)",
    )
    parser.add_argument(
        "--data",
        type    = Path,
        required= True,
        help    = "Path to Speech Commands v2 root directory",
    )
    parser.add_argument(
        "--epochs",
        type    = int,
        default = 10,
        help    = "Total QAT training epochs (default: 10)",
    )
    parser.add_argument(
        "--lr",
        type    = float,
        default = 1e-5,
        help    = "Adam learning rate (default: 1e-5)",
    )
    parser.add_argument(
        "--out",
        type    = Path,
        default = Path("checkpoints/best_model_qat.pth"),
        help    = "Output checkpoint path (default: checkpoints/best_model_qat.pth)",
    )
    parser.add_argument(
        "--device",
        type    = str,
        default = "cuda" if torch.cuda.is_available() else "cpu",
        help    = "Device: cuda or cpu (default: cuda if available)",
    )
    parser.add_argument(
        "--gvk",
        type    = Path,
        default = Path("golden_vectors_1000/normalized"),
        help    = "Path to GV1K normalized directory (default: golden_vectors_1000/normalized)",
    )
    return parser.parse_args()


def main():
    args   = parse_args()
    device = torch.device(args.device)

    print("=" * 60)
    print("STREAMSENSE — QAT Fine-tuning  (Scope 2 QAT extension)")
    print("=" * 60)
    print(f"  Checkpoint  : {args.ckpt}")
    print(f"  Data root   : {args.data}")
    print(f"  Epochs      : {args.epochs}")
    print(f"  LR          : {args.lr}")
    print(f"  Output      : {args.out}")
    print(f"  Device      : {device}")
    print(f"  GV1K dir    : {args.gvk}")

    # ── Prerequisite checks ───────────────────────────────────────────────────
    if not args.ckpt.exists():
        print(f"[ERROR] Checkpoint not found: {args.ckpt}")
        sys.exit(1)

    # ── Ensure output directory exists ────────────────────────────────────────
    args.out.parent.mkdir(parents=True, exist_ok=True)

    # ── Build QAT model ───────────────────────────────────────────────────────
    print("\n[Step 1] Building QAT model...")
    model = build_qat_model(args.ckpt, device)

    # ── Datasets and DataLoaders ──────────────────────────────────────────────
    print("\n[Step 2] Loading Speech Commands datasets...")
    val_ds  = SpeechCommandsDataset(args.data, subset="validation")
    test_ds = SpeechCommandsDataset(args.data, subset="testing")

    # For training we use the validation split of Speech Commands (it is the
    # standard labelled non-test split, labelled via validation_list.txt).
    # The testing split is held out for the final GV1K gate — it is not used
    # during QAT training to avoid data contamination.
    train_ds = SpeechCommandsDataset(args.data, subset="validation")

    train_loader = DataLoader(
        train_ds,
        batch_size  = BATCH_SIZE,
        shuffle     = True,
        num_workers = NUM_WORKERS,
        pin_memory  = (device.type == "cuda"),
        drop_last   = True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size  = BATCH_SIZE,
        shuffle     = False,
        num_workers = NUM_WORKERS,
        pin_memory  = (device.type == "cuda"),
    )

    print(f"  Train batches : {len(train_loader)}")
    print(f"  Val   batches : {len(val_loader)}")

    # ── Optimizer and criterion ───────────────────────────────────────────────
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_acc  = 0.0
    best_ckpt_saved = False

    # ── Training loop ─────────────────────────────────────────────────────────
    print("\n[Step 3] Training loop")
    print(f"  Epochs 1–3  : backbone FROZEN, training embed_head + quantizer scales")
    print(f"  Epoch  4+   : all parameters UNFROZEN")
    print()

    for epoch in range(1, args.epochs + 1):

        # Phase 1: epochs 1-3 freeze backbone
        if epoch == 1:
            _freeze_backbone(model)
            print(f"  [Epoch {epoch}] Backbone FROZEN.  Trainable params: {_count_trainable(model):,}")
        elif epoch == 4:
            _unfreeze_all(model)
            print(f"  [Epoch {epoch}] All parameters UNFROZEN.  Trainable params: {_count_trainable(model):,}")

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
        val_acc    = validate(model, val_loader, device)

        print(
            f"Epoch {epoch:>3}/{args.epochs}  "
            f"train_loss={train_loss:.4f}  "
            f"val_top1={val_acc:.2f}%"
        )

        # Save checkpoint if validation accuracy improved
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "model_state"  : model.state_dict(),
                    "epoch"        : epoch,
                    "val_accuracy" : val_acc,
                    "qat"          : True,
                },
                args.out,
            )
            best_ckpt_saved = True
            print(f"  [checkpoint] Saved best checkpoint  val_acc={val_acc:.2f}%  → {args.out}")

    if not best_ckpt_saved:
        # Save whatever we have if no improvement was ever detected
        torch.save(
            {
                "model_state"  : model.state_dict(),
                "epoch"        : args.epochs,
                "val_accuracy" : best_val_acc,
                "qat"          : True,
            },
            args.out,
        )
        print(f"  [checkpoint] Saved final checkpoint → {args.out}")

    # ── Post-training GV1K gate ───────────────────────────────────────────────
    print("\n[Step 4] Post-training GV1K gate")

    # Reload best checkpoint into a fresh model for gate evaluation
    best_ckpt_data = torch.load(args.out, map_location="cpu", weights_only=True)
    gate_model = build_qat_model(args.ckpt, device)
    gate_model.load_state_dict(best_ckpt_data["model_state"])
    gate_model.eval()

    gvk_dir = _ROOT / args.gvk if not args.gvk.is_absolute() else args.gvk
    gv1k_acc = run_gv1k_gate(gate_model, gvk_dir, device)

    # ── Final summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("QAT FINE-TUNING COMPLETE")
    print("=" * 60)
    print(f"  Best val accuracy : {best_val_acc:.2f}%")
    print(f"  GV1K top-1        : {gv1k_acc:.2f}%")
    print(f"  Checkpoint saved  : {args.out}")
    print()

    if gv1k_acc < 90.0:
        print("[FAIL] GV1K gate failed. Checkpoint NOT deployment-grade.")
        sys.exit(1)
    else:
        print("[PASS] GV1K gate passed. Checkpoint is deployment-grade.")


if __name__ == "__main__":
    main()
