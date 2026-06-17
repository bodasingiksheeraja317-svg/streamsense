"""
train_qat_brevitas.py — A5.2 STRETCH
Quantization-Aware Training (QAT) for StreamSenseNet using Brevitas.
Targets Zynq-7000 FPGA via FINN/QONNX export.

Produces two checkpoints:
  checkpoints_qat/qat_w8a8_best.pth   — W8A8 (INT8 weights + INT8 activations)
  checkpoints_qat/qat_w4a4_best.pth   — W4A4 (INT4 weights + INT4 activations)

Usage:
  # From C:\STREAMSENSE, with streamsense-env-win activated:
  python training/train_qat_brevitas.py --bits 8   # W8A8
  python training/train_qat_brevitas.py --bits 4   # W4A4
  python training/train_qat_brevitas.py --bits all # both (default)

Install (run once):
  pip install brevitas==0.10.2
  pip install qonnx         # for export_qonnx.py
"""

import argparse
import os
import sys
import csv
import random
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

# ---------------------------------------------------------------------------
# Brevitas imports
# ---------------------------------------------------------------------------
try:
    import brevitas.nn as qnn
    from brevitas.quant import (
        Int8WeightPerTensorFloat,
        Int8ActPerTensorFloat,
        Int4WeightPerTensorFloat,
    )
except ImportError:
    sys.exit(
        "\n[ERROR] Brevitas not found.\n"
        "Install with:  pip install brevitas==0.10.2\n"
    )

# ---------------------------------------------------------------------------
# Path setup — works whether run from C:\STREAMSENSE or training/
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR) if os.path.basename(SCRIPT_DIR) == "training" else SCRIPT_DIR
sys.path.insert(0, os.path.join(ROOT, "training"))

from dataset import MelSpectrogramDataset

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED = 42

def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Helper: safely unwrap a Brevitas QuantTensor to a plain torch.Tensor.
# QuantReLU with return_quant_tensor=True returns a QuantTensor namedtuple.
# Standard PyTorch ops (BN, MaxPool, Dropout, GAP) do not accept QuantTensor,
# so we must unwrap before passing to them.
# ---------------------------------------------------------------------------
def _unwrap(x):
    """Return x.value if x is a QuantTensor, else x unchanged."""
    return x.value if hasattr(x, "value") else x


# ---------------------------------------------------------------------------
# Quantized model definition
# ---------------------------------------------------------------------------

class QuantConvBlock(nn.Module):
    """
    Double-Conv block with Brevitas quantized Conv2d + QuantReLU layers.
    Mirrors the float StreamSenseNet block structure exactly.

    weight_bit : bit-width for weight quantization  (4 or 8)
    act_bit    : bit-width for activation quantization (4 or 8)

    Data flow per sub-block:
      QuantConv2d → unwrap → BN → QuantReLU(return_quant_tensor=True)
                                        ↓
                             next QuantConv2d accepts QuantTensor directly
    The second QuantReLU output is unwrapped before MaxPool/Dropout.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        weight_bit: int = 8,
        act_bit: int = 8,
        dropout: float = 0.25,
    ):
        super().__init__()

        if weight_bit == 8:
            WeightQuant = Int8WeightPerTensorFloat
        elif weight_bit == 4:
            WeightQuant = Int4WeightPerTensorFloat
        else:
            raise ValueError(f"Unsupported weight bit-width: {weight_bit}")

        # QuantConv2d: accepts plain tensor or QuantTensor input.
        # output_quant is left at default (None) — quantization of activations
        # is handled explicitly by the following QuantReLU, which gives FINN
        # a cleaner graph (ReLU + quant fused into one node).
        self.conv1 = qnn.QuantConv2d(
            in_ch, out_ch, kernel_size=3, padding=1, bias=False,
            weight_bit_width=weight_bit,
            weight_quant=WeightQuant,
        )
        self.bn1  = nn.BatchNorm2d(out_ch)
        # return_quant_tensor=True: passes QuantTensor to next QuantConv2d so
        # Brevitas can propagate scale/zero-point through the graph correctly.
        self.act1 = qnn.QuantReLU(bit_width=act_bit, return_quant_tensor=True)

        self.conv2 = qnn.QuantConv2d(
            out_ch, out_ch, kernel_size=3, padding=1, bias=False,
            weight_bit_width=weight_bit,
            weight_quant=WeightQuant,
        )
        self.bn2  = nn.BatchNorm2d(out_ch)
        # Last QuantReLU in the block: return_quant_tensor=False so the output
        # is a plain tensor — MaxPool2d and Dropout2d do not accept QuantTensor.
        self.act2 = qnn.QuantReLU(bit_width=act_bit, return_quant_tensor=False)

        self.pool    = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout2d(dropout)

    def forward(self, x):
        # Conv1 branch
        x = self.conv1(x)       # QuantTensor or plain tensor in → QuantTensor out
        x = self.bn1(_unwrap(x))  # BN needs plain tensor
        x = self.act1(x)          # plain tensor in → QuantTensor out

        # Conv2 branch — QuantConv2d accepts QuantTensor directly
        x = self.conv2(x)
        x = self.bn2(_unwrap(x))
        x = self.act2(x)          # plain tensor out (return_quant_tensor=False)

        x = self.pool(x)
        x = self.dropout(x)
        return x                  # plain float tensor


class StreamSenseNetQAT(nn.Module):
    """
    Quantization-Aware Training version of StreamSenseNet.
    All Conv2d blocks replaced with QuantConv2d + QuantReLU (Brevitas).
    Classifier head kept in float32 (lives on ARM PS side in FINN flow).

    Input:  [B, 1, 64, 97]  float32 normalized mel spectrogram
    Output: [B, 10]         float32 raw logits
    """

    def __init__(self, num_classes: int = 10, weight_bit: int = 8, act_bit: int = 8):
        super().__init__()
        self.weight_bit = weight_bit
        self.act_bit    = act_bit

        # Input quantization node — quantizes incoming float32 tensor.
        # return_quant_tensor=False: block1's QuantConv2d accepts plain tensor.
        self.input_quant = qnn.QuantIdentity(
            bit_width=act_bit, return_quant_tensor=False
        )

        self.block1 = QuantConvBlock(1,   32, weight_bit, act_bit, dropout=0.25)
        self.block2 = QuantConvBlock(32,  64, weight_bit, act_bit, dropout=0.25)
        self.block3 = QuantConvBlock(64, 128, weight_bit, act_bit, dropout=0.25)

        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, qnn.QuantConv2d):
                # In Brevitas 0.10.2, QuantConv2d.weight is a plain nn.Parameter
                # (the quantization wrapper is applied at forward time, not on
                # the stored tensor). Kaiming init is safe to apply directly.
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.input_quant(x)   # float → quantized float (fake quant)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.gap(x)
        x = x.flatten(1)
        return self.classifier(x)


# ---------------------------------------------------------------------------
# Weight transfer from float checkpoint
# ---------------------------------------------------------------------------

def load_float_weights(qat_model: StreamSenseNetQAT, ckpt_path: str):
    """
    Transfer weights from the frozen float StreamSenseNet checkpoint into the
    QAT model. Conv weights and BN params are copied by matching key names.
    Brevitas-specific quant params (scale, zero_point) are left at their
    initialized values and learned during fine-tuning.
    """
    from model import StreamSenseNet

    float_model = StreamSenseNet(num_classes=10)
    ckpt  = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("model_state_dict", ckpt)
    float_model.load_state_dict(state)
    float_model.eval()

    float_sd = float_model.state_dict()
    qat_sd   = qat_model.state_dict()

    transferred = 0
    skipped     = 0
    for k, v in float_sd.items():
        if k in qat_sd and qat_sd[k].shape == v.shape:
            qat_sd[k] = v.clone()
            transferred += 1
        else:
            skipped += 1

    qat_model.load_state_dict(qat_sd, strict=False)
    print(f"[Weight transfer] Transferred: {transferred}  Skipped/quant-only: {skipped}")
    return qat_model


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_qat(
    bits: int,
    root: str,
    epochs: int = 15,
    batch_size: int = 32,
    lr: float = 1e-4,
    patience: int = 6,
    device: str = None,
):
    tag = f"w{bits}a{bits}"
    print(f"\n{'='*60}")
    print(f"  QAT Training — {tag.upper()}  ({bits}-bit weights + {bits}-bit activations)")
    print(f"{'='*60}")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device : {device}")
    print(f"  Epochs : {epochs}  |  LR : {lr}  |  Batch : {batch_size}")

    set_seed(SEED)

    splits_dir = os.path.join(root, "data", "splits")
    train_ds = MelSpectrogramDataset(os.path.join(splits_dir, "train_files.txt"), augment=True)
    val_ds   = MelSpectrogramDataset(os.path.join(splits_dir, "val_files.txt"),   augment=False)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=2, pin_memory=(device == "cuda"),
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=(device == "cuda"),
    )

    model = StreamSenseNetQAT(num_classes=10, weight_bit=bits, act_bit=bits)
    ckpt_path = os.path.join(root, "checkpoints", "best_model.pth")
    if not os.path.exists(ckpt_path):
        print(f"[WARN] Checkpoint not found at {ckpt_path} — starting from random init")
    else:
        model = load_float_weights(model, ckpt_path)
    model = model.to(device)

    optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-7, verbose=True
    )
    criterion = nn.CrossEntropyLoss()

    ckpt_dir  = os.path.join(root, "checkpoints_qat")
    os.makedirs(ckpt_dir, exist_ok=True)
    log_path  = os.path.join(ckpt_dir, f"qat_{tag}_log.csv")
    best_path = os.path.join(ckpt_dir, f"qat_{tag}_best.pth")

    best_val_acc = 0.0
    no_improve   = 0

    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "val_acc", "lr"])

        for epoch in range(1, epochs + 1):
            model.train()
            total_loss = 0.0
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                loss = criterion(model(xb), yb)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                total_loss += loss.item() * xb.size(0)
            train_loss = total_loss / len(train_ds)

            model.eval()
            val_loss = 0.0
            correct  = 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    logits = model(xb)
                    val_loss += criterion(logits, yb).item() * xb.size(0)
                    correct  += (logits.argmax(1) == yb).sum().item()
            val_loss /= len(val_ds)
            val_acc   = 100.0 * correct / len(val_ds)

            current_lr = optimizer.param_groups[0]["lr"]
            print(
                f"Epoch {epoch:02d}/{epochs}  "
                f"train_loss={train_loss:.4f}  "
                f"val_loss={val_loss:.4f}  "
                f"val_acc={val_acc:.2f}%  "
                f"lr={current_lr:.2e}"
            )
            writer.writerow([epoch, f"{train_loss:.4f}", f"{val_loss:.4f}", f"{val_acc:.4f}", f"{current_lr:.2e}"])

            scheduler.step(val_loss)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                no_improve   = 0
                torch.save(
                    {
                        "epoch":            epoch,
                        "val_acc":          val_acc,
                        "weight_bit":       bits,
                        "act_bit":          bits,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state":  optimizer.state_dict(),
                    },
                    best_path,
                )
                print(f"  ✔ New best saved → {best_path}")
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"  Early stop triggered (patience={patience}).")
                    break

    print(f"\n[{tag.upper()}] Best val accuracy: {best_val_acc:.2f}%")
    print(f"  Log        → {log_path}")
    print(f"  Checkpoint → {best_path}")
    return best_val_acc, best_path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="StreamSenseNet QAT with Brevitas")
    parser.add_argument("--bits",    choices=["4", "8", "all"], default="all")
    parser.add_argument("--epochs",  type=int,   default=15)
    parser.add_argument("--batch",   type=int,   default=32)
    parser.add_argument("--lr",      type=float, default=1e-4)
    parser.add_argument("--patience",type=int,   default=6)
    parser.add_argument(
        "--root", type=str,
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."),
    )
    args = parser.parse_args()

    root    = os.path.abspath(args.root)
    results = {}
    targets = {"all": [8, 4], "8": [8], "4": [4]}[args.bits]

    for b in targets:
        acc, path = train_qat(
            bits=b, root=root, epochs=args.epochs,
            batch_size=args.batch, lr=args.lr, patience=args.patience,
        )
        results[f"W{b}A{b}"] = {"val_acc": acc, "checkpoint": path}

    print("\n" + "="*60)
    print("  QAT SUMMARY")
    print("="*60)
    print(f"  Float baseline (FP32): 95.97%")
    for tag, r in results.items():
        drop = 95.97 - r["val_acc"]
        print(f"  {tag} QAT val acc:  {r['val_acc']:.2f}%  (drop: {drop:+.2f}%)")
    print("="*60)
    print("\nNext step: run  python training/export_qonnx.py  to export to QONNX.")


if __name__ == "__main__":
    main()
