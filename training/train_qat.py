# =============================================================================
# train_qat.py — Quantization-Aware Training (QAT) for StreamSenseNet
# Project STREAMSENSE (OSL-PRG-2026-SE) | Track A | Epic A5.2 STRETCH
# Run on: Google Colab T4 GPU
# =============================================================================
# WHAT THIS DOES:
#   1. Loads your pretrained best_model.pth (FP32) from Google Drive
#   2. Replaces all Conv2d, Linear layers with Brevitas quantized equivalents
#      (W8A8 — INT8 weights + INT8 activations)
#   3. Fine-tunes for a few epochs (QAT fine-tuning, not full retraining)
#   4. Saves QAT checkpoint to Drive: STREAMSENSE_outputs/best_model_qat.pth
# =============================================================================

# -----------------------------------------------------------------------------
# CELL 1 — Install dependencies (run once per Colab session)
# -----------------------------------------------------------------------------
# !pip install brevitas onnx onnxruntime --quiet

# -----------------------------------------------------------------------------
# CELL 2 — Mount Drive & clone repo
# -----------------------------------------------------------------------------
# from google.colab import drive
# drive.mount('/content/drive')
#
# import subprocess
# subprocess.run([
#     "git", "clone",
#     "https://<YOUR_TOKEN>@github.com/<YOUR_USERNAME>/STREAMSENSE.git",
#     "/content/STREAMSENSE"
# ], check=True)
#
# import sys
# sys.path.insert(0, '/content/STREAMSENSE/training')

# -----------------------------------------------------------------------------
# CELL 3 — Imports
# -----------------------------------------------------------------------------
import os
import sys
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Brevitas quantized layer replacements
import brevitas.nn as qnn
from brevitas.quant import Int8WeightPerTensorFloat, Int8ActPerTensorFloat

# Your existing modules (available after git clone + sys.path)
from dataset import get_dataloaders   # reuse existing dataloader
from mel_pipeline import preprocess   # MPIC v1.0 — do not modify

# =============================================================================
# SECTION 1 — Paths (edit these if your Drive layout differs)
# =============================================================================

DRIVE_ROOT        = "/content/drive/MyDrive"
DRIVE_OUTPUTS     = f"{DRIVE_ROOT}/STREAMSENSE_outputs"
PRETRAINED_CKPT   = f"{DRIVE_OUTPUTS}/best_model.pth"       # FP32 checkpoint
QAT_CKPT_OUT      = f"{DRIVE_OUTPUTS}/best_model_qat.pth"  # QAT output

DATA_RAW_ZIP      = f"{DRIVE_ROOT}/data_raw.zip"            # raw dataset zip
DATA_EXTRACT_DIR  = "/content/data"                         # local extract path
SPLITS_DIR        = "/content/STREAMSENSE/data/splits"      # split .txt files

# =============================================================================
# SECTION 2 — Hyperparameters
# =============================================================================

BATCH_SIZE        = 32
QAT_EPOCHS        = 10          # fine-tuning epochs (not full retrain)
LR_QAT            = 1e-4        # lower LR for fine-tuning
WEIGHT_DECAY      = 1e-4
GRAD_CLIP         = 1.0
SEED              = 42
NUM_CLASSES       = 10
DEVICE            = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# W8A8 — INT8 weights, INT8 activations (FINN-compatible starting point)
WEIGHT_BIT_WIDTH  = 8
ACT_BIT_WIDTH     = 8

# =============================================================================
# SECTION 3 — Brevitas quantized model definition
# =============================================================================
# This mirrors StreamSenseNet exactly but replaces:
#   nn.Conv2d   → qnn.QuantConv2d
#   nn.Linear   → qnn.QuantLinear
#   ReLU        → qnn.QuantReLU
# BatchNorm and pooling layers are NOT quantized (standard FINN practice).

class QuantConvBlock(nn.Module):
    """Single conv block: QuantConv → BN → QuantReLU → QuantConv → BN → QuantReLU → MaxPool → Dropout."""
    def __init__(self, in_ch, out_ch, dropout=0.25):
        super().__init__()
        self.block = nn.Sequential(
            qnn.QuantConv2d(
                in_ch, out_ch, kernel_size=3, padding=1,
                weight_bit_width=WEIGHT_BIT_WIDTH,
                weight_quant=Int8WeightPerTensorFloat,
                bias=False
            ),
            nn.BatchNorm2d(out_ch),
            qnn.QuantReLU(bit_width=ACT_BIT_WIDTH, act_quant=Int8ActPerTensorFloat),

            qnn.QuantConv2d(
                out_ch, out_ch, kernel_size=3, padding=1,
                weight_bit_width=WEIGHT_BIT_WIDTH,
                weight_quant=Int8WeightPerTensorFloat,
                bias=False
            ),
            nn.BatchNorm2d(out_ch),
            qnn.QuantReLU(bit_width=ACT_BIT_WIDTH, act_quant=Int8ActPerTensorFloat),

            nn.MaxPool2d(2, 2),
            nn.Dropout2d(dropout),
        )

    def forward(self, x):
        return self.block(x)


class StreamSenseNetQAT(nn.Module):
    """
    Brevitas W8A8 quantized version of StreamSenseNet.
    Input:  [B, 1, 64, 97]  float32 normalized mel spectrogram
    Output: [B, 10]          raw logits (same as FP32 model)
    """
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()

        # Input quantizer — quantizes the float32 input tensor before first conv
        self.input_quant = qnn.QuantIdentity(
            act_quant=Int8ActPerTensorFloat,
            bit_width=ACT_BIT_WIDTH,
            return_quant_tensor=True
        )

        # Three conv blocks (mirrors FP32 architecture exactly)
        self.block1 = QuantConvBlock(1,   32, dropout=0.25)   # → [B, 32, 32, 48]
        self.block2 = QuantConvBlock(32,  64, dropout=0.25)   # → [B, 64, 16, 24]
        self.block3 = QuantConvBlock(64, 128, dropout=0.25)   # → [B,128,  8, 12]

        # Global Average Pooling — not quantized (passthrough)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))               # → [B, 128, 1, 1]

        # Classifier head
        self.classifier = nn.Sequential(
            qnn.QuantLinear(
                128, 64,
                weight_bit_width=WEIGHT_BIT_WIDTH,
                weight_quant=Int8WeightPerTensorFloat,
                bias=True
            ),
            qnn.QuantReLU(bit_width=ACT_BIT_WIDTH, act_quant=Int8ActPerTensorFloat),
            nn.Dropout(0.5),
            # Final linear — NOT quantized (logits need full range for softmax)
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.input_quant(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.gap(x)
        x = x.flatten(1)        # [B, 128]
        x = self.classifier(x)  # [B, 10]
        return x


# =============================================================================
# SECTION 4 — Weight transfer from FP32 checkpoint
# =============================================================================

def load_fp32_weights_into_qat(qat_model, fp32_ckpt_path, device):
    """
    Load FP32 best_model.pth weights into the QAT model.
    Only transfers weights for matching layer names — quantizer params
    are initialised fresh (Brevitas handles their initialisation).
    """
    fp32_state = torch.load(fp32_ckpt_path, map_location=device)

    # Handle checkpoint dict vs raw state dict
    if "model_state_dict" in fp32_state:
        fp32_state = fp32_state["model_state_dict"]
    elif "state_dict" in fp32_state:
        fp32_state = fp32_state["state_dict"]

    qat_state  = qat_model.state_dict()
    transferred = 0
    skipped     = 0

    for k, v in fp32_state.items():
        if k in qat_state and qat_state[k].shape == v.shape:
            qat_state[k] = v
            transferred += 1
        else:
            skipped += 1

    qat_model.load_state_dict(qat_state, strict=False)
    print(f"[Weight transfer] Transferred: {transferred} | Skipped: {skipped}")
    return qat_model


# =============================================================================
# SECTION 5 — Training & evaluation helpers
# =============================================================================

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss   = criterion(logits, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        correct    += (logits.argmax(1) == y).sum().item()
        total      += x.size(0)
    return total_loss / total, correct / total * 100.0


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss   = criterion(logits, y)
        total_loss += loss.item() * x.size(0)
        correct    += (logits.argmax(1) == y).sum().item()
        total      += x.size(0)
    return total_loss / total, correct / total * 100.0


# =============================================================================
# SECTION 6 — Main QAT training loop
# =============================================================================

def main():
    torch.manual_seed(SEED)
    os.makedirs(DRIVE_OUTPUTS, exist_ok=True)

    print(f"Device: {DEVICE}")
    print(f"Loading data from: {DATA_EXTRACT_DIR}")

    # ------------------------------------------------------------------
    # Extract dataset (only needed first time)
    # ------------------------------------------------------------------
    if not os.path.exists(DATA_EXTRACT_DIR):
        print("Extracting data_raw.zip ...")
        import zipfile
        with zipfile.ZipFile(DATA_RAW_ZIP, 'r') as z:
            z.extractall(DATA_EXTRACT_DIR)
        print("Extraction done.")

    # ------------------------------------------------------------------
    # Dataloaders — reuse existing dataset.py
    # ------------------------------------------------------------------
    # dataset.py expects: data_dir, splits_dir, batch_size, augment(bool)
    train_loader, val_loader, _ = get_dataloaders(
        data_dir   = DATA_EXTRACT_DIR,
        splits_dir = SPLITS_DIR,
        batch_size = BATCH_SIZE,
        augment    = True,
        seed       = SEED,
    )
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    # ------------------------------------------------------------------
    # Build QAT model and transfer FP32 weights
    # ------------------------------------------------------------------
    print("\nBuilding QAT model (W8A8) ...")
    qat_model = StreamSenseNetQAT(num_classes=NUM_CLASSES).to(DEVICE)

    print(f"Loading FP32 weights from: {PRETRAINED_CKPT}")
    qat_model = load_fp32_weights_into_qat(qat_model, PRETRAINED_CKPT, DEVICE)

    total_params = sum(p.numel() for p in qat_model.parameters())
    print(f"QAT model parameters: {total_params:,}")

    # ------------------------------------------------------------------
    # Optimizer and loss
    # ------------------------------------------------------------------
    optimizer = optim.Adam(
        qat_model.parameters(), lr=LR_QAT, weight_decay=WEIGHT_DECAY
    )
    scheduler = ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-6, verbose=True
    )
    criterion = nn.CrossEntropyLoss()

    # ------------------------------------------------------------------
    # QAT fine-tuning loop
    # ------------------------------------------------------------------
    best_val_acc  = 0.0
    best_state    = None

    print(f"\n{'='*60}")
    print(f"Starting QAT fine-tuning: {QAT_EPOCHS} epochs, lr={LR_QAT}")
    print(f"{'='*60}")

    for epoch in range(1, QAT_EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(
            qat_model, train_loader, optimizer, criterion, DEVICE
        )
        val_loss, val_acc = evaluate(
            qat_model, val_loader, criterion, DEVICE
        )
        scheduler.step(val_loss)

        print(
            f"Epoch {epoch:02d}/{QAT_EPOCHS} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.2f}% | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.2f}%"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state   = copy.deepcopy(qat_model.state_dict())
            print(f"  ✓ New best val_acc={best_val_acc:.2f}% — checkpoint saved")

    # ------------------------------------------------------------------
    # Save best QAT checkpoint to Drive
    # ------------------------------------------------------------------
    print(f"\nBest QAT val_acc: {best_val_acc:.2f}%")
    print(f"Saving QAT checkpoint to: {QAT_CKPT_OUT}")

    torch.save({
        "model_state_dict" : best_state,
        "val_acc"          : best_val_acc,
        "qat_epochs"       : QAT_EPOCHS,
        "weight_bit_width" : WEIGHT_BIT_WIDTH,
        "act_bit_width"    : ACT_BIT_WIDTH,
        "architecture"     : "StreamSenseNetQAT_W8A8",
    }, QAT_CKPT_OUT)

    print("QAT training complete.")
    print(f"  FP32 baseline   : 95.97%")
    print(f"  QAT best val_acc: {best_val_acc:.2f}%")
    print(f"  Next step       : run export_qonnx.py to export to QONNX")


if __name__ == "__main__":
    main()
