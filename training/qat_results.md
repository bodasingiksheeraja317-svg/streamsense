# QAT Results — A5.2 STRETCH
# StreamSenseNet × Brevitas × QONNX
# OSL-PRG-2026-SE | Track A | Ksheeraja

---

## Overview

This document records the results of Quantization-Aware Training (QAT) performed
on StreamSenseNet using the Brevitas library, targeting the Zynq-7000 FPGA via
the FINN toolchain (QONNX export path).

This is the **FPGA research path** (A5.2 STRETCH). The production deployment
artifact remains the PTQ INT8 ONNX model (`onnx_models/streamsense_model_int8.onnx`),
which is GV1K-green and consumed by Track B.

---

## Baseline Reference

| Metric                     | Value          |
|----------------------------|----------------|
| Float FP32 test accuracy   | 95.97%         |
| PTQ INT8 test accuracy     | 95.86%         |
| PTQ INT8 accuracy drop     | −0.11%         |
| PTQ INT8 model size        | 306 KB (73.6% smaller than FP32) |
| PTQ INT8 inference speedup | 8.29×          |

---

## QAT Configuration

| Parameter          | Value                          |
|--------------------|-------------------------------|
| Library            | Brevitas 0.10.2                |
| Export format      | QONNX (opset 13)               |
| Init strategy      | Transfer weights from `checkpoints/best_model.pth` (epoch 26) |
| Optimizer          | Adam, lr=1e-4, weight_decay=1e-4 |
| Scheduler          | ReduceLROnPlateau (factor=0.5, patience=3) |
| Max epochs         | 15 (fine-tuning, not full retraining) |
| Early stop         | 6 epochs no improvement        |
| Augmentation       | SpecAugment + time-domain (same as float training) |
| Seed               | 42                             |
| Quantization scope | All Conv2d layers + activations (post-ReLU) |
| Classifier head    | Float32 (kept full-precision)  |
| Quantization type  | Symmetric per-tensor, QAT (fake quant during training) |

---

## Results

> **Fill in after running `train_qat_brevitas.py` and `export_qonnx.py`**

### W8A8 (INT8 weights + INT8 activations)

| Metric                        | Value          |
|-------------------------------|----------------|
| QAT val accuracy              | ___.____%      |
| QAT test accuracy             | ___.____%      |
| Accuracy drop vs FP32         | −___.___%      |
| Accuracy drop vs PTQ INT8     | ___.___%       |
| Best epoch                    | ___            |
| QONNX export size             | ___ KB         |
| GV regression (10-vector)     | ___/10 PASS    |
| ONNX Runtime sanity check     | PASS / FAIL    |

### W4A4 (INT4 weights + INT4 activations)

| Metric                        | Value          |
|-------------------------------|----------------|
| QAT val accuracy              | ___.____%      |
| QAT test accuracy             | ___.____%      |
| Accuracy drop vs FP32         | −___.___%      |
| Accuracy drop vs PTQ INT8     | ___.___%       |
| Best epoch                    | ___            |
| QONNX export size             | ___ KB         |
| GV regression (10-vector)     | ___/10 PASS    |
| ONNX Runtime sanity check     | PASS / FAIL    |

---

## Design Decisions & Trade-offs

### Why W4A4 for the FPGA path?

The Zynq-7000 (xc7z020) has limited DSP48E1 slices (~220). FINN's HLS
transformation pipeline maps quantized multiply-accumulate operations to
DSP slices most efficiently at INT4:

- **INT8 (W8A8):** Requires 2 DSP48 slices per MAC (18-bit multiplier width)
- **INT4 (W4A4):** Fits 1 MAC per DSP48 (4×4 = 8-bit product, fits in 18-bit multiplier)
- Net effect: W4A4 roughly halves DSP utilization vs W8A8 on Zynq-7000

W8A8 is provided as a safety fallback — if W4A4 accuracy drop is unacceptable,
W8A8 gives a better accuracy/resource middle ground before falling back to PTQ INT8.

### Why fine-tuning rather than full retraining?

The float model already converged well (96.11% val, 95.97% test). QAT
fine-tuning from the float checkpoint typically recovers quantization loss
faster and with fewer epochs than training from scratch with fake quantization,
since the weights are already in a "quantization-friendly" region.

### Why keep the classifier head in float?

The FC head (128→64→10) has minimal parameter count (128×64 + 64×10 = 8,832 params)
relative to the Conv blocks (286,720 params). On FPGA, the small head is typically
implemented in the ARM PS side (soft processor), not the PL fabric. Keeping it
float avoids quantization error accumulation at the final decision layer.

### Quantization scope

All three QuantConvBlock layers are quantized (weights + activations).
An input quantization node (`QuantIdentity`) is inserted before Block 1 to
quantize the float32 input tensor. This matches FINN's expected graph structure.

---

## Exported Artifacts

| File | Description |
|------|-------------|
| `checkpoints_qat/qat_w8a8_best.pth` | W8A8 QAT PyTorch checkpoint |
| `checkpoints_qat/qat_w8a8_log.csv`  | W8A8 per-epoch training log |
| `checkpoints_qat/qat_w4a4_best.pth` | W4A4 QAT PyTorch checkpoint |
| `checkpoints_qat/qat_w4a4_log.csv`  | W4A4 per-epoch training log |
| `onnx_models/streamsense_qat_w8a8.onnx` | W8A8 QONNX export (for FINN) |
| `onnx_models/streamsense_qat_w4a4.onnx` | W4A4 QONNX export (for FINN) |

---

## FINN Integration Notes (for Track E — Prikshit)

The QONNX models are the input to the FINN synthesis flow targeting Zynq-7000:

```python
# FINN workflow sketch (Track E reference)
from finn.core.modelwrapper import ModelWrapper
from finn.transformation.qonnx.convert_qonnx_to_finn import ConvertQONNXtoFINN
from finn.transformation.streamline import Streamline

model = ModelWrapper("onnx_models/streamsense_qat_w4a4.onnx")
model = model.transform(ConvertQONNXtoFINN())
model = model.transform(Streamline())
# ... HLS + synthesis for xc7z020
```

Key constraints for FINN compatibility (already satisfied):
- ✅ Symmetric per-tensor weight quantization
- ✅ Static input shape [1, 1, 64, 97]
- ✅ No dynamic axes
- ✅ ONNX opset 13
- ✅ No framework-private ops

BatchNorm is folded into Conv weights by `export_qonnx` automatically.

---

## Limitations & Known Issues

1. **GV1K gate is NOT enforced on QONNX models.** The QONNX export introduces
   FINN-specific operator annotations that are not compatible with standard
   ONNX Runtime evaluation. The GV1K gate remains the responsibility of the
   PTQ INT8 model (the production artifact).

2. **W4A4 accuracy may degrade significantly.** 4-bit activations are aggressive
   for a mel-spectrogram input with dynamic range −80 dB to 0 dB. If accuracy
   drop exceeds ~3%, consider W4A8 (mixed precision) as an alternative.

3. **Brevitas API compatibility.** Brevitas 0.10.2 is pinned. Later versions may
   change the `QuantConv2d` / `QuantReLU` API. Test before upgrading.

4. **This is a STRETCH deliverable.** Per Scope 1 (Work Package A Rev 1.0):
   *"Partial, well-documented progress on STRETCH is a good outcome."*
   The core quantization deliverable (A5.1 PTQ INT8) is complete and GV1K-green.

---

*Generated: 17 June 2026 | Track A — Ksheeraja | Project STREAMSENSE (OSL-PRG-2026-SE)*
