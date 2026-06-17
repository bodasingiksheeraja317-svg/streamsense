# Architecture Decision Record — ADR-001
## Representation & Model Architecture for StreamSenseNet

**Project:** STREAMSENSE (OSL-PRG-2026-SE)  
**Track:** A — Host-side ML  
**Author:** Ksheeraja (OSL-IPL-2026-INT-002)  
**Date:** 17 June 2026  
**Status:** ACCEPTED  
**Supports:** Epic A3.3 (Scope 1 — Work Package A Rev 1.0)

---

## 1. Context

Project STREAMSENSE requires a fixed-arity signal classifier that:

- Accepts a 1-second, 16 kHz mono frame (16,000 float32 samples) over TCP/IP (NSP v1.2)
- Classifies it into one of 10 categories in real time
- Exports to ONNX for cross-platform deployment (host CPU and, later, FPGA via FINN)
- Is modality-agnostic — the architecture must not hard-code audio-domain assumptions
- Operates within a tight parameter and latency budget suited to edge deployment

Two candidate architectures were prototyped and evaluated end-to-end on the Google Speech Commands v2 dataset (10 classes, 5,779-sample test split):

| Candidate | Input representation | Model |
|-----------|---------------------|-------|
| **A — Chosen** | Log-mel spectrogram `[1, 64, 97]` | VGG-style 2D CNN (`StreamSenseNet`) |
| B — Rejected | Raw waveform `[1, 16000]` | Strided 1D CNN (`StreamSenseNet1D`) |

Both were trained under identical conditions (Adam, lr=0.001, ReduceLROnPlateau, SpecAugment, seed=42, 30 epochs max, early stopping patience=8) and evaluated with the same test split and golden-vector regression suite (GV1K — 1,000 vectors, 100 per class).

---

## 2. Decision

**Chosen architecture: VGG-style 2D CNN on log-mel spectrogram (StreamSenseNet)**

The model receives a pre-computed log-mel spectrogram of shape `[1, 64, 97]` produced by the frozen MPIC v1.0 front-end pipeline. It applies three convolutional blocks (32 → 64 → 128 filters), global average pooling, and a two-layer fully-connected head to produce `[1, 10]` raw logits.

---

## 3. Quantitative Evidence

### 3.1 Overall accuracy and efficiency

| Metric | 2D CNN (Chosen) | 1D CNN (Rejected) |
|--------|-----------------|-------------------|
| Test accuracy | **95.97%** | 95.76% |
| Test loss | **0.1273** | 0.1396 |
| Parameters | **295,786** | 591,210 |
| Parameter ratio | 1.00× (baseline) | **2.00× larger** |
| Accuracy / 100k params | **32.45%** | 16.18% |

The 2D CNN achieves higher accuracy with half the parameters — a 2× efficiency advantage on a per-parameter basis.

### 3.2 Per-class accuracy

| Class | 2D CNN | 1D CNN | Delta (1D − 2D) |
|-------|--------|--------|-----------------|
| yes | 98.84% | 98.02% | −0.82% |
| no | 96.79% | 93.91% | **−2.88%** |
| up | 95.17% | 95.71% | +0.54% |
| down | 94.04% | 95.91% | +1.87% |
| left | 96.67% | 94.56% | **−2.11%** |
| right | 99.29% | 98.59% | −0.70% |
| on | 95.66% | 96.35% | +0.69% |
| off | 94.47% | 91.62% | **−2.85%** |
| stop | 93.80% | 96.21% | +2.41% |
| go | 94.85% | 96.56% | +1.71% |

The 1D CNN shows notably larger deficits on spectrally similar short-duration words (no/off/left), where explicit frequency-axis structure in the mel representation gives the 2D CNN a consistent advantage. The 1D model's wins (down/stop/go) are on longer-voiced or more temporally distinctive words where raw waveform cues suffice.

### 3.3 Quantization results (2D CNN only — INT8 PTQ via ONNX Runtime)

| Metric | FP32 | INT8 |
|--------|------|------|
| Test accuracy | 95.97% | 95.86% |
| Accuracy drop | — | **−0.11%** (budget: ≤1.0% ✅) |
| Model size | 1.13 MB | 306 KB (73.6% reduction) |
| Inference speed | 14.48 ms/sample | 1.75 ms/sample (**8.29× faster**) |
| GV1K regression | 1000/1000 PASS | 1000/1000 PASS |

The 1D CNN was not carried through quantization evaluation; the 2D CNN's quantization profile was a factor in confirming the final decision.

---

## 4. Rationale

### 4.1 Explicit time-frequency structure

A log-mel spectrogram is a compact, well-conditioned 2D representation of signal energy across both time and frequency. The 2D CNN can apply spatially local filters over this structure from the first layer. The 1D CNN must learn to construct equivalent frequency-selective representations implicitly from raw samples via large receptive fields — a significantly harder learning problem, requiring more parameters to approximate the same inductive bias.

### 4.2 Parameter efficiency

The 2D CNN achieves better accuracy with 295,786 parameters vs 591,210 for the 1D CNN — half the model footprint. For edge and FPGA deployment, smaller models mean lower BRAM consumption, shorter synthesis times, and more predictable latency.

### 4.3 ONNX portability

Both architectures export cleanly to ONNX opset 17. The 2D CNN uses only standard ops (Conv2d, BatchNorm, ReLU, MaxPool, AdaptiveAvgPool, Linear) with no audio-library dependencies in the graph itself. The MPIC v1.0 preprocessing pipeline (mel computation) is a separate, versioned contract — it does not live inside the ONNX graph, keeping the model truly modality-agnostic at the graph level.

### 4.4 Preprocessing separability

Decoupling the front-end transform (MPIC v1.0) from the model graph means:
- The ONNX graph is portable to any runtime that can feed a `[1, 1, 64, 97]` float32 tensor
- The preprocessing contract can be versioned independently of the model weights
- Any downstream modality (vibration, ECG, RF) only needs to produce a conformant tensor — the model graph is unchanged

### 4.5 Quantization fitness

The 2D CNN quantizes to INT8 with a 0.11% accuracy drop — well within the 1.0% budget — and achieves 8.29× inference speedup. This profile is compatible with the FPGA deployment target (Zynq-7000 via FINN) and validates the choice for the QAT / QONNX stretch path (A5.2).

---

## 5. Alternatives Considered and Rejected

### 5.1 1D CNN on raw waveform (StreamSenseNet1D) — Rejected

**Reason:** 2× parameter count for −0.21% lower accuracy. Larger deficits on spectrally similar classes (no, off, left) with up to −2.88% per-class delta. No quantization evaluation was run; the larger graph would likely be harder to quantize without accuracy loss. Not carried forward.

### 5.2 MATLAB Deep Learning Toolbox model — Not prototyped

**Reason:** MATLAB DL Toolbox models do not export to ONNX opset 17 without toolchain-specific adapters. The deployment target (Track B C++ runtime via ONNX Runtime, Track E FPGA via FINN) requires a clean, framework-agnostic ONNX graph. MATLAB was used for exploratory signal analysis only (MPIC parameter selection) and was explicitly excluded from the model development path.

### 5.3 ResNet / MobileNet style architectures — Not prototyped

**Reason:** Residual connections and depthwise-separable convolutions introduce additional ONNX ops (elementwise add, grouped convolutions) that complicate FINN synthesis. The VGG-style block (Conv → BN → ReLU → MaxPool) maps to a minimal, well-understood set of FPGA-friendly primitives. Given that the simpler architecture already achieves 95.97% accuracy at 295k parameters, introducing architectural complexity was not warranted.

### 5.4 Global Average Pooling vs Flatten

GAP (`AdaptiveAvgPool2d(1,1)`) collapses the spatial map regardless of input dimensions, making the classifier head input size invariant to upstream spatial changes. Flatten would tie the head's input dimension to the exact spatial output of Block 3 (`8 × 12 = 96` values per channel), breaking the architecture if the mel shape or pooling stride changes. GAP also acts as a mild spatial regularizer. GAP was chosen unconditionally.

---

## 6. Implications and Open Items

### Confirmed implications
- MPIC v1.0 is frozen. Any change to mel parameters (`n_fft`, `hop_length`, `n_mels`, normalization stats) requires a version bump and joint sign-off, as it changes the input tensor distribution.
- The ONNX graph (`streamsense_model_fp32.onnx`, `streamsense_model_int8.onnx`) takes `float32[1,1,64,97]` and produces `float32[1,10]`. These shapes are static and pinned.
- GV1K (1,000-vector golden regression set) remains the acceptance gate for any model change going forward.

### FPGA path (Scope 2 / A5.2)
The QAT / Brevitas stretch (A5.2) targets INT8 or lower bit-width quantization using Quantization-Aware Training, exporting to QONNX format for synthesis via FINN. The 2D CNN's small size, standard op set, and demonstrated PTQ fitness make it the correct starting point for this path. The 1D CNN is not carried forward to A5.2.

### Multi-head extension (Scope 2 / WA-2)
Scope 2 adds embedding and novelty heads alongside the existing logits output. The GAP layer (`[B, 128]` feature vector) is the natural attachment point for the projection head — this was a deliberate design choice in the 2D CNN that would not have been as clean in the 1D strided architecture.

---

## 7. Decision Summary

| Dimension | Outcome |
|-----------|---------|
| Architecture | VGG-style 2D CNN |
| Input | Log-mel spectrogram `[1, 1, 64, 97]` float32 |
| Front-end | MPIC v1.0 (separate, versioned, modality-agnostic) |
| Parameters | 295,786 |
| Test accuracy (FP32) | 95.97% |
| INT8 accuracy drop | 0.11% |
| ONNX opset | 17 |
| Status | **ACCEPTED** |
| Supersedes | None (first ADR for this project) |
| Next review trigger | MPIC version bump OR accuracy regression >0.5% on GV1K |

---

*ADR-001 — StreamSenseNet Architecture | Track A | OSL-PRG-2026-SE | 17 June 2026*
