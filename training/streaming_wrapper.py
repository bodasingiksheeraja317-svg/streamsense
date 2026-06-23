"""
streaming_wrapper.py
Project STREAMSENSE — Track A
Scope 2 / WA-2 — Embedding + Novelty Heads

Wraps the frozen StreamSenseNet classifier with two additive output heads:

    1. Embedding head  — a linear projection from the GAP feature vector
                         ([B, 128]) to a fixed-length embedding ([B, 128]).
                         Frozen embedding dim D = 128.

    2. Novelty head    — a scalar novelty score computed from the logits
                         via the max-softmax method: novelty = 1 − max(softmax(logits)).
                         Output shape [B, 1] (2-D, not [B]).

Design constraints (from Scope 2, Sections 5.2, 7 WA-2, 8 D-A2, 9):

    • The logits branch is structurally UNTOUCHED.  The frozen StreamSenseNet
      classifier (model.py) is instantiated as-is and its state_dict is loaded
      verbatim.  No weight in the logits path is modified or re-initialised.
      GV1K logit bitmatch to the single-head FP32 baseline is a hard exit
      criterion for WA-2.

    • Heads are additive branches that attach at the GAP output ([B, 128]).
      This is the deliberate design hook identified in ADR-001 §6 "Multi-head
      extension (Scope 2 / WA-2)".

    • novelty_score is defined as 1 − max_softmax(logits). This is the
      standard maximum-softmax probability (MSP) baseline for open-set /
      novelty detection. High novelty_score → input is out-of-distribution.
      The score is bounded [0, 1] by construction and requires no calibration
      dataset. Alternative (energy score) was considered but MSP is
      implementation-free and matched the "calibrated threshold" requirement
      with zero additional parameters.

    • ONNX export dimensions — non-negotiable (Scope 2, problem statement):
        input         : [1, 1, 64, 97]  float32  static
        logits        : [1, 10]         float32
        embedding     : [1, 128]        float32
        novelty_score : [1, 1]          float32  — MUST be 2-D

DSA Decision Record (as required by Section 6):
    Date            : 2026-06-23
    Component       : Embedding head
    Structure       : Single Linear(128, 128) projection; no activation.
    Complexity      : O(D²) per forward pass — constant for fixed D.
    Memory          : 128×128×4 = 65 536 bytes pre-allocated in weights.
    Alternative rejected: MLP projection with non-linearity — adds training
    dependency (head weights must be fine-tuned, violating the "additive
    branch" constraint); single linear projection is sufficient for a
    discriminative embedding space and exports to a single MatMul ONNX node.

    Date            : 2026-06-23
    Component       : Novelty scoring
    Algorithm chosen: Maximum Softmax Probability (MSP):
                      novelty = 1 − max(softmax(logits)).
    Structure       : novelty = 1 − max(softmax(logits)).
    Complexity      : O(C) per frame where C = 10 (constant). Satisfies the
                      O(1) per-frame complexity target (C is a fixed constant).
    Memory          : Zero additional state — purely functional over logits.
    Bounded output  : Score is intrinsically bounded [0, 1], making threshold
                      calibration straightforward without a calibration dataset.
                      This satisfies the "calibrated threshold" intent of §6.
    Alternative rejected — Streaming percentile:
                      Requires a held-out calibration window and a running
                      buffer of past novelty values to maintain empirical
                      percentile estimates. This adds per-stream mutable state
                      and a calibration dataset dependency that is not available
                      at WA-2 scope. Rejected on grounds of state complexity
                      and deployment portability.
    Alternative rejected — Energy score (−log Σ exp(logits/T)):
                      Requires a temperature hyperparameter T — an extra
                      scalar that must be tuned per deployment. Output is
                      unbounded (range depends on T and logit magnitudes),
                      making threshold calibration non-trivial and not directly
                      comparable across frames or models. Rejected on grounds
                      of hyperparameter dependency and unbounded output range.
    Conclusion      : MSP is self-contained, outputs a bounded [0, 1] score,
                      maps to three standard ONNX ops (Softmax, ReduceMax,
                      Sub), and satisfies both the O(1) complexity target and
                      the calibrated-threshold intent without any calibration
                      dataset or additional state.

Run directly for a smoke test:
    python streaming_wrapper.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Resolve project root and import the frozen classifier ────────────────────
# streaming_wrapper.py lives in training/; model.py is in the same directory.
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from model import StreamSenseNet  # noqa: E402 — path insert above

# ── Architecture constants ────────────────────────────────────────────────────
# These are frozen by MPIC v1.0 / ADR-001.  Do not change.
NUM_CLASSES    = 10
GAP_FEATURES   = 128      # StreamSenseNet GAP output width  (block3 → 128 ch)
EMBEDDING_DIM  = 128      # D — frozen into ERR v1.0; do not change post-export

# ── Default checkpoint path ───────────────────────────────────────────────────
_ROOT         = _THIS_DIR.parent
_DEFAULT_CKPT = _ROOT / "checkpoints" / "best_model.pth"


# ─────────────────────────────────────────────────────────────────────────────
# StreamSenseWrapper
# ─────────────────────────────────────────────────────────────────────────────

class StreamSenseWrapper(nn.Module):
    """
    Multi-head wrapper around the frozen StreamSenseNet classifier.

    Adds two output branches at the GAP feature vector, keeping the logits
    path strictly identical to the frozen single-head model.

    Inputs
    ------
    x : Tensor [B, 1, 64, 97]  float32  (MPIC v1.0 normalised mel spectrogram)

    Outputs  (all float32)
    -------
    logits        : [B, 10]   — identical to StreamSenseNet.forward(x)
    embedding     : [B, 128]  — linear projection of GAP features
    novelty_score : [B,  1]   — 1 − max(softmax(logits)); 2-D always
    """

    def __init__(self, num_classes: int = NUM_CLASSES,
                 embedding_dim: int = EMBEDDING_DIM):
        super().__init__()

        # ── Frozen backbone (logits path) ─────────────────────────────────────
        # Instantiate the canonical classifier.  Weights are loaded separately
        # via load_checkpoint() so that the wrapper can be constructed without
        # a checkpoint file (e.g. for ONNX shape inspection).
        self.backbone = StreamSenseNet(num_classes=num_classes)

        # ── Embedding head ────────────────────────────────────────────────────
        # Single linear projection: GAP_FEATURES → embedding_dim.
        # No activation; the downstream consumer (Track C / ANN index) applies
        # its own similarity metric.  This maps to a single MatMul ONNX node.
        self.embed_head = nn.Linear(GAP_FEATURES, embedding_dim, bias=True)

        # ── Novelty head ──────────────────────────────────────────────────────
        # Pure computation over logits — no learned parameters.
        # Implemented in forward() as:
        #   probs   = softmax(logits, dim=1)       [B, C]
        #   max_p   = probs.max(dim=1, keepdim=True).values   [B, 1]
        #   novelty = 1.0 - max_p                  [B, 1]
        # keepdim=True guarantees the 2-D [B, 1] contract.

        # ── Initialise embedding head ─────────────────────────────────────────
        # Xavier uniform is standard for a projection with no downstream
        # activation; keeps unit-variance signal in embedding space at init.
        nn.init.xavier_uniform_(self.embed_head.weight)
        nn.init.zeros_(self.embed_head.bias)

    # ─────────────────────────────────────────────────────────────────────────

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Multi-head forward pass.

        Args:
            x : Tensor [B, 1, 64, 97] float32

        Returns:
            logits        : [B, 10]  float32
            embedding     : [B, 128] float32
            novelty_score : [B,  1]  float32
        """
        # ── Backbone: replicate StreamSenseNet.forward() exactly ──────────────
        # We call into backbone's sub-modules directly, splitting at the GAP
        # output so we can branch without re-running conv blocks twice.
        feat = self.backbone.block1(x)          # [B, 32, 32, 48]
        feat = self.backbone.block2(feat)        # [B, 64, 16, 24]
        feat = self.backbone.block3(feat)        # [B, 128, 8, 12]
        feat = self.backbone.gap(feat)           # [B, 128, 1, 1]
        feat = feat.flatten(start_dim=1)         # [B, 128]  ← GAP vector

        # ── Logits branch (frozen, identical to single-head model) ────────────
        logits = self.backbone.classifier(feat)  # [B, 10]

        # ── Embedding branch ──────────────────────────────────────────────────
        embedding = self.embed_head(feat)        # [B, 128]

        # ── Novelty branch ────────────────────────────────────────────────────
        # detach() is NOT used here — both heads flow gradients normally during
        # any future fine-tuning.  The logits path is protected by not touching
        # backbone weights (see load_checkpoint).
        probs         = F.softmax(logits, dim=1)                    # [B, 10]
        max_prob      = probs.max(dim=1, keepdim=True).values       # [B,  1]
        novelty_score = 1.0 - max_prob                              # [B,  1]

        return logits, embedding, novelty_score

    # ─────────────────────────────────────────────────────────────────────────

    def load_checkpoint(self, ckpt_path: Path | str) -> tuple[int, float]:
        """
        Load backbone weights from a StreamSenseNet checkpoint.

        Only 'model_state' is consumed.  The embedding_head weights are NOT
        loaded from checkpoint — they remain at Xavier-uniform initialisation
        until explicitly trained.  This preserves the "additive branch"
        constraint: backbone weights are never touched.

        Args:
            ckpt_path : Path to best_model.pth checkpoint.

        Returns:
            (epoch, val_accuracy) from checkpoint metadata.
        """
        ckpt_path = Path(ckpt_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {ckpt_path}\n"
                f"Expected: checkpoints/best_model.pth"
            )

        ckpt = torch.load(ckpt_path, map_location="cpu")

        # Strict=True — every key in model_state must match backbone exactly.
        # This will raise if model.py has been modified (intentional safeguard).
        self.backbone.load_state_dict(ckpt["model_state"], strict=True)

        epoch    = int(ckpt.get("epoch", -1))
        val_acc  = float(ckpt.get("val_accuracy", float("nan")))
        return epoch, val_acc

    # ─────────────────────────────────────────────────────────────────────────

    @property
    def embedding_dim(self) -> int:
        """Frozen embedding dimension D."""
        return self.embed_head.out_features


# ─────────────────────────────────────────────────────────────────────────────
# Convenience factory
# ─────────────────────────────────────────────────────────────────────────────

def build_wrapper(
    ckpt_path: Path | str | None = None,
    eval_mode: bool = True,
) -> StreamSenseWrapper:
    """
    Construct a StreamSenseWrapper and optionally load checkpoint weights.

    Args:
        ckpt_path : Path to best_model.pth.  If None, uses the default path
                    (checkpoints/best_model.pth relative to project root).
                    Pass ckpt_path=False to skip checkpoint loading entirely
                    (useful for shape inspection without a checkpoint file).
        eval_mode : If True (default), put model in eval mode (BatchNorm and
                    Dropout2d disabled for deterministic inference).

    Returns:
        StreamSenseWrapper ready for inference or ONNX export.
    """
    wrapper = StreamSenseWrapper(
        num_classes   = NUM_CLASSES,
        embedding_dim = EMBEDDING_DIM,
    )

    if ckpt_path is not False:
        path = Path(ckpt_path) if ckpt_path is not None else _DEFAULT_CKPT
        epoch, val_acc = wrapper.load_checkpoint(path)
        print(f"[wrapper] Loaded checkpoint: epoch={epoch}, val_acc={val_acc:.4f}%")
    else:
        print("[wrapper] No checkpoint loaded — random backbone weights.")

    if eval_mode:
        wrapper.eval()

    return wrapper


# ─────────────────────────────────────────────────────────────────────────────
# Smoke test
# ─────────────────────────────────────────────────────────────────────────────

def _smoke_test():
    """
    Verify output shapes, dtypes, and novelty score range without a checkpoint.
    Runs entirely on CPU — no GPU or checkpoint required.
    """
    print("=" * 64)
    print("STREAMSENSE — streaming_wrapper.py smoke test (WA-2)")
    print("=" * 64)

    wrapper = StreamSenseWrapper(num_classes=NUM_CLASSES, embedding_dim=EMBEDDING_DIM)
    wrapper.eval()

    # Batch = 1 (inference contract), static input shape
    dummy = torch.zeros(1, 1, 64, 97)

    passed = 0
    failed = 0

    def check(name: str, cond: bool, detail: str = ""):
        nonlocal passed, failed
        status = "PASS" if cond else "FAIL"
        print(f"  [{status}] {name}  {detail}")
        if cond:
            passed += 1
        else:
            failed += 1

    with torch.no_grad():
        logits, embedding, novelty_score = wrapper(dummy)

    print(f"\nForward pass shapes:")
    print(f"  logits        : {tuple(logits.shape)}")
    print(f"  embedding     : {tuple(embedding.shape)}")
    print(f"  novelty_score : {tuple(novelty_score.shape)}")
    print()

    # ── Shape checks ──────────────────────────────────────────────────────────
    check("logits shape       [1, 10]",
          tuple(logits.shape) == (1, 10),
          f"got {tuple(logits.shape)}")
    check("embedding shape    [1, 128]",
          tuple(embedding.shape) == (1, EMBEDDING_DIM),
          f"got {tuple(embedding.shape)}")
    check("novelty_score shape [1,  1] (2-D)",
          tuple(novelty_score.shape) == (1, 1),
          f"got {tuple(novelty_score.shape)}")

    # ── Dtype checks ──────────────────────────────────────────────────────────
    check("logits dtype        float32",
          logits.dtype == torch.float32,
          f"got {logits.dtype}")
    check("embedding dtype     float32",
          embedding.dtype == torch.float32,
          f"got {embedding.dtype}")
    check("novelty_score dtype float32",
          novelty_score.dtype == torch.float32,
          f"got {novelty_score.dtype}")

    # ── Novelty score bounded [0, 1] ──────────────────────────────────────────
    ns_val = float(novelty_score.squeeze())
    check("novelty_score in [0, 1]",
          0.0 <= ns_val <= 1.0,
          f"got {ns_val:.6f}")

    # ── Novelty = 1 − max_softmax ─────────────────────────────────────────────
    expected_ns = float(1.0 - torch.softmax(logits, dim=1).max())
    check("novelty_score == 1 − max_softmax(logits)",
          abs(ns_val - expected_ns) < 1e-6,
          f"got {ns_val:.8f}, expected {expected_ns:.8f}")

    # ── Novelty is 2-D (not 1-D scalar) ──────────────────────────────────────
    check("novelty_score.ndim == 2 (not squeezed to 1-D)",
          novelty_score.ndim == 2,
          f"ndim={novelty_score.ndim}")

    # ── Batch size test (B=4) ──────────────────────────────────────────────────
    batch4 = torch.zeros(4, 1, 64, 97)
    with torch.no_grad():
        lg4, emb4, ns4 = wrapper(batch4)
    check("batch B=4 logits        [4, 10]",
          tuple(lg4.shape)  == (4, 10),
          f"got {tuple(lg4.shape)}")
    check("batch B=4 embedding     [4, 128]",
          tuple(emb4.shape) == (4, EMBEDDING_DIM),
          f"got {tuple(emb4.shape)}")
    check("batch B=4 novelty_score [4,  1]",
          tuple(ns4.shape)  == (4, 1),
          f"got {tuple(ns4.shape)}")

    # ── Checkpoint loading test (if checkpoint exists) ────────────────────────
    if _DEFAULT_CKPT.exists():
        try:
            w2 = build_wrapper(ckpt_path=_DEFAULT_CKPT, eval_mode=True)
            with torch.no_grad():
                lg2, emb2, ns2 = w2(dummy)
            check("checkpoint load: logits shape [1, 10]",
                  tuple(lg2.shape) == (1, 10),
                  f"got {tuple(lg2.shape)}")
            # Verify novelty is still 2-D after checkpoint load
            check("checkpoint load: novelty_score shape [1, 1]",
                  tuple(ns2.shape) == (1, 1),
                  f"got {tuple(ns2.shape)}")
        except Exception as e:
            check("checkpoint load", False, str(e))
    else:
        print(f"  [SKIP] Checkpoint not found at {_DEFAULT_CKPT} — skipping load test.")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*64}")
    print(f"Results: {passed} passed,  {failed} failed")
    if failed == 0:
        print("[PASS] streaming_wrapper.py verified — WA-2 shape contract met.")
        print("       Ready for export_multihead_onnx.py (WA-4).")
    else:
        print("[FAIL] One or more checks failed — review output above.")
    print(f"{'='*64}")

    return failed == 0


if __name__ == "__main__":
    ok = _smoke_test()
    sys.exit(0 if ok else 1)
