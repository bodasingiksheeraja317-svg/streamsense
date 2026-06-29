"""
model_speaker.py
================
OSL-IPL-2026-INT-002  |  Track A  |  WA-3

SpeakerNet: convolutional speaker-embedding model for fingerprinting.

Architecture
------------
Backbone  : StreamSenseNet conv blocks 1-3 + GAP  (weights loaded from
            best_model.pth; fine-tuned end-to-end during speaker training)
Projection: Linear(128→256) + BN + ReLU → Linear(256→128) → L2-norm
Output    : unit-norm embedding  [B, 128]  float32

Training head (attached only during training, not exported):
ArcFaceHead: cosine-based angular margin classifier over N_speakers

The embedding dimension is frozen at 128 to match ERR v1.0  embed_dim.

Usage
-----
    from training.model_speaker import SpeakerNet, ArcFaceHead

    # Build and load pretrained backbone
    net = SpeakerNet()
    net.load_backbone("checkpoints/best_model.pth")

    # Forward (inference)
    emb = net(mel)           # [B, 128] unit-norm

    # Training
    head = ArcFaceHead(in_dim=128, n_classes=n_speakers)
    logits = head(emb, labels)
    loss = F.cross_entropy(logits, labels)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

# ── Embedding dimension (frozen into ERR v1.0) ───────────────────────────────
EMBED_DIM = 128


# ── Conv block (identical to StreamSenseNet for weight compatibility) ─────────

class _ConvBlock(nn.Module):
    """Two Conv2d + BN + ReLU + MaxPool + SpatialDropout2d."""

    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.25) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch,  out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(dropout),   # spatial dropout
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# ── SpeakerNet ────────────────────────────────────────────────────────────────

class SpeakerNet(nn.Module):
    """
    Speaker embedding network.

    Input  : [B, 1, 64, 97]  (MPIC v1.0 mel tensor)
    Output : [B, 128]         unit-norm embedding

    The conv backbone is identical in structure to StreamSenseNet so
    that pre-trained weights can be loaded directly.
    """

    def __init__(self) -> None:
        super().__init__()

        # ── backbone (mirrors StreamSenseNet blocks 1-3 + GAP) ────────────
        self.block1 = _ConvBlock(1,  32)
        self.block2 = _ConvBlock(32, 64)
        self.block3 = _ConvBlock(64, 128)
        self.gap    = nn.AdaptiveAvgPool2d(1)   # [B, 128, 1, 1]

        # ── projection head (new, speaker-specific) ───────────────────────
        # Two-layer MLP: 128 → 256 → 128, with BN between layers
        self.proj = nn.Sequential(
            nn.Linear(128, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, EMBED_DIM, bias=False),
        )

        self._init_projection()

    def _init_projection(self) -> None:
        """Kaiming init for projection layers."""
        for m in self.proj.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    # ── weight loading ────────────────────────────────────────────────────────

    def load_backbone(self, ckpt_path: str | Path) -> None:
        """
        Load conv backbone weights from a StreamSenseNet checkpoint.
        Only copies block1/block2/block3/gap weights; projection head
        keeps its fresh initialisation.

        Args:
            ckpt_path : path to best_model.pth
        """
        ckpt_path = Path(ckpt_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        raw = torch.load(ckpt_path, map_location="cpu")

        # handle both bare state_dict and {"model_state": ...} formats
        if isinstance(raw, dict) and "model_state" in raw:
            src_sd = raw["model_state"]
        elif isinstance(raw, dict) and all(isinstance(k, str) for k in raw):
            src_sd = raw
        else:
            raise ValueError("Unrecognised checkpoint format.")

        # keys that belong to the backbone
        backbone_keys = {"block1", "block2", "block3", "gap"}
        dst_sd = self.state_dict()

        loaded, skipped = 0, 0
        for k, v in src_sd.items():
            prefix = k.split(".")[0]
            if prefix in backbone_keys and k in dst_sd:
                if dst_sd[k].shape == v.shape:
                    dst_sd[k] = v
                    loaded += 1
                else:
                    print(f"  [WARN] shape mismatch for {k}: src {v.shape} vs dst {dst_sd[k].shape}")
                    skipped += 1
            else:
                skipped += 1

        self.load_state_dict(dst_sd)
        print(f"[SpeakerNet] Backbone loaded: {loaded} tensors copied, {skipped} skipped.")

    # ── forward ───────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : [B, 1, 64, 97]  MPIC v1.0 mel tensor
        Returns:
            embedding : [B, 128]  L2-normalised  (unit-norm)
        """
        # backbone
        x = self.block1(x)          # [B,  32, 32, 48]
        x = self.block2(x)          # [B,  64, 16, 24]
        x = self.block3(x)          # [B, 128,  8, 12]
        x = self.gap(x)             # [B, 128,  1,  1]
        x = x.flatten(1)            # [B, 128]

        # projection
        x = self.proj(x)            # [B, 128]

        # L2 normalise → unit-norm embeddings for cosine similarity
        x = F.normalize(x, p=2, dim=1)
        return x


# ── ArcFace head ──────────────────────────────────────────────────────────────

class ArcFaceHead(nn.Module):
    """
    Additive Angular Margin (ArcFace) classification head.

    Reference: Deng et al. 2019  "ArcFace: Additive Angular Margin Loss
    for Deep Face Recognition"  (CVPR 2019)

    The weight matrix W is L2-normalised so that the logit for class c is:
        s · cos(θ_c + m)   for the ground-truth class
        s · cos(θ_c)       for all other classes

    Args:
        in_dim    : embedding dimension (128)
        n_classes : number of speaker classes in the training split
        s         : feature scale  (default 32.0)
        m         : angular margin in radians  (default 0.50 ≈ 28.6°)
    """

    def __init__(
        self,
        in_dim: int,
        n_classes: int,
        s: float = 32.0,
        m: float = 0.50,
    ) -> None:
        super().__init__()
        self.in_dim    = in_dim
        self.n_classes = n_classes
        self.s         = s
        self.m         = m

        # learnable weight matrix; each row is the class prototype
        self.weight = nn.Parameter(torch.empty(n_classes, in_dim))
        nn.init.xavier_uniform_(self.weight)

        # pre-compute margin constants
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th    = math.cos(math.pi - m)   # threshold: cos(π - m)
        self.mm    = math.sin(math.pi - m) * m

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            embeddings : [B, in_dim]  unit-norm  (output of SpeakerNet)
            labels     : [B]          integer speaker IDs (0-indexed)
        Returns:
            logits     : [B, n_classes]  scaled cosine logits with margin
        """
        # normalise weight prototypes
        w = F.normalize(self.weight, p=2, dim=1)   # [n_classes, in_dim]

        # cosine similarity  [B, n_classes]
        cosine = F.linear(embeddings, w)
        cosine = cosine.clamp(-1.0 + 1e-7, 1.0 - 1e-7)

        # sin(θ)  via identity  sin² + cos² = 1
        sine = torch.sqrt(1.0 - cosine.pow(2))

        # cos(θ + m) = cos θ · cos m − sin θ · sin m
        phi = cosine * self.cos_m - sine * self.sin_m

        # numerical guard: if cos θ < cos(π - m), use linear approx
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # apply margin only to the ground-truth class
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.unsqueeze(1), 1.0)

        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output = output * self.s

        return output


# ── Quick smoke test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("SpeakerNet smoke test …")

    net  = SpeakerNet()
    head = ArcFaceHead(in_dim=EMBED_DIM, n_classes=200)

    dummy_mel    = torch.randn(8, 1, 64, 97)
    dummy_labels = torch.randint(0, 200, (8,))

    emb    = net(dummy_mel)
    logits = head(emb, dummy_labels)
    loss   = F.cross_entropy(logits, dummy_labels)

    print(f"  Input  : {dummy_mel.shape}")
    print(f"  Embed  : {emb.shape}   norm={emb.norm(dim=1).mean():.4f} (should be ≈1.0)")
    print(f"  Logits : {logits.shape}")
    print(f"  Loss   : {loss.item():.4f}")
    print("Smoke test PASSED.")
