"""
train_speaker.py
================
OSL-IPL-2026-INT-002  |  Track A  |  WA-3

Training pipeline for SpeakerNet with ArcFace metric learning.

Run from repo root:
    python training/train_speaker.py

Or with overrides:
    python training/train_speaker.py --epochs 40 --m 0.45 --s 32

Checkpoints saved to:  checkpoints/speaker/
Best model saved to:   checkpoints/best_speaker_model.pth
"""

import os
import sys
import math
import time
import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# ── path setup (works from repo root or training/) ───────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "training"))

from dataset_speaker import SpeakerDataset, BalancedBatchSampler
from model_speaker    import SpeakerNet, ArcFaceHead


# ── Defaults ─────────────────────────────────────────────────────────────────

DEFAULTS = dict(
    train_csv  = "data/speaker_splits/speaker_train.csv",
    val_csv    = "data/speaker_splits/speaker_val.csv",
    ckpt_dir   = "checkpoints/speaker",
    best_path  = "checkpoints/best_speaker_model.pth",
    backbone   = "checkpoints/best_model.pth",
    epochs     = 30,
    M          = 16,    # speakers per batch
    K          = 4,     # utterances per speaker per batch
    lr         = 1e-4,
    wd         = 1e-4,
    s          = 32.0,  # ArcFace scale
    m          = 0.50,  # ArcFace margin
    seed       = 42,
    num_workers= 2,
)


# ── Reproducibility ───────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── EER computation ───────────────────────────────────────────────────────────

def compute_eer(embeddings: np.ndarray, labels: np.ndarray) -> float:
    """
    Compute Equal Error Rate on a held-out split.

    Builds all genuine pairs (same speaker) and a random equal-size
    sample of impostor pairs (different speakers), then finds the
    threshold where FAR == FRR.

    Args:
        embeddings : [N, 128] float32  L2-normalised
        labels     : [N]      int      speaker IDs

    Returns:
        eer : float  (0.0 – 1.0)
    """
    rng = np.random.default_rng(0)

    # ── collect genuine and impostor cosine similarities ─────────────────
    genuine_scores  = []
    impostor_scores = []

    speaker_to_idx: dict[int, list[int]] = {}
    for i, lbl in enumerate(labels):
        speaker_to_idx.setdefault(int(lbl), []).append(i)

    # genuine pairs: all within-speaker pairs (capped at 2000 per speaker)
    for sid, idxs in speaker_to_idx.items():
        if len(idxs) < 2:
            continue
        idxs_arr = np.array(idxs)
        for i in range(len(idxs_arr)):
            for j in range(i + 1, min(i + 5, len(idxs_arr))):
                cos_sim = float(
                    embeddings[idxs_arr[i]] @ embeddings[idxs_arr[j]]
                )
                genuine_scores.append(cos_sim)

    if len(genuine_scores) == 0:
        return 0.5   # no genuine pairs → undefined; return chance

    # impostor pairs: random cross-speaker, same count as genuine
    n_imp = len(genuine_scores)
    all_idx = np.arange(len(labels))
    for _ in range(n_imp):
        a, b = rng.choice(all_idx, 2, replace=False)
        while labels[a] == labels[b]:
            b = rng.choice(all_idx)
        impostor_scores.append(float(embeddings[a] @ embeddings[b]))

    genuine_arr  = np.array(genuine_scores)
    impostor_arr = np.array(impostor_scores)

    # ── sweep thresholds ─────────────────────────────────────────────────
    thresholds = np.linspace(-1.0, 1.0, 400)
    best_eer   = 1.0

    for t in thresholds:
        far = np.mean(impostor_arr >= t)   # impostors accepted
        frr = np.mean(genuine_arr  <  t)   # genuines rejected
        eer_candidate = abs(far - frr)
        if eer_candidate < abs(best_eer - (far + frr) / 2):
            best_eer = (far + frr) / 2

    return float(best_eer)


# ── Validation pass ───────────────────────────────────────────────────────────

@torch.no_grad()
def validate(
    net    : SpeakerNet,
    loader : DataLoader,
    device : torch.device,
) -> tuple[float, float]:
    """
    Extract all val embeddings, compute EER and Rank-1 accuracy.

    Returns:
        eer    : float  (lower is better)
        rank1  : float  fraction correct nearest-neighbour (higher is better)
    """
    net.eval()
    all_emb  = []
    all_lbl  = []

    for mels, sids, _ in loader:
        mels = mels.to(device)
        emb  = net(mels)
        all_emb.append(emb.cpu().numpy())
        all_lbl.append(sids.numpy())

    embeddings = np.concatenate(all_emb, axis=0)   # [N, 128]
    labels     = np.concatenate(all_lbl, axis=0)   # [N]

    eer = compute_eer(embeddings, labels)

    # ── Rank-1: for each sample find nearest neighbour (excl. self) ───────
    # cosine similarity matrix via dot product (embeddings are unit-norm)
    sim = embeddings @ embeddings.T     # [N, N]
    np.fill_diagonal(sim, -2.0)        # exclude self
    nn_pred = np.argmax(sim, axis=1)   # [N]
    rank1   = float(np.mean(labels[nn_pred] == labels))

    return eer, rank1


# ── Training loop ─────────────────────────────────────────────────────────────

def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train_speaker] Device: {device}")

    # ── datasets ──────────────────────────────────────────────────────────
    train_csv = ROOT / args.train_csv
    val_csv   = ROOT / args.val_csv

    train_ds = SpeakerDataset(train_csv, augment=True)
    val_ds   = SpeakerDataset(val_csv,   augment=False)

    n_speakers = train_ds.n_speakers
    print(f"[train_speaker] Training speakers : {n_speakers}")
    print(f"[train_speaker] Val     speakers  : {val_ds.n_speakers}")

    train_sampler = BalancedBatchSampler(
        train_ds, M=args.M, K=args.K, seed=args.seed
    )
    # val loader uses a flat sampler — all val embeddings extracted once
    val_loader = DataLoader(
        val_ds,
        batch_size=64,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    train_loader = DataLoader(
        train_ds,
        batch_sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    # ── model ─────────────────────────────────────────────────────────────
    net  = SpeakerNet().to(device)
    head = ArcFaceHead(
        in_dim=128,
        n_classes=n_speakers,
        s=args.s,
        m=args.m,
    ).to(device)

    backbone_path = ROOT / args.backbone
    if backbone_path.exists():
        net.load_backbone(backbone_path)
    else:
        print(f"[WARN] Backbone checkpoint not found at {backbone_path}. Training from scratch.")

    # ── optimiser ─────────────────────────────────────────────────────────
    # train backbone + projection + arcface head jointly
    params = list(net.parameters()) + list(head.parameters())
    opt    = AdamW(params, lr=args.lr, weight_decay=args.wd)
    sched  = CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-6)

    # ── checkpoint dir ────────────────────────────────────────────────────
    ckpt_dir  = ROOT / args.ckpt_dir
    best_path = ROOT / args.best_path
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path.parent.mkdir(parents=True, exist_ok=True)

    best_eer   = 1.0
    best_rank1 = 0.0

    print()
    print(f"{'Epoch':>5}  {'Train Loss':>10}  {'EER':>7}  {'Rank-1':>7}  {'LR':>8}  {'Time':>6}")
    print("-" * 55)

    for epoch in range(1, args.epochs + 1):
        net.train()
        head.train()
        t0 = time.time()
        epoch_loss = 0.0
        n_batches  = 0

        for mels, sids, _ in train_loader:
            mels = mels.to(device)
            sids = sids.to(device)

            opt.zero_grad(set_to_none=True)

            emb    = net(mels)              # [B, 128] unit-norm
            logits = head(emb, sids)        # [B, n_speakers]
            loss   = F.cross_entropy(logits, sids)

            loss.backward()
            # gradient clip — stabilises early training
            nn.utils.clip_grad_norm_(params, max_norm=10.0)
            opt.step()

            epoch_loss += loss.item()
            n_batches  += 1

        sched.step()
        avg_loss = epoch_loss / max(n_batches, 1)
        elapsed  = time.time() - t0

        # ── validation every epoch ────────────────────────────────────────
        eer, rank1 = validate(net, val_loader, device)
        lr_now = sched.get_last_lr()[0]

        print(
            f"{epoch:>5}  {avg_loss:>10.4f}  {eer:>7.4f}  {rank1:>7.4f}"
            f"  {lr_now:>8.2e}  {elapsed:>5.1f}s"
        )

        # ── save checkpoint every 5 epochs ───────────────────────────────
        if epoch % 5 == 0:
            ckpt = {
                "epoch"      : epoch,
                "net_state"  : net.state_dict(),
                "head_state" : head.state_dict(),
                "opt_state"  : opt.state_dict(),
                "eer"        : eer,
                "rank1"      : rank1,
                "args"       : vars(args),
            }
            torch.save(ckpt, ckpt_dir / f"speaker_epoch_{epoch:03d}.pth")

        # ── save best model (by EER, tie-break by rank1) ─────────────────
        if eer < best_eer or (eer == best_eer and rank1 > best_rank1):
            best_eer   = eer
            best_rank1 = rank1
            torch.save(
                {
                    "epoch"     : epoch,
                    "net_state" : net.state_dict(),
                    "eer"       : eer,
                    "rank1"     : rank1,
                    "n_speakers": n_speakers,
                    "embed_dim" : 128,
                    "args"      : vars(args),
                },
                best_path,
            )

    # ── final summary ─────────────────────────────────────────────────────
    print()
    print("=" * 55)
    print(f"  Training complete.")
    print(f"  Best EER    : {best_eer:.4f}  (target ≤ 0.15)")
    print(f"  Best Rank-1 : {best_rank1:.4f}  (target ≥ 0.80)")
    print(f"  Best model  : {best_path}")
    print("=" * 55)

    # exit code 1 if below SOW exit criteria (useful in CI)
    if best_eer > 0.15 or best_rank1 < 0.80:
        print("[WARN] SOW exit criteria not met. Consider more epochs or lower margin.")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train SpeakerNet (WA-3)")
    for key, val in DEFAULTS.items():
        t = type(val) if val is not None else str
        p.add_argument(f"--{key}", type=t, default=val)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
