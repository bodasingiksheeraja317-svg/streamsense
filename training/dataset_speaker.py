"""
dataset_speaker.py
==================
OSL-IPL-2026-INT-002  |  Track A  |  WA-3

PyTorch Dataset and BalancedBatchSampler for speaker fingerprinting.

Reads the speaker CSV manifests produced by build_speaker_dataset.py.
Applies MPIC v1.0 mel preprocessing (mel_pipeline.preprocess) — no new
normalization statistics; the frozen global stats remain in effect.

Classes exported:
    SpeakerDataset        — standard Map-style Dataset
    BalancedBatchSampler  — yields M-speaker × K-utterance batches
                            required for ArcFace / triplet training

Usage:
    from training.dataset_speaker import SpeakerDataset, BalancedBatchSampler

    train_ds = SpeakerDataset("data/speaker_splits/speaker_train.csv")
    sampler  = BalancedBatchSampler(train_ds, M=16, K=4)
    loader   = DataLoader(train_ds, batch_sampler=sampler, num_workers=2)

    for mel_tensors, speaker_ids, class_labels in loader:
        ...   # mel_tensors: [B, 1, 64, 97], speaker_ids: [B], class_labels: [B]
"""

import csv
import random
from pathlib import Path
from collections import defaultdict
from typing import Iterator
import sys, os

import torch
from torch.utils.data import Dataset, Sampler

# ── mel_pipeline import ───────────────────────────────────────────────────────
# mel_pipeline.py lives in training/ and exports a plain function `preprocess`.
# Insert both this file's directory AND cwd/training so it resolves correctly
# whether run locally (repo root) or in Colab (Drive path).
_this_dir    = str(Path(__file__).resolve().parent)
_cwd_training = str(Path(os.getcwd()) / 'training')
for _p in [_this_dir, _cwd_training]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from mel_pipeline import preprocess as _mel_preprocess


# ── SpeakerDataset ────────────────────────────────────────────────────────────

class SpeakerDataset(Dataset):
    """
    Map-style Dataset over speaker CSV manifests.

    Each item returns:
        mel  : torch.Tensor  shape [1, 64, 97]  float32   (MPIC v1.0)
        sid  : int            speaker integer ID
        cls  : int            command class index (0-9)
    """

    def __init__(self, csv_path: str | Path, augment: bool = False) -> None:
        """
        Args:
            csv_path : path to speaker_train/val/test.csv
            augment  : if True, apply time-domain augmentation
                       (same as dataset.py: circular shift, noise, amplitude)
                       Only set True for the training split.
        """
        self.csv_path = Path(csv_path)
        self.augment  = augment

        # ── parse CSV ─────────────────────────────────────────────────────
        self.records: list[tuple[str, int, int]] = []  # (filepath, speaker_id, class_label)
        self.speaker_to_indices: dict[int, list[int]] = defaultdict(list)

        with open(self.csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row_idx, row in enumerate(reader):
                filepath    = row["filepath"]
                speaker_id  = int(row["speaker_id"])
                class_label = int(row["class_label"])
                self.records.append((filepath, speaker_id, class_label))
                self.speaker_to_indices[speaker_id].append(row_idx)

        self.n_speakers: int = len(self.speaker_to_indices)
        print(
            f"[SpeakerDataset] {self.csv_path.name}: "
            f"{len(self.records)} utterances, {self.n_speakers} speakers"
        )

    # ── Dataset interface ─────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        filepath, speaker_id, class_label = self.records[idx]

        # ── load waveform ─────────────────────────────────────────────────
        import torchaudio
        waveform, sr = torchaudio.load(filepath)

        if sr != 16_000:
            waveform = torchaudio.functional.resample(waveform, sr, 16_000)

        # mono downmix
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # pad / crop to exactly 16 000 samples
        n = waveform.shape[-1]
        if n < 16_000:
            waveform = torch.nn.functional.pad(waveform, (0, 16_000 - n))
        else:
            waveform = waveform[..., :16_000]

        # ── optional time-domain augmentation (training only) ─────────────
        if self.augment:
            waveform = _augment_waveform(waveform)

        # ── MPIC v1.0 mel preprocessing ───────────────────────────────────
        # mel_pipeline.preprocess takes np.ndarray [16000] float32
        # and returns np.ndarray [1, 1, 64, 97]
        wav_np = waveform.squeeze(0).numpy()   # [16000]
        mel    = _mel_preprocess(wav_np)        # [1, 1, 64, 97]

        mel_tensor = torch.from_numpy(mel).squeeze(0)  # [1, 64, 97]

        return mel_tensor, speaker_id, class_label

    # ── Convenience ───────────────────────────────────────────────────────────

    @property
    def speaker_ids(self) -> list[int]:
        """Sorted list of unique speaker integer IDs in this split."""
        return sorted(self.speaker_to_indices.keys())


# ── Time-domain augmentation (mirrors dataset.py) ─────────────────────────────

def _augment_waveform(waveform: torch.Tensor) -> torch.Tensor:
    """
    waveform: [1, 16000] float32
    Applies circular shift, gaussian noise, amplitude scaling.
    Same parameters as the original dataset.py.
    """
    wav = waveform.clone()
    shift = random.randint(-3200, 3200)
    wav = torch.roll(wav, shifts=shift, dims=-1)
    wav = wav + torch.randn_like(wav) * 0.005
    scale = random.uniform(0.8, 1.2)
    wav = wav * scale
    return wav


# ── BalancedBatchSampler ──────────────────────────────────────────────────────

class BalancedBatchSampler(Sampler):
    """
    Yields index lists of size M*K.

    Each batch contains exactly M distinct speaker classes, each
    contributing exactly K utterances. This guarantees valid
    anchor-positive pairs in every batch — required by ArcFace and
    hard-negative triplet mining.

    Args:
        dataset  : SpeakerDataset instance
        M        : speakers per batch  (recommend 16)
        K        : utterances per speaker per batch  (recommend 4)
        drop_last: drop incomplete last batch
        seed     : random seed
    """

    def __init__(
        self,
        dataset: SpeakerDataset,
        M: int = 16,
        K: int = 4,
        drop_last: bool = True,
        seed: int = 42,
    ) -> None:
        self.dataset = dataset
        self.M       = M
        self.K       = K
        self._rng    = random.Random(seed)

        self._eligible_speakers: list[int] = [
            sid
            for sid, indices in dataset.speaker_to_indices.items()
            if len(indices) >= K
        ]

        dropped = dataset.n_speakers - len(self._eligible_speakers)
        if dropped:
            print(
                f"[BalancedBatchSampler] Dropped {dropped} speakers with < {K} utterances. "
                f"Eligible: {len(self._eligible_speakers)}"
            )

        if len(self._eligible_speakers) < M:
            raise ValueError(
                f"Need at least M={M} eligible speakers, found {len(self._eligible_speakers)}."
            )

        self._n_batches = len(self._eligible_speakers) // M

    def __len__(self) -> int:
        return self._n_batches

    def __iter__(self) -> Iterator[list[int]]:
        speakers = list(self._eligible_speakers)
        self._rng.shuffle(speakers)

        for batch_start in range(0, len(speakers) - self.M + 1, self.M):
            batch_speakers = speakers[batch_start : batch_start + self.M]
            batch_indices: list[int] = []
            for sid in batch_speakers:
                pool   = self.dataset.speaker_to_indices[sid]
                chosen = self._rng.sample(pool, self.K)
                batch_indices.extend(chosen)
            yield batch_indices
