"""PyTorch Dataset for EEG spectrogram windows."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class EEGSpectrogramDataset(Dataset):
    """Map-style dataset wrapping pre-computed spectrogram records.

    Parameters
    ----------
    records : list[tuple[str, str, int, int, np.ndarray]]
        Each element is ``(basename, subject_id, label, window_idx, spec)``
        where *spec* has shape ``(n_channels, n_mels, time_frames)``.
    """

    def __init__(self, records: list[tuple[str, str, int, int, np.ndarray]]) -> None:
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        _basename, _subject_id, label, _widx, spec = self.records[idx]

        # Per-example z-score normalisation
        spec = (spec - spec.mean()) / (spec.std() + 1e-8)
        spec = spec.astype(np.float32)

        x = torch.from_numpy(spec)
        y = torch.tensor(label, dtype=torch.long)
        return x, y
