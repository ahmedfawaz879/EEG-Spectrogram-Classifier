"""Tests for the PyTorch Dataset."""

import numpy as np
import torch

from eeg_classifier.data.dataset import EEGSpectrogramDataset


class TestEEGSpectrogramDataset:
    def _make_records(self, n=5, channels=2, n_mels=64, t_frames=20):
        records = []
        for i in range(n):
            spec = np.random.randn(channels, n_mels, t_frames)
            records.append((f"file_{i}.edf", f"subj_{i}", i % 2, i, spec))
        return records

    def test_length(self):
        ds = EEGSpectrogramDataset(self._make_records(10))
        assert len(ds) == 10

    def test_item_types(self):
        ds = EEGSpectrogramDataset(self._make_records(3))
        x, y = ds[0]
        assert isinstance(x, torch.Tensor)
        assert isinstance(y, torch.Tensor)
        assert x.dtype == torch.float32
        assert y.dtype == torch.long

    def test_normalization(self):
        """Dataset should z-score normalize each spectrogram."""
        ds = EEGSpectrogramDataset(self._make_records(1))
        x, _ = ds[0]
        assert abs(x.mean().item()) < 0.1  # approximately zero-mean
