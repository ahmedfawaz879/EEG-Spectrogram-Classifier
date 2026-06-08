"""Tests for spectrogram feature extraction."""

import numpy as np

from eeg_classifier.features.spectrogram import compute_mel_spectrogram


class TestMelSpectrogram:
    def test_output_shape(self):
        """Verify (n_channels, n_mels, time) shape."""
        segment = np.random.randn(2, 256 * 10).astype(np.float32)
        spec = compute_mel_spectrogram(segment, sr=256, n_mels=64, hop_length=128)
        assert spec.shape[0] == 2  # channels preserved
        assert spec.shape[1] == 64  # n_mels

    def test_single_channel(self):
        segment = np.random.randn(1, 256 * 10).astype(np.float32)
        spec = compute_mel_spectrogram(segment, sr=256, n_mels=32, hop_length=64)
        assert spec.shape[0] == 1
        assert spec.shape[1] == 32

    def test_output_is_finite(self):
        segment = np.random.randn(1, 256 * 5).astype(np.float32)
        spec = compute_mel_spectrogram(segment, sr=256, n_mels=64, hop_length=128)
        assert np.all(np.isfinite(spec))
