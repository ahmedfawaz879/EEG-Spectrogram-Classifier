"""Tests for the windowing module."""

import numpy as np

from eeg_classifier.data.windowing import windows_from_raw


class _FakeRaw:
    """Minimal MNE Raw mock for testing."""

    def __init__(self, n_channels: int, n_samples: int, sfreq: int = 256):
        self._data = np.random.randn(n_channels, n_samples)
        self.info = {"sfreq": sfreq}

    def get_data(self) -> np.ndarray:
        return self._data


class TestWindowsFromRaw:
    def test_basic_windowing(self):
        """10s windows with 5s stride at 256 Hz on a 30s recording."""
        raw = _FakeRaw(n_channels=2, n_samples=256 * 30, sfreq=256)
        windows = windows_from_raw(raw, window_sec=10, step_sec=5)
        # 30s recording, 10s window, 5s step → windows at 0,5,10,15,20 = 5
        assert len(windows) == 5

    def test_window_shape(self):
        raw = _FakeRaw(n_channels=4, n_samples=256 * 10, sfreq=256)
        windows = windows_from_raw(raw, window_sec=10, step_sec=10)
        assert len(windows) == 1
        segment, t0, t1 = windows[0]
        assert segment.shape == (4, 256 * 10)
        assert t0 == 0.0
        assert t1 == 10.0

    def test_no_windows_if_too_short(self):
        raw = _FakeRaw(n_channels=1, n_samples=256 * 5, sfreq=256)
        windows = windows_from_raw(raw, window_sec=10, step_sec=5)
        assert len(windows) == 0

    def test_time_stamps(self):
        raw = _FakeRaw(n_channels=1, n_samples=256 * 20, sfreq=256)
        windows = windows_from_raw(raw, window_sec=10, step_sec=5)
        assert windows[0][1] == 0.0
        assert windows[0][2] == 10.0
        assert windows[1][1] == 5.0
        assert windows[1][2] == 15.0
