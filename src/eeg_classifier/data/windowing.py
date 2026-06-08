"""EEG signal windowing – sliding-window segmentation of continuous recordings."""

from __future__ import annotations

import mne
import numpy as np


def windows_from_raw(
    raw: mne.io.BaseRaw,
    window_sec: float = 10,
    step_sec: float = 5,
) -> list[tuple[np.ndarray, float, float]]:
    """Segment a continuous MNE Raw object into overlapping windows.

    Parameters
    ----------
    raw : mne.io.BaseRaw
        Loaded EEG recording.
    window_sec : float
        Duration of each window in seconds.
    step_sec : float
        Stride between consecutive windows in seconds.

    Returns
    -------
    list[tuple[np.ndarray, float, float]]
        Each element is ``(segment, t_start, t_end)`` where *segment*
        has shape ``(n_channels, n_samples)``.
    """
    sfreq = int(raw.info["sfreq"])
    window_samples = int(window_sec * sfreq)
    step_samples = int(step_sec * sfreq)

    data = raw.get_data()  # (n_channels, n_total_samples)
    _, n_samples = data.shape

    windows: list[tuple[np.ndarray, float, float]] = []
    start = 0
    while start + window_samples <= n_samples:
        segment = data[:, start : start + window_samples]
        t0 = start / sfreq
        t1 = (start + window_samples) / sfreq
        windows.append((segment, t0, t1))
        start += step_samples

    return windows
