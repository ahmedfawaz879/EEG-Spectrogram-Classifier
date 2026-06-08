"""Mel-spectrogram feature extraction from EEG segments.

Uses librosa for mel-spectrogram computation with EEG-appropriate
defaults (0.5–40 Hz bandpass via fmin/fmax).
"""

from __future__ import annotations

import librosa
import numpy as np


def compute_mel_spectrogram(
    segment: np.ndarray,
    sr: int = 256,
    n_mels: int = 64,
    hop_length: int = 128,
    n_fft: int = 512,
    fmin: float = 0.5,
    fmax: float = 40.0,
) -> np.ndarray:
    """Compute a multi-channel mel-spectrogram from an EEG segment.

    Parameters
    ----------
    segment : np.ndarray
        EEG window of shape ``(n_channels, n_samples)``.
    sr : int
        Sampling rate in Hz.
    n_mels : int
        Number of mel filter banks.
    hop_length : int
        Hop length for STFT.
    n_fft : int
        FFT window size.
    fmin : float
        Minimum frequency for the mel filterbank (Hz).
    fmax : float
        Maximum frequency for the mel filterbank (Hz).

    Returns
    -------
    np.ndarray
        Log-scaled mel-spectrogram of shape ``(n_channels, n_mels, time_frames)``.
    """
    specs = []
    for channel in segment:
        channel = channel.astype(np.float32)
        mel_spec = librosa.feature.melspectrogram(
            y=channel,
            sr=sr,
            n_mels=n_mels,
            hop_length=hop_length,
            n_fft=n_fft,
            fmin=fmin,
            fmax=fmax,
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        specs.append(mel_spec_db)

    return np.stack(specs, axis=0)  # (n_channels, n_mels, time_frames)
