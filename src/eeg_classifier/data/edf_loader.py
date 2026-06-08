"""EDF file discovery and loading utilities.

Supports CHB-MIT and TUH directory layouts.  Uses MNE-Python for robust
EDF parsing with optional channel selection and resampling.
"""

from __future__ import annotations

import glob
import os
import re
from pathlib import Path

import mne


def find_edf_files(data_dir: str | Path) -> list[str]:
    """Recursively discover all ``*.edf`` files under *data_dir*.

    Parameters
    ----------
    data_dir : str | Path
        Root directory to search.

    Returns
    -------
    list[str]
        Sorted, deduplicated list of absolute paths.
    """
    data_dir = str(data_dir)
    patterns = [
        os.path.join(data_dir, "**", "*.edf"),
        os.path.join(data_dir, "**", "*.EDF"),
    ]
    files: list[str] = []
    for p in patterns:
        files.extend(glob.glob(p, recursive=True))
    return sorted(set(files))


def extract_subject_id(filepath: str) -> str:
    """Heuristically extract a subject / patient identifier from the path.

    CHB-MIT layout: ``chb01/chb01_01.edf`` → ``chb01``
    TUH layout:     ``00000258/s001_2003_07_21/…`` → ``00000258``
    Fallback:       basename without extension.

    Parameters
    ----------
    filepath : str
        Path to an EDF file.

    Returns
    -------
    str
        Subject identifier string.
    """
    parts = Path(filepath).parts
    # CHB-MIT: look for ``chbXX`` folder
    for part in parts:
        if re.match(r"^chb\d+$", part, re.IGNORECASE):
            return part.lower()
    # TUH: look for a purely numeric folder
    for part in parts:
        if re.match(r"^\d{5,}$", part):
            return part
    # Fallback
    return Path(filepath).stem


def read_raw_edf(
    path: str | Path,
    picks: list[str] | None = None,
    resample_sfreq: int | None = 256,
) -> mne.io.BaseRaw:
    """Read a single EDF file, optionally selecting channels and resampling.

    Parameters
    ----------
    path : str | Path
        Path to the ``.edf`` file.
    picks : list[str] | None
        Channel names to keep.  ``None`` keeps all channels.
    resample_sfreq : int | None
        Target sampling frequency in Hz.  ``None`` skips resampling.

    Returns
    -------
    mne.io.BaseRaw
        Loaded (and optionally resampled) MNE Raw object.
    """
    raw = mne.io.read_raw_edf(str(path), preload=True, verbose=False)

    if picks is not None:
        picks_present = [ch for ch in picks if ch in raw.ch_names]
        if picks_present:
            raw.pick_channels(picks_present)

    if resample_sfreq is not None and raw.info["sfreq"] != resample_sfreq:
        raw.resample(resample_sfreq)

    return raw
