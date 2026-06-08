"""Label assignment from seizure annotations.

Supports two annotation sources:
1. MNE annotations embedded in the EDF file.
2. An external CSV with columns ``filename, start_sec, end_sec``.
"""

from __future__ import annotations

import os


def load_annotations_csv(csv_path: str) -> dict[str, list[tuple[float, float]]]:
    """Parse seizure interval annotations from a CSV file.

    Expected format (header optional, ``#``-comments allowed)::

        filename, start_sec, end_sec
        chb01_01.edf, 2996, 3036
        chb01_03.edf, 2866, 2908

    Parameters
    ----------
    csv_path : str
        Path to the annotation CSV.

    Returns
    -------
    dict[str, list[tuple[float, float]]]
        Mapping from EDF **basename** to a list of ``(start, end)`` pairs
        in seconds.
    """
    annotations: dict[str, list[tuple[float, float]]] = {}
    if not os.path.exists(csv_path):
        return annotations

    with open(csv_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(",")
            if len(parts) < 3:
                continue
            filename = os.path.basename(parts[0].strip())
            start = float(parts[1])
            end = float(parts[2])
            annotations.setdefault(filename, []).append((start, end))

    return annotations


def label_window(
    t0: float,
    t1: float,
    annotations: list[tuple[float, float]],
) -> int:
    """Determine whether a time window overlaps any seizure annotation.

    Parameters
    ----------
    t0 : float
        Window start time in seconds.
    t1 : float
        Window end time in seconds.
    annotations : list[tuple[float, float]]
        Seizure intervals as ``(onset, offset)`` pairs.

    Returns
    -------
    int
        ``1`` if any overlap exists, ``0`` otherwise.
    """
    for start, end in annotations:
        if start < t1 and end > t0:
            return 1
    return 0
