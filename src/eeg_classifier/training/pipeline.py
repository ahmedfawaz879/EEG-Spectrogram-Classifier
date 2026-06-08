"""Data pipeline – build records and DataLoaders from config.

Implements subject-aware splitting using :class:`GroupShuffleSplit` to
prevent data leakage across train / val / test sets.
"""

from __future__ import annotations

import os

import numpy as np
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from torch.utils.data import DataLoader

from eeg_classifier.config import Config
from eeg_classifier.data.dataset import EEGSpectrogramDataset
from eeg_classifier.data.edf_loader import extract_subject_id, find_edf_files, read_raw_edf
from eeg_classifier.data.labeling import label_window, load_annotations_csv
from eeg_classifier.data.windowing import windows_from_raw
from eeg_classifier.features.spectrogram import compute_mel_spectrogram

# Record schema: (basename, subject_id, label, window_idx, spectrogram_array)
Record = tuple[str, str, int, int, np.ndarray]


def build_records(cfg: Config) -> list[Record]:
    """Build spectrogram records from EDF files according to *cfg*.

    Parameters
    ----------
    cfg : Config
        Full configuration object.

    Returns
    -------
    list[Record]
        Each element is ``(basename, subject_id, label, window_idx, spec)``.
    """
    files = find_edf_files(cfg.data.data_dir)
    if not files:
        raise RuntimeError(f"No EDF files found in {cfg.data.data_dir}")
    print(f"Found {len(files)} EDF files")

    ann_map = (
        load_annotations_csv(cfg.data.ann_csv) if cfg.data.ann_csv else {}
    )

    max_files = cfg.data.max_files
    records: list[Record] = []

    for i, path in enumerate(files):
        if max_files is not None and i >= max_files:
            break

        basename = os.path.basename(path)
        subject_id = extract_subject_id(path)
        print(f"  [{i + 1}] Reading {basename} (subject={subject_id})")

        try:
            raw = read_raw_edf(
                path,
                picks=cfg.data.picks,
                resample_sfreq=cfg.signal.sample_rate,
            )
        except Exception as exc:
            print(f"    ⚠ Failed to read {path}: {exc}")
            continue

        windows = windows_from_raw(
            raw,
            window_sec=cfg.signal.window_sec,
            step_sec=cfg.signal.window_step_sec,
        )

        # Annotations – prefer MNE-embedded, fall back to CSV
        annotations = ann_map.get(basename, [])
        if (
            hasattr(raw, "annotations")
            and raw.annotations is not None
            and len(raw.annotations) > 0
        ):
            annotations = [
                (ann["onset"], ann["onset"] + ann["duration"])
                for ann in raw.annotations
            ]

        for widx, (segment, t0, t1) in enumerate(windows):
            lbl = label_window(t0, t1, annotations) if annotations else 0
            spec = compute_mel_spectrogram(
                segment,
                sr=cfg.signal.sample_rate,
                n_mels=cfg.features.n_mels,
                hop_length=cfg.features.hop_length,
                n_fft=cfg.features.n_fft,
                fmin=cfg.features.fmin,
                fmax=cfg.features.fmax,
            )
            records.append((basename, subject_id, lbl, widx, spec))

    print(f"Built {len(records)} windows")
    return records


def _subject_aware_split(
    records: list[Record],
    test_size: float,
    random_seed: int,
) -> tuple[list[Record], list[Record]]:
    """Split records ensuring no subject appears in both sets."""
    subjects = np.array([r[1] for r in records])
    indices = np.arange(len(records))

    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_seed)
    train_idx, test_idx = next(splitter.split(indices, groups=subjects))

    train_recs = [records[i] for i in train_idx]
    test_recs = [records[i] for i in test_idx]
    return train_recs, test_recs


def build_dataloaders(
    records: list[Record],
    cfg: Config,
) -> dict[str, DataLoader]:
    """Split records and wrap in DataLoaders.

    Uses subject-aware splitting when ``cfg.data.subject_aware_split``
    is ``True`` (default) to prevent data leakage.

    Parameters
    ----------
    records : list[Record]
        Output of :func:`build_records`.
    cfg : Config
        Configuration object.

    Returns
    -------
    dict[str, DataLoader]
        Keys: ``"train"``, ``"val"``, ``"test"``.
    """
    seed = cfg.data.random_seed
    use_group_split = cfg.data.subject_aware_split

    if use_group_split:
        print("Using subject-aware (GroupShuffleSplit) to prevent data leakage")
        train_val_recs, test_recs = _subject_aware_split(
            records, test_size=cfg.data.test_size, random_seed=seed
        )
        train_recs, val_recs = _subject_aware_split(
            train_val_recs, test_size=cfg.data.val_size, random_seed=seed
        )
    else:
        labels = [r[2] for r in records]
        train_val_recs, test_recs = train_test_split(
            records, test_size=cfg.data.test_size, random_state=seed, stratify=labels
        )
        train_labels = [r[2] for r in train_val_recs]
        train_recs, val_recs = train_test_split(
            train_val_recs, test_size=cfg.data.val_size, random_state=seed, stratify=train_labels
        )

    batch_size = cfg.classifier.batch_size

    loaders: dict[str, DataLoader] = {
        "train": DataLoader(
            EEGSpectrogramDataset(train_recs),
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
        ),
        "val": DataLoader(
            EEGSpectrogramDataset(val_recs),
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        ),
        "test": DataLoader(
            EEGSpectrogramDataset(test_recs),
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        ),
    }

    print(
        f"Split → train={len(train_recs)}, val={len(val_recs)}, test={len(test_recs)}"
    )
    return loaders
