"""Single-file inference pipeline.

Usage::

    python -m eeg_classifier.inference.predict sample.edf --checkpoint runs/clf.pth
    eeg-predict sample.edf --checkpoint runs/clf.pth

Outputs JSON with per-window predictions.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch
import torch.nn as nn

from eeg_classifier.config import load_config
from eeg_classifier.data.edf_loader import read_raw_edf
from eeg_classifier.data.windowing import windows_from_raw
from eeg_classifier.features.spectrogram import compute_mel_spectrogram
from eeg_classifier.models.autoencoder import ConvAutoencoder
from eeg_classifier.models.classifier import SeizureClassifier

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def predict_edf(
    edf_path: str,
    checkpoint_path: str,
    cfg=None,
) -> list[dict]:
    """Run inference on an EDF file.

    Parameters
    ----------
    edf_path : str
        Path to the ``.edf`` file to classify.
    checkpoint_path : str
        Path to the saved classifier checkpoint.
    cfg : Config | None
        Configuration object.  Uses defaults if ``None``.

    Returns
    -------
    list[dict]
        Per-window predictions with keys:
        ``window_idx``, ``start_sec``, ``end_sec``,
        ``prediction``, ``confidence``.
    """
    if cfg is None:
        from eeg_classifier.config import load_config as _lc
        cfg = _lc()

    raw = read_raw_edf(edf_path, resample_sfreq=cfg.signal.sample_rate)
    windows = windows_from_raw(
        raw,
        window_sec=cfg.signal.window_sec,
        step_sec=cfg.signal.window_step_sec,
    )

    in_channels = raw.get_data().shape[0]
    ae = ConvAutoencoder(
        in_channels=in_channels,
        embedding_dim=cfg.autoencoder.embedding_dim,
    )
    clf = SeizureClassifier(
        autoencoder=ae,
        n_classes=2,
        dropout=cfg.classifier.dropout,
        freeze_encoder=False,
    )
    clf.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    clf = clf.to(DEVICE)
    clf.eval()

    results = []
    with torch.no_grad():
        for widx, (segment, t0, t1) in enumerate(windows):
            spec = compute_mel_spectrogram(
                segment,
                sr=cfg.signal.sample_rate,
                n_mels=cfg.features.n_mels,
                hop_length=cfg.features.hop_length,
                n_fft=cfg.features.n_fft,
                fmin=cfg.features.fmin,
                fmax=cfg.features.fmax,
            )
            spec = (spec - spec.mean()) / (spec.std() + 1e-8)
            x = torch.from_numpy(spec.astype(np.float32)).unsqueeze(0).to(DEVICE)
            x = nn.functional.interpolate(
                x, size=(64, 64), mode="bilinear", align_corners=False
            )

            logits = clf(x)
            probs = torch.softmax(logits, dim=1)[0]
            pred_idx = probs.argmax().item()
            confidence = probs[pred_idx].item()

            results.append({
                "window_idx": widx,
                "start_sec": round(t0, 2),
                "end_sec": round(t1, 2),
                "prediction": "seizure" if pred_idx == 1 else "non-seizure",
                "confidence": round(confidence, 4),
            })

    return results


def main() -> None:
    """CLI entry-point: ``eeg-predict``."""
    parser = argparse.ArgumentParser(description="Run seizure inference on an EDF file")
    parser.add_argument("edf_path", type=str, help="Path to the .edf file")
    parser.add_argument("--checkpoint", type=str, default="runs/clf.pth")
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config) if args.config else load_config()
    preds = predict_edf(args.edf_path, args.checkpoint, cfg)

    print(json.dumps(preds, indent=2))

    seizure_windows = [p for p in preds if p["prediction"] == "seizure"]
    print(f"\n{len(seizure_windows)}/{len(preds)} windows classified as seizure")


if __name__ == "__main__":
    main()
