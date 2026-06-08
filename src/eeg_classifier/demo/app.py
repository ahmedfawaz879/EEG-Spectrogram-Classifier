"""Gradio demo – interactive EEG seizure detection.

Launch with::

    python -m eeg_classifier.demo.app
    # or
    gradio eeg_classifier/demo/app.py

Upload an EDF file to visualise:
- Raw EEG signal
- Mel spectrogram
- Per-window seizure prediction with confidence
"""

from __future__ import annotations

import os

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np

from eeg_classifier.config import load_config
from eeg_classifier.data.edf_loader import read_raw_edf
from eeg_classifier.data.windowing import windows_from_raw
from eeg_classifier.features.spectrogram import compute_mel_spectrogram


def _plot_raw_eeg(raw, max_channels: int = 4, max_sec: float = 30) -> plt.Figure:
    """Plot first few channels of raw EEG."""
    data = raw.get_data()
    sfreq = int(raw.info["sfreq"])
    n_ch = min(data.shape[0], max_channels)
    n_samp = min(data.shape[1], int(max_sec * sfreq))
    t = np.arange(n_samp) / sfreq

    fig, axes = plt.subplots(n_ch, 1, figsize=(12, 2 * n_ch), sharex=True)
    if n_ch == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        ax.plot(t, data[i, :n_samp], linewidth=0.5, color="#2563eb")
        ax.set_ylabel(f"Ch {i + 1}")
        ax.set_xlim(0, t[-1])
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Raw EEG Signal", fontsize=14, fontweight="bold")
    fig.tight_layout()
    return fig


def _plot_spectrogram(spec: np.ndarray) -> plt.Figure:
    """Plot the first channel mel-spectrogram."""
    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(
        spec[0], aspect="auto", origin="lower", cmap="magma", interpolation="nearest"
    )
    ax.set_ylabel("Mel Bin")
    ax.set_xlabel("Time Frame")
    ax.set_title("Mel Spectrogram (Channel 1)", fontsize=14, fontweight="bold")
    fig.colorbar(im, ax=ax, label="dB")
    fig.tight_layout()
    return fig


def analyse_edf(edf_file, checkpoint_path: str | None = None):
    """Main Gradio handler."""
    if edf_file is None:
        return None, None, "Please upload an EDF file."

    cfg = load_config()
    raw = read_raw_edf(edf_file.name, resample_sfreq=cfg.signal.sample_rate)
    windows = windows_from_raw(
        raw, window_sec=cfg.signal.window_sec, step_sec=cfg.signal.window_step_sec
    )

    # Plot raw EEG
    raw_fig = _plot_raw_eeg(raw)

    # Compute and plot spectrogram of first window
    if windows:
        spec = compute_mel_spectrogram(
            windows[0][0],
            sr=cfg.signal.sample_rate,
            n_mels=cfg.features.n_mels,
            hop_length=cfg.features.hop_length,
        )
        spec_fig = _plot_spectrogram(spec)
    else:
        spec_fig = None

    # If a checkpoint is provided, run inference
    summary = f"**Recording:** {os.path.basename(edf_file.name)}\n"
    summary += f"**Channels:** {raw.get_data().shape[0]}\n"
    summary += f"**Duration:** {raw.get_data().shape[1] / raw.info['sfreq']:.1f}s\n"
    summary += f"**Windows:** {len(windows)}\n\n"

    if checkpoint_path and os.path.exists(checkpoint_path):
        from eeg_classifier.inference.predict import predict_edf

        preds = predict_edf(edf_file.name, checkpoint_path, cfg)
        seizure_count = sum(1 for p in preds if p["prediction"] == "seizure")
        summary += "### Predictions\n"
        summary += f"**Seizure windows:** {seizure_count}/{len(preds)}\n\n"
        for p in preds[:10]:
            emoji = "🔴" if p["prediction"] == "seizure" else "🟢"
            summary += (
                f"{emoji} Window {p['window_idx']} "
                f"({p['start_sec']}s–{p['end_sec']}s): "
                f"**{p['prediction']}** ({p['confidence']:.1%})\n"
            )
        if len(preds) > 10:
            summary += f"\n_... and {len(preds) - 10} more windows_"
    else:
        summary += "_No checkpoint loaded — showing signal analysis only._"

    return raw_fig, spec_fig, summary


def create_app() -> gr.Blocks:
    """Build and return the Gradio app."""
    with gr.Blocks(
        title="EEG Seizure Detector",
        theme=gr.themes.Soft(primary_hue="blue"),
    ) as app:
        gr.Markdown(
            "# 🧠 EEG Seizure Detection\n"
            "Upload an EDF recording to visualise the raw signal, "
            "mel-spectrogram, and seizure predictions."
        )

        with gr.Row():
            edf_input = gr.File(label="Upload EDF File", file_types=[".edf"])
            ckpt_input = gr.Textbox(
                label="Checkpoint path (optional)",
                placeholder="runs/clf.pth",
                value="runs/clf.pth",
            )

        btn = gr.Button("Analyse", variant="primary")

        with gr.Row():
            raw_plot = gr.Plot(label="Raw EEG")
            spec_plot = gr.Plot(label="Mel Spectrogram")

        output_md = gr.Markdown(label="Results")

        btn.click(
            fn=analyse_edf,
            inputs=[edf_input, ckpt_input],
            outputs=[raw_plot, spec_plot, output_md],
        )

    return app


if __name__ == "__main__":
    app = create_app()
    app.launch()
