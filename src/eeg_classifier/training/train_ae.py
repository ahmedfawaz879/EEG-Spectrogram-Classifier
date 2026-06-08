"""Autoencoder pretraining loop with MLflow experiment tracking."""

from __future__ import annotations

import argparse
import os

import mlflow
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from eeg_classifier.config import load_config
from eeg_classifier.models.autoencoder import ConvAutoencoder
from eeg_classifier.training.pipeline import build_dataloaders, build_records

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_autoencoder(
    model: ConvAutoencoder,
    train_loader: DataLoader,
    *,
    epochs: int = 10,
    lr: float = 1e-3,
    checkpoint_path: str | None = None,
    tracking_enabled: bool = True,
) -> ConvAutoencoder:
    """Train the autoencoder with MSE reconstruction loss.

    Parameters
    ----------
    model : ConvAutoencoder
        The autoencoder to train.
    train_loader : DataLoader
        Training data loader.
    epochs : int
        Number of training epochs.
    lr : float
        Learning rate.
    checkpoint_path : str | None
        Where to save the best model weights.
    tracking_enabled : bool
        Whether to log metrics to MLflow.

    Returns
    -------
    ConvAutoencoder
        The trained autoencoder.
    """
    model = model.to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        pbar = tqdm(train_loader, desc=f"AE Epoch {epoch + 1}/{epochs}")
        for x, _y in pbar:
            x = x.to(DEVICE)
            # Resize to fixed spatial dimensions expected by the encoder
            x = nn.functional.interpolate(x, size=(64, 64), mode="bilinear", align_corners=False)

            reconstruction, _embedding = model(x)
            # Match reconstruction to input spatial size
            reconstruction = nn.functional.interpolate(
                reconstruction, size=(64, 64), mode="bilinear", align_corners=False
            )
            loss = criterion(reconstruction, x)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = epoch_loss / max(n_batches, 1)
        print(f"  AE Epoch {epoch + 1} — avg loss: {avg_loss:.4f}")

        if tracking_enabled:
            mlflow.log_metric("ae_train_loss", avg_loss, step=epoch)

        if avg_loss < best_loss:
            best_loss = avg_loss
            if checkpoint_path:
                os.makedirs(os.path.dirname(checkpoint_path) or ".", exist_ok=True)
                torch.save(model.state_dict(), checkpoint_path)

    return model


def main() -> None:
    """CLI entry-point: ``eeg-train-ae``."""
    parser = argparse.ArgumentParser(description="Pretrain EEG autoencoder")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config override")
    args = parser.parse_args()

    cfg = load_config(args.config)

    records = build_records(cfg)
    loaders = build_dataloaders(records, cfg)

    in_channels = records[0][4].shape[0]
    ae = ConvAutoencoder(
        in_channels=in_channels,
        embedding_dim=cfg.autoencoder.embedding_dim,
    )

    tracking_enabled = cfg.tracking.enabled
    if tracking_enabled:
        mlflow.set_tracking_uri(cfg.tracking.tracking_uri)
        mlflow.set_experiment(cfg.tracking.experiment_name)
        mlflow.start_run(run_name="autoencoder-pretraining")
        mlflow.log_params(cfg.autoencoder.to_dict())

    try:
        train_autoencoder(
            ae,
            loaders["train"],
            epochs=cfg.autoencoder.epochs,
            lr=cfg.autoencoder.learning_rate,
            checkpoint_path=cfg.autoencoder.checkpoint,
            tracking_enabled=tracking_enabled,
        )
    finally:
        if tracking_enabled:
            mlflow.end_run()


if __name__ == "__main__":
    main()
