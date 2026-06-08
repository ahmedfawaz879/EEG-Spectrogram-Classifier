"""End-to-end training script: autoencoder pretraining → classifier training → evaluation.

Usage
-----
::

    # With default config
    python -m eeg_classifier.training.train

    # With custom config
    python -m eeg_classifier.training.train --config configs/custom.yaml

    # Via the installed CLI entry-point
    eeg-train --config configs/custom.yaml
"""

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
from eeg_classifier.evaluation.metrics import evaluate_model, log_metrics_to_mlflow
from eeg_classifier.models.autoencoder import ConvAutoencoder
from eeg_classifier.models.classifier import SeizureClassifier
from eeg_classifier.training.pipeline import build_dataloaders, build_records
from eeg_classifier.training.train_ae import train_autoencoder

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_classifier(
    model: SeizureClassifier,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    epochs: int = 20,
    lr: float = 1e-4,
    checkpoint_path: str | None = None,
    tracking_enabled: bool = True,
) -> SeizureClassifier:
    """Train the seizure classifier.

    Parameters
    ----------
    model : SeizureClassifier
        Classifier model to train.
    train_loader : DataLoader
        Training data.
    val_loader : DataLoader
        Validation data.
    epochs : int
        Number of training epochs.
    lr : float
        Learning rate.
    checkpoint_path : str | None
        Path to save the best model weights.
    tracking_enabled : bool
        Whether to log to MLflow.

    Returns
    -------
    SeizureClassifier
        The trained classifier.
    """
    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr
    )

    best_val_loss = float("inf")

    for epoch in range(epochs):
        # ---- Training -------------------------------------------------------
        model.train()
        train_loss = 0.0
        n_batches = 0

        pbar = tqdm(train_loader, desc=f"CLF Epoch {epoch + 1}/{epochs}")
        for x, y in pbar:
            x = x.to(DEVICE)
            x = nn.functional.interpolate(x, size=(64, 64), mode="bilinear", align_corners=False)
            y = y.to(DEVICE)

            logits = model(x)
            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            n_batches += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_train_loss = train_loss / max(n_batches, 1)

        # ---- Validation -----------------------------------------------------
        model.eval()
        val_loss = 0.0
        val_batches = 0

        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(DEVICE)
                x = nn.functional.interpolate(
                    x, size=(64, 64), mode="bilinear", align_corners=False
                )
                y = y.to(DEVICE)
                logits = model(x)
                loss = criterion(logits, y)
                val_loss += loss.item()
                val_batches += 1

        avg_val_loss = val_loss / max(val_batches, 1)
        print(
            f"  Epoch {epoch + 1} — train_loss: {avg_train_loss:.4f}, "
            f"val_loss: {avg_val_loss:.4f}"
        )

        if tracking_enabled:
            mlflow.log_metrics(
                {
                    "clf_train_loss": avg_train_loss,
                    "clf_val_loss": avg_val_loss,
                },
                step=epoch,
            )

        # Checkpoint best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            if checkpoint_path:
                os.makedirs(os.path.dirname(checkpoint_path) or ".", exist_ok=True)
                torch.save(model.state_dict(), checkpoint_path)

    return model


def main() -> None:
    """CLI entry-point: ``eeg-train``."""
    parser = argparse.ArgumentParser(description="Full EEG seizure detection training pipeline")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config override")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # ---- Data ---------------------------------------------------------------
    records = build_records(cfg)
    loaders = build_dataloaders(records, cfg)

    in_channels = records[0][4].shape[0]

    # ---- MLflow -------------------------------------------------------------
    tracking = cfg.tracking.enabled
    if tracking:
        mlflow.set_tracking_uri(cfg.tracking.tracking_uri)
        mlflow.set_experiment(cfg.tracking.experiment_name)
        mlflow.start_run(run_name="full-pipeline")
        # Log all config
        flat = cfg.to_dict()
        mlflow.log_dict(flat, "config.yaml")

    try:
        # ---- Autoencoder pretraining ----------------------------------------
        print("\n" + "=" * 60)
        print("Phase 1: Autoencoder Pretraining")
        print("=" * 60)
        ae = ConvAutoencoder(
            in_channels=in_channels,
            embedding_dim=cfg.autoencoder.embedding_dim,
        )
        ae = train_autoencoder(
            ae,
            loaders["train"],
            epochs=cfg.autoencoder.epochs,
            lr=cfg.autoencoder.learning_rate,
            checkpoint_path=cfg.autoencoder.checkpoint,
            tracking_enabled=tracking,
        )

        # ---- Classifier training --------------------------------------------
        print("\n" + "=" * 60)
        print("Phase 2: Classifier Training")
        print("=" * 60)
        clf = SeizureClassifier(
            autoencoder=ae,
            n_classes=2,
            dropout=cfg.classifier.dropout,
            freeze_encoder=cfg.classifier.freeze_encoder,
        )
        clf = train_classifier(
            clf,
            loaders["train"],
            loaders["val"],
            epochs=cfg.classifier.epochs,
            lr=cfg.classifier.learning_rate,
            checkpoint_path=cfg.classifier.checkpoint,
            tracking_enabled=tracking,
        )

        # ---- Evaluation on test set -----------------------------------------
        print("\n" + "=" * 60)
        print("Phase 3: Test Set Evaluation")
        print("=" * 60)
        results = evaluate_model(clf, loaders["test"], device=DEVICE)
        if tracking:
            log_metrics_to_mlflow(results)
            plots_dir = cfg.evaluation.get("plots_dir", "runs/plots")
            os.makedirs(plots_dir, exist_ok=True)
            if results.get("confusion_matrix") is not None:
                from eeg_classifier.evaluation.metrics import plot_confusion_matrix

                cm_path = os.path.join(plots_dir, "confusion_matrix.png")
                plot_confusion_matrix(results["confusion_matrix"], save_path=cm_path)
                mlflow.log_artifact(cm_path)

    finally:
        if tracking:
            mlflow.end_run()


if __name__ == "__main__":
    main()
