"""Model evaluation metrics and visualisation utilities.

Computes standard classification metrics (accuracy, F1, precision, recall,
ROC-AUC) and generates confusion-matrix plots.
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader


def evaluate_model(
    model: nn.Module,
    data_loader: DataLoader,
    *,
    device: torch.device | str = "cpu",
) -> dict[str, Any]:
    """Evaluate *model* on the given DataLoader.

    Parameters
    ----------
    model : nn.Module
        Trained classifier.
    data_loader : DataLoader
        Evaluation data.
    device : torch.device | str
        Device to run inference on.

    Returns
    -------
    dict[str, Any]
        Keys: ``accuracy``, ``f1``, ``precision``, ``recall``, ``roc_auc``,
        ``confusion_matrix``, ``classification_report``.
    """
    model = model.to(device)
    model.eval()

    all_labels: list[int] = []
    all_preds: list[int] = []
    all_probs: list[float] = []

    with torch.no_grad():
        for x, y in data_loader:
            x = x.to(device)
            x = nn.functional.interpolate(x, size=(64, 64), mode="bilinear", align_corners=False)

            logits = model(x)
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            preds = logits.argmax(dim=1).cpu().numpy()

            all_labels.extend(y.numpy().tolist())
            all_preds.extend(preds.tolist())
            all_probs.extend(probs.tolist())

    results: dict[str, Any] = {
        "accuracy": accuracy_score(all_labels, all_preds),
        "f1": f1_score(all_labels, all_preds, zero_division=0),
        "precision": precision_score(all_labels, all_preds, zero_division=0),
        "recall": recall_score(all_labels, all_preds, zero_division=0),
        "confusion_matrix": confusion_matrix(all_labels, all_preds),
        "classification_report": classification_report(all_labels, all_preds),
    }

    try:
        results["roc_auc"] = roc_auc_score(all_labels, all_probs)
    except ValueError:
        results["roc_auc"] = None

    # Print summary
    print(f"\n{'─' * 50}")
    print(f"  Accuracy  : {results['accuracy']:.4f}")
    print(f"  F1 Score  : {results['f1']:.4f}")
    print(f"  Precision : {results['precision']:.4f}")
    print(f"  Recall    : {results['recall']:.4f}")
    print(f"  ROC-AUC   : {results['roc_auc']}")
    print(f"{'─' * 50}")
    print(results["classification_report"])

    return results


def log_metrics_to_mlflow(results: dict[str, Any]) -> None:
    """Log evaluation results to the active MLflow run.

    Parameters
    ----------
    results : dict[str, Any]
        Output of :func:`evaluate_model`.
    """
    numeric_keys = ["accuracy", "f1", "precision", "recall", "roc_auc"]
    for key in numeric_keys:
        value = results.get(key)
        if value is not None:
            mlflow.log_metric(f"test_{key}", value)


def plot_confusion_matrix(
    cm: np.ndarray,
    labels: list[str] | None = None,
    save_path: str | None = None,
) -> None:
    """Plot and optionally save a confusion matrix.

    Parameters
    ----------
    cm : np.ndarray
        Confusion matrix from sklearn.
    labels : list[str] | None
        Class labels. Defaults to ``["Non-Seizure", "Seizure"]``.
    save_path : str | None
        If provided, save figure to this path.
    """
    if labels is None:
        labels = ["Non-Seizure", "Seizure"]

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=labels,
        yticklabels=labels,
        ylabel="True label",
        xlabel="Predicted label",
        title="Confusion Matrix",
    )

    # Annotate cells
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"Confusion matrix saved to {save_path}")
    plt.close(fig)
