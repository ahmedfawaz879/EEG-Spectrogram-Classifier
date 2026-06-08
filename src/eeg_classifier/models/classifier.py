"""CNN classifier head for seizure detection.

Uses the encoder from a pretrained :class:`ConvAutoencoder` as the
feature backbone, followed by fully-connected layers for binary
classification (seizure / non-seizure).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from eeg_classifier.models.autoencoder import ConvAutoencoder


class _EncoderWrapper(nn.Module):
    """Thin wrapper that exposes only the *encode* path of an autoencoder."""

    def __init__(self, autoencoder: ConvAutoencoder) -> None:
        super().__init__()
        self.encoder = autoencoder.encoder
        self.fc_enc = autoencoder.fc_enc

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        z = self.fc_enc(z)
        return z


class SeizureClassifier(nn.Module):
    """Encoder + FC classification head.

    Parameters
    ----------
    autoencoder : ConvAutoencoder
        A (pre-)trained autoencoder whose encoder will be reused.
    n_classes : int
        Number of target classes (default ``2``: seizure / non-seizure).
    dropout : float
        Dropout probability in the classifier head.
    freeze_encoder : bool
        If ``True``, encoder weights are frozen during classifier training.
    """

    def __init__(
        self,
        autoencoder: ConvAutoencoder,
        n_classes: int = 2,
        dropout: float = 0.4,
        freeze_encoder: bool = True,
    ) -> None:
        super().__init__()
        self.encoder = _EncoderWrapper(autoencoder)
        self.embedding_dim = autoencoder.embedding_dim

        if freeze_encoder:
            for param in self.encoder.encoder.parameters():
                param.requires_grad = False
            # Keep fc_enc trainable for fine-tuning the projection
            for param in self.encoder.fc_enc.parameters():
                param.requires_grad = True

        self.classifier = nn.Sequential(
            nn.Linear(self.embedding_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return class logits for input *x*."""
        z = self.encoder(x)
        return self.classifier(z)
