"""Convolutional Autoencoder for unsupervised EEG representation learning.

The encoder learns a compact embedding from mel-spectrogram images;
the decoder reconstructs the input as a self-supervised pretext task.
The trained encoder can then be plugged into a downstream classifier.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ConvAutoencoder(nn.Module):
    """Symmetric convolutional autoencoder.

    Architecture
    ------------
    Encoder: Conv2d → BN → ReLU  ×3  → Flatten → FC
    Decoder: FC → Unflatten → ConvTranspose2d → BN → ReLU  ×3

    Parameters
    ----------
    in_channels : int
        Number of input channels (= EEG channels per spectrogram).
    embedding_dim : int
        Size of the bottleneck latent vector.
    """

    def __init__(self, in_channels: int = 1, embedding_dim: int = 128) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim

        # ---- Encoder --------------------------------------------------------
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Flatten(),
        )

        # Projection to embedding space.
        # Input will be resized to (64, 64) so spatial dims after encoder = 8×8.
        self.fc_enc = nn.Linear(64 * 8 * 8, embedding_dim)

        # ---- Decoder --------------------------------------------------------
        self.fc_dec = nn.Linear(embedding_dim, 64 * 8 * 8)
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (64, 8, 8)),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, in_channels, kernel_size=4, stride=2, padding=1),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return the latent embedding for input *x*."""
        z = self.encoder(x)
        z = self.fc_enc(z)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Reconstruct from a latent embedding *z*."""
        out = self.fc_dec(z)
        out = self.decoder(out)
        return out

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Full forward pass.

        Returns
        -------
        tuple[Tensor, Tensor]
            ``(reconstruction, embedding)``
        """
        z = self.encode(x)
        reconstruction = self.decode(z)
        return reconstruction, z
