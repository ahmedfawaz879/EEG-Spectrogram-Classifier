"""Tests for model architectures – verify shapes and forward-pass consistency."""

import torch

from eeg_classifier.models.autoencoder import ConvAutoencoder
from eeg_classifier.models.classifier import SeizureClassifier


class TestConvAutoencoder:
    def test_forward_shape(self):
        model = ConvAutoencoder(in_channels=1, embedding_dim=128)
        x = torch.randn(4, 1, 64, 64)
        recon, z = model(x)
        assert z.shape == (4, 128)
        assert recon.shape[0] == 4

    def test_encode_decode(self):
        model = ConvAutoencoder(in_channels=2, embedding_dim=64)
        x = torch.randn(2, 2, 64, 64)
        z = model.encode(x)
        assert z.shape == (2, 64)
        recon = model.decode(z)
        assert recon.shape[0] == 2

    def test_multichannel(self):
        model = ConvAutoencoder(in_channels=4, embedding_dim=256)
        x = torch.randn(1, 4, 64, 64)
        recon, z = model(x)
        assert z.shape == (1, 256)


class TestSeizureClassifier:
    def test_forward_shape(self):
        ae = ConvAutoencoder(in_channels=1, embedding_dim=128)
        clf = SeizureClassifier(ae, n_classes=2)
        x = torch.randn(4, 1, 64, 64)
        logits = clf(x)
        assert logits.shape == (4, 2)

    def test_frozen_encoder(self):
        ae = ConvAutoencoder(in_channels=1, embedding_dim=128)
        clf = SeizureClassifier(ae, freeze_encoder=True)
        frozen = sum(1 for p in clf.encoder.encoder.parameters() if not p.requires_grad)
        assert frozen > 0  # at least some params frozen

    def test_unfrozen_encoder(self):
        ae = ConvAutoencoder(in_channels=1, embedding_dim=128)
        clf = SeizureClassifier(ae, freeze_encoder=False)
        all_trainable = all(p.requires_grad for p in clf.parameters())
        assert all_trainable
