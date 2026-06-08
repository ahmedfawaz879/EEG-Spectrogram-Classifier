"""Tests for Config loading."""

from eeg_classifier.config import Config, load_config


class TestConfig:
    def test_load_default(self):
        cfg = load_config()
        assert cfg.signal.sample_rate == 256
        assert cfg.features.n_mels == 64

    def test_attribute_access(self):
        cfg = Config({"a": {"b": 1}})
        assert cfg.a.b == 1

    def test_to_dict(self):
        cfg = Config({"x": {"y": 2}})
        d = cfg.to_dict()
        assert d == {"x": {"y": 2}}
