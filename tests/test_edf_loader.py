"""Tests for EDF loader utilities."""

from eeg_classifier.data.edf_loader import extract_subject_id


class TestExtractSubjectId:
    def test_chbmit_path(self):
        path = "data/chb-mit/chb01/chb01_03.edf"
        assert extract_subject_id(path) == "chb01"

    def test_tuh_path(self):
        path = "data/tuh/00000258/s001_2003/file.edf"
        assert extract_subject_id(path) == "00000258"

    def test_fallback(self):
        path = "data/unknown/recording.edf"
        assert extract_subject_id(path) == "recording"
