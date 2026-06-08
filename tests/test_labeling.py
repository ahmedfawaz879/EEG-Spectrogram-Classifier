"""Tests for labeling utilities."""

from eeg_classifier.data.labeling import label_window


class TestLabelWindow:
    def test_overlap(self):
        assert label_window(5.0, 15.0, [(10.0, 20.0)]) == 1

    def test_no_overlap(self):
        assert label_window(0.0, 5.0, [(10.0, 20.0)]) == 0

    def test_exact_boundary(self):
        assert label_window(10.0, 20.0, [(10.0, 20.0)]) == 1

    def test_empty_annotations(self):
        assert label_window(0.0, 10.0, []) == 0

    def test_partial_overlap(self):
        assert label_window(15.0, 25.0, [(10.0, 20.0)]) == 1

    def test_multiple_annotations(self):
        assert label_window(5.0, 8.0, [(1.0, 3.0), (7.0, 10.0)]) == 1
