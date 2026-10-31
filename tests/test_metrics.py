"""Accuracy / precision / recall / F1 and the confusion-matrix plot."""

import pytest

from surname_classifier.metrics import classification_report, confusion_matrix, plot_confusion_matrix


def test_confusion_matrix_counts_pairs():
    assert confusion_matrix([0, 0, 1, 2, 2], [0, 1, 1, 2, 0], 3) == [[1, 1, 0], [0, 1, 0], [1, 0, 1]]


def test_report_matches_hand_computation():
    # class a: tp=2, fp=1, fn=1 -> p=2/3 r=2/3 f1=2/3
    # class b: tp=1, fp=1, fn=0 -> p=1/2 r=1   f1=2/3
    # class c: tp=0, fp=0, fn=1 -> p=0   r=0   f1=0
    y_true = [0, 0, 0, 1, 2]
    y_pred = [0, 0, 1, 1, 0]
    r = classification_report(y_true, y_pred, ["a", "b", "c"])
    assert r["accuracy"] == pytest.approx(3 / 5)
    assert r["per_class"]["a"] == pytest.approx({"precision": 2 / 3, "recall": 2 / 3, "f1": 2 / 3, "support": 3})
    assert r["per_class"]["b"] == pytest.approx({"precision": 0.5, "recall": 1.0, "f1": 2 / 3, "support": 1})
    assert r["per_class"]["c"] == {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 1}
    assert r["macro_f1"] == pytest.approx((2 / 3 + 2 / 3 + 0) / 3)
    assert r["weighted_f1"] == pytest.approx((3 * 2 / 3 + 1 * 2 / 3) / 5)


def test_majority_class_baseline_has_high_accuracy_but_low_macro_f1():
    # Why macro F1 is reported: always predicting the big class looks good on accuracy only.
    y_true = [0] * 90 + [1] * 5 + [2] * 5
    r = classification_report(y_true, [0] * 100, ["big", "x", "y"])
    assert r["accuracy"] == pytest.approx(0.9)
    assert r["macro_f1"] < 0.35


def test_perfect_predictions():
    r = classification_report([0, 1, 2], [0, 1, 2], ["a", "b", "c"])
    assert r["accuracy"] == r["macro_f1"] == r["weighted_f1"] == 1.0


def test_plot_writes_a_png(tmp_path):
    pytest.importorskip("matplotlib")
    path = tmp_path / "cm.png"
    plot_confusion_matrix([[3, 1], [0, 0]], ["a", "b"], str(path))  # includes an empty row
    assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
