"""The numbers in the README must match results/metrics.json, and the metrics must match labels.json.

Guards against the README drifting from the measured results (a typo, or a
retrain whose numbers were not copied over).
"""

import json
import re

import pytest

from .conftest import RESULTS_DIR, ROOT, shipped_labels

README = (ROOT / "README.md").read_text(encoding="utf-8")
METRICS = json.loads((RESULTS_DIR / "metrics.json").read_text())
TEST = METRICS["test"]


def test_metrics_cover_exactly_the_shipped_labels():
    assert list(TEST["per_class"]) == shipped_labels()
    assert len(TEST["confusion_matrix"]) == len(shipped_labels())


def test_confusion_matrix_image_exists():
    assert (RESULTS_DIR / "confusion_matrix.png").read_bytes()[:4] == b"\x89PNG"


def test_headline_numbers_match():
    assert f"{100 * TEST['accuracy']:.1f} %" in README
    assert f"{TEST['macro_f1']:.3f}" in README
    assert f"{TEST['weighted_f1']:.3f}" in README
    assert f"{METRICS['data']['test_names']:,}" in README


@pytest.mark.parametrize("label", shipped_labels())
def test_per_class_table_row_matches(label):
    m = TEST["per_class"][label]
    names = METRICS["data"]["names_per_language"][label]
    row = (
        rf"\|\s*{label}\s*\|\s*{names:,}\s*\|\s*{m['support']}\s*\|\s*{m['precision']:.3f}\s*\|"
        rf"\s*{m['recall']:.3f}\s*\|\s*{m['f1']:.3f}\s*\|"
    )
    assert re.search(row, README), f"README row for {label} does not match metrics.json"
