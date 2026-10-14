"""Model loading (with and without labels.json) and prediction."""
import json
import shutil

import pytest

from surname_classifier import inference
from surname_classifier.preprocessing import LABELS

from .conftest import MODELS_DIR


@pytest.fixture(scope="module")
def shipped():
    return inference.load_model(str(MODELS_DIR))


def test_shipped_model_falls_back_to_builtin_labels(shipped):
    _, labels = shipped
    assert labels == LABELS  # no labels.json is committed for the shipped model


def test_labels_json_overrides_builtin_order(tmp_path):
    shutil.copy(MODELS_DIR / inference.SCRIPTED_MODEL, tmp_path)
    custom = sorted(LABELS)
    (tmp_path / inference.LABELS_FILE).write_text(json.dumps(custom))
    _, labels = inference.load_model(str(tmp_path))
    assert labels == custom


@pytest.mark.parametrize("name", ["nakamura", "Rossi", "Müller", "o'brien"])
def test_predict_returns_a_known_language(shipped, name):
    model, labels = shipped
    assert inference.predict(model, name, labels) in LABELS


def test_predict_is_deterministic(shipped):
    model, labels = shipped
    assert inference.predict(model, "tanaka", labels) == inference.predict(model, "tanaka", labels)


@pytest.mark.parametrize("name", ["", "123", "#!?", "ABC"])
def test_predict_rejects_names_with_no_usable_letters(shipped, name):
    model, labels = shipped
    with pytest.raises(ValueError):
        inference.predict(model, name, labels)
