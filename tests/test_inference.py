"""Model loading (labels.json is required) and prediction."""
import json
import shutil

import pytest

from surname_classifier import inference

from .conftest import LANGUAGES, MODELS_DIR, shipped_labels


@pytest.fixture(scope="module")
def shipped():
    return inference.load_model(str(MODELS_DIR))


def test_labels_json_lists_the_18_languages_in_sorted_order():
    assert shipped_labels() == LANGUAGES


def test_load_model_uses_labels_json(shipped):
    _, labels = shipped
    assert labels == shipped_labels()


def test_custom_labels_json_is_respected(tmp_path):
    shutil.copy(MODELS_DIR / inference.SCRIPTED_MODEL, tmp_path)
    custom = list(reversed(LANGUAGES))
    (tmp_path / inference.LABELS_FILE).write_text(json.dumps(custom))
    _, labels = inference.load_model(str(tmp_path))
    assert labels == custom


def test_missing_labels_json_is_an_error_not_a_guess(tmp_path):
    shutil.copy(MODELS_DIR / inference.SCRIPTED_MODEL, tmp_path)
    with pytest.raises(FileNotFoundError):
        inference.load_model(str(tmp_path))


def test_model_has_one_output_per_label(shipped):
    import torch

    from surname_classifier.preprocessing import line_to_tensor

    model, labels = shipped
    with torch.no_grad():
        assert model(line_to_tensor("tanaka")).shape == (1, len(labels))


@pytest.mark.parametrize("name", ["nakamura", "Rossi", "Müller", "O'Brien"])
def test_predict_returns_a_known_language(shipped, name):
    model, labels = shipped
    assert inference.predict(model, name, labels) in LANGUAGES


def test_capitalisation_does_not_change_the_prediction(shipped):
    model, labels = shipped
    for name in ["nakamura", "rossi", "ivanov"]:
        assert inference.predict(model, name, labels) == inference.predict(model, name.upper(), labels)


@pytest.mark.parametrize("name", ["", "123", "#!?"])
def test_predict_rejects_names_with_no_usable_letters(shipped, name):
    model, labels = shipped
    with pytest.raises(ValueError):
        inference.predict(model, name, labels)
