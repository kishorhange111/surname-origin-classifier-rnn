"""Smoke tests of the Streamlit UI with the real shipped model (no network needed)."""

from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from surname_classifier import inference  # noqa: E402

from .conftest import MODELS_DIR, shipped_labels  # noqa: E402

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def run_app(text):
    at = AppTest.from_file(APP, default_timeout=30).run()
    at.text_input[0].input(text)
    return at.run()


def shown_language(at):
    """The language the app printed in its answer line."""
    for m in at.markdown:
        if "Your Surname is similar to" in str(m.value):
            return str(m.value)
    raise AssertionError("no prediction shown")


def test_app_labels_match_labels_json():
    # The app's prediction for each name must be exactly what the model + labels.json give, i.e.
    # the app maps output units to languages with labels.json and not with any other list.
    model, _ = inference.load_model(str(MODELS_DIR))
    labels = shipped_labels()
    for name in ["Nakamura", "Rossi", "Ivanov", "Nguyen"]:
        expected = inference.predict(model, name, labels)
        text = shown_language(run_app(name))
        assert expected in text
        # no other language name appears in the answer
        assert [lang for lang in labels if lang in text] == [expected]


def test_input_without_letters_shows_warning():
    at = run_app("1234")
    assert not at.exception
    assert "Please enter a surname" in at.warning[0].value
