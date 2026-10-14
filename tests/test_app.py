"""Smoke tests of the Streamlit UI with the real shipped model (no network needed)."""
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from surname_classifier.preprocessing import LABELS  # noqa: E402

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def run_app(text):
    at = AppTest.from_file(APP, default_timeout=30).run()
    at.text_input[0].input(text)
    return at.run()


def test_prediction_is_shown():
    at = run_app("nakamura")
    assert not at.exception
    shown = " ".join(str(m.value) for m in at.markdown)
    assert any(lang in shown for lang in LABELS)


def test_input_without_letters_shows_warning():
    at = run_app("1234")
    assert not at.exception
    assert "Please enter a surname" in at.warning[0].value
