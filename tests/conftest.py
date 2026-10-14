"""Shared fixtures: a tiny on-disk surnames dataset, so no test downloads the real one."""

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"

# The 18 languages of the dataset (README), in sorted order.
LANGUAGES = [
    "Arabic",
    "Chinese",
    "Czech",
    "Dutch",
    "English",
    "French",
    "German",
    "Greek",
    "Irish",
    "Italian",
    "Japanese",
    "Korean",
    "Polish",
    "Portuguese",
    "Russian",
    "Scottish",
    "Spanish",
    "Vietnamese",
]


def shipped_labels():
    """The label order saved next to the shipped model."""
    return json.loads((MODELS_DIR / "labels.json").read_text())


@pytest.fixture
def names_dir(tmp_path):
    """Three toy 'languages' whose names are trivially separable by their letters."""
    files = {
        "Alpha.txt": "aaab\nabaa\nbaaa\naaba\nabab\n",
        "Zulu.txt": "zzzy\nzyzz\nyzzz\nzzyz\nzyzy\n",
        "Mixed.txt": "Café\nmüller\n",  # accents to strip, plus an upper-case initial
    }
    for name, content in files.items():
        (tmp_path / name).write_text(content, encoding="utf-8")
    return tmp_path
