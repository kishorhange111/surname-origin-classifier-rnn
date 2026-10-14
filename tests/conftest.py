"""Shared fixtures: a tiny on-disk surnames dataset, so no test downloads the real one."""
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"


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
