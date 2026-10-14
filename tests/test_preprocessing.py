"""Name cleaning, one-hot encoding and label decoding."""
import pytest
import torch

from surname_classifier.preprocessing import (
    LABELS,
    allowed_characters,
    label_from_output,
    letter_to_index,
    line_to_tensor,
    n_letters,
    unicode_to_ascii,
)


def test_vocabulary_is_32_symbols():
    assert n_letters == len(allowed_characters) == 32
    assert len(set(allowed_characters)) == 32


@pytest.mark.parametrize("raw,clean", [
    ("müller", "muller"),          # accent stripped
    ("josé", "jose"),
    ("o'neil", "o'neil"),          # apostrophe is in the vocabulary
    ("van der berg", "van der berg"),
    ("smith-jones", "smithjones"), # hyphen is not
    ("nguyen2", "nguyen"),         # digits dropped
    ("", ""),
])
def test_unicode_to_ascii(raw, clean):
    assert unicode_to_ascii(raw) == clean


def test_upper_case_letters_are_dropped_as_trained():
    # Pinned on purpose: the vocabulary check happens before lower-casing, so capitals are
    # removed. The shipped model was trained this way; changing it needs a retrain.
    assert unicode_to_ascii("Müller") == "uller"
    assert unicode_to_ascii("NAKAMURA") == ""


def test_letter_to_index_maps_unknown_to_underscore():
    assert letter_to_index("a") == 0
    assert letter_to_index("z") == 25
    assert letter_to_index("'") == allowed_characters.index("'")
    assert letter_to_index("#") == letter_to_index("Q") == allowed_characters.index("_")


def test_line_to_tensor_is_one_hot_per_character():
    t = line_to_tensor("kim")
    assert t.shape == (3, 1, n_letters)
    assert t.sum(-1).flatten().tolist() == [1.0, 1.0, 1.0]
    assert t[:, 0].argmax(-1).tolist() == [letter_to_index(c) for c in "kim"]


def test_line_to_tensor_empty():
    assert line_to_tensor("").shape == (0, 1, n_letters)


def test_label_from_output_picks_argmax():
    out = torch.full((1, len(LABELS)), -10.0)
    out[0, 3] = -0.1
    assert label_from_output(out, LABELS) == (LABELS[3], 3)


def test_labels_cover_the_18_languages_in_the_readme():
    assert sorted(LABELS) == [
        "Arabic", "Chinese", "Czech", "Dutch", "English", "French", "German", "Greek", "Irish",
        "Italian", "Japanese", "Korean", "Polish", "Portuguese", "Russian", "Scottish", "Spanish", "Vietnamese",
    ]
