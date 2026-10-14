"""
Text cleaning and one-hot encoding of surnames, character by character.

A neural network needs numbers, not strings. A character-level model reads a
name one character at a time, so each character becomes a vector:

    vocabulary  = "abcdefghijklmnopqrstuvwxyz .,;'_"     (32 symbols)
    'c'         -> [0, 0, 1, 0, ..., 0]                  (one-hot, length 32)
    "kim"       -> tensor of shape (3, 1, 32)            (time, batch, symbol)

The middle dimension is a batch of size 1, which is the layout nn.RNN
expects by default (sequence first).
"""
import string
import unicodedata

import torch

# The 32-symbol vocabulary. "_" doubles as the "unknown character" slot.
allowed_characters = string.ascii_lowercase + " .,;'" + "_"
n_letters = len(allowed_characters)


def unicode_to_ascii(s):
    """Normalise a name to the model's vocabulary.

    Steps, in order:
      1. lower-case ("Müller" -> "müller"), so capitals are kept as letters;
      2. NFD normalisation splits an accented letter into base letter +
         combining mark ("ü" -> "u" + "\u0308");
      3. drop the combining marks (Unicode category "Mn") and any character
         outside the vocabulary.

    Example: "Müller" -> "muller", "O'Brien" -> "o'brien".

    Args:
        s: the raw surname.

    Returns:
        The cleaned string (possibly empty, e.g. for "123").
    """
    return ''.join(
        c for c in unicodedata.normalize('NFD', s.lower())
        if unicodedata.category(c) != 'Mn'
        and c in allowed_characters
    )


def letter_to_index(letter):
    """Position of a character in the vocabulary.

    Args:
        letter: a single character.

    Returns:
        Its index in `allowed_characters`; unknown characters map to the
        index of "_".
    """
    if letter not in allowed_characters:
        return allowed_characters.find("_")
    return allowed_characters.find(letter)


def line_to_tensor(line):
    """One-hot encode a name, one time step per character.

    Args:
        line: a (cleaned) name.

    Returns:
        A float tensor of shape (len(line), 1, n_letters) with exactly one 1
        per time step.
    """
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][letter_to_index(letter)] = 1
    return tensor


def label_from_output(predicted_output, output_labels):
    """Turn the model's output into a language name.

    Args:
        predicted_output: (1, n_labels) log-probabilities from the model.
        output_labels: language names in the order of the model's output units.

    Returns:
        (label, index) of the most likely language.
    """
    top_n, top_i = predicted_output.topk(1)  # highest log-probability and its position
    label_i = top_i[0].item()
    return output_labels[label_i], label_i
