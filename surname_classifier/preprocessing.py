"""Text cleaning and one-hot encoding of surnames, character by character."""
import string
import unicodedata

import torch

# Label order used by the shipped model in models/. Retraining writes the order it used to models/labels.json,
# which the app prefers over this list.
LABELS = ['Dutch', 'Arabic', 'German', 'Japanese', 'Irish', 'Polish', 'Korean', 'French', 'Greek', 'Portuguese',
          'Vietnamese', 'Russian', 'Czech', 'Chinese', 'Spanish', 'Scottish', 'English', 'Italian']
alldata = LABELS  # backwards-compatible name

allowed_characters = string.ascii_lowercase + " .,;'" + "_"
n_letters = len(allowed_characters)


def unicode_to_ascii(s):
    """Strip accents (é -> e), lower-case, and drop characters outside the vocabulary."""
    return ''.join(
        c.lower() for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in allowed_characters
    )


def letter_to_index(letter):
    """Index of a character in the vocabulary; unknown characters map to '_'."""
    if letter not in allowed_characters:
        return allowed_characters.find("_")
    return allowed_characters.find(letter)


def line_to_tensor(line):
    """One-hot encode a name as a (length, 1, n_letters) tensor - one time step per character."""
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][letter_to_index(letter)] = 1
    return tensor


def label_from_output(predicted_output, output_labels):
    """Most likely label and its index from the model's log-probabilities."""
    top_n, top_i = predicted_output.topk(1)
    label_i = top_i[0].item()
    return output_labels[label_i], label_i
