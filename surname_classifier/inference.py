"""
Load the exported model and predict a surname's language: the serving path used by app.py.

The app does not import CharRNN at all. It loads the TorchScript file, a
self-contained, serialised graph of the model, which is what makes the
serving code independent of the training code.
"""
import json
import os

import torch

from .preprocessing import LABELS, label_from_output, line_to_tensor, unicode_to_ascii

SCRIPTED_MODEL = "CharRNN_scripted.pt"
LABELS_FILE = "labels.json"


def load_model(models_dir):
    """Load the TorchScript model and the label order it was trained with.

    Args:
        models_dir: folder with CharRNN_scripted.pt and, after a retrain,
            labels.json.

    Returns:
        (model, labels): the model in eval mode, and the language names in
        output-unit order. labels.json (written by train.py) wins over the
        built-in LABELS list, so a retrained model always gets its own order.
    """
    model = torch.jit.load(os.path.join(models_dir, SCRIPTED_MODEL))
    model.eval()
    labels_file = os.path.join(models_dir, LABELS_FILE)
    if os.path.exists(labels_file):
        with open(labels_file) as f:
            labels = json.load(f)
    else:
        labels = LABELS
    return model, labels


def predict(model, surname, labels):
    """Predict the most likely language of origin of a surname.

    Args:
        model: a loaded CharRNN (eager or TorchScript).
        surname: raw user input; it is cleaned with unicode_to_ascii first.
        labels: language names in output-unit order.

    Returns:
        The predicted language name.

    Raises:
        ValueError: if nothing is left after cleaning (e.g. "123"); an RNN
            cannot run on an empty sequence.
    """
    clean = unicode_to_ascii(surname)
    if not clean:
        raise ValueError(f"{surname!r} contains no letters the model understands")
    with torch.no_grad():  # inference only: no gradient bookkeeping
        output = model(line_to_tensor(clean))
    return label_from_output(output, labels)[0]
