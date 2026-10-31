"""
Character-level RNN that predicts the language of origin of a surname.

  preprocessing  clean a name and one-hot encode it character by character
  model          CharRNN: RNN over characters -> 18 language log-probabilities
  dataset        PyTorch Dataset over a folder of <Language>.txt files + stratified split
  metrics        accuracy, per-class precision/recall/F1, confusion-matrix plot
  inference      load the TorchScript export + labels.json and predict (used by app.py)
"""

from .model import CharRNN
from .preprocessing import label_from_output, line_to_tensor, n_letters, unicode_to_ascii

__all__ = ["CharRNN", "label_from_output", "line_to_tensor", "n_letters", "unicode_to_ascii"]
