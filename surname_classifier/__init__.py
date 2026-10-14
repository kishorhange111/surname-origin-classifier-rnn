"""
Character-level RNN that predicts the language of origin of a surname.

  preprocessing  clean a name and one-hot encode it character by character
  model          CharRNN: RNN over characters -> 18 language log-probabilities
  dataset        PyTorch Dataset over a folder of <Language>.txt files
  inference      load the TorchScript export and predict (used by app.py)
"""
from .model import CharRNN
from .preprocessing import LABELS, label_from_output, line_to_tensor, n_letters, unicode_to_ascii

__all__ = ["CharRNN", "LABELS", "label_from_output", "line_to_tensor", "n_letters", "unicode_to_ascii"]
