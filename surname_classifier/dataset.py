"""
PyTorch Dataset for the surnames data: one text file per language, one surname per line.

    data/names/Japanese.txt   ->  label "Japanese"
        Nakamura
        Tanaka
        ...

Every name is cleaned and one-hot encoded once, up front, so training epochs
only index into ready-made tensors.
"""
import glob
import os

import torch
from torch.utils.data import Dataset

from .preprocessing import line_to_tensor, unicode_to_ascii


class SurnamesDataset(Dataset):
    """All surnames from a folder of <Language>.txt files.

    Args:
        data_dir: folder containing one UTF-8 .txt file per language.

    Attributes:
        labels_uniq: sorted list of language names; the position of a
            language in this list is the index of its model output unit.
        data / data_tensors: cleaned names and their one-hot tensors.
        labels / labels_tensors: language names and their class indices.
    """

    def __init__(self, data_dir):
        self.data_dir = data_dir
        labels_set = set()

        self.data = []
        self.data_tensors = []
        self.labels = []
        self.labels_tensors = []

        text_files = glob.glob(os.path.join(data_dir, '*.txt'))
        for filename in text_files:
            label = os.path.splitext(os.path.basename(filename))[0]  # "Japanese.txt" -> "Japanese"
            labels_set.add(label)
            lines = open(filename, encoding='utf-8').read().strip().split('\n')
            for name in lines:
                clean_name = unicode_to_ascii(name)
                self.data.append(clean_name)
                self.data_tensors.append(line_to_tensor(clean_name))
                self.labels.append(label)

        # sorted(): a plain list(set) gives a different order on every Python run, which would silently
        # scramble the mapping between output neurons and language names after retraining.
        self.labels_uniq = sorted(labels_set)
        for idx in range(len(self.labels)):
            # Shape (1,): NLLLoss expects a batch of class indices, and each example is a batch of one.
            temp_tensor = torch.tensor([self.labels_uniq.index(self.labels[idx])], dtype=torch.long)
            self.labels_tensors.append(temp_tensor)

    def __len__(self):
        """Number of surnames."""
        return len(self.data)

    def __getitem__(self, idx):
        """Return (label_tensor, name_tensor, label, name) for one surname."""
        return self.labels_tensors[idx], self.data_tensors[idx], self.labels[idx], self.data[idx]
