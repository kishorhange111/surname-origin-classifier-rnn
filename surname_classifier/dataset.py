"""PyTorch Dataset for the surnames data: one text file per language, one surname per line."""
import glob
import os

import torch
from torch.utils.data import Dataset

from .preprocessing import line_to_tensor, unicode_to_ascii


class SurnamesDataset(Dataset):

    def __init__(self, data_dir):
        self.data_dir = data_dir
        labels_set = set()

        self.data = []
        self.data_tensors = []
        self.labels = []
        self.labels_tensors = []

        text_files = glob.glob(os.path.join(data_dir, '*.txt'))
        for filename in text_files:
            label = os.path.splitext(os.path.basename(filename))[0]
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
            temp_tensor = torch.tensor([self.labels_uniq.index(self.labels[idx])], dtype=torch.long)
            self.labels_tensors.append(temp_tensor)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.labels_tensors[idx], self.data_tensors[idx], self.labels[idx], self.data[idx]
