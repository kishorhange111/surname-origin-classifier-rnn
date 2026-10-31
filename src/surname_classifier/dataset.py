"""
PyTorch Dataset for the surnames data: one text file per language, one surname per line.

    data/names/Japanese.txt   ->  label "Japanese"
        Nakamura
        Tanaka
        ...

Every name is cleaned and one-hot encoded once, up front, so training epochs
only index into ready-made tensors.

Duplicates are removed on load. The raw files repeat many names (Arabic.txt
has 2,000 lines but only 108 distinct names), and with duplicates the same
name lands in both train and test, so the test score measures memorisation.

stratified_split() makes the held-out test set: the classes are very
imbalanced (Russian has ~9k names, Vietnamese ~70), and a plain random split
could leave the rarest languages with almost no test examples. Passing
``groups`` keeps every copy of a name on one side of the split, which matters
for the few hundred names listed under two languages.
"""

import glob
import os
import random
from collections import defaultdict

import torch
from torch.utils.data import Dataset

from .preprocessing import line_to_tensor, unicode_to_ascii


class SurnamesDataset(Dataset):
    """All surnames from a folder of <Language>.txt files.

    Args:
        data_dir: folder containing one UTF-8 .txt file per language.
        dedupe: drop repeated (name, language) pairs, counted in ``duplicates_removed``.

    Attributes:
        labels_uniq: sorted list of language names; the position of a
            language in this list is the index of its model output unit.
        data / data_tensors: cleaned names and their one-hot tensors.
        labels / labels_tensors: language names and their class indices.
    """

    def __init__(self, data_dir, dedupe=True):
        self.data_dir = data_dir
        self.duplicates_removed = 0
        seen = set()
        labels_set = set()

        self.data = []
        self.data_tensors = []
        self.labels = []
        self.labels_tensors = []

        text_files = sorted(glob.glob(os.path.join(data_dir, "*.txt")))  # sorted: same example order on every OS
        for filename in text_files:
            label = os.path.splitext(os.path.basename(filename))[0]  # "Japanese.txt" -> "Japanese"
            labels_set.add(label)
            lines = open(filename, encoding="utf-8").read().strip().split("\n")
            for name in lines:
                clean_name = unicode_to_ascii(name)
                if dedupe:
                    if (clean_name, label) in seen:
                        self.duplicates_removed += 1
                        continue
                    seen.add((clean_name, label))
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


def stratified_split(labels, test_fraction, seed, groups=None):
    """Split example indices so every class keeps the same train/test proportion.

    Args:
        labels: the class label of every example (any hashable values).
        test_fraction: share of each class that goes to the test set, in (0, 1).
        seed: RNG seed, so the split is identical on every run.
        groups: optional group key per example (e.g. the name itself). All
            examples of a group go to the same side; a group is stratified
            under the label of its first example.

    Returns:
        (train_idx, test_idx): sorted lists of indices that together cover
        every example exactly once. Each class with at least 2 examples puts
        at least 1 in test and at least 1 in train.
    """
    if groups is None:
        groups = range(len(labels))  # every example is its own group
    members = defaultdict(list)  # group key -> example indices, in first-seen order
    for i, g in enumerate(groups):
        members[g].append(i)
    by_class = defaultdict(list)  # label -> group keys
    for g, idx in members.items():
        by_class[labels[idx[0]]].append(g)
    rng = random.Random(seed)
    train_idx, test_idx = [], []
    for label in sorted(by_class):  # sorted: iteration order must not depend on hashing
        keys = by_class[label]
        rng.shuffle(keys)
        n_test = round(len(keys) * test_fraction)
        if len(keys) >= 2:
            n_test = min(max(n_test, 1), len(keys) - 1)
        for k in keys[:n_test]:
            test_idx += members[k]
        for k in keys[n_test:]:
            train_idx += members[k]
    return sorted(train_idx), sorted(test_idx)
