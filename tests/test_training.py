"""train() and evaluate() from train.py on a tiny, separable dataset (CPU, seconds)."""
import random

import pytest
import torch
from torch.utils.data import Subset

from surname_classifier.dataset import SurnamesDataset
from surname_classifier.model import CharRNN
from surname_classifier.preprocessing import n_letters
from train import evaluate, train


@pytest.fixture
def separable(names_dir):
    """Only the 'a...' vs 'z...' languages: easy enough to learn perfectly."""
    ds = SurnamesDataset(str(names_dir))
    keep = [i for i, label in enumerate(ds.labels) if label != "Mixed"]
    return ds, Subset(ds, keep)


def test_training_reduces_loss_and_learns_the_task(separable):
    ds, data = separable
    random.seed(0)
    torch.manual_seed(0)
    model = CharRNN(n_letters, 32, len(ds.labels_uniq))
    losses = train(model, data, n_epoch=30, n_batch_size=2, report_every=100, learning_rate=0.1)
    assert len(losses) == 30
    assert losses[-1] < losses[0] / 5
    assert evaluate(model, data, ds.labels_uniq) == pytest.approx(1.0)


def test_evaluate_returns_accuracy_in_unit_interval(separable):
    ds, data = separable
    torch.manual_seed(0)
    acc = evaluate(CharRNN(n_letters, 8, len(ds.labels_uniq)), data, ds.labels_uniq)
    assert 0.0 <= acc <= 1.0
