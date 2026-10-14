"""
Train, evaluate and export the character-level RNN surname classifier.

Pipeline:
  1. load every surname from data/names/<Language>.txt (SurnamesDataset)
  2. stratified, seeded 85/15 split into train / held-out test (stratified_split)
  3. train CharRNN with mini-batch SGD + gradient clipping (train)
  4. test accuracy, per-language precision/recall/F1 and macro F1 (evaluate)
  5. export the weights, a TorchScript model for the app, the label order and the metrics

Usage:  python train.py [--data data/names] [--epochs 27] [--seed 2024] [--test-fraction 0.15]
Writes models/CharRNN.pt (weights), models/CharRNN_scripted.pt (TorchScript, used by app.py),
models/labels.json (the label order the model was trained with), results/metrics.json and
results/confusion_matrix.png.
"""

import argparse
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Subset

from surname_classifier.dataset import SurnamesDataset, stratified_split
from surname_classifier.metrics import classification_report, plot_confusion_matrix
from surname_classifier.model import CharRNN
from surname_classifier.preprocessing import label_from_output, line_to_tensor, n_letters, unicode_to_ascii

ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(ROOT, "models")
RESULTS_DIR = os.path.join(ROOT, "results")


def train(model, training_data, n_epoch=10, n_batch_size=64, report_every=50, learning_rate=0.2, criterion=None):
    """Mini-batch SGD with gradient clipping.

    Names have different lengths, so instead of padding them into one tensor
    each name in a batch is run separately and the losses are summed; one
    optimizer step is taken per batch.

    Args:
        model: the CharRNN to train (updated in place).
        training_data: indexable dataset of (label_tensor, name_tensor, label, name).
        n_epoch: passes over the data.
        n_batch_size: names per optimizer step (approximate: np.array_split
            makes len(data) // n_batch_size nearly equal batches).
        report_every: print the loss every this many epochs.
        learning_rate: SGD step size.
        criterion: loss on log-probabilities; defaults to NLLLoss, which pairs with LogSoftmax.

    Returns:
        List with the average per-name loss of each epoch.
    """
    if criterion is None:
        criterion = nn.NLLLoss()
    current_loss = 0
    all_losses = []
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    print(f"training on data set with n = {len(training_data)}")

    for epoch in range(1, n_epoch + 1):
        model.zero_grad()
        # Shuffle example indices, then cut them into nearly equal batches.
        batches = list(range(len(training_data)))
        random.shuffle(batches)
        batches = np.array_split(batches, len(batches) // n_batch_size)

        for batch in batches:
            batch_loss = 0
            for i in batch:
                label_tensor, text_tensor, _, _ = training_data[i]
                output = model.forward(text_tensor)
                batch_loss += criterion(output, label_tensor)
            batch_loss.backward()
            # RNNs are prone to exploding gradients on long sequences; cap the gradient norm.
            nn.utils.clip_grad_norm_(model.parameters(), 3)
            optimizer.step()
            optimizer.zero_grad()
            current_loss += batch_loss.item() / len(batch)  # per-name average, for reporting

        all_losses.append(current_loss / len(batches))
        if epoch % report_every == 0:
            print(f"{epoch} ({epoch / n_epoch:.0%}): \t average batch loss = {all_losses[-1]:.4f}")
        current_loss = 0
    return all_losses


def predict_all(model, data, classes):
    """Run the model on every example.

    Args:
        model: the trained model.
        data: dataset of (label_tensor, name_tensor, label, name).
        classes: language names in output-unit order.

    Returns:
        (y_true, y_pred): lists of class indices.
    """
    y_true, y_pred = [], []
    model.eval()
    with torch.no_grad():
        for i in range(len(data)):
            _, text_tensor, label, _ = data[i]
            _, guess_i = label_from_output(model(text_tensor), classes)
            y_true.append(classes.index(label))
            y_pred.append(guess_i)
    return y_true, y_pred


def evaluate(model, testing_data, classes):
    """Accuracy, per-class precision/recall/F1 and macro F1 on held-out data.

    Args:
        model: the trained model.
        testing_data: dataset of (label_tensor, name_tensor, label, name).
        classes: language names in output-unit order.

    Returns:
        The report dict from surname_classifier.metrics.classification_report.
    """
    report = classification_report(*predict_all(model, testing_data, classes), classes)
    print(f"{'language':<12} {'prec':>6} {'recall':>6} {'f1':>6} {'n':>5}")
    for label, m in report["per_class"].items():
        print(f"{label:<12} {m['precision']:6.3f} {m['recall']:6.3f} {m['f1']:6.3f} {m['support']:5d}")
    print(
        f"test accuracy: {report['accuracy']:.1%}   macro F1: {report['macro_f1']:.3f}   "
        f"weighted F1: {report['weighted_f1']:.3f}"
    )
    return report


def main():
    """Parse arguments, then load data, split, train, evaluate and export the model and metrics."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/names", help="folder with one <Language>.txt file per language")
    parser.add_argument("--epochs", type=int, default=27)
    parser.add_argument("--test-fraction", type=float, default=0.15, help="share of each language held out")
    parser.add_argument("--seed", type=int, default=2024)
    args = parser.parse_args()

    # Seed every RNG that training touches (split, batch order, weight init) so a re-run reproduces the model.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_device(device)  # every tensor created from here on lives on this device
    print(f"Using device = {torch.get_default_device()}")

    alldata = SurnamesDataset(args.data)
    classes = alldata.labels_uniq
    print(f"loaded {len(alldata)} items of data")
    # Stratified, seeded split: every language keeps the same share in the held-out test set, and the
    # same names land there on every run. The test set is used once, after training.
    train_idx, test_idx = stratified_split(alldata.labels, args.test_fraction, args.seed)
    train_set, test_set = Subset(alldata, train_idx), Subset(alldata, test_idx)
    print(f"train examples = {len(train_set)}, test examples = {len(test_set)}")

    rnn = CharRNN(n_letters, 256, len(classes))
    print(rnn)

    start = time.time()
    losses = train(rnn, train_set, n_epoch=args.epochs, learning_rate=0.15, report_every=5)
    train_seconds = time.time() - start
    print(f"training took {train_seconds:.0f}s")
    report = evaluate(rnn, test_set, classes)

    # Export: weights, TorchScript model for the app, and the label order.
    os.makedirs(MODELS_DIR, exist_ok=True)
    torch.save(rnn.state_dict(), os.path.join(MODELS_DIR, "CharRNN.pt"))
    # Reload the saved weights into a fresh model before scripting, which also checks the file round-trips.
    model = CharRNN(n_letters, 256, len(classes))
    model.load_state_dict(torch.load(os.path.join(MODELS_DIR, "CharRNN.pt"), weights_only=True))
    model.eval()
    # TorchScript compiles the model into a standalone graph the app can load without this code.
    torch.jit.script(model).save(os.path.join(MODELS_DIR, "CharRNN_scripted.pt"))
    with open(os.path.join(MODELS_DIR, "labels.json"), "w") as f:
        json.dump(classes, f)

    # Metrics + confusion matrix for the README.
    os.makedirs(RESULTS_DIR, exist_ok=True)
    counts = {c: alldata.labels.count(c) for c in classes}
    metrics = {
        "test": report,
        "data": {
            "total_names": len(alldata),
            "train_names": len(train_set),
            "test_names": len(test_set),
            "names_per_language": counts,
            "largest_class_share": max(counts.values()) / len(alldata),
        },
        "training": {
            "epochs": args.epochs,
            "batch_size": 64,
            "learning_rate": 0.15,
            "hidden_size": 256,
            "test_fraction": args.test_fraction,
            "seed": args.seed,
            "device": str(device),
            "train_seconds": round(train_seconds),
            "final_epoch_loss": losses[-1] if losses else None,
            "torch_version": torch.__version__,
        },
    }
    with open(os.path.join(RESULTS_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    plot_confusion_matrix(
        report["confusion_matrix"],
        classes,
        os.path.join(RESULTS_DIR, "confusion_matrix.png"),
        title=f"Test-set confusion matrix (row-normalised), accuracy {report['accuracy']:.1%}",
    )

    print("example:", label_from_output(model(line_to_tensor(unicode_to_ascii("Nakamura"))), classes))


if __name__ == "__main__":
    main()
