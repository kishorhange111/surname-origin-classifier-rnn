"""Train, evaluate and export the character-level RNN surname classifier.

Usage:  python train.py [--data data/names] [--epochs 27]
Writes models/CharRNN.pt (weights), models/CharRNN_scripted.pt (TorchScript, used by app.py)
and models/labels.json (the label order the model was trained with).
"""
import argparse
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn

from surname_classifier.dataset import SurnamesDataset
from surname_classifier.model import CharRNN
from surname_classifier.preprocessing import label_from_output, line_to_tensor, n_letters, unicode_to_ascii

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")


def train(model, training_data, n_epoch=10, n_batch_size=64, report_every=50, learning_rate=0.2,
          criterion=nn.NLLLoss()):
    """Mini-batch SGD with gradient clipping; returns the average loss per epoch."""
    current_loss = 0
    all_losses = []
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    print(f"training on data set with n = {len(training_data)}")

    for epoch in range(1, n_epoch + 1):
        model.zero_grad()
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
            nn.utils.clip_grad_norm_(model.parameters(), 3)
            optimizer.step()
            optimizer.zero_grad()
            current_loss += batch_loss.item() / len(batch)

        all_losses.append(current_loss / len(batches))
        if epoch % report_every == 0:
            print(f"{epoch} ({epoch / n_epoch:.0%}): \t average batch loss = {all_losses[-1]:.4f}")
        current_loss = 0
    return all_losses


def evaluate(model, testing_data, classes):
    """Row-normalised confusion matrix and overall accuracy on held-out data."""
    confusion = torch.zeros(len(classes), len(classes))
    model.eval()
    with torch.no_grad():
        for i in range(len(testing_data)):
            _, text_tensor, label, _ = testing_data[i]
            _, guess_i = label_from_output(model(text_tensor), classes)
            confusion[classes.index(label)][guess_i] += 1
    accuracy = (confusion.diag().sum() / confusion.sum()).item()
    for i in range(len(classes)):
        if confusion[i].sum() > 0:
            confusion[i] = confusion[i] / confusion[i].sum()
    print(confusion.cpu())
    print(f"validation accuracy: {accuracy:.1%}")
    return accuracy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/names", help="folder with one <Language>.txt file per language")
    parser.add_argument("--epochs", type=int, default=27)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_device(device)
    print(f"Using device = {torch.get_default_device()}")

    alldata = SurnamesDataset(args.data)
    print(f"loaded {len(alldata)} items of data")
    train_set, test_set = torch.utils.data.random_split(
        alldata, [.85, .15], generator=torch.Generator(device=device).manual_seed(2024))
    print(f"train examples = {len(train_set)}, validation examples = {len(test_set)}")

    rnn = CharRNN(n_letters, 256, len(alldata.labels_uniq))
    print(rnn)

    start = time.time()
    train(rnn, train_set, n_epoch=args.epochs, learning_rate=0.15, report_every=5)
    print(f"training took {time.time() - start:.0f}s")
    evaluate(rnn, test_set, classes=alldata.labels_uniq)

    # Export: weights, TorchScript model for the app, and the label order.
    os.makedirs(MODELS_DIR, exist_ok=True)
    torch.save(rnn.state_dict(), os.path.join(MODELS_DIR, "CharRNN.pt"))
    model = CharRNN(n_letters, 256, len(alldata.labels_uniq))
    model.load_state_dict(torch.load(os.path.join(MODELS_DIR, "CharRNN.pt"), weights_only=True))
    model.eval()
    torch.jit.script(model).save(os.path.join(MODELS_DIR, "CharRNN_scripted.pt"))
    json.dump(alldata.labels_uniq, open(os.path.join(MODELS_DIR, "labels.json"), "w"))

    print("example:", label_from_output(model(line_to_tensor(unicode_to_ascii("Nakamura"))), alldata.labels_uniq))


if __name__ == "__main__":
    main()
