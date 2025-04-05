import torch
import time
import torch.nn as nn
import random
import numpy as np
from model import CharRNN
from preprocessing import n_letters, line_to_tensor, unicode_to_ascii, label_from_output
from surname_dataset import SurnamesDataset
import os

device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')

torch.set_default_device(device)
print(f"Using device = {torch.get_default_device()}")

alldata = SurnamesDataset("data/names")
print(f"loaded {len(alldata)} items of data")
print(f"example = {alldata[0]}")

train_set, test_set = torch.utils.data.random_split(alldata, [.85, .15],
                                                    generator=torch.Generator(device=device).manual_seed(2024))

print(f"train examples = {len(train_set)}, validation examples = {len(test_set)}")


n_hidden = 256
rnn = CharRNN(n_letters, n_hidden, len(alldata.labels_uniq))
print(rnn)


def train(model, training_data, n_epoch=10, n_batch_size=64, report_every=50, learning_rate=0.2, criterion=nn.NLLLoss()):
    current_loss = 0
    all_losses = []
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    print(f"training on data set with n = {len(training_data)}")

    for iter in range(1, n_epoch + 1):
        model.zero_grad()

        batches = list(range(len(training_data)))
        random.shuffle(batches)
        batches = np.array_split(batches, len(batches) // n_batch_size)

        for idx, batch in enumerate(batches):
            batch_loss = 0
            for i in batch:
                (label_tensor, text_tensor, label, text) = training_data[i]
                output = model.forward(text_tensor)
                loss = criterion(output, label_tensor)
                batch_loss += loss

            batch_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 3)
            optimizer.step()
            optimizer.zero_grad()

            current_loss += batch_loss.item() / len(batch)

        all_losses.append(current_loss / len(batches))
        if iter % report_every == 0:
            print(f"{iter} ({iter / n_epoch:.0%}): \t average batch loss :  = {all_losses[-1]}")
        current_loss = 0

    return all_losses


start = time.time()
all_losses = train(rnn, train_set, n_epoch=27, learning_rate=0.15, report_every=5)
end = time.time()
print(f"training took {end-start}s")


def evaluate(model, testing_data, classes):
    confusion = torch.zeros(len(classes), len(classes))

    model.eval()
    with torch.no_grad():
        for i in range(len(testing_data)):
            (label_tensor, text_tensor, label, text) = testing_data[i]
            output = model(text_tensor)
            guess, guess_i = label_from_output(output, classes)
            label_i = classes.index(label)
            confusion[label_i][guess_i] += 1

    for i in range(len(classes)):
        denominator = confusion[i].sum()
        if denominator > 0:
            confusion[i] = confusion[i] / denominator

    print(confusion.cpu())


evaluate(rnn, test_set, classes=alldata.labels_uniq)

input_line = line_to_tensor(unicode_to_ascii('Cho'))
output = rnn(input_line)
print(output)
print(label_from_output(output, alldata.labels_uniq))

model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),"model")

torch.save(rnn.state_dict(), os.path.join(model_path,"CharRNN.pt"))

model = CharRNN(n_letters,256,len(alldata.labels_uniq))
model.load_state_dict(torch.load(os.path.join(model_path,"CharRNN.pt"), weights_only=True))
model.eval()

input_line = line_to_tensor(unicode_to_ascii('Cho'))
output = model(input_line)
print(output)
print(label_from_output(output, alldata.labels_uniq))

model_scripted = torch.jit.script(model)
model_scripted.save(os.path.join(model_path,"CharRNN_scripted.pt"))

