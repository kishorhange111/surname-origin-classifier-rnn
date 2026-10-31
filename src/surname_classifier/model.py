"""
The character-level recurrent neural network (RNN).

An RNN reads a sequence one step at a time and carries a *hidden state*, a
vector that summarises everything read so far:

    h_t = tanh(W_ih x_t + W_hh h_(t-1) + b)

After the last character, h_T is a fixed-size summary of the whole name,
whatever its length. Two linear layers turn it into one score per language,
and LogSoftmax turns the scores into log-probabilities (the input NLLLoss
expects during training).

    (len, 1, 32) one-hot --RNN(256)--> h_T (1, 256) --Linear--> (1, 128) --Linear--> (1, 18) --LogSoftmax
"""

import torch.nn as nn


class CharRNN(nn.Module):
    """Single-layer RNN classifier over one-hot character sequences.

    Args:
        input_size: size of each one-hot character vector (the vocabulary size, 32).
        hidden_size: size of the RNN hidden state (256 for the shipped model).
        output_size: number of languages (18).
    """

    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()

        self.rnn = nn.RNN(input_size, hidden_size)
        self.h2o1 = nn.Linear(hidden_size, 128)
        self.h2o2 = nn.Linear(128, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, line_tensor):
        """Classify one name.

        Args:
            line_tensor: (seq_len, 1, input_size) one-hot tensor from line_to_tensor.

        Returns:
            (1, output_size) log-probabilities over languages.
        """
        # nn.RNN returns the output at every step and the final hidden state of
        # shape (num_layers=1, batch=1, hidden). Only the final state is needed.
        rnn_out, hidden = self.rnn(line_tensor)
        hidden = self.h2o1(hidden[0])  # hidden[0]: drop the num_layers dim -> (1, hidden)
        output = self.h2o2(hidden)
        output = self.softmax(output)

        return output
