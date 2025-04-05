import torch.nn as nn

class CharRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(CharRNN, self).__init__()

        self.rnn = nn.RNN(input_size, hidden_size)
        self.h2o1 = nn.Linear(hidden_size, 128)
        self.h2o2 = nn.Linear(128, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, line_tensor):
        rnn_out, hidden = self.rnn(line_tensor)
        hidden = self.h2o1(hidden[0])
        output = self.h2o2(hidden)
        output = self.softmax(output)

        return output