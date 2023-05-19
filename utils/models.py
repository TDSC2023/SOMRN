import torch
import torch.nn as nn


class LSTMClassifier(nn.Module):
    def __init__(self, in_features, hidden_neurons, num_layers, category, batch_first=True, dropout=0):
        super().__init__()
        self.rnn = nn.LSTM(
            input_size=in_features,
            hidden_size=hidden_neurons,
            num_layers=num_layers,
            batch_first=batch_first,
            dropout=dropout
        )
        self.fc = nn.Linear(hidden_neurons, category)

    def forward(self, x):
        output, (hn, cn) = self.rnn(x)
        return self.fc(hn[-1])


class GRUClassifier(nn.Module):
    def __init__(self, in_features, hidden_neurons, num_layers, category, batch_first=True, dropout=0):
        super().__init__()
        self.rnn = nn.GRU(
            input_size=in_features,
            hidden_size=hidden_neurons,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=batch_first
        )
        self.fc = nn.Linear(hidden_neurons, category)

    def forward(self, x):
        output, hn = self.rnn(x)
        return self.fc(hn[-1])
