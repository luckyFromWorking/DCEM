import torch
import torch.nn as nn


class LSTM1(nn.Module):
    def __init__(self, input_dim, num_layers):
        super(LSTM1, self).__init__()
        self.input_dim = input_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(self.input_dim + 4, self.input_dim + 8, self.num_layers,
                            bidirectional=False, batch_first=True)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.input_dim + 8).requires_grad_().cuda()
        c0 = torch.zeros(self.num_layers, x.size(0), self.input_dim + 8).requires_grad_().cuda()
        out, _ = self.lstm(x, (h0, c0))

        return out
