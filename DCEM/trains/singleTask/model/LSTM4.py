import torch
import torch.nn as nn


class LSTM4(nn.Module):
    def __init__(self, input_dim, output_dim, num_layers, size):
        super(LSTM4, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.size = size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(self.input_dim, output_dim + self.size, self.num_layers,
                            bidirectional=False, batch_first=True)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.output_dim + self.size).requires_grad_().cuda()
        c0 = torch.zeros(self.num_layers, x.size(0), self.output_dim + self.size).requires_grad_().cuda()
        out, _ = self.lstm(x, (h0, c0))

        return out
