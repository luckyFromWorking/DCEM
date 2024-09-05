import torch
import torch.nn as nn


class LSTM3(nn.Module):
    def __init__(self, input_dim, num_layers, mode):
        super(LSTM3, self).__init__()
        self.input_dim = input_dim
        self.mode = mode - 1
        self.mode_x2 = 2 * self.mode
        self.num_layers = num_layers
        self.layer_norm = nn.LayerNorm(self.input_dim + self.mode_x2)
        self.lstm = nn.LSTM(self.input_dim + self.mode, self.input_dim + self.mode_x2, self.num_layers,
                            bidirectional=False, batch_first=True)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.input_dim + self.mode_x2).requires_grad_().cuda()
        c0 = torch.zeros(self.num_layers, x.size(0), self.input_dim + self.mode_x2).requires_grad_().cuda()
        out, _ = self.lstm(x, (h0, c0))
        out = self.layer_norm(out)
        return out
