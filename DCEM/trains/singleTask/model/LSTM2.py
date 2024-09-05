import torch
import torch.nn as nn


class LSTM2(nn.Module):
    def __init__(self, input_dim, output_dim, num_layers, padding, layer_norm):
        super(LSTM2, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.padding = padding
        self.layer_norm = layer_norm
        self.lstm = nn.LSTM(self.input_dim, output_dim + self.padding, self.num_layers,
                            bidirectional=False, batch_first=True)
        self.layer_norm = nn.LayerNorm(self.output_dim + self.padding)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.output_dim + self.padding).requires_grad_().cuda()
        c0 = torch.zeros(self.num_layers, x.size(0), self.output_dim + self.padding).requires_grad_().cuda()
        out, _ = self.lstm(x, (h0, c0))
        if self.layer_norm:
            out = self.layer_norm(out)
        return out
