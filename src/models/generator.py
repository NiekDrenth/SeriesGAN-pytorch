#!/usr/bin/env python3.13
import torch.nn as nn
import torch

class Generator(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(Generator, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.gru = nn.GRU(
            input_size = input_dim,
            hidden_size = hidden_dim,
            num_layers = num_layers,
            batch_first=True,
            dropout=0.0 if num_layers == 1 else 0.1
        )
        self.fc = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        """
        Returns generated sample into embedded space
        """
        gru_out, _ = self.gru(x)
        E = self.fc(gru_out)
        return E

if __name__ == "__main__":

    input_dim = 6
    hidden_dim = 6
    num_layers = 4
    seq_len = 16
    batch_size = 64

    generator = Generator(input_dim = input_dim, hidden_dim = hidden_dim, num_layers=num_layers)
    x = torch.randn(batch_size, seq_len, input_dim)
    generated = generator(x)
    print(f"Input shape: {generated.shape}")
