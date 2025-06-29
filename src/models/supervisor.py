#!/usr/bin/env python3.13

import torch.nn as nn

class Supervisor(nn.Module):
    def __init__(self, hidden_dim, num_layers, output_dim):
        super(Supervisor, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim

        self.gru = nn.GRU(input_size = hidden_dim, hidden_size = hidden_dim, num_layers =  num_layers, batch_first=True, dropout=0.0 if num_layers == 1 else 0)

        self.fc = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, H):
        """
        Input: Embedded sequence
        Output Supervised embedded sequence"""

        gru_out, _ = self.gru(H)
        S = self.fc(gru_out)
        return S
