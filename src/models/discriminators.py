#!/usr/bin/env python3.13

import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, hidden_dim, num_layers):
        super(Discriminator, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.gru = nn.GRU(
            input_size = hidden_dim,
            hidden_size = hidden_dim,
            num_layers = num_layers,
            batch_first = True,
            dropout = 0.0 if num_layers ==1 else 0.1
        )

        self.fc = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, H):
        """
        Input: embedded sequence
        Output: probability of being real for each variable in sequence.
        """
        gru_out, _ = self.gru(H)
        Y_hat = self.fc(gru_out)

        return Y_hat


class AE_Discriminator(nn.Module):
    def __init__(self, hidden_dim, num_layers, max_seq_len = 1):
        super(AE_Discriminator, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        self.gru = nn.GRU(
            input_size = hidden_dim,
            hidden_size = hidden_dim,
            num_layers  = num_layers,
            batch_first = True,
            dropout = 0.0 if num_layers == 1 else 0.1
        )
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(max_seq_len*hidden_dim, hidden_dim)

    def forward(self, X):
        """
        Input: Real data or reconstructed AE data.
        Output: Probability of real data for each variable. Only one value per sequence
        """
        d_output, _ = self.gru(X)

        flattened_output = self.flatten(d_output)

        Y_hat_ae = self.fc(flattened_output)
        return Y_hat_ae
