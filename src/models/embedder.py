#!/usr/bin/env python3.13
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class Embedder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(Embedder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Multi-layer GRU
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.0 if num_layers == 1 else 0.1
        )

        # Fully connected layer applied to all timesteps
        self.fc = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, lengths=None):
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            lengths: Optional tensor of actual sequence lengths for each sample
        Returns:
            H: Embedded representation of shape (batch_size, seq_len, hidden_dim)
        """
        if lengths is not None:
            # Pack sequences for variable length handling
            x_packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
            gru_out_packed, _ = self.gru(x_packed)
            gru_out, _ = pad_packed_sequence(gru_out_packed, batch_first=True)
        else:
            gru_out, _ = self.gru(x)

        # Apply fully connected layer to all timesteps
        # gru_out shape: (batch_size, seq_len, hidden_dim)
        H = self.fc(gru_out)

        return H


class Recovery(nn.Module):
    def __init__(self, hidden_dim, num_layers, output_dim):
        super(Recovery, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim

        # Multi-layer GRU
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.0 if num_layers == 1 else 0.1
        )

        # Fully connected layer applied to all timesteps
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, H, lengths=None):
        """
        Args:
            H: Embedded representation of shape (batch_size, seq_len, hidden_dim)
            lengths: Optional tensor of actual sequence lengths for each sample
        Returns:
            X_tilde: Recovered sequences of shape (batch_size, seq_len, output_dim)
        """
        if lengths is not None:
            # Pack sequences for variable length handling
            H_packed = pack_padded_sequence(H, lengths, batch_first=True, enforce_sorted=False)
            gru_out_packed, _ = self.gru(H_packed)
            gru_out, _ = pad_packed_sequence(gru_out_packed, batch_first=True)
        else:
            gru_out, _ = self.gru(H)

        # Apply fully connected layer to all timesteps
        # gru_out shape: (batch_size, seq_len, hidden_dim)
        X_tilde = self.fc(gru_out)

        return X_tilde


class EmbedderRecoveryNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim=None):
        super(EmbedderRecoveryNetwork, self).__init__()
        if output_dim is None:
            output_dim = input_dim

        self.embedder = Embedder(input_dim, hidden_dim, num_layers)
        self.recovery = Recovery(hidden_dim, num_layers, output_dim)

    def forward(self, x, lengths=None):
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            lengths: Optional tensor of actual sequence lengths
        Returns:
            X_tilde: Recovered sequences
            H: Embedded representation
        """
        H = self.embedder(x, lengths)
        X_tilde = self.recovery(H, lengths)
        return X_tilde, H

    def embed(self, x, lengths=None):
        """Embed sequences to hidden representation"""
        return self.embedder(x, lengths)

    def recover(self, H, lengths=None):
        """Recover sequences from hidden representation"""
        return self.recovery(H, lengths)


# Example usage
if __name__ == "__main__":
    # Hyperparameters
    input_dim = 64
    hidden_dim = 64
    num_layers = 2
    batch_size = 32
    seq_len = 50

    # Create the network
    network = EmbedderRecoveryNetwork(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers
    )

    # Example input
    x = torch.randn(batch_size, seq_len, input_dim)
    lengths = torch.randint(20, seq_len + 1, (batch_size,))

    # Full forward pass
    recovered, embedded = network(x, lengths)
    print(f"Input shape: {x.shape}")
    print(f"Embedded shape: {embedded.shape}")
    print(f"Recovered shape: {recovered.shape}")

    # Use embedder and recovery separately
    embedded_separate = network.embed(x, lengths)
    recovered_separate = network.recover(embedded_separate, lengths)
    print(f"Separate embedding shape: {embedded_separate.shape}")
    print(f"Separate recovery shape: {recovered_separate.shape}")

    # Use components directly
    embedder = network.embedder
    recovery = network.recovery

    embedded_only = embedder(x, lengths)
    recovered_only = recovery(embedded_only, lengths)
    print(f"Embedder only output shape: {embedded_only.shape}")
    print(f"Recovery only output shape: {recovered_only.shape}")
