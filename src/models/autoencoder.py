#!/usr/bin/env python3.13
import torch.nn as nn
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class TemporalEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, temporal_dimension):
        super(TemporalEncoder, self).__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.0 if num_layers == 1 else 0.1
        )

        # Fully connected layer to compress to temporal dimension
        self.fc_compress = nn.Linear(hidden_dim, temporal_dimension)

    def forward(self, x, lengths=None):
        """
        args:
           x: Input tensor with dimension (batch_size, seq_len, input_dim)
           lengths: Optional sequence lengths for each sample.

        output:
           H: Compressed representation of sequence.
        """
        if lengths is not None:
            # Pack sequences for variable length handling
            x_packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
            gru_out_packed, hidden = self.gru(x_packed)
            gru_out, _ = pad_packed_sequence(gru_out_packed, batch_first=True)
        else:
            gru_out, hidden = self.gru(x)

        # Use the last hidden state from the last layer
        # hidden shape: (num_layers, batch_size, hidden_dim)
        last_hidden = hidden[-1]  # Take the last layer's hidden state

        # Compress to temporal dimension
        H = self.fc_compress(last_hidden)

        return H
class TemporalDecoder(nn.Module):
    def __init__(self, temporal_dimension, hidden_dim, num_layers, output_dim, max_seq_len):
        super(TemporalDecoder, self).__init__()
        self.temporal_dimension = temporal_dimension
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.max_seq_len = max_seq_len

        # Fully connected layer to expand compressed representation
        self.fc_expand = nn.Linear(temporal_dimension, max_seq_len * hidden_dim)

        # Multi-layer GRU
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.0 if num_layers == 1 else 0.1
        )

        # Fully connected layer to reconstruct original feature dimensions
        self.fc_reconstruct = nn.Linear(hidden_dim, output_dim)

    def forward(self, H_t, target_seq_len=None):
        """
        Args:
            H_t: Compressed temporal representation of shape (batch_size, temporal_dimension)
            target_seq_len: Optional target sequence length (defaults to max_seq_len)
        Returns:
            X_tilde: Reconstructed sequences of shape (batch_size, seq_len, output_dim)
        """
        batch_size = H_t.size(0)
        seq_len = target_seq_len if target_seq_len is not None else self.max_seq_len

        # Expand the compressed representation
        expanded_H = self.fc_expand(H_t)

        # Reshape to (batch_size, max_seq_len, hidden_dim)
        expanded_H = expanded_H.view(batch_size, self.max_seq_len, self.hidden_dim)

        # If target sequence length is different from max_seq_len, truncate or pad
        if seq_len != self.max_seq_len:
            if seq_len < self.max_seq_len:
                expanded_H = expanded_H[:, :seq_len, :]
            else:
                # Pad with zeros if needed
                padding = torch.zeros(batch_size, seq_len - self.max_seq_len, self.hidden_dim,
                                    device=expanded_H.device, dtype=expanded_H.dtype)
                expanded_H = torch.cat([expanded_H, padding], dim=1)

        # Pass through GRU
        gru_out, _ = self.gru(expanded_H)

        # Reconstruct original feature dimensions
        X_tilde = self.fc_reconstruct(gru_out)

        return X_tilde


class TemporalAutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, temporal_dimension, max_seq_len):
        super(TemporalAutoEncoder, self).__init__()
        self.encoder = TemporalEncoder(input_dim, hidden_dim, num_layers, temporal_dimension)
        self.decoder = TemporalDecoder(temporal_dimension, hidden_dim, num_layers, input_dim, max_seq_len)

    def forward(self, x, lengths=None, target_seq_len=None):
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            lengths: Optional tensor of actual sequence lengths
            target_seq_len: Optional target sequence length for reconstruction
        Returns:
            X_tilde: Reconstructed sequences
            H: Encoded temporal representation
        """
        H = self.encoder(x, lengths)
        X_tilde = self.decoder(H, target_seq_len)
        return X_tilde, H

    def encode(self, x, lengths=None):
        """Encode sequences to temporal representation"""
        return self.encoder(x, lengths)

    def decode(self, H_t, target_seq_len=None):
        """Decode temporal representation to sequences"""
        return self.decoder(H_t, target_seq_len)


if __name__ == '__main__':
    input_dim = 6  
    hidden_dim = 6 
    num_layers = 4  
    temporal_dimension = 16
    max_seq_len = 24
    batch_size = 64
    seq_len = 24

    autoencoder = TemporalAutoEncoder(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        temporal_dimension=temporal_dimension,
        max_seq_len=max_seq_len
    )
    x = torch.randn(batch_size, seq_len, input_dim)
    lengths = torch.randint(20, seq_len + 1, (batch_size,))  # Variable sequence lengths

    # Full autoencoder forward pass
    reconstructed, encoded = autoencoder(x, lengths)
    print(f"Input shape: {x.shape}")
    print(f"Encoded shape: {encoded.shape}")
    print(f"Reconstructed shape: {reconstructed.shape}")

    # Use encoder and decoder separately
    encoded_separate = autoencoder.encode(x, lengths)
    reconstructed_separate = autoencoder.decode(encoded_separate)
    print(f"Separate encoding shape: {encoded_separate.shape}")
    print(f"Separate decoding shape: {reconstructed_separate.shape}")

    # Use just the encoder
    encoder = autoencoder.encoder
    encoded_only = encoder(x, lengths)
    print(f"Encoder only output shape: {encoded_only.shape}")

    # Use just the decoder
    decoder = autoencoder.decoder
    decoded_only = decoder(encoded_only)
    print(f"Decoder only output shape: {decoded_only.shape}")
