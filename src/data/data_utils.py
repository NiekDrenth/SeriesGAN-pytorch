#!/usr/bin/env python3.13

import polars as pl
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader

class TimeSeriesDataset(Dataset):
    def __init__(self, data, seq_len):
        self.data = data
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data) - self.seq_len + 1

    def __getitem__(self, idx):
        return self.data[idx:idx + self.seq_len]

def get_dataloader(
    csv_file: str,
    seq_len: int,
    batch_size: int = 64,
    normalize: bool = True,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    test_size: float = 0.2,
) -> tuple[DataLoader, DataLoader, pl.DataFrame, pl.DataFrame, MinMaxScaler]:
    """
    Loads CSV time series, splits into train/test, normalizes and sequences it, returns DataLoaders and DataFrames.

    Args:
        csv_file (str): Path to CSV file.
        seq_len (int): Length of each sequence.
        batch_size (int): Batch size for training.
        normalize (bool): Whether to apply MinMax scaling.
        shuffle (bool): Shuffle the dataset or not.
        num_workers (int): DataLoader worker processes.
        pin_memory (bool): Pin memory for faster GPU transfer.
        test_size (float): Proportion of dataset to include in test split (0.0-1.0).

    Returns:
        tuple: (train_dataloader, test_dataloader, train_df, test_df, scaler)
    """
    # Load data
    df = pl.read_csv(csv_file)

    # Calculate split index for time series (no shuffling for temporal order)
    n_samples = len(df)
    split_idx = int(n_samples * (1 - test_size))

    # Split dataframes
    train_df = df[:split_idx]
    test_df = df[split_idx:]

    # Convert to numpy
    train_data = train_df.to_numpy()
    test_data = test_df.to_numpy()

    # Normalize if requested
    scaler = None
    if normalize:
        scaler = MinMaxScaler()
        train_data = scaler.fit_transform(train_data)
        test_data = scaler.transform(test_data)  # Use fitted scaler on test data

    # Convert to tensors
    train_tensor = torch.tensor(train_data, dtype=torch.float32)
    test_tensor = torch.tensor(test_data, dtype=torch.float32)

    # Create datasets
    train_dataset = TimeSeriesDataset(train_tensor, seq_len)
    test_dataset = TimeSeriesDataset(test_tensor, seq_len)

    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,  # Don't shuffle test data
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    return train_dataloader, test_dataloader, train_df, test_df, scaler

if __name__ == '__main__':
    data_test, _, _, _, _ = get_dataloader("../../data/stock_data.csv", seq_len = 24, normalize = False)
    print(F"Amount of batches in stock dataset {len(data_test.dataset)}")
