#!/usr/bin/env python3.13

from data.data_utils import get_dataloader
from models.autoencoder import TemporalAutoEncoder
from models.embedder import EmbedderRecoveryNetwork
from models.discriminators import Discriminator, AE_Discriminator
from models.generator import Generator
from models.supervisor import Supervisor
import argparse
import sys
import os
import subprocess
from train import train
import torch
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Parse arguments for your model."
    )

    parser.add_argument(
        "--seq_len",
        type=int,
        default=24,
        help="Sequence length (default: 24)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=100,
        help="Batch size (default: 100)"
    )
    parser.add_argument(
        "--dim",
        type=int,
        default=6,
        help="Dimension size (default: 6)"
    )
    parser.add_argument(
        "--num_layers",
        type=int,
        default=4,
        help="Number of layers (default: 4)"
    )
    parser.add_argument(
        "--temporal_dimension",
        type=int,
        default=16,
        help="Temporal dimension (default: 16)"
    )
    parser.add_argument(
        "--num_epoch",
        type=int,
        default=50,
        help="Number of epochs (default: 50)"
    )
    parser.add_argument(
        "--num_generations",
        type=int,
        default=10,
        help="Number of generations (default: 10)"
    )
    parser.add_argument(
        "--filename",
        type=str,
        default="../data/stock_data.csv",
        help='Filename (default: "stock_data.csv")'
    )

    args = parser.parse_args()
    return args

args = parse_args()

# Network parameters
seq_len = args.seq_len
batch_size = args.batch_size
dim = args.dim
num_layers = args.num_layers
temporal_dimension = args.temporal_dimension
num_epoch = args.num_epoch
num_generations = args.num_generations
filename = args.filename


basename = os.path.basename(filename)

# Remove the extension
namefile, _ = os.path.splitext(basename)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_loader, train_df, test_loader, test_df, scaler = get_dataloader(
    csv_file=filename, seq_len=seq_len, batch_size=batch_size
)

save_dir = "../model_weights"
os.makedirs(save_dir, exist_ok=True)

# Build a filename stem from hyperparameters
filename_stem = (
    f"data{namefile}_"
    f"seq{args.seq_len}_"
    f"batch{args.batch_size}_"
    f"dim{args.dim}_"
    f"layers{args.num_layers}_"
    f"tempdim{args.temporal_dimension}_"
    f"epochs{args.num_epoch}_"
)

model_dir = os.path.join("../model_weights", filename_stem)
print(model_dir)
if not os.path.exists(model_dir):
    print(f"[WARNING] Model directory does not exist: {model_dir}")
    response = input("Do you want to train this model and generate data? [Y/n]: ").strip().lower()
    if response == 'y':
        print("Starting training")
        train(train_loader, seq_len, batch_size, dim, num_layers, temporal_dimension, num_epoch, filename)
    else:
        sys.exit(0)

autoencoder = TemporalAutoEncoder(
    dim,
    dim,
    num_layers=num_layers,
    temporal_dimension=temporal_dimension,
    max_seq_len=24,
).to(device)

embedder = EmbedderRecoveryNetwork(dim, dim, num_layers).to(device)
discriminator = Discriminator(dim, num_layers).to(device)
ae_discriminator = AE_Discriminator(dim, num_layers, seq_len).to(device)
generator = Generator(dim, dim, num_layers).to(device)
supervisor = Supervisor(dim, num_layers, dim).to(device)

# Load weights
autoencoder.load_state_dict(
    torch.load(os.path.join(model_dir, f"autoencoder_{filename_stem}.pt"), map_location=device)
)
embedder.load_state_dict(
    torch.load(os.path.join(model_dir, f"embedder_{filename_stem}.pt"), map_location=device)
)
discriminator.load_state_dict(
    torch.load(os.path.join(model_dir, f"discriminator_{filename_stem}.pt"), map_location=device)
)
ae_discriminator.load_state_dict(
    torch.load(os.path.join(model_dir, f"ae_discriminator_{filename_stem}.pt"), map_location=device)
)
generator.load_state_dict(
    torch.load(os.path.join(model_dir, f"generator_{filename_stem}.pt"), map_location=device)
)
supervisor.load_state_dict(
    torch.load(os.path.join(model_dir, f"supervisor_{filename_stem}.pt"), map_location=device)
)

# Optional: switch all models to eval mode if you're running inference
autoencoder.eval()
embedder.eval()
discriminator.eval()
ae_discriminator.eval()
generator.eval()
supervisor.eval()

all_data = []
all_probs = []

with torch.no_grad():
    for _ in range(num_generations):
        z = torch.rand([batch_size, seq_len, dim]).to(device)
        generated = generator(z)
        supervised = supervisor(generated)
        synthetic_recovered = embedder.recover(supervised)
        prob_real = ae_discriminator(synthetic_recovered)

        # Move tensors to CPU and detach
        recovered_np = synthetic_recovered.cpu().numpy()
        prob_real_np = prob_real.cpu().numpy()

        recovered_2d = recovered_np.reshape(-1, dim)

        # Apply inverse scaling
        recovered_scaled_2d = scaler.inverse_transform(recovered_2d)

        # Reshape back to [batch_size, seq_len, dim]
        recovered_scaled = recovered_scaled_2d.reshape(
            recovered_np.shape[0],
            recovered_np.shape[1],
            recovered_np.shape[2]
        )
        for i in range(recovered_scaled.shape[0]):
            flattened_recovered = recovered_scaled[i].flatten()   # shape: seq_len * dim
            probs = prob_real_np[i]                           # shape: dim
            combined = list(flattened_recovered) + list(probs)
            all_data.append(combined)

# Create a DataFrame
df = pd.DataFrame(all_data)

# Optionally name the columns
columns = [f"feature_{t}_{d}" for t in range(seq_len) for d in range(dim)]
prob_cols = [f"prob_real_d{d}" for d in range(dim)]
all_cols = columns + prob_cols
df.columns = all_cols


# Save CSV
csv_path = os.path.join(model_dir, f"generated_data_{filename_stem}.csv")
df.to_csv(csv_path, index=False)




print(f"Saved generated data to: {csv_path}")
