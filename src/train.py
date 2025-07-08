#!/usr/bin/env python3.13
from enum import auto
from data.data_utils import get_dataloader
from models.autoencoder import  TemporalAutoEncoder
from models.embedder import EmbedderRecoveryNetwork
from models.discriminators import Discriminator, AE_Discriminator
from models.generator import Generator
from models.supervisor import Supervisor
import torch
import torch.nn as nn
import argparse
import os
import sys

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
        default=1000,
        help="Number of generations (default: 1000)"
    )
    parser.add_argument(
        "--filename",
        type=str,
        default="stock_data.csv",
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



save_dir = "model_weights"
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
    f"gens{args.num_generations}"
)

model_dir = os.path.join("model_weights", filename_stem)

if os.path.exists(model_dir):
    print(f"[WARNING] Model directory already exists: {model_dir}")
    response = input("Do you want to overwrite it? [y/N]: ").strip().lower()
    if response != 'y':
        print("Exiting without training.")
        sys.exit(0)

# Create the directory if it doesn't exist (or was just confirmed for overwrite)
os.makedirs(model_dir, exist_ok=True)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print("Training on GPU")
else:
    print("Training on CPU")

# device = torch.device("cpu")
train_loader, train_df, test_loader, test_df, scaler = get_dataloader(
    csv_file = filename, seq_len=seq_len, batch_size=batch_size
)


# Initialize the models
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

mseLoss = nn.MSELoss()

# temporal_optimizer = torch.optim.Adam(autoencoder.parameters(), lr=1e-3)
# embedder_optimizer = torch.optim.Adam(embedder.parameters(), lr=1e-4, betas=(0.5,0.9))
# discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=1e-4)
# ae_discriminator_optimizer = torch.optim.Adam(ae_discriminator.parameters(), lr=1e-4)
# generator_optimizer = torch.optim.Adam(generator.parameters(), lr=1e-5)
# supervisor_optimizer = torch.optim.Adam(supervisor.parameters(), lr=1e-5)
temporal_optimizer = torch.optim.RMSprop(autoencoder.parameters(), lr=1e-4, alpha=0.99)
embedder_optimizer = torch.optim.RMSprop(embedder.parameters(), lr=1e-4, alpha=0.99)
discriminator_optimizer = torch.optim.RMSprop(discriminator.parameters(), lr=1e-4, alpha=0.99)
ae_discriminator_optimizer = torch.optim.RMSprop(ae_discriminator.parameters(), lr=1e-4, alpha=0.99)
generator_optimizer = torch.optim.RMSprop(generator.parameters(), lr=1e-4, alpha=0.99)
supervisor_optimizer = torch.optim.RMSprop(supervisor.parameters(), lr=1e-4, alpha=0.99)


# Train temporal autoencoder
for epoch in range(num_epoch):
    for batch in train_loader:
        batch = batch.to(device)

        predictions = autoencoder(batch)
        temporal_loss = mseLoss(predictions[0], batch)

        temporal_optimizer.zero_grad()
        temporal_loss.backward()
        temporal_optimizer.step()

    print(f"Epoch {epoch + 1}, Loss: {temporal_loss.item():.4f}")

print(
    f"predictions shape: {predictions[0].shape} \n embedded shape: {predictions[1].shape}"
)

print("Start embedder network training")
# Embedding network training
for epoch in range(num_epoch):
    for batch in train_loader:
        batch = batch.to(device)

        reconstruction, embedding = embedder(batch)
        Y_ae_fake = ae_discriminator(reconstruction)
        fake_data = torch.randn_like(batch)
        Y_ae_real = ae_discriminator(batch)

        lambda_c = 0.001
        embedder_loss = mseLoss(reconstruction, batch) + lambda_c * mseLoss(
            Y_ae_fake, torch.ones_like(Y_ae_fake)
        )

        embedder_optimizer.zero_grad()
        embedder_loss.backward()
        embedder_optimizer.step()

        Y_ae_fake = ae_discriminator(reconstruction.detach())
        ae_discriminator_loss = mseLoss(
            Y_ae_real, torch.ones_like(Y_ae_real)
        ) + mseLoss(Y_ae_fake, torch.zeros_like(Y_ae_fake))

        if ae_discriminator_loss > 0.15:
            ae_discriminator_optimizer.zero_grad()
            ae_discriminator_loss.backward()
            ae_discriminator_optimizer.step()

    print(
        f"Epoch {epoch + 1}, Loss: {embedder_loss.item():.4f}, AE discriminator loss: {ae_discriminator_loss.item():.4f}"
    )

print("Start training with supervised loss only")
for epoch in range(num_epoch):
    for batch in train_loader:
        batch = batch.to(device)
        random_batch = torch.rand_like(batch)

        _, real_embedded = embedder(batch)
        generated = generator(random_batch)
        supervised = supervisor(generated)

        supervised_loss = mseLoss(supervised[:, :-2, :], real_embedded[:, 2:, :])

        generator_optimizer.zero_grad()
        supervisor_optimizer.zero_grad()

        # Backward pass
        supervised_loss.backward()

        # Update parameters
        generator_optimizer.step()
        supervisor_optimizer.step()

    print(f"Epoch {epoch + 1}, Supervised loss: {supervised_loss}")

print("Start Joint training")

for epoch in range(num_epoch*2):
    i = 0
    for batch in train_loader:
        batch = batch.to(device)
        random_batch = torch.rand_like(batch)

        generated = generator(random_batch)
        supervised = supervisor(generated)
        real_recovered, real_embedded = embedder(batch)

        # G_loss_u_totall: diff ones vs y_fake
        Y_fake = discriminator(supervised)
        Y_real = discriminator(real_embedded)

        Y_ae_fake = ae_discriminator(real_recovered)
        Y_ae_real = ae_discriminator(batch)

        Y_fake_e = discriminator(generated)
        synthetic_recovered = embedder.recover(supervised)
        synthetic_recovered_second = embedder.recover(generated)
        Y_ae_fake_e = ae_discriminator(synthetic_recovered)
        Y_ae_fake_e_second = ae_discriminator(synthetic_recovered_second)

        X_t, H_t = autoencoder(batch)
        H_t_mean = torch.mean(H_t)
        H_t_hat = autoencoder.encode(supervised)
        H_t_hat_mean = torch.mean(H_t_hat)
        H_t_std = torch.std(H_t)
        H_t_hat_std = torch.std(H_t_hat)
        # generator_loss = (
        #     mseLoss(Y_fake, torch.ones_like(Y_fake))
        #     + mseLoss(Y_fake_e, torch.ones_like(Y_fake_e))
        #     + mseLoss(Y_ae_fake_e, torch.ones_like(Y_ae_fake_e))
        #     + mseLoss(Y_ae_fake_e_second, torch.ones_like(Y_ae_fake_e_second))
        # )
        gamma = 1.0
        beta = 1.0

        # Suppose these are already computed earlier
        G_loss_U = mseLoss(Y_fake, torch.ones_like(Y_fake))
        G_loss_U_e = mseLoss(Y_fake_e, torch.ones_like(Y_fake_e))
        G_loss_U_ae = mseLoss(Y_ae_fake_e, torch.ones_like(Y_ae_fake_e))
        G_loss_U_ae_e = mseLoss(Y_ae_fake_e_second, torch.ones_like(Y_ae_fake_e_second))

        G_loss_S = mseLoss(supervised[:, :-2, :], real_embedded[:, 2:, :])
        G_loss_ts = mseLoss(H_t_mean, H_t_hat_mean) + mseLoss(H_t_std, H_t_hat_std)

        std_X_hat = torch.sqrt(
            torch.var(synthetic_recovered, dim=0, unbiased=False) + 1e-6
        )
        std_X = torch.sqrt(torch.var(batch, dim=0, unbiased=False) + 1e-6)
        G_loss_V1 = torch.mean(torch.abs(std_X_hat - std_X))

        mean_X_hat = torch.mean(synthetic_recovered, dim=0)
        mean_X = torch.mean(batch, dim=0)
        G_loss_V2 = torch.mean(torch.abs(mean_X_hat - mean_X))

        G_loss_V = G_loss_V1 + G_loss_V2

        G_loss = (
            G_loss_U
            + gamma * G_loss_U_e
            + beta * (G_loss_U_ae + gamma * G_loss_U_ae_e)
            + 20 * torch.sqrt(G_loss_S)
            + 10 * G_loss_V
            + 20 * G_loss_ts
        )

        generator_optimizer.zero_grad()
        supervisor_optimizer.zero_grad()
        G_loss.backward()
        generator_optimizer.step()
        supervisor_optimizer.step()

        lambda_e = 0.001
        # #train embedder
        # supervised_loss = mseLoss(supervised[:, :-2, :], real_embedded[:, 2:, :])
        # embedder_loss = torch.sqrt(
        #     mseLoss(real_recovered, batch)
        #     + lambda_e* mseLoss(Y_ae_fake, torch.ones_like(Y_ae_fake))
        #     + lambda_e* 0.1* mseLoss(Y_ae_fake_e, torch.ones_like(Y_ae_fake_e))
        #     + lambda_e* 0.1* mseLoss(Y_ae_fake_e_second, torch.ones_like(Y_ae_fake_e_second))
        # ) + 0.01* supervised_loss

        # embedder_optimizer.zero_grad()
        # embedder_loss.backward()
        # embedder_optimizer.step()
        supervised_detached = supervised.detach()
        generated_detached = generated.detach()
        synthetic_recovered_detached = synthetic_recovered.detach()
        synthetic_recovered_second_detached = synthetic_recovered_second.detach()
        real_embedded_detached = real_embedded.detach()
        Y_ae_fake_detached = ae_discriminator(real_recovered.detach())
        Y_ae_fake_e_detached = ae_discriminator(synthetic_recovered_detached)
        Y_ae_fake_e_second_detached = ae_discriminator(generated_detached)
        real_recovered_detached = real_recovered.detach()
        supervised_loss = mseLoss(
            supervised_detached[:, :-2, :], real_embedded_detached[:, 2:, :]
        )

        embedder_loss = (
            torch.sqrt(
                mseLoss(real_recovered_detached, batch)
                + lambda_e
                * 0.1
                * mseLoss(Y_ae_fake_detached, torch.ones_like(Y_ae_fake_detached))
                + lambda_e
                * 0.1
                * mseLoss(Y_ae_fake_e_detached, torch.ones_like(Y_ae_fake_e_detached))
                + lambda_e
                * 0.1
                * mseLoss(
                    Y_ae_fake_e_second_detached,
                    torch.ones_like(Y_ae_fake_e_second_detached),
                )
            )
            + 0.01 * supervised_loss
        )

        embedder_optimizer.zero_grad()
        embedder_loss.backward()
        embedder_optimizer.step()

        if i % 2 == 0:
            Y_real = discriminator(real_embedded_detached)
            Y_fake = discriminator(supervised_detached)
            Y_fake_e = discriminator(generated_detached)
            discriminator_loss = (
                mseLoss(Y_real, torch.ones_like(Y_real))
                + mseLoss(Y_fake, torch.zeros_like(Y_fake))
                + mseLoss(Y_fake_e, torch.zeros_like(Y_fake_e))
            )
            if discriminator_loss>0.15:
                discriminator_optimizer.zero_grad()
                discriminator_loss.backward()
                discriminator_optimizer.step()

            Y_ae_real = ae_discriminator(batch)
            Y_ae_fake = ae_discriminator(real_recovered_detached)
            Y_ae_fake_e = ae_discriminator(synthetic_recovered_detached)

            Y_ae_fake_e_second = ae_discriminator(generated_detached)

            ae_loss = (
                mseLoss(Y_ae_real, torch.ones_like(Y_ae_real))
                + mseLoss(Y_ae_fake, torch.zeros_like(Y_ae_fake))
                + mseLoss(Y_ae_fake_e, torch.zeros_like(Y_ae_fake_e))
                + mseLoss(Y_ae_fake_e_second, torch.zeros_like(Y_ae_fake_e_second))
            )
            if ae_loss >0.15:
                ae_discriminator_optimizer.zero_grad()
                ae_loss.backward()
                ae_discriminator_optimizer.step()
        i+=1
    print(
        f"Epoch: {epoch + 1}. \n Generator_loss: {G_loss}, \n embedder_loss: {embedder_loss}, \n Discriminator loss: {discriminator_loss}, \n Ae_discriminator_loss: {ae_loss}"
    )





# Save each model's weights separately
torch.save(autoencoder.state_dict(), os.path.join(model_dir, f"autoencoder_{filename_stem}.pt"))
torch.save(embedder.state_dict(), os.path.join(model_dir, f"embedder_{filename_stem}.pt"))
torch.save(discriminator.state_dict(), os.path.join(model_dir, f"discriminator_{filename_stem}.pt"))
torch.save(ae_discriminator.state_dict(), os.path.join(model_dir, f"ae_discriminator_{filename_stem}.pt"))
torch.save(generator.state_dict(), os.path.join(model_dir, f"generator_{filename_stem}.pt"))
torch.save(supervisor.state_dict(), os.path.join(model_dir, f"supervisor_{filename_stem}.pt"))

print("All model weights saved successfully.")
