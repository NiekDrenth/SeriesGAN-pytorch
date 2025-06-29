#!/usr/bin/env python3.13
from enum import auto
from data.data_utils import get_dataloader
from models.autoencoder import TemporalEncoder, TemporalDecoder, TemporalAutoEncoder
from models.embedder import EmbedderRecoveryNetwork
from models.discriminators import Discriminator, AE_Discriminator
from models.generator import Generator
from models.supervisor import Supervisor
import torch
import torch.nn as nn


# Network parameters
seq_len = 24
batch_size = 100
dim = 6
num_layers = 4
temporal_dimension = 16
num_epoch = 50

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_loader, train_df, test_loader, test_df, scaler = get_dataloader(
    "../data/stock_data.csv", seq_len=seq_len, batch_size=batch_size
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
