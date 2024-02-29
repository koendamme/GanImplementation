from GAN import Discriminator, Generator
from dataset import Dataset
import torch
from utils import weights_init
from torch.utils.data import DataLoader
from tqdm import tqdm


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

noise_vector_length = 100
n_epochs = 100
lr = .0001

dataset = Dataset()
dataloader = DataLoader(dataset.get_subset(5), batch_size=16, shuffle=True)

G = Generator(noise_vector_length, dataset.get_subset(5)[0][0].shape)
G.apply(weights_init)
D = Discriminator(dataset.get_subset(5)[0][0].shape)
D.apply(weights_init)

G = G.to(device)
D = D.to(device)

criterion = torch.nn.BCELoss()

gen_optimizer = torch.optim.Adam(G.parameters(), lr=lr)
discr_optimizer = torch.optim.Adam(D.parameters(), lr=lr)

for i in tqdm(range(n_epochs)):
    running_discr_loss, running_gen_loss = 0.0, 0.0

    for batch, _ in dataloader:
        batch = batch.to(device)

        # Update Discriminator
        D.zero_grad()
        # Train with real batch
        discr_real_output = D(batch.flatten(start_dim=1))
        loss_D_real = criterion(discr_real_output, torch.ones(batch.shape[0], 1, device=device))
        loss_D_real.backward()

        # Train with fake batch
        noise_batch = torch.normal(0, 1, size=(batch.shape[0], noise_vector_length), device=device)
        fake = G(noise_batch)
        discr_fake_output = D(fake.detach())

        loss_D_fake = criterion(discr_fake_output, torch.zeros(batch.shape[0], 1, device=device))
        loss_D_fake.backward()
        loss_D = loss_D_real + loss_D_fake
        running_discr_loss += loss_D.item()

        discr_optimizer.step()

        # Update generator
        G.zero_grad()
        discr_output = D(fake)
        loss_G = criterion(discr_output, torch.ones(batch.shape[0], 1, device=device))
        running_gen_loss += loss_G.item()

        loss_G.backward()
        gen_optimizer.step()

    # print(f"Discriminiator loss: {running_discr_loss/(len(dataloader))}, Generator loss: {running_gen_loss/len(dataloader)}")