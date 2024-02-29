from GAN import Discriminator, Generator
from dataset import Dataset
import torch
from utils import weights_init, generate_images
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb


config = dict(
    noise_vector_length=150,
    n_epochs=100,
    lr=.001,
    batch_size=32,
    p_dropout_G=.2,
    p_dropout_D=.2,
    digit=5,
    architecture="GAN"
)

wandb.init(project="MNIST-Simple-GAN", config=config)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

dataset = Dataset()
dataloader = DataLoader(dataset.get_subset(5), batch_size=config["batch_size"], shuffle=True, pin_memory=True)

G = Generator(config["noise_vector_length"], dataset.get_subset(5)[0][0].shape, p_dropout=config["p_dropout_G"]).to(device)
G.apply(weights_init)
D = Discriminator(dataset.get_subset(5)[0][0].shape, p_dropout=config["p_dropout_G"]).to(device)
D.apply(weights_init)

criterion = torch.nn.BCELoss()

gen_optimizer = torch.optim.Adam(G.parameters(), lr=config["lr"])
discr_optimizer = torch.optim.Adam(D.parameters(), lr=config["lr"])

wandb.watch(G, criterion, log="all", log_freq=10)
wandb.watch(D, criterion, log="all", log_freq=10)

for i_epoch in tqdm(range(config["n_epochs"])):
    running_discr_loss, running_gen_loss = 0.0, 0.0

    for i_batch, (batch, _) in enumerate(dataloader):
        batch = batch.to(device)

        # Update Discriminator
        D.zero_grad()
        # Train with real batch
        discr_real_output = D(batch.flatten(start_dim=1))
        loss_D_real = criterion(discr_real_output, torch.ones(batch.shape[0], 1, device=device))
        loss_D_real.backward()

        # Train with fake batch
        noise_batch = torch.normal(0, 1, size=(batch.shape[0], config["noise_vector_length"]), device=device)
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

        if (i_batch + 1) % 10 == 0:
            wandb.log({
                "D_loss": loss_D,
                "G_loss": loss_G,
                "epoch": i_epoch
            })

    images = generate_images(8, G, dataset.get_subset(5)[0][0].shape, config["noise_vector_length"], device)
    images_to_log = [wandb.Image(image, caption="Example image") for image in images]
    wandb.log({
        "example_batch": images_to_log
    })