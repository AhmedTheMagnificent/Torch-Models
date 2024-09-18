import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# Define the Discriminator
class Discriminator(nn.Module):
    def __init__(self, img_dim):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            nn.Linear(img_dim, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )
        
    def forward(self, x):
        return self.disc(x)

# Define the Generator
class Generator(nn.Module):
    def __init__(self, z_dim, img_dim):
        super().__init__()
        self.gen = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.LeakyReLU(0.1),
            nn.Linear(256, img_dim),
            nn.Tanh()
        )
        
    def forward(self, x):
        return self.gen(x)

# Plot function to visualize real and fake images
def show_images(fake, real, epoch):
    fake = fake.reshape(-1, 1, 28, 28).detach().cpu()
    real = real.reshape(-1, 1, 28, 28).detach().cpu()

    fig, axs = plt.subplots(2, 8, figsize=(10, 5))
    for i in range(8):
        axs[0, i].imshow(fake[i].squeeze(), cmap='gray')
        axs[0, i].axis('off')
        axs[1, i].imshow(real[i].squeeze(), cmap='gray')
        axs[1, i].axis('off')
    fig.suptitle(f"Epoch {epoch}")
    plt.show()

# Hyperparameters
device = "cuda" if torch.cuda.is_available() else "cpu"
lr = 3e-4
z_dim = 64
image_dim = 28 * 28
batch_size = 32
num_epoch = 1011

# Initialize models
disc = Discriminator(image_dim).to(device)
gen = Generator(z_dim, image_dim).to(device)
fixed_noise = torch.randn((batch_size, z_dim)).to(device)

# Data loader and optimizer setup
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)
dataset = datasets.MNIST(root="dataset/", transform=transform, download=True)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
opt_disc = optim.Adam(disc.parameters(), lr=lr)
opt_gen = optim.Adam(gen.parameters(), lr=lr)
criterion = nn.BCELoss()

# Training loop
for epoch in range(num_epoch):
    for batch_idx, (real, _) in enumerate(loader):
        real = real.view(-1, 784).to(device)
        batch_size = real.shape[0]

        # Train Discriminator
        noise = torch.randn(batch_size, z_dim).to(device)
        fake = gen(noise)
        disc_real = disc(real).view(-1)
        lossD_real = criterion(disc_real, torch.ones_like(disc_real))
        disc_fake = disc(fake).view(-1)
        lossD_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
        lossD = (lossD_real + lossD_fake) / 2
        disc.zero_grad()
        lossD.backward(retain_graph=True)
        opt_disc.step()

        # Train Generator
        output = disc(fake).view(-1)
        lossG = criterion(output, torch.ones_like(output))
        gen.zero_grad()
        lossG.backward()
        opt_gen.step()

    # Display progress and images every 10 epochs
    if epoch % 10 == 0:
        print(
            f"Epoch [{epoch}/{num_epoch}] "
            f"Loss D: {lossD:.4f}, Loss G: {lossG:.4f}"
        )
        with torch.no_grad():
            fake = gen(fixed_noise)
            show_images(fake, real, epoch)
