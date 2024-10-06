import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# DCGAN Discriminator
class Discriminator(nn.Module):
    def __init__(self, channels_img, features_d, num_classes, img_size):
        super(Discriminator, self).__init__()
        self.img_size = img_size
        self.disc = nn.Sequential(
            nn.Conv2d(channels_img+1, features_d, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            self._block(features_d, features_d * 2, 4, 2, 1),
            self._block(features_d * 2, features_d * 4, 4, 2, 1),
            self._block(features_d * 4, features_d * 8, 4, 2, 1),
            nn.Conv2d(features_d * 8, 1, kernel_size=4, stride=2, padding=0),
        )
        self.embed = nn.Embedding(num_classes, img_size*img_size)

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            ),
            nn.InstanceNorm2d(out_channels, affine=True),  # Added batch normalization
            nn.LeakyReLU(0.2),
        )

    def forward(self, x, labels):
        embeddings = self.embed(labels).view(labels.shape[0], 1, self.img_size, self.img_size)
        x = torch.cat([x, embeddings], dim=1 )
        return self.disc(x)


# DCGAN Generator
class Generator(nn.Module):
    def __init__(self, channels_noise, channels_img, features_g, num_classes, img_size, embed_size):
        super(Generator, self).__init__()
        self.img_size = img_size
        self.gen = nn.Sequential(
            self._block(channels_noise+embed_size, features_g * 16, 4, 1, 0),  # 1x1 -> 4x4
            self._block(features_g * 16, features_g * 8, 4, 2, 1),  # 4x4 -> 8x8
            self._block(features_g * 8, features_g * 4, 4, 2, 1),  # 8x8 -> 16x16
            self._block(features_g * 4, features_g * 2, 4, 2, 1),  # 16x16 -> 32x32
            nn.ConvTranspose2d(
                features_g * 2, channels_img, kernel_size=4, stride=2, padding=1
            ),  # 32x32 -> 64x64
            nn.Tanh(),  # Output: 64x64
        )
        self.embed = nn.Embedding(num_classes, embed_size)

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),  # Added batch normalization
            nn.ReLU(),
        )

    def forward(self, x, labels):
        embeddings = self.embed(labels).unsqueeze(2).unsqueeze(3)
        x = torch.cat([x, embeddings], dim=1)
        return self.gen(x)


def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)


# Displaying real and fake images using matplotlib
def show_images(fake, real, epoch):
    fake = fake.detach().cpu()
    real = real.detach().cpu()

    fig, axs = plt.subplots(2, 8, figsize=(10, 5))
    for i in range(8):
        axs[0, i].imshow(fake[i].squeeze(), cmap='gray')
        axs[0, i].axis('off')
        axs[1, i].imshow(real[i].squeeze(), cmap='gray')
        axs[1, i].axis('off')
    fig.suptitle(f"Epoch {epoch}")
    plt.show()


# Hyperparameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LEARNING_RATE = 2e-4  # Adjusted learning rate for both generator and discriminator
BATCH_SIZE = 128
IMAGE_SIZE = 64
CHANNELS_IMG = 1
NUM_CLASSES = 10
GEN_EMBEDDINGS = 100
NOISE_DIM = 100
NUM_EPOCHS = 5
FEATURES_DISC = 64
FEATURES_GEN = 64

# Dataset and DataLoader
transforms = transforms.Compose(
    [
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]
)

dataset = datasets.MNIST(root="dataset/", train=True, transform=transforms, download=True)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Initialize models and optimizer
gen = Generator(NOISE_DIM, CHANNELS_IMG, FEATURES_GEN, NUM_CLASSES, IMAGE_SIZE, GEN_EMBEDDINGS).to(device)
disc = Discriminator(CHANNELS_IMG, FEATURES_DISC, NUM_CLASSES, IMAGE_SIZE).to(device)
initialize_weights(gen)
initialize_weights(disc)

opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
opt_disc = optim.Adam(disc.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))  # Adjusted learning rate
criterion = nn.BCELoss()

# Fixed noise for consistent visual comparison
fixed_noise = torch.randn(32, NOISE_DIM, 1, 1).to(device)

# Training loop
gen.train()
disc.train()

for epoch in range(NUM_EPOCHS):
    for batch_idx, (real, label) in enumerate(dataloader):
        real = real.to(device)
        noise = torch.randn(BATCH_SIZE, NOISE_DIM, 1, 1).to(device)
        fake = gen(noise, label)

        ### Train Discriminator: max log(D(x)) + log(1 - D(G(z)))
        disc_real = disc(real, label).reshape(-1)
        loss_disc_real = criterion(disc_real, torch.full_like(disc_real, 0.9))  # Label smoothing
        disc_fake = disc(fake.detach(), label).reshape(-1)
        loss_disc_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
        loss_disc = (loss_disc_real + loss_disc_fake) / 2
        disc.zero_grad()
        loss_disc.backward()
        opt_disc.step()

        ### Train Generator: min log(1 - D(G(z))) <-> max log(D(G(z)))
        output = disc(fake, label).reshape(-1)
        loss_gen = criterion(output, torch.ones_like(output))
        gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()

        # Print losses occasionally and show generated images using matplotlib
        if batch_idx % 100 == 0:
            print(
                f"Epoch [{epoch}/{NUM_EPOCHS}] Batch {batch_idx}/{len(dataloader)} "
                f"Loss D: {loss_disc:.4f}, Loss G: {loss_gen:.4f}"
            )

            with torch.no_grad():
                fake = gen(noise, label)
                show_images(fake[:8], real[:8], epoch)
