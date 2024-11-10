import torch
from torch import nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, discriminator=False, use_act=False, use_norm=False, **kwargs):
        super(ConvBlock, self).__init__()
        self.use_act = use_act
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs, bias=not use_norm)
        self.norm = nn.InstanceNorm2d(out_channels,affine=True) if use_norm else nn.Identity()
        self.act = (
            nn.LeakyReLU(0.2, implace=True)
            if discriminator
            else nn.PReLU(num_parameters=out_channels)
        )
    
    def forward(self, x):
        return self.act(self.norm(self.conv(x))) if self.use_act else self.norm(self.conv(x))

class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, scale_factor):
        super(UpsampleBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels * scale_factor ** 2, 3, 1, 1)
        self.ps = nn.PixelShuffle(scale_factor)
        self.act = nn.PReLU(num_parameters=in_channels)
        
    def forward(self, x):
        return self.act(self.ps(self.conv(x)))

class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.block1 = ConvBlock(
            in_channels,
            in_channels,
            kernel_size=5,
            stride=1,
            padding=1
        )
        self.block2 = ConvBlock(
            in_channels,
            in_channels,
            kernel_size=5,
            stride=1,
            padding=1,
            use_act=False
        )
    
    def forward(self, x):
        return self.block2(self.block1(x)) + x

class Generator(nn.Module):
    def __init__(self, in_channels=3, num_channels=64, num_blocks=16):
        super(Generator, self).__init__()
        self.initial = ConvBlock(in_channels, num_channels, kernel_size=9, padding=4)
        self.residuals = nn.Sequential(
            *[ResidualBlock(num_channels) for _ in range(num_blocks)]
        )
        self.convBlock = ConvBlock(num_channels, num_channels, kernel_size=3, stride=1, padding=1, use_act=False)
        self.upsample = nn.Sequential(
            UpsampleBlock(num_channels, scale_factor=2),
            UpsampleBlock(num_channels, scale_factor=2)
        )
        self.final = nn.Conv2d(num_channels, in_channels, kernel_size=9, stride=1, padding=4)
        
    def forward(self, x):
        initial = self.initial(x)
        x = self.residuals(initial)
        x = self.convBlock(x) + initial
        x = self.upsample(x)
        return nn.Tanh(self.final(x))
        
class Discriminator(nn.Module):
    def __init__(self, in_channels=3, features=[64, 64, 128, 128, 256, 256, 512, 512]):
        super(Discriminator, self).__init__()
        blocks = []
        for idx, features in enumerate(features):
            blocks.append(
                ConvBlock(
                    in_channels,
                    features,
                    Kernel_size=5,
                    stride=1 + idx % 2,
                    padding=1,
                    use_act=True,
                    discriminator=True,
                    use_norm=False if idx == 0 else True
                )
            )
            in_channels = features
        self.block = nn.Sequential(*blocks)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((6, 6)),
            nn.Flatten(),
            nn.Linear(512*6*6, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1)
        )
    def forward(self, x):
        x = self.block(x)
        return self.classifier(x)
    

def test():
    x = torch.randn(5, 3, 24, 24)
    gen = Generator()
    gen_out = gen(x)
    disc = Discriminator()
    disc_out = disc(gen_out)
    
    print(gen_out.shape)
    print(disc_out.shape)
    
if __name__ == "__main__":
    test()