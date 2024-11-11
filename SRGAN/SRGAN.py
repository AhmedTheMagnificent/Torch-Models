import torch
from torch import nn
from torchvision.models import vgg19
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

LEARNING_RATE = 2e-4
NUM_EPOCHS = 16
HIGH_RES = 96
LOW_RES = HIGH_RES // 4
IMG_CHANNELS = 3

highres_transform = A.Compose(
    [
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ToTensorV2()
    ]
)

lowres_transform = A.Compose(
    [
        A.Resize(width=LOW_RES, height=LOW_RES, interpolation=Image.BICUBIC),
        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1]),
        ToTensorV2()
    ]
)

both_transforms = A.Compose(
    [
        A.RandomCrop(width=HIGH_RES, height=HIGH_RES),
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
    ]
)

test_transform = A.Compose(
    [
        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1]),
        ToTensorV2(),
    ]
)

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, discriminator=False, use_act=False, use_norm=False, **kwargs):
        super(ConvBlock, self).__init__()
        self.use_act = use_act
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs, bias=not use_norm)
        self.norm = nn.InstanceNorm2d(out_channels, affine=True) if use_norm else nn.Identity()
        self.act = nn.LeakyReLU(0.2, inplace=True) if discriminator else nn.PReLU(num_parameters=out_channels)
    
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
        self.block1 = ConvBlock(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.block2 = ConvBlock(in_channels, in_channels, kernel_size=3, stride=1, padding=1, use_act=False)
    
    def forward(self, x):
        return self.block2(self.block1(x)) + x

class Generator(nn.Module):
    def __init__(self, in_channels=3, num_channels=64, num_blocks=16):
        super(Generator, self).__init__()
        self.initial = ConvBlock(in_channels, num_channels, kernel_size=9, padding=4)
        self.residuals = nn.Sequential(*[ResidualBlock(num_channels) for _ in range(num_blocks)])
        self.convBlock = ConvBlock(num_channels, num_channels, kernel_size=3, stride=1, padding=1, use_act=False)
        self.upsample = nn.Sequential(UpsampleBlock(num_channels, scale_factor=2), UpsampleBlock(num_channels, scale_factor=2))
        self.final = nn.Conv2d(num_channels, in_channels, kernel_size=9, stride=1, padding=4)
        
    def forward(self, x):
        initial = self.initial(x)
        x = self.residuals(initial)
        x = self.convBlock(x) + initial
        x = self.upsample(x)
        return nn.Tanh()(self.final(x))

class Discriminator(nn.Module):
    def __init__(self, in_channels=3, features=[64, 64, 128, 128, 256, 256, 512, 512]):
        super(Discriminator, self).__init__()
        blocks = []
        for idx, feature in enumerate(features):
            blocks.append(ConvBlock(in_channels, feature, kernel_size=5, stride=1 + idx % 2, padding=1, use_act=True, discriminator=True, use_norm=False if idx == 0 else True))
            in_channels = feature
        self.block = nn.Sequential(*blocks)
        self.classifier = nn.Sequential(nn.AdaptiveAvgPool2d((6, 6)), nn.Flatten(), nn.Linear(512*6*6, 1024), nn.LeakyReLU(0.2, inplace=True), nn.Linear(1024, 1))
    
    def forward(self, x):
        x = self.block(x)
        return self.classifier(x)
    
class VGGLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.vgg = vgg19(pretrained=True).features[:36].eval()
        self.loss = nn.MSELoss()
        for param in self.vgg.parameters():
            param.requires_grad = False
        
    def forward(self, input, target):
        vgg_input_features = self.vgg(input)
        vgg_target_features = self.vgg(target)
        return self.loss(vgg_input_features, vgg_target_features)

class MyImageFolder(Dataset):
    def __init__(self, root_dir):
        super(MyImageFolder, self).__init__()
        self.data = []
        self.root_dir = root_dir
        self.class_names = os.listdir(root_dir)
        
        for idx, name in enumerate(self.class_names):
            files = os.listdir(os.path.join(root_dir, name))
            self.data += list(zip(files, [idx] * len(files)))
            
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_file, label = self.data[idx]
        root_and_dir = os.path.join(self.root_dir, self.class_names[label])

        image = np.array(Image.open(os.path.join(root_and_dir, img_file)))
        image = both_transforms(image=image)["image"]
        high_res = highres_transform(image=image)["image"]
        low_res = lowres_transform(image=image)["image"]
        return low_res, high_res

def test():
    dataset = MyImageFolder(root_dir=r"A:\ProgrmmingStuff\new_data")
    loader = DataLoader(dataset, batch_size=1, num_workers=8)

    for low_res, high_res in loader:
        print(low_res.shape)
        print(high_res.shape)
    
if __name__ == "__main__":
    os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
    test()
