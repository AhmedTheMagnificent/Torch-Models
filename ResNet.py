import torch
import torch.nn as nn

class block(nn.Module):
    def __init__(self, inChannels, outChannels, identity_downsample=None, stride=1):
        super(block, self).__init__()
        self.expansion = 4
        self.conv1 = nn.Conv2d(inChannels, outChannels, kernel_size=1, stride=1, padding=0)
        self.batchnorm1 = nn.BatchNorm2d(outChannels)
        self.conv2 = nn.Conv2d(outChannels, outChannels, kernel_size=3, stride=stride, padding=1)
        self.batchnorm2 = nn.BatchNorm2d(outChannels)
        self.conv3 = nn.Conv2d(outChannels, outChannels * self.expansion, kernel_size=1, stride=1, padding=0)
        self.batchnorm3 = nn.BatchNorm2d(outChannels * self.expansion)
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample
        
    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.batchnorm3(x)
        
        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)
            
        x += identity
        return self.relu(x)

class ResNet(nn.Module):
    def __init__(self, block, layers, image_channels, num_classes):
        super(ResNet, self).__init__()      
        self.inChannels = 64
        self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3)
        self.batchnorm1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self.make_layers(block, layers[0], out_channels=64, stride=1)
        self.layer2 = self.make_layers(block, layers[1], out_channels=128, stride=2)
        self.layer3 = self.make_layers(block, layers[2], out_channels=256, stride=2)
        self.layer4 = self.make_layers(block, layers[3], out_channels=512, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * 4, num_classes)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x
        
    def make_layers(self, block, num_residual_blocks, out_channels, stride):
        identity_downsample = None
        layers = []

        if stride != 1 or self.inChannels != out_channels * 4:
            identity_downsample = nn.Sequential(
                nn.Conv2d(self.inChannels, out_channels * 4, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels * 4)
            )
            
        layers.append(block(self.inChannels, out_channels, identity_downsample, stride))
        self.inChannels = out_channels * 4
        
        for i in range(num_residual_blocks - 1):
            layers.append(block(self.inChannels, out_channels))
            
        return nn.Sequential(*layers)
    
def ResNet50(imgChannels=3, numClasses=1000):
    return ResNet(block, [3, 4, 6, 3], imgChannels, numClasses)

x = torch.randn(2, 3, 224, 224)
model = ResNet50()
output = model(x)
print(output.shape)