import torch
import torch.nn as nn

class InceptionNet(nn.Module):
    def __init__(self, inChannels=3, numClasses=1000):
        super(InceptionNet, self).__init__()
        self.conv1 = convBlock(inChannels, 64, kernel_size=(7, 7), stride=(2, 2), padding=3)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2 = convBlock(64, 192, kernel_size=3)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        
        self.inception3a = inceptionBlock(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = inceptionBlock(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2)
        
        self.inception4a = inceptionBlock(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = inceptionBlock(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = inceptionBlock(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = inceptionBlock(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = inceptionBlock(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(kernel_size=3, stride=2)
        
        self.inception5a = inceptionBlock(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = inceptionBlock(832, 384, 192, 384, 48, 128, 128)
        self.avgpool = nn.AvgPool2d(kernel_size=3)
        self.dropout = nn.Dropout(p=0.4)
        self.fc1 = nn.Linear(1024, numClasses)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)
        
        x = self.inception4a(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        x = self.inception4e(x)
        x = self.maxpool4(x)
        
        x = self.inception5a(x)
        x = self.inception5b(x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor before feeding into fully connected layer
        x = self.dropout(x)
        x = self.fc1(x)
        return x
        
class inceptionBlock(nn.Module):
    def __init__(self, inChannels, out_1x1, red_3x3, out_3x3, red_5x5, out_5x5, out_1x1_pool):
        super(inceptionBlock, self).__init__()
        self.branch1 = convBlock(inChannels, out_1x1, kernel_size=1)
        self.branch2 = nn.Sequential(
            convBlock(inChannels, red_3x3, kernel_size=1),
            convBlock(red_3x3, out_3x3, kernel_size=3, stride=1, padding=1)
        )
        self.branch3 = nn.Sequential(
            convBlock(inChannels, red_5x5, kernel_size=1),
            convBlock(red_5x5, out_5x5, kernel_size=5, padding=2)
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            convBlock(inChannels, out_1x1_pool, kernel_size=1)
        )
        
    def forward(self, x):
        # Concatenate the outputs from all branches
        return torch.cat([self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)], 1)

class convBlock(nn.Module):
    def __init__(self, inChannels, outChannels, **kwargs):
        super(convBlock, self).__init__()
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(inChannels, outChannels, **kwargs)
        self.batchnorm = nn.BatchNorm2d(outChannels)
        
    def forward(self, x):
        return self.relu(self.batchnorm(self.conv(x)))


model = InceptionNet()

# Set the model to evaluation mode (not necessary here but good practice)
model.eval()

# Create a random input tensor (1 image, 3 channels, 224x224 pixels)
input_tensor = torch.randn(4, 3, 224, 224)

# Perform a forward pass through the network
output = model(input_tensor)

# Print the output shape
print(f"Output shape: {output}")  # Should be [1, 1000]