import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# VGG16 architecture
VGG16 = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M']

class VGG(nn.Module):
    def __init__(self, inChannels, numClasses):
        super(VGG, self).__init__()
        self.inChannels = inChannels
        self.convLayers = self.createConvLayers(VGG16)  # Use self.createConvLayers
        self.fcs = nn.Sequential(
            nn.Linear(512*7*7, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, numClasses)
        )
        
    def forward(self, x):
        x = self.convLayers(x)
        print(x.shape)  # Print the shape of the feature map before flattening
        x = x.reshape(x.shape[0], -1)
        x = self.fcs(x)
        return x

    
    def createConvLayers(self, architecture):
        layers = []
        inChannels = self.inChannels
        for i in architecture:
            if type(i) == int:
                outChannels = i
                layers += [
                    nn.Conv2d(inChannels, outChannels, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
                    nn.BatchNorm2d(outChannels),
                    nn.ReLU()
                ]
                inChannels = outChannels
            else:
                layers += [nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))]
            
        return nn.Sequential(*layers)  # Correct return indentation

# Example usage with MNIST dataset (change to your needs)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the dataset (you might want to use a different one for VGG16)
trainDataset = datasets.MNIST(root='datasets/', train=True, transform=transforms.ToTensor(), download=True)
trainLoader = DataLoader(dataset=trainDataset, batch_size=64, shuffle=True)
testDataset = datasets.MNIST(root='datasets/', train=False, transform=transforms.ToTensor(), download=True)
testLoader = DataLoader(dataset=testDataset, batch_size=64, shuffle=True)

# Model, loss function, and optimizer
model = VGG(inChannels=1, numClasses=10).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 2
for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(trainLoader):
        data = data.to(device)
        targets = targets.to(device)
        
        # Forward pass
        scores = model(data)
        loss = criterion(scores, targets)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient descent or Adam step
        optimizer.step()

# Function to check accuracy
def checkAccuracy(loader, model):
    numCorrect = 0
    numSamples = 0
    model.eval()
    
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            
            scores = model(x)
            _, predictions = scores.max(1)
            numCorrect += (predictions == y).sum()
            numSamples += predictions.size(0)
        
        print(f"Accuracy: {float(numCorrect)/float(numSamples)*100:.2f}%")
    
    model.train()

# Check accuracy on training and test set
checkAccuracy(trainLoader, model)
checkAccuracy(testLoader, model)
