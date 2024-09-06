import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F 
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

class CNN(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=8, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.pool = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3,3), stride=(1,1), padding=(1,1))  # Renamed to conv2
        self.fc1 = nn.Linear(16*7*7, num_classes)  # Adjust input size to the fc layer
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)  # Flatten the tensor
        x = self.fc1(x)
        return x

# Hyperparameters
in_channels = 1
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 5

# Load Data
trainDataset = datasets.MNIST(root='datasets/', train=True, transform=transforms.ToTensor(), download=True)
trainLoader = DataLoader(dataset=trainDataset, batch_size=batch_size, shuffle=True)
testDataset = datasets.MNIST(root='datasets/', train=False, transform=transforms.ToTensor(), download=True)
testLoader = DataLoader(dataset=testDataset, batch_size=batch_size, shuffle=True)

# Initialize network
model = CNN(in_channels=in_channels, num_classes=num_classes)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(trainLoader):
        # Forward pass
        scores = model(data)
        loss = criterion(scores, targets)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
def checkAccuracy(loader, model):
    numCorrect = 0
    numSamples = 0
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            scores = model(x)
            _, predictions = scores.max(dim=1)  # Specify dim=1 here
            numCorrect += (predictions == y).sum()
            numSamples += predictions.size(0)
        accuracy = float(numCorrect) / float(numSamples) * 100
        print(f"Got {numCorrect} / {numSamples} with accuracy {accuracy:.2f}")
    model.train()
    return accuracy

# Check accuracy
checkAccuracy(trainLoader, model)
checkAccuracy(testLoader, model)
