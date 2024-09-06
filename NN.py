import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F 
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

class NN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, num_classes)
        
    def forward(self, x):  # Corrected typo here
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x
    
input_size = 28*28
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 1

trainDataset = datasets.MNIST(root='datasets/', train=True, transform=transforms.ToTensor(), download=True)
trainLoader = DataLoader(dataset=trainDataset, batch_size=batch_size, shuffle=True)
testDataset = datasets.MNIST(root='datasets/', train=False, transform=transforms.ToTensor(), download=True)
testLoader = DataLoader(dataset=testDataset, batch_size=batch_size, shuffle=True)

model = NN(input_size=input_size, num_classes=num_classes)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(trainLoader):
        data = data.reshape(data.shape[0], -1)
        scores = model(data)
        loss = criterion(scores, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
def checkAccuracy(loader, model):
    numCorrect = 0
    numSamples = 0
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.reshape(x.shape[0], -1)
            scores = model(x)
            _, predictions = scores.max(dim=1)  # Specify dim=1 here
            numCorrect += (predictions == y).sum()
            numSamples += predictions.size(0)
        accuracy = float(numCorrect) / float(numSamples) * 100
        print(f"Got {numCorrect} / {numSamples} with accuracy {accuracy:.2f}")
    model.train()
    return accuracy  # Return accuracy instead of 'acc'
        
checkAccuracy(trainLoader, model)
checkAccuracy(testLoader, model)
