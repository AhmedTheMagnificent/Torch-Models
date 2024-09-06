import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# Define the LeNet architecture
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.relu = nn.ReLU()
        self.pool = nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.conv1 = nn.Conv2d(1, 6, (5, 5))  # Input: (28x28x1), Output: (24x24x6)
        self.conv2 = nn.Conv2d(6, 16, (5, 5))  # Input: (12x12x6), Output: (8x8x16)
        self.fc1 = nn.Linear(16*4*4, 120)  # 16 channels with size 4x4 after pooling
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)  # 10 classes for output
        
    def forward(self, x):
        x = self.relu(self.conv1(x))  # Convolution 1 + ReLU
        x = self.pool(x)              # Pooling 1
        x = self.relu(self.conv2(x))  # Convolution 2 + ReLU
        x = self.pool(x)              # Pooling 2
        x = x.view(x.size(0), -1)     # Flatten
        x = self.relu(self.fc1(x))    # Fully connected layer 1 + ReLU
        x = self.relu(self.fc2(x))    # Fully connected layer 2 + ReLU
        x = self.fc3(x)               # Output layer (no activation, handled by loss function)
        return x

# Hyperparameters
learning_rate = 0.001
batch_size = 64
num_epochs = 10

# Load MNIST dataset
trainDataset = datasets.MNIST(root='datasets/', train=True, transform=transforms.ToTensor(), download=True)
trainLoader = DataLoader(dataset=trainDataset, batch_size=batch_size, shuffle=True)

testDataset = datasets.MNIST(root='datasets/', train=False, transform=transforms.ToTensor(), download=True)
testLoader = DataLoader(dataset=testDataset, batch_size=batch_size, shuffle=False)

# Initialize network
model = LeNet()

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training function
def train(model, loader, criterion, optimizer, num_epochs):
    model.train()  # Set the model to training mode
    for epoch in range(num_epochs):
        for batch_idx, (data, targets) in enumerate(loader):
            # Forward pass
            scores = model(data)
            loss = criterion(scores, targets)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Accuracy check function
def checkAccuracy(loader, model):
    numCorrect = 0
    numSamples = 0
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        for x, y in loader:
            scores = model(x)
            _, predictions = scores.max(1)
            numCorrect += (predictions == y).sum()
            numSamples += predictions.size(0)
        
        accuracy = float(numCorrect) / float(numSamples) * 100
        print(f'Accuracy: {accuracy:.2f}%')
    
    model.train()  # Set the model back to training mode

# Train the model
train(model, trainLoader, criterion, optimizer, num_epochs)

# Check accuracy on the training set
print("Checking accuracy on training set:")
checkAccuracy(trainLoader, model)

# Check accuracy on the test set
print("Checking accuracy on test set:")
checkAccuracy(testLoader, model)
