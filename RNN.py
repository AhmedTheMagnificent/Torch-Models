import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)  # Use hidden_size instead of hidden_size*sequence_length
        
    def forward(self, x):
        # Initialize hidden state for first input with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        
        # Forward propagate the RNN
        out, _ = self.rnn(x, h0)
        
        # Get the output from the last time step
        out = out[:, -1, :]  # Only use the output from the last time step
        
        # Pass through fully connected layer
        out = self.fc(out)
        return out

# Hyperparameters
input_size = 28  # Each row of the MNIST image is considered as a sequence input
sequence_length = 28  # MNIST images are 28x28
num_classes = 10
num_layers = 2
hidden_size = 256  # Fixing variable name consistency
learning_rate = 0.001
batch_size = 64
num_epochs = 5

# Load MNIST dataset
trainDataset = datasets.MNIST(root='datasets/', train=True, transform=transforms.ToTensor(), download=True)
trainLoader = DataLoader(dataset=trainDataset, batch_size=batch_size, shuffle=True)
testDataset = datasets.MNIST(root='datasets/', train=False, transform=transforms.ToTensor(), download=True)
testLoader = DataLoader(dataset=testDataset, batch_size=batch_size, shuffle=True)

# Initialize the model, loss function, and optimizer
model = RNN(input_size, hidden_size, num_layers, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(trainLoader):
        # Reshape data into [batch_size, sequence_length, input_size]
        data = data.reshape(data.shape[0], sequence_length, input_size)
        
        # Forward pass
        scores = model(data)
        loss = criterion(scores, targets)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# Check accuracy function
def checkAccuracy(loader, model):
    numCorrect = 0
    numSamples = 0
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            # Reshape input into [batch_size, sequence_length, input_size]
            x = x.reshape(x.shape[0], sequence_length, input_size)
            
            # Forward pass
            scores = model(x)
            _, predictions = scores.max(dim=1)  # Get the index of the max log-probability
            
            numCorrect += (predictions == y).sum()
            numSamples += predictions.size(0)
        
        accuracy = float(numCorrect) / float(numSamples) * 100
        print(f"Got {numCorrect} / {numSamples} with accuracy {accuracy:.2f}")
    model.train()
    return accuracy

# Checking accuracy on training and test data
checkAccuracy(trainLoader, model)
checkAccuracy(testLoader, model)
