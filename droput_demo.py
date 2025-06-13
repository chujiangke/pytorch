from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from loguru import logger
import numpy as np
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_digits
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch import optim


class SimpleNN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)  # Dropout layer with 50% probability

        self.fc2 = nn.Linear(128, 128)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=0.5)  # Another dropout layer

        self.fc3 = nn.Linear(128, 128)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(p=0.5)  # Another dropout layer

        self.fc4 = nn.Linear(128, 128)
        self.relu4 = nn.ReLU()
        self.dropout4 = nn.Dropout(p=0.5)

        self.fc5 = nn.Linear(128, num_classes)  # Corrected output layer

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.relu2(x)       
        x = self.dropout2(x)

        x = self.fc3(x)
        x = self.relu3(x)
        x = self.dropout3(x)
        
        x = self.fc4(x)
        x = self.relu4(x)
        x = self.dropout4(x)
        x = self.fc5(x)  # Use the correct output layer
        return x

def train_model(model, train_loader, criterion, optimizer, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Print statistics
            running_loss += loss.item()
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')

# Example usage
if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1)),  # Flatten the image
    ])

    train_dataset = datasets.MNIST(root='./mnist_data', train=True, download=True, transform=transform)
    train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)

    input_size = 784  # MNIST images are 28x28
    num_classes = 10
    model = SimpleNN(input_size, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_model(model, train_loader, criterion, optimizer, num_epochs=30)
 
