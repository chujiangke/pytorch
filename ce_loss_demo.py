import torch
import torch.nn as nn
import torch.optim as optim

# Define a neural network class
class SimpleClassifier(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SimpleClassifier, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)

# Training and testing process
class Trainer:
    def __init__(self, model, criterion, optimizer):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer

    def train(self, inputs, targets, epochs=10):
        self.model.train()
        for epoch in range(epochs):
            # Forward pass
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)

            # Backward pass and optimization
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

    def test(self, inputs, targets):
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(inputs)
            predicted = torch.argmax(outputs, dim=1)
            accuracy = (predicted == targets).float().mean().item()
            print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Define input and target tensors
inputs = torch.randn(5, 4)  # 5 samples, 4 features
targets = torch.tensor([0, 2, 1, 0, 2])  # 3 classes (0, 1, 2)

# Initialize model, loss function, and optimizer
model = SimpleClassifier(input_dim=4, output_dim=3)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Create Trainer instance
trainer = Trainer(model, criterion, optimizer)

# Train the model
trainer.train(inputs, targets, epochs=20)

# Test the model
trainer.test(inputs, targets)