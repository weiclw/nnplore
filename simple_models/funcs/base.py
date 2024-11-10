import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Generate data points for f(x) = sin(x)
x_data = np.linspace(-10, 10, 1000)
y_data = np.sin(x_data)

# Convert to PyTorch tensors
x_tensor = torch.tensor(x_data, dtype=torch.float32).view(-1, 1)  # Input needs to be a column vector
y_tensor = torch.tensor(y_data, dtype=torch.float32).view(-1, 1)  # Target needs to be a column vector

# Step 2: Define a simple neural network model with a single hidden layer and Sigmoid activation
class SimpleNNSigmoid(nn.Module):
    def __init__(self):
        super(SimpleNNSigmoid, self).__init__()
        # Define a single hidden layer with 64 neurons
        self.hidden = nn.Linear(1, 64)  # 1 input feature, 64 neurons in the hidden layer
        self.output = nn.Linear(64, 1)   # Output layer, 1 output (for the scalar function value)

    def forward(self, x):
        # Apply the hidden layer with Sigmoid activation
        x = torch.sigmoid(self.hidden(x))
        # Output layer
        x = self.output(x)
        return x

# Step 3: Create an instance of the neural network and define the loss function and optimizer
model = SimpleNNSigmoid()

# Use Mean Squared Error Loss since it's a regression problem
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Step 4: Train the neural network
num_epochs = 1000
for epoch in range(num_epochs):
    # Forward pass: Compute predicted y by passing x to the model
    y_pred = model(x_tensor)
    
    # Compute loss
    loss = criterion(y_pred, y_tensor)
    
    # Backward pass: Compute gradients
    optimizer.zero_grad()
    loss.backward()
    
    # Update weights
    optimizer.step()
    
    # Print loss every 100 epochs for monitoring
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Step 5: Test the model and plot the results
with torch.no_grad():
    y_pred = model(x_tensor).numpy()

# Plot the original sine function and the neural network approximation
plt.figure(figsize=(10, 6))
plt.plot(x_data, y_data, label='sin(x)', color='blue', linewidth=2)
plt.plot(x_data, y_pred, label='NN Approximation', color='red', linestyle='dashed', linewidth=2)
plt.legend()
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Function Approximation: f(x) = sin(x) using a Neural Network (Sigmoid Activation)')
plt.grid(True)
plt.show()
