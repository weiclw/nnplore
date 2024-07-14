import math
import torch
import torch.nn as nn
import torch.optim as optim


class SimpleSingleLayerNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(SimpleSingleLayerNN, self).__init__()
        self.fc = nn.Linear(input_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc(x)
        x = self.sigmoid(x)
        return x


class Runner():
    def __init__(self):
        self.num_epochs = 2000
        self.num_samples = 200
        self.input_size = 2
        self.output_size = 1
        self.data = None
        self.target = None
        self.model = SimpleSingleLayerNN(self.input_size, self.output_size)
        # Binary Cross Entropy Loss for binary classification.
        self.criterion = nn.BCELoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.25)

    def run(self):
        self.prepare_data()
        for epoch in range(self.num_epochs):
            self.run_epoch(epoch)
        self.validate()

    def prepare_data(self):
        self.data = torch.randn(self.num_samples, self.input_size)
        self.target = (torch.sum(self.data, dim=1) > 0).float().unsqueeze(1)

    def run_epoch(self, epoch):
        self.model.train()
        # Forward pass
        outputs = self.model(self.data)
        loss = self.criterion(outputs, self.target)
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if epoch % 20 == 0 or epoch == self.num_epochs - 1:
            print(f'Epoch [{epoch}], loss {loss.item():.4f}, prob {math.exp(-1.0*loss.item()):.4f}')

    def validate(self):
        print('Done without validation')


Runner().run()
