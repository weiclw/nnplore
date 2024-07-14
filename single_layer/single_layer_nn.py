import math
import torch
import torch.nn as nn
import torch.optim as optim


class SingleLayerNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(SingleLayerNN, self).__init__()
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
        self.num_status_display = 10
        self.input_size = 2
        self.output_size = 1
        self.learning_rate = 0.25
        self.data = None
        self.target = None
        self.model = None
        self.criterion = None
        self.optimizer = None

    def run(self):
        self.prepare_model()
        self.prepare_data()
        for epoch in range(self.num_epochs):
            self.run_epoch(epoch)
        self.validate()

    def prepare_model(self):
        self.model = SingleLayerNN(self.input_size, self.output_size)
        # Binary Cross Entropy Loss for binary classification.
        self.criterion = nn.BCELoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate)
        

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
        # Maybe display status
        step = self.num_epochs / self.num_status_display
        if step == 0:
            step = 1
        if epoch % step == 0 or epoch == self.num_epochs - 1:
            print(f'Epoch [{epoch}], loss {loss.item():.4f}, prob {math.exp(-1.0*loss.item()):.4f}')

    def validate(self):
        print('Done without validation')


