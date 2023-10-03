"""
    @Time    : 25/06/2023
    @Author  : qinghua
    @Software: PyCharm
    @File    : uncertainty_quantification.py

"""
from pytorch_lightning import Trainer
import torch
import torch.nn as nn
import torch.optim as optim
from simple_uq.models.pnn import PNN


class UTScorer():
    def __init__(self, dataloader):
        pnn = PNN(
            input_dim=1,
            output_dim=1,
            encoder_hidden_sizes=[64, 64],
            encoder_output_dim=64,
            mean_hidden_sizes=[],
            logvar_hidden_sizes=[],
        )
        trainer = Trainer()
        trainer.fit(pnn, dataloader, dataloader)
        self.trainer = trainer
        self.pnn = pnn
        self.x,self.y=dataloader

    def get_uncertainty_scores(self, dataloader):
        results = self.trainer.test(self.pnn, dataloader)
        pred_mean, pred_std = self.pnn.get_mean_and_standard_deviation(self.x.reshape(-1, 1))
        pred_mean = pred_mean.flatten()
        pred_std = pred_std.flatten()





class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, dropout_probability=0.0):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 50)
        self.fc2 = nn.Linear(50, 1)
        self.dropout = nn.Dropout(p=dropout_probability)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Bayesian Neural Network (with dropout)
class BayesianNN:
    def __init__(self, input_dim):
        self.model = NeuralNetwork(input_dim, dropout_probability=0.5)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.01)

    def train(self, dataloader, epochs=100):
        self.model.train()
        for epoch in range(epochs):
            for batch_X, batch_y in dataloader:
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def get_uncertainty_sample(self, sample):
        self.model.eval()
        with torch.no_grad():
            predictions = [self.model(sample.unsqueeze(0)).item() for _ in range(100)]
            uncertainty = torch.std(torch.tensor(predictions))
        return uncertainty.item()

    def get_uncertainty_batch(self, batch):
        return [self.get_uncertainty_sample(sample) for sample in batch]

class EnsembleNN:
    def __init__(self, input_dim, n_models=5):
        self.models = [NeuralNetwork(input_dim) for _ in range(n_models)]
        self.criterion = nn.MSELoss()
        self.optimizers = [optim.Adam(model.parameters(), lr=0.01) for model in self.models]

    def train(self, dataloader, epochs=100):
        for model, optimizer in zip(self.models, self.optimizers):
            model.train()
            for epoch in range(epochs):
                for batch_X, batch_y in dataloader:
                    outputs = model(batch_X)
                    loss = self.criterion(outputs, batch_y)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

    def get_uncertainty_sample(self, sample):
        with torch.no_grad():
            predictions = [model(sample.unsqueeze(0)).item() for model in self.models]
            uncertainty = torch.std(torch.tensor(predictions))
        return uncertainty.item()

    def get_uncertainty_batch(self, batch):
        return [self.get_uncertainty_sample(sample) for sample in batch]

