import flwr as fl
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.model import IDSModel

# load dataset
X_test = np.load("data/processed/X_test.npy")
y_test = np.load("data/processed/y_test.npy")

X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

X_train = np.load("data/processed/X_train.npy")
y_train = np.load("data/processed/y_train.npy")

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)

dataset = TensorDataset(X_train, y_train)
loader = DataLoader(dataset, batch_size=64, shuffle=True)

model = IDSModel(X_train.shape[1])


class IDSClient(fl.client.NumPyClient):

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        model.load_state_dict(state_dict)

    def fit(self, parameters, config):

        self.set_parameters(parameters)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = torch.nn.BCELoss()

        model.train()

        for X_batch, y_batch in loader:

            optimizer.zero_grad()

            outputs = model(X_batch)

            loss = criterion(outputs, y_batch)

            loss.backward()

            optimizer.step()

        import random

        # simulate malicious client
        if random.random() < 0.3:
            print("⚠️ Malicious client sending poisoned weights")

            for param in model.parameters():
                param.data += torch.randn_like(param) * 5

        return self.get_parameters(config={}), len(loader.dataset), {}

    def evaluate(self, parameters, config):

        self.set_parameters(parameters)

        model.eval()

        with torch.no_grad():

            outputs = model(X_test)

            predictions = (outputs > 0.5).float()

            correct = (predictions == y_test).sum().item()

            accuracy = correct / len(y_test)

        print("Evaluation Accuracy:", accuracy)

        return 0.0, len(X_test), {"accuracy": accuracy}

        with torch.no_grad():

            outputs = model(X_test)

            predictions = (outputs > 0.5).float()

            correct = (predictions == y_test).sum().item()

            accuracy = correct / len(y_test)

        print("Evaluation Accuracy:", accuracy)

        return 0.0, len(X_test), {"accuracy": accuracy}


fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=IDSClient())