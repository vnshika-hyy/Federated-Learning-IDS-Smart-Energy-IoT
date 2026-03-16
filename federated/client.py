import flwr as fl
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import sys
import os
import random
import uuid

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.model import IDSModel


# ==========================
# Load dataset
# ==========================

X_train = np.load("data/processed/X_train.npy")
y_train = np.load("data/processed/y_train.npy")

X_test = np.load("data/processed/X_test.npy")
y_test = np.load("data/processed/y_test.npy")


# ==========================
# Non-IID Client Simulation
# ==========================

CLIENT_ID = str(uuid.uuid4())[:8]
print("Client ID:", CLIENT_ID)

normal_idx = np.where(y_train == 0)[0]
attack_idx = np.where(y_train == 1)[0]

if int(CLIENT_ID[-1], 16) % 3 == 0:

    print("Client type: Mostly Normal Traffic")

    selected = np.concatenate([
        normal_idx[:4000],
        attack_idx[:1000]
    ])

elif int(CLIENT_ID[-1], 16) % 3 == 1:

    print("Client type: Mostly Attack Traffic")

    selected = np.concatenate([
        normal_idx[:1000],
        attack_idx[:4000]
    ])

else:

    print("Client type: Balanced Traffic")

    size = min(len(normal_idx), len(attack_idx))

    selected = np.concatenate([
        normal_idx[:size],
        attack_idx[:size]
    ])


X_train = X_train[selected]
y_train = y_train[selected]


# ==========================
# Convert to Torch Tensors
# ==========================

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)

X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)


dataset = TensorDataset(X_train, y_train)

loader = DataLoader(dataset, batch_size=128, shuffle=True)


# ==========================
# Model
# ==========================

model = IDSModel(X_train.shape[1])


# ==========================
# Flower Client
# ==========================

class IDSClient(fl.client.NumPyClient):

    def get_parameters(self, config):

        return [val.cpu().numpy() for _, val in model.state_dict().items()]


    def set_parameters(self, parameters):

        params_dict = zip(model.state_dict().keys(), parameters)

        state_dict = {k: torch.tensor(v) for k, v in params_dict}

        model.load_state_dict(state_dict, strict=True)


    # ======================
    # Local Training
    # ======================

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


        # ======================
        # Byzantine Attack Simulation
        # ======================

        if random.random() < 0.3:

            print("⚠️ Malicious client sending poisoned weights")

            for param in model.parameters():

                param.data += torch.randn_like(param) * 0.5


        return self.get_parameters(config={}), len(loader.dataset), {}


    # ======================
    # Evaluation
    # ======================

    def evaluate(self, parameters, config):

        self.set_parameters(parameters)

        criterion = torch.nn.BCELoss()

        model.eval()

        with torch.no_grad():

            outputs = model(X_test)

            loss = criterion(outputs, y_test).item()

            probs = outputs.cpu().numpy()

            predictions = (outputs > 0.5).float().cpu().numpy()

            y_true = y_test.cpu().numpy()


        accuracy = accuracy_score(y_true, predictions)

        precision = precision_score(y_true, predictions)

        recall = recall_score(y_true, predictions)

        f1 = f1_score(y_true, predictions)

        roc_auc = roc_auc_score(y_true, probs)


        print("\n===== Evaluation =====")

        print("Loss:", loss)

        print("Accuracy:", accuracy)

        print("Precision:", precision)

        print("Recall:", recall)

        print("F1 Score:", f1)

        print("ROC AUC:", roc_auc)


        return loss, len(X_test), {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "roc_auc": roc_auc
        }


# ==========================
# Start Flower Client
# ==========================

fl.client.start_numpy_client(
    server_address="127.0.0.1:8080",
    client=IDSClient(),
)