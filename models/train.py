import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from model import IDSModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load dataset
X_train = np.load("data/processed/X_train.npy")
y_train = np.load("data/processed/y_train.npy")

# convert to tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)

dataset = TensorDataset(X_train, y_train)

loader = DataLoader(dataset, batch_size=256, shuffle=True)

input_size = X_train.shape[1]

model = IDSModel(input_size).to(device)

criterion = torch.nn.BCELoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 10

print("Starting training...")

for epoch in range(epochs):

    total_loss = 0

    for X_batch, y_batch in loader:

        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()

        outputs = model(X_batch)

        loss = criterion(outputs, y_batch)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")

torch.save(model.state_dict(), "models/ids_model.pth")

print("Training complete. Model saved!")