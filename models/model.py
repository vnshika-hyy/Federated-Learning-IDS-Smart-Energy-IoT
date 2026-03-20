import torch
import torch.nn as nn
import torch.nn.functional as F


class IDSModel(nn.Module):

    def __init__(self, input_size):

        super(IDSModel, self).__init__()

        # CNN layers
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)

        self.pool = nn.MaxPool1d(2)

        # LSTM
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=64,
            num_layers=1,
            batch_first=True
        )

        # Fully connected layers
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 1)

        self.dropout = nn.Dropout(0.3)

    def forward(self, x):

        # reshape for CNN
        x = x.unsqueeze(1)

        x = F.relu(self.conv1(x))
        x = self.pool(x)

        x = F.relu(self.conv2(x))
        x = self.pool(x)

        # reshape for LSTM
        x = x.permute(0, 2, 1)

        lstm_out, _ = self.lstm(x)

        x = lstm_out[:, -1, :]

        x = self.dropout(F.relu(self.fc1(x)))

        x = torch.sigmoid(self.fc2(x))

        return x