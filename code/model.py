"""1D-CNN classifier for gas discrimination (H2S / CH4 / CO2 / air)."""
import torch.nn as nn
from torch.nn import functional as F


class CNN1D(nn.Module):
    """Two-conv + two-FC 1D-CNN operating on the 500-point waveform.

    Architecture (2,059,204 trainable parameters):
        Conv1d(1->32, k=7, pad=3) -> ReLU
        Conv1d(32->64, k=5, pad=2) -> ReLU
        MaxPool1d(2) -> Dropout(0.5)
        Flatten(16000) -> Linear(16000->128) -> ReLU
        Linear(128->num_classes)
    """

    def __init__(self, input_length, num_classes=4):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=7, padding=3)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(0.5)
        self.flattened_size = (input_length // 2) * 64
        self.fc1 = nn.Linear(self.flattened_size, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)
