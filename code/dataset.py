"""Gas discrimination dataset (resistance-time waveforms)."""
import numpy as np
import torch
from torch.utils.data import Dataset


class GasDataset(Dataset):
    """Wraps preprocessed 500-point waveforms + labels.

    Training-only augmentation (applied when augment=True): Gaussian noise
    (sigma=0.01, p=0.5) and random 10% crop with zero-padding (p=0.5),
    matching the CNN training protocol. Validation/test data use augment=False.
    """

    def __init__(self, X, y, augment=False):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.augment = augment

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]
        if self.augment:
            if np.random.rand() < 0.5:
                x = x + torch.randn_like(x) * 0.01
            if np.random.rand() < 0.5:
                crop_len = int(0.9 * x.shape[0])
                start = np.random.randint(0, x.shape[0] - crop_len)
                cropped = x[start:start + crop_len]
                pad_len = x.shape[0] - cropped.shape[0]
                x = torch.cat([cropped, torch.zeros(pad_len)], dim=0)
        return x.unsqueeze(0), y
