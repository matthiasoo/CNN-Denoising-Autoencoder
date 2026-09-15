import numpy as np
import torch
from torch.utils.data import Dataset

class SignalDataset(Dataset):
    def __init__(self):
        self.fs = 48000
        self.chunk_size = 128
        self.t = torch.arange(self.chunk_size, dtype=torch.float32) / self.fs
        self.channels = 1
        self.n_samples = 40000
        self.f_min = 50
        self.f_max = 1000
        self.phase_min = 0.0
        self.phase_max = 2.0
        self.std_min = 0.1
        self.std_max = 0.5

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        freq = float(np.random.uniform(self.f_min, self.f_max))
        phase = float(np.random.uniform(self.phase_min, self.phase_max) * np.pi)
        std = float(np.random.uniform(self.std_min, self.std_max))

        x_clean = torch.sin(2 * np.pi * freq * self.t + phase).unsqueeze(0)
        noise = torch.randn_like(x_clean) * std
        x_noisy = x_clean + noise

        return x_noisy, x_clean