# src/data_loader.py

import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class EmotionDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        self.labels = self.data['label'].astype('category').cat.codes
        self.features = self.data.iloc[:, 2:].values.astype(np.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.features[idx]), torch.tensor(self.labels[idx])