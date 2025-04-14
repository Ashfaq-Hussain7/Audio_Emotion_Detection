# data_loader.py

import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from extract_features import extract_features

class EmotionDataset(Dataset):
    def __init__(self, csv_path):
        data = pd.read_csv(csv_path)
        file_paths = data.iloc[:, 0].values
        labels = data.iloc[:, -1].values

        self.features = np.array([extract_features(fp) for fp in file_paths], dtype=np.float32)
        self.labels = LabelEncoder().fit_transform(labels)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return torch.tensor(self.features[idx]), torch.tensor(self.labels[idx], dtype=torch.long)
