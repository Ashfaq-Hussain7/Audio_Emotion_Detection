import torch
import pandas as pd
from torch.utils.data import Dataset

class EmotionDataset(Dataset):
    def __init__(self, csv_path, augment=True):
        """Load features and labels from CSV and apply augmentation if needed"""
        self.data = pd.read_csv(csv_path)
        self.augment = augment

        # Ensure CSV contains necessary columns
        if 'label' not in self.data.columns:
            raise ValueError("CSV file must contain a 'label' column.")

        # Extract numerical features (assuming features start from the 3rd column)
        self.features = self.data.iloc[:, 2:].values  # Adjust if needed
        self.labels = self.data['label'].astype('category').cat.codes.values  # Convert labels to numerical categories

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        # Reshape to (1, feature_length) for 1D convolution
        feature_tensor = torch.tensor(self.features[idx], dtype=torch.float32).unsqueeze(0)
        label_tensor = torch.tensor(self.labels[idx], dtype=torch.long)
        return feature_tensor, label_tensor