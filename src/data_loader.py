import torch
import pandas as pd
import torchaudio.transforms as T
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

        # Define SpecAugment transformations
        self.time_masking = T.TimeMasking(time_mask_param=20)
        self.freq_masking = T.FrequencyMasking(freq_mask_param=5)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature_tensor = torch.tensor(self.features[idx], dtype=torch.float32)
    
        # Reshape to (1, height, width) for CNN input
        feature_tensor = feature_tensor.unsqueeze(0)  # Adds a channel dimension
    
        label_tensor = torch.tensor(self.labels[idx], dtype=torch.long)
        return feature_tensor, label_tensor

