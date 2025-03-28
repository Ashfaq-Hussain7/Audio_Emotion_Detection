import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader

# Hyperparameters
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001

# Ensure the models directory exists
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# Custom Dataset Class
class EmotionDataset(Dataset):
    def __init__(self, csv_file):
        df = pd.read_csv(csv_file)
        self.features = df.iloc[:, 2:].values.astype(np.float32)
        self.labels = LabelEncoder().fit_transform(df['label'])
        
    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return torch.tensor(self.features[idx]), torch.tensor(self.labels[idx], dtype=torch.long)

# Define Model
class EmotionModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(EmotionModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Training Function
def train_model(csv_path):
    dataset = EmotionDataset(csv_path)
    input_size = dataset.features.shape[1]
    num_classes = len(set(dataset.labels))
    
    train_data, val_data = train_test_split(dataset, test_size=0.2, random_state=42)
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)
    
    model = EmotionModel(input_size, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        correct = 0
        
        for features, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()
        
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss / len(train_loader)}, Accuracy: {correct / len(train_data)}")
    
    # Save the model inside the "models" folder
    MODEL_PATH = os.path.join(MODEL_DIR, "emotion_model.pth")
    torch.save(model.state_dict(), MODEL_PATH)

    print(f"✅ Model saved at {MODEL_PATH}")

if __name__ == "__main__":
    train_model("C:/Users/ashfa/OneDrive/Desktop/Audio_Emotion_Detection/features.csv")
