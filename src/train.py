import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import WeightedRandomSampler

# Hyperparameters
BATCH_SIZE = 32
EPOCHS = 100  # Increased max epochs but will use early stopping
LEARNING_RATE = 0.001

# Create model directory
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# 🧠 Simplified CNN Architecture
class ImprovedEmotionCNN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(ImprovedEmotionCNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(32)

        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)

        self.pool = nn.MaxPool1d(kernel_size=2)
        self.dropout = nn.Dropout(0.5)  # Increased dropout rate

        # Calculate flattened size based on input
        self.flattened_size = (input_size // 4) * 64

        self.fc1 = nn.Linear(self.flattened_size, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = x.view(x.size(0), -1)
        x = self.dropout(torch.relu(self.fc1(x)))
        return self.fc2(x)

# Dataset class
class EmotionDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features.astype(np.float32)
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return torch.tensor(self.features[idx]), torch.tensor(self.labels[idx], dtype=torch.long)

# Training function
def train_model(train_csv_path, val_csv_path=None):
    # Load training data
    df_train = pd.read_csv(train_csv_path)
    print(f"Training dataset shape: {df_train.shape}")
    print(f"Number of unique classes: {df_train['label'].nunique()}")
    print(f"Emotion distributions: {df_train['label'].value_counts()}")

    if df_train.isna().sum().sum() > 0:
        print("Removing rows with NaN values")
        df_train.dropna(inplace=True)

    # Initialize scaler and encoders
    scaler = StandardScaler()
    label_encoder = LabelEncoder()
    
    # Extract features and labels from training data
    X_train = df_train.iloc[:, 2:].values
    X_train = scaler.fit_transform(X_train)
    y_train = label_encoder.fit_transform(df_train['label'])
    class_names = label_encoder.classes_
    print(f"Class mapping: {dict(zip(class_names, range(len(class_names))))}")

    # Handle validation data
    if val_csv_path:
        # Use separate validation file
        df_val = pd.read_csv(val_csv_path)
        if df_val.isna().sum().sum() > 0:
            df_val.dropna(inplace=True)
        
        X_val = df_val.iloc[:, 2:].values
        X_val = scaler.transform(X_val)  # Use same scaler as training
        y_val = label_encoder.transform(df_val['label'])  # Use same encoder
        
        print(f"Validation dataset shape: {df_val.shape}")
        print(f"Using separate validation set")
    else:
        # Split training data for validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
        )
        print(f"Split validation from training data")

    print(f"Training set size: {len(X_train)}, Validation set size: {len(X_val)}")

    # Create PyTorch datasets
    train_dataset = EmotionDataset(X_train, y_train)
    val_dataset = EmotionDataset(X_val, y_val)

    # Handle class imbalance with weighted sampling
    class_counts = np.bincount(y_train)
    class_weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)
    sample_weights = class_weights[y_train]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Create model
    input_size = X_train.shape[1]
    num_classes = len(class_names)
    model = ImprovedEmotionCNN(input_size, num_classes)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)  # Increased weight decay
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.5, verbose=True)

    # Initialize tracking variables
    train_losses, train_accuracies, val_accuracies = [], [], []
    best_val_acc = 0
    patience = 7  # Early stopping patience
    patience_counter = 0

    # Training loop
    for epoch in range(EPOCHS):
        # Training phase
        model.train()
        running_loss = 0.0
        correct, total = 0, 0

        for features_batch, labels_batch in train_loader:
            features_batch = features_batch.unsqueeze(1)  # shape: [B, 1, F]
            
            optimizer.zero_grad()
            outputs = model(features_batch)
            loss = criterion(outputs, labels_batch)
            loss.backward()
            
            # Add gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels_batch.size(0)
            correct += (predicted == labels_batch).sum().item()

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = correct / total
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc)

        # Validation phase
        model.eval()
        val_correct, val_total = 0, 0
        val_losses = []
        
        confusion_matrix = torch.zeros(num_classes, num_classes)
        
        with torch.no_grad():
            for features_batch, labels_batch in val_loader:
                features_batch = features_batch.unsqueeze(1)
                outputs = model(features_batch)
                val_loss = criterion(outputs, labels_batch)
                val_losses.append(val_loss.item())
                
                _, predicted = torch.max(outputs, 1)
                val_total += labels_batch.size(0)
                val_correct += (predicted == labels_batch).sum().item()
                
                # Update confusion matrix
                for t, p in zip(labels_batch.view(-1), predicted.view(-1)):
                    confusion_matrix[t.long(), p.long()] += 1

        val_acc = val_correct / val_total
        val_accuracies.append(val_acc)
        
        # Update learning rate scheduler
        scheduler.step(val_acc)
        
        # Print progress
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}, Val Acc: {val_acc:.4f}")
        
        # Class-wise accuracy
        class_acc = confusion_matrix.diag() / confusion_matrix.sum(1)
        for i, acc in enumerate(class_acc):
            if not torch.isnan(acc):  # Avoid printing NaN values
                print(f"  Class {class_names[i]}: {acc.item():.4f}")
        
        # Early stopping check
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'train_acc': epoch_acc,
            }, os.path.join(MODEL_DIR, "best_emotion_model.pth"))
            print(f"✅ New best model saved with validation accuracy: {val_acc:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break

    # Save final model and metadata
    model_path = os.path.join(MODEL_DIR, "final_emotion_model.pth")
    torch.save(model.state_dict(), model_path)
    print(f"✅ Final model saved at {model_path}")

    mapping_path = os.path.join(MODEL_DIR, "class_mapping.csv")
    pd.DataFrame(list(enumerate(class_names)), columns=['index', 'emotion']).to_csv(mapping_path, index=False)
    print(f"✅ Class mapping saved at {mapping_path}")
    
    # Save scaler for future inference
    scaler_path = os.path.join(MODEL_DIR, "feature_scaler.pkl")
    import pickle
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"✅ Feature scaler saved at {scaler_path}")

    # Plot training curves
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label="Train Loss")
    plt.title("Loss over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    
    plt.subplot(1, 3, 2)
    plt.plot(train_accuracies, label="Train Accuracy")
    plt.plot(val_accuracies, label="Val Accuracy")
    plt.title("Accuracy over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    
    # Plot class-wise accuracies for the final epoch
    plt.subplot(1, 3, 3)
    class_acc_values = [acc.item() if not torch.isnan(acc) else 0 for acc in class_acc]
    plt.bar(class_names, class_acc_values)
    plt.title("Final Class-wise Accuracy")
    plt.xlabel("Emotion Class")
    plt.ylabel("Accuracy")
    plt.xticks(rotation=45)

    plt.tight_layout()
    plot_path = os.path.join(MODEL_DIR, "training_curves.png")
    plt.savefig(plot_path)
    print(f"✅ Training curves saved to {plot_path}")

if __name__ == "__main__":
    # Check if we have separate validation features
    if os.path.exists("features_val.csv"):
        train_model("features.csv", "features_val.csv")
    else:
        train_model("features.csv")