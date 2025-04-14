# extract_features.py

import librosa
import numpy as np
import os
import pandas as pd
from augmentation import augment_audio

# Emotion code to label mapping
EMOTION_MAP = {
    "ANG": "angry",
    "HAP": "happy",
    "SAD": "sad",
    "NEU": "neutral",
    "FEA": "fear",
    "DIS": "disgust",
    "SUR": "surprise"
}

def extract_label_from_filename(filename):
    parts = filename.split('_')
    if len(parts) >= 3:
        return EMOTION_MAP.get(parts[2], "unknown")
    return "unknown"

def extract_features(file_path, n_mfcc=40, augment=False):
    try:
        y, sr = librosa.load(file_path, sr=None)

        if augment:
            y = augment_audio(y, sr)

        # Base MFCCs
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        
        # Add delta and delta-delta features
        delta_mfccs = librosa.feature.delta(mfccs)
        delta2_mfccs = librosa.feature.delta(mfccs, order=2)
        
        # Extract spectral features
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        
        # Combine features
        features = np.concatenate([
            np.mean(mfccs.T, axis=0),           # Mean of MFCCs
            np.std(mfccs.T, axis=0),            # Standard deviation of MFCCs
            np.mean(delta_mfccs.T, axis=0),     # Mean of delta MFCCs
            np.mean(delta2_mfccs.T, axis=0),    # Mean of delta-delta MFCCs
            np.mean(spectral_centroid.T, axis=0),
            np.mean(spectral_bandwidth.T, axis=0),
            np.mean(spectral_rolloff.T, axis=0)
        ])
        
        return features
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def process_dataset(dataset_path, n_mfcc=40, augment=False):
    data = []
    for file in os.listdir(dataset_path):
        if file.endswith(".wav"):
            file_path = os.path.join(dataset_path, file)
            label = extract_label_from_filename(file)
            features = extract_features(file_path, n_mfcc, augment=augment)
            if features is not None and label != "unknown":
                data.append([file_path, label] + list(features))

    # Create columns based on feature length
    if data:
        feature_count = len(data[0]) - 2  # Exclude file_path and label
        columns = ['file_path', 'label'] + [f'feature_{i}' for i in range(feature_count)]
        return pd.DataFrame(data, columns=columns)
    else:
        print("No valid data extracted. Check your dataset path and file formats.")
        return pd.DataFrame()

if __name__ == "__main__":
    dataset_path = "dataset"  # assumes relative to project root
    
    # Create augmented training data
    df = process_dataset(dataset_path, n_mfcc=40, augment=True)
    df.to_csv("features.csv", index=False)
    print("✅ Feature extraction completed. Saved to features.csv")
    
    # Create non-augmented features for validation
    df_val = process_dataset(dataset_path, n_mfcc=40, augment=False)
    df_val.to_csv("features_val.csv", index=False)
    print("✅ Validation feature extraction completed. Saved to features_val.csv")