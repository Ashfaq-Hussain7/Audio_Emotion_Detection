# src/extract_features.py

import librosa
import numpy as np
import os
import pandas as pd

def extract_features(file_path, n_mfcc=40):
    try:
        y, sr = librosa.load(file_path, sr=None)  # Load audio
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        return np.mean(mfccs.T, axis=0)  # Average over time
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def process_dataset(dataset_path, n_mfcc=40):  # Add n_mfcc as a parameter
    data = []
    for root, _, files in os.walk(dataset_path):
        for file in files:
            if file.endswith(".wav"):
                label = os.path.basename(root)  # Folder name as label
                file_path = os.path.join(root, file)
                features = extract_features(file_path, n_mfcc)
                if features is not None:
                    data.append([file_path, label] + list(features))

    columns = ['file_path', 'label'] + [f'mfcc_{i}' for i in range(n_mfcc)]  # Use the parameter here
    return pd.DataFrame(data, columns=columns)

if __name__ == "__main__":
    dataset_path = "../dataset"
    df = process_dataset(dataset_path, n_mfcc=40)  # Pass n_mfcc explicitly
    df.to_csv("../features.csv", index=False)
    print("Feature extraction completed. Saved to features.csv")
