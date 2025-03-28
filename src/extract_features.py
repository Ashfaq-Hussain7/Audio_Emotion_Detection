import librosa
import numpy as np
import os
import pandas as pd

# Emotion label mapping from filename
EMOTION_MAP = {
    "ANG": "angry",
    "HAP": "happy",
    "SAD": "sad",
    "NEU": "neutral",
    "FEA": "fear",
    "DIS": "disgust",
    "SUR": "surprise"
}

def get_emotion_from_filename(filename):
    """Extract emotion from filename based on predefined emotion codes."""
    parts = filename.split("_")  # Split by underscore
    for part in parts:
        if part in EMOTION_MAP:
            return EMOTION_MAP[part]  # Map to emotion name
    return "unknown"  # Default if no match found

def extract_features(file_path, n_mfcc=40):
    try:
        y, sr = librosa.load(file_path, sr=None)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        return np.mean(mfccs.T, axis=0)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def process_dataset(dataset_path, n_mfcc=40):
    data = []
    for root, _, files in os.walk(dataset_path):
        for file in files:
            if file.endswith(".wav"):
                file_path = os.path.join(root, file)
                label = get_emotion_from_filename(file)  # Extract label from filename
                features = extract_features(file_path, n_mfcc)
                if features is not None:
                    data.append([file_path, label] + list(features))

    columns = ['file_path', 'label'] + [f'mfcc_{i}' for i in range(n_mfcc)]
    return pd.DataFrame(data, columns=columns)

if __name__ == "__main__":
    dataset_path = "C:/Users/ashfa/OneDrive/Desktop/Audio_Emotion_Detection/dataset"
    df = process_dataset(dataset_path, n_mfcc=40)
    df.to_csv("C:/Users/ashfa/OneDrive/Desktop/Audio_Emotion_Detection/features.csv", index=False)
    print("Feature extraction completed. Saved to features.csv")
