import librosa
import soundfile as sf
import numpy as np
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from augmentation import augment_audio  # Ensure this module is correctly implemented
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

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
    parts = filename.split("_")  
    for part in parts:
        if part in EMOTION_MAP:
            return EMOTION_MAP[part]
    return "unknown"

def extract_advanced_features(file_path, n_mfcc=40, augment=False):
    """
    Extract multiple acoustic features from an audio file.

    Args:
        file_path (str): Path to the audio file
        n_mfcc (int): Number of MFCCs to extract
        augment (bool): Whether to apply augmentation

    Returns:
        np.ndarray: Combined feature vector or None if extraction fails.
    """
    try:
        if not os.path.exists(file_path):
            print(f"❌ File not found: {file_path}")
            return None

        with sf.SoundFile(file_path) as sound_file:
            sr = sound_file.samplerate
            duration = len(sound_file) / sr
            if duration < 0.1 or duration > 10:
                print(f"⚠️ Skipping abnormal duration file: {file_path}")
                return None

        y, sr = librosa.load(file_path, sr=None)

        if y is None or len(y) == 0:
            print(f"❌ Empty or corrupted file: {file_path}")
            return None

        if augment:
            try:
                y = augment_audio(y, sr)
            except Exception as aug_err:
                print(f"⚠️ Augmentation error in {file_path}: {aug_err}")

        # Feature Extraction
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        mfccs_mean = np.mean(mfccs.T, axis=0)
        mfccs_std = np.std(mfccs.T, axis=0)

        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]

        zero_crossing_rate = librosa.feature.zero_crossing_rate(y)[0]
        rms_energy = librosa.feature.rms(y=y)[0]

        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma = np.nan_to_num(chroma)

        # Feature Aggregation
        features = np.concatenate([
            mfccs_mean,  
            mfccs_std,   
            [np.mean(spectral_centroid)],
            [np.std(spectral_centroid)],
            [np.mean(spectral_rolloff)],
            [np.std(spectral_rolloff)],
            [np.mean(spectral_bandwidth)],
            [np.std(spectral_bandwidth)],
            [np.mean(zero_crossing_rate)],
            [np.mean(rms_energy)],
            [np.mean(chroma)]
        ])

        return features

    except Exception as e:
        print(f"❌ Unexpected error in {file_path}: {e}")
        return None

def process_dataset(dataset_path, output_csv, n_mfcc=40, augment=False):
    """Processes a dataset directory and extracts features."""
    data = []
    
    for root, _, files in os.walk(dataset_path):
        for file in files:
            if file.endswith(".wav"):
                file_path = os.path.join(root, file)
                label = get_emotion_from_filename(file)
                features = extract_advanced_features(file_path, n_mfcc, augment)
                if features is not None:
                    data.append([file_path, label] + list(features))

    if not data:
        print("❌ No valid audio files found. Exiting.")
        return
    
    base_columns = ['file_path', 'label']
    feature_columns = [f'feature_{i}' for i in range(len(data[0]) - 2)]
    columns = base_columns + feature_columns
    
    df = pd.DataFrame(data, columns=columns)

    # Normalize Features (excluding file_path and label)
    scaler = StandardScaler()
    df.iloc[:, 2:] = scaler.fit_transform(df.iloc[:, 2:])  

    df.to_csv(output_csv, index=False)
    print(f"✅ Feature extraction completed. Saved to {output_csv}")

if __name__ == "__main__":
    dataset_path = "C:/Users/ashfa/OneDrive/Desktop/Audio_Emotion_Detection/dataset"
    output_csv = "C:/Users/ashfa/OneDrive/Desktop/Audio_Emotion_Detection/features.csv"

    process_dataset(dataset_path, output_csv, n_mfcc=40, augment=True)
