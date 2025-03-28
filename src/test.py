import torch
import librosa
import numpy as np
import torch.nn as nn
import pandas as pd
from train import EmotionModel  # Ensure this matches your model file
from extract_features import extract_advanced_features

# Load the trained model
MODEL_PATH = "C:/Users/ashfa/OneDrive/Desktop/Audio_Emotion_Detection/models/emotion_model.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Find the input size and number of classes from your features CSV
def get_model_params():
    features_path = "C:/Users/ashfa/OneDrive/Desktop/Audio_Emotion_Detection/features.csv"
    df = pd.read_csv(features_path)
    input_size = df.iloc[:, 2:].shape[1]
    
    # Determine the number of unique classes
    num_classes = len(df['label'].unique())
    
    return input_size, num_classes

# Get input size and number of classes
input_size, num_classes = get_model_params()

# Define the model with the correct number of classes
model = EmotionModel(input_size=input_size, num_classes=num_classes)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()  # Set to evaluation mode

# Update class labels based on the actual unique labels in the dataset
EMOTION_CLASSES = list(pd.read_csv("C:/Users/ashfa/OneDrive/Desktop/Audio_Emotion_Detection/features.csv")['label'].unique())

def predict_emotion(audio_path):
    """Extract features from an audio file and predict emotion."""
    features = extract_advanced_features(audio_path, n_mfcc=40)
    
    if features is None:
        print(f"❌ Failed to process {audio_path}")
        return
    
    # Convert to PyTorch tensor
    features = torch.tensor(features, dtype=torch.float32).to(device).unsqueeze(0)  # Add batch dimension

    # Get model prediction
    with torch.no_grad():
        output = model(features)
        predicted_class = torch.argmax(output, dim=1).item()  # Get class index

    predicted_emotion = EMOTION_CLASSES[predicted_class]
    print(f"🎤 Prediction for {audio_path}: {predicted_emotion}")

# Test on a new audio file
TEST_AUDIO = "C:/Users/ashfa/OneDrive/Desktop/Audio_Emotion_Detection/generated_audio.wav"  # REPLACE WITH ACTUAL AUDIO FILE PATH
predict_emotion(TEST_AUDIO)