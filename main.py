# main.py

import os

if __name__ == "__main__":
    # Ensure dataset and output directories exist
    os.makedirs("dataset", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    print("Step 1: Extracting features from dataset...")
    os.system("python src/extract_features.py")

    print("Step 2: Training the model...")
    os.system("python src/train.py")

    print("Speech Emotion Recognition pipeline completed!")