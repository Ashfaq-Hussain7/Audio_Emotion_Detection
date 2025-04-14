#Audio Emotion Detection
Project Overview
This project implements an audio emotion detection model using deep learning techniques.
Setup Instructions
Prerequisites

Python 3.8+
CUDA (optional, for GPU acceleration)

Installation

Clone the repository
Create a virtual environment

bashCopypython -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

Install dependencies

bashCopypip install -r requirements.txt
Dataset

Place audio files in the dataset folder
Supported formats: WAV, MP3

Training
bashCopypython main.py train
Inference
bashCopypython main.py predict --audio_path path/to/audio.wav
Project Structure
Copyaudio-emotion-detection/
│
├── dataset/
│   └── (audio files)
│
├── models/
│   └── emotion_model.pth
│
├── src/
│   ├── data_loader.py
│   ├── extract_feature.py
│   ├── augmentation.py
│   ├── train.py
│   └── test.py
│
├── main.py
├── requirements.txt
└── README.md
Model Performance

Accuracy: 91%
Emotions Detected: Neutral, Happy, Sad, Angry, Fear, Disgust, Surprise

Copy