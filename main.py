# main.py

import argparse
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.train import train_model
from src.test import predict_emotion
from src.config import Config

def main():
    parser = argparse.ArgumentParser(description='Audio Emotion Detection')
    parser.add_argument('mode', choices=['train', 'predict'], 
                        help='Mode of operation')
    parser.add_argument('--audio_path', type=str, 
                        help='Path to audio file for prediction')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        train_model(Config.DATASET_DIR)
    
    elif args.mode == 'predict':
        if not args.audio_path:
            print("Please provide an audio path for prediction")
            sys.exit(1)
        
        predict_emotion(args.audio_path)

if __name__ == '__main__':
    main()
