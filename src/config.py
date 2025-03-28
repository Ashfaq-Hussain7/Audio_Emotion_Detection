import os

class Config:
    # Paths
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    DATASET_DIR = os.path.join(PROJECT_ROOT, 'dataset')
    MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
    
    # Model Hyperparameters
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    EPOCHS = 50
    
    # Feature Extraction
    N_MFCC = 40
    SAMPLE_RATE = 22050
    
    # Emotion Classes
    EMOTIONS = ['neutral', 'happy', 'sad', 'angry', 'fear', 'disgust', 'surprise']

    # Validation Split
    TEST_SIZE = 0.2
    RANDOM_SEED = 42