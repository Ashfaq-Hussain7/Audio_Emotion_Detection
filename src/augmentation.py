import numpy as np
import librosa
import random
import nlpaug.augmenter.audio as naa  # Optional: for more advanced augmentations

def add_noise(y, noise_level=0.005):
    """Add Gaussian noise to audio"""
    noise = np.random.randn(len(y))
    return y + noise_level * noise

def change_pitch(y, sr, n_steps=None):
    """Pitch shift with random steps if not specified"""
    if n_steps is None:
        n_steps = random.uniform(-2, 2)  # Reduced range from (-3, 3)
    return librosa.effects.pitch_shift(y, sr, n_steps=n_steps)

def time_stretch(y, rate=None):
    """Time stretching with more moderate values"""
    if rate is None:
        rate = random.uniform(0.8, 1.2)  # Less aggressive than before
    return librosa.effects.time_stretch(y, rate)

def add_background_noise(y, noise_level=0.01):
    """Add background noise pattern"""
    noise = np.random.uniform(-noise_level, noise_level, len(y))
    return y + noise

def random_eq(y, sr):
    """Apply random equalization"""
    n_bands = random.randint(2, 4)  # Reduced from 3-6
    gains = np.random.uniform(-8, 8, n_bands)  # Reduced from -12, 12
    center_freqs = np.logspace(np.log10(80), np.log10(8000), n_bands)
    for cf, gain in zip(center_freqs, gains):
        y = librosa.effects.preemphasis(y, coef=min(0.95, max(0.7, 1 - (gain/100))))
    return y

def time_mask(y, max_mask_length=0.1):
    """Randomly mask portions of the audio"""
    mask_length = int(len(y) * random.uniform(0.05, max_mask_length))
    mask_start = random.randint(0, len(y) - mask_length)
    y[mask_start:mask_start+mask_length] = 0
    return y

def augment_audio(y, sr, augmentation_prob=0.9):
    """
    Enhanced audio augmentation pipeline
    Args:
        y: audio signal
        sr: sample rate
        augmentation_prob: probability to apply any augmentation
    Returns:
        augmented audio signal
    """
    if random.random() > augmentation_prob:
        return y  # Return original with small probability
    
    # Apply core augmentations (less aggressive settings)
    augmentations = [
        lambda y: add_noise(y, noise_level=random.uniform(0.001, 0.01)),
        lambda y: change_pitch(y, sr, n_steps=random.uniform(-2, 2)),
        lambda y: time_stretch(y, rate=random.uniform(0.8, 1.2)),
        lambda y: add_background_noise(y, noise_level=0.01),
        lambda y: random_eq(y, sr),
        lambda y: time_mask(y, max_mask_length=0.1)
    ]
    
    # Apply 1-2 augmentations instead of 2-4
    random.shuffle(augmentations)
    for aug in augmentations[:random.randint(1, 2)]:
        try:
            y = aug(y)
        except:
            continue  # Skip if augmentation fails
    
    # Normalize to prevent clipping
    y = librosa.util.normalize(y)
    
    return y