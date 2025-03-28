import librosa
import numpy as np
import random
import torch
import torchaudio.transforms as T

def add_noise(y, noise_level=0.005):
    """Adds Gaussian noise to audio."""
    noise = np.random.randn(len(y)) * noise_level
    return y + noise

def time_stretch(y, sr, rate=1.0):
    """Applies time stretching to audio."""
    try:
        rate = max(0.5, min(2.0, rate))  
        len_stretched = int(len(y) / rate)
        
        stretched = librosa.resample(y, orig_sr=sr, target_sr=int(sr * rate))
        
        if len(stretched) > len_stretched:
            stretched = stretched[:len_stretched]
        else:
            stretched = np.pad(stretched, (0, len_stretched - len(stretched)), mode='constant')
        
        return stretched
    except Exception as e:
        print(f"Time stretch error: {e}. Returning original audio.")
        return y

def pitch_shift(y, sr, n_steps=0):
    """Shifts pitch by `n_steps` semitones."""
    try:
        return librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)
    except Exception as e:
        print(f"Pitch shift error: {e}. Returning original audio.")
        return y

def change_volume(y, factor=1.0):
    """Increases or decreases volume."""
    return y * factor

def spec_augment(mel_spectrogram):
    """Applies SpecAugment to mel spectrograms."""
    try:
        freq_mask = T.FrequencyMasking(freq_mask_param=10)
        time_mask = T.TimeMasking(time_mask_param=10)
        mel_spectrogram = freq_mask(mel_spectrogram)
        mel_spectrogram = time_mask(mel_spectrogram)
        return mel_spectrogram
    except Exception as e:
        print(f"SpecAugment error: {e}. Returning original spectrogram.")
        return mel_spectrogram

def augment_audio(y, sr):
    """Randomly applies augmentation techniques to an audio sample with error handling."""
    try:
        augmentations = [
            (add_noise, {'noise_level': random.uniform(0.002, 0.01)}, 0.3),
            (time_stretch, {'rate': random.uniform(0.8, 1.2)}, 0.3),
            (pitch_shift, {'n_steps': random.randint(-2, 2)}, 0.3),
            (change_volume, {'factor': random.uniform(0.5, 1.5)}, 0.3)
        ]
        
        for aug_func, params, prob in augmentations:
            if random.random() < prob:
                if aug_func in [time_stretch, pitch_shift]:  
                    y = aug_func(y, sr=sr, **params)  # Explicitly pass 'sr'
                else:
                    y = aug_func(y, **params)  # Don't pass 'sr' if not needed

        return y
    except Exception as e:
        print(f"Augmentation error: {e}. Returning original audio.")
        return y
