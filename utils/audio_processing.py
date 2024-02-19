# audio_processing.py
import librosa
import numpy as np
from audiomentations import Compose, AddGaussianNoise, PitchShift, Shift

# Function to load audio files
def load_audio(filepath, sr=22050):
    """
    Load an audio file into a waveform.

    Parameters:
    - filepath: str, path to the audio file.
    - sr: int, sampling rate to use. Default is 22050Hz.

    Returns:
    - waveform: np.ndarray, audio time series.
    - sr: int, sampling rate of the audio file.
    """
    waveform, sr = librosa.load(filepath, sr=sr)
    return waveform, sr

# Function for audio preprocessing
def preprocess_audio(waveform):
    """
    Preprocess the audio waveform.

    Parameters:
    - waveform: np.ndarray, audio time series.

    Returns:
    - processed_waveform: np.ndarray, preprocessed audio time series.
    """
    # Normalize the waveform
    processed_waveform = librosa.util.normalize(waveform)
    return processed_waveform

# Function for audio augmentation (if needed)
def augment_audio(waveform, sample_rate=22050):
    """
    Apply audio augmentation techniques to the waveform.

    Parameters:
    - waveform: np.ndarray, audio time series.
    - sample_rate: int, sampling rate of the waveform.

    Returns:
    - augmented_waveform: np.ndarray, augmented audio time series.
    """
    augment = Compose([
        AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
        PitchShift(min_semitones=-4, max_semitones=4, p=0.5),
        Shift(min_fraction=-0.5, max_fraction=0.5, p=0.5),
    ])

    augmented_waveform = augment(samples=waveform, sample_rate=sample_rate)
    return augmented_waveform
