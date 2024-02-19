# audio_processing.py
import torch
import torchaudio
from torchaudio.transforms import Resample, Vol, PitchShift

# Ensure torchaudio's backend is set appropriately for your system (e.g., SoX or SoundFile)
torchaudio.set_audio_backend("sox_io") # or "soundfile" depending on your installation

def load_audio(filepath, sr=22050):
    """
    Load an audio file into a waveform using torchaudio.

    Parameters:
    - filepath: str, path to the audio file.
    - sr: int, desired sampling rate.

    Returns:
    - waveform: torch.Tensor, audio time series.
    - sr: int, sampling rate of the audio file.
    """
    waveform, orig_sr = torchaudio.load(filepath)
    if orig_sr != sr:
        resample = Resample(orig_sr, sr)
        waveform = resample(waveform)
    return waveform, sr

def preprocess_audio(waveform):
    """
    Normalize the audio waveform using PyTorch operations.

    Parameters:
    - waveform: torch.Tensor, audio time series.

    Returns:
    - processed_waveform: torch.Tensor, normalized audio time series.
    """
    processed_waveform = waveform / torch.max(torch.abs(waveform))
    return processed_waveform

def augment_audio(waveform, sample_rate=22050):
    """
    Apply audio augmentation techniques to the waveform using torchaudio.

    Parameters:
    - waveform: torch.Tensor, audio time series.
    - sample_rate: int, sampling rate of the waveform.

    Returns:
    - augmented_waveform: torch.Tensor, augmented audio time series.
    """
    # Note: torchaudio's augmentation features might be limited compared to audiomentations.
    # You may need to implement custom augmentations for some features.

    # Adding Gaussian Noise
    waveform = waveform + 0.015 * torch.randn(waveform.size())

    # Pitch shifting
    # Note: torchaudio's pitch shift operation requires specifying the number of steps to shift, which may vary based on the sample rate.
    pitch_shift = PitchShift(sample_rate, n_steps=4) # Example: shift by 4 semitones; adjust as needed.
    waveform = pitch_shift(waveform)

    # Shifting the audio waveform in time is not directly supported by torchaudio as of my last update.
    # This would require a custom implementation, for example, by slicing the tensor.

    return waveform
