# perceptual_test.py
import torch
import torchaudio

def spectral_distance(original_audio_path, cloaked_audio_path):
    """
    Calculate an objective metric representing the spectral distance between two audio files.

    Parameters:
    - original_audio_path: Path to the original audio file.
    - cloaked_audio_path: Path to the cloaked (modified) audio file.

    Returns:
    - distance: A float representing the spectral distance between the two audio files.
    """
    # Load the audio files
    orig_waveform, orig_sr = torchaudio.load(original_audio_path)
    cloaked_waveform, cloaked_sr = torchaudio.load(cloaked_audio_path)

    # Ensure the sample rates are equal or resample if necessary
    if orig_sr != cloaked_sr:
        cloaked_waveform = torchaudio.transforms.Resample(orig_sr, cloaked_sr)(cloaked_waveform)

    # Compute Mel spectrograms
    mel_spectrogram = torchaudio.transforms.MelSpectrogram()
    orig_spec = mel_spectrogram(orig_waveform)
    cloaked_spec = mel_spectrogram(cloaked_waveform)

    # Calculate the spectral distance (e.g., Euclidean distance between spectrograms)
    distance = torch.norm(orig_spec - cloaked_spec, p=2).item()

    return distance

# Example usage
if __name__ == "__main__":
    original_audio_path = "path/to/original/audio.wav"
    cloaked_audio_path = "path/to/cloaked/audio.wav"
    distance = spectral_distance(original_audio_path, cloaked_audio_path)
    print(f"Spectral Distance: {distance}")
