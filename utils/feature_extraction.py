# feature_extraction.py
import librosa
import numpy as np

def extract_features(audio_data, model, sr=22050, n_fft=2048, hop_length=512, n_mels=128):
    """
    Extract features from audio data using a pre-trained model.

    Parameters:
    - audio_data: np.ndarray, audio time series.
    - model: Pre-trained model for feature extraction.
    - sr: int, sampling rate of the audio file.
    - n_fft: int, length of the FFT window.
    - hop_length: int, number of samples between successive frames.
    - n_mels: int, number of Mel bands to generate.

    Returns:
    - features: np.ndarray, extracted features from the model.
    """
    # Convert the waveform to a Mel spectrogram.
    S = librosa.feature.melspectrogram(audio_data, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    # Convert to decibel units.
    S_DB = librosa.power_to_db(S, ref=np.max)

    # Assuming the model expects a 4D input (batch_size, height, width, channels),
    # we need to add two extra dimensions to S_DB.
    S_DB = S_DB[np.newaxis, ..., np.newaxis]

    # Pass the spectrogram through the pre-trained model to extract features.
    features = model.predict(S_DB)

    return features
