# feature_extraction.py
import torch
import torchaudio

def extract_features_pytorch(audio_data, model, sr=22050, n_fft=2048, hop_length=512, n_mels=128):
    audio_data = torch.tensor(audio_data).float()
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(sample_rate=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)(audio_data[None, :])
    S_DB = torchaudio.transforms.AmplitudeToDB()(mel_spectrogram)
    features = model(S_DB)
    return features.detach()
