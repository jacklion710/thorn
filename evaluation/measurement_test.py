# technical_test.py
import numpy as np
from scipy.signal import spectrogram
from utils.audio_processing import load_audio

def spectral_content_similarity(original_audio, cloaked_audio):
    """
    Calculate similarity in spectral content between original and cloaked audio.
    
    Parameters:
    - original_audio: Filepath to the original audio file.
    - cloaked_audio: Filepath to the cloaked audio file.
    
    Returns:
    - similarity_score: A similarity score indicating how close the spectral content
                        of the two audio signals are.
    """
    # Load the audio files
    orig_waveform = load_audio(original_audio)
    cloaked_waveform = load_audio(cloaked_audio)
    
    # Compute spectrograms
    f_orig, t_orig, Sxx_orig = spectrogram(orig_waveform)
    f_cloak, t_cloak, Sxx_cloak = spectrogram(cloaked_waveform)
    
    # Normalize spectrograms for comparison
    norm_Sxx_orig = Sxx_orig / np.amax(Sxx_orig)
    norm_Sxx_cloak = Sxx_cloak / np.amax(Sxx_cloak)
    
    # Calculate the similarity score, for simplicity, using Euclidean distance
    similarity_score = np.linalg.norm(norm_Sxx_orig - norm_Sxx_cloak)
    
    return similarity_score

def conduct_technical_test(original_audio, cloaked_audio):
    """
    Conducts various technical tests to compare the original and cloaked audio.
    
    Parameters:
    - original_audio: Filepath to the original audio file.
    - cloaked_audio: Filepath to the cloaked audio file.
    
    Returns:
    - test_results: A dictionary containing results of various technical tests.
    """
    test_results = {}
    
    # Spectral content similarity
    spectral_similarity = spectral_content_similarity(original_audio, cloaked_audio)
    test_results['Spectral Similarity'] = spectral_similarity
    
    # Other tests like frequency response, dynamic range, etc., could be added here
    
    return test_results

# Example usage
if __name__ == "__main__":
    original_audio_path = "path/to/your/original/audio.wav"
    cloaked_audio_path = "path/to/your/cloaked/audio.wav"
    
    results = conduct_technical_test(original_audio_path, cloaked_audio_path)
    print("Technical Test Results:")
    for test_name, result in results.items():
        print(f"{test_name}: {result}")
