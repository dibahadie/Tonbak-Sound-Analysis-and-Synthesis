import librosa
import numpy as np

def load_and_preprocess(file_path, target_sr=44100, target_dbfs=-1.0):
    """Loads audio, resamples, trims silence, and peak normalizes."""
    # Load and resample
    y, sr = librosa.load(file_path, sr=target_sr)
    
    # Dynamic silence trimming (top_db=30 means 30dB below peak)
    y_trimmed, _ = librosa.effects.trim(y, top_db=30)
    
    # Peak normalization
    peak = np.max(np.abs(y_trimmed))
    if peak > 0:
        target_linear = 10 ** (target_dbfs / 20)
        y_norm = y_trimmed * (target_linear / peak)
    else:
        y_norm = y_trimmed
        
    return y_norm, target_sr