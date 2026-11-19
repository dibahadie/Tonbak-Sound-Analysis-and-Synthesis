import librosa
import numpy as np

def safe_delta(feature_matrix):
    """Fixes the librosa crash by dynamically sizing the delta window for very short segments."""
    n_frames = feature_matrix.shape[1]
    if n_frames < 3:
        return np.zeros_like(feature_matrix)
    width = 9 if n_frames >= 9 else (n_frames if n_frames % 2 != 0 else n_frames - 1)
    if width < 3:
        return np.zeros_like(feature_matrix)
    return librosa.feature.delta(feature_matrix, width=width)

def extract_features(y, sr):
    """Extracts temporal, spectral, and percussive features across phases."""
    # Base Spectral Matrices
    S = np.abs(librosa.stft(y))
    H, P = librosa.decompose.hpss(S)
    mel = librosa.feature.melspectrogram(y=y, sr=sr)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    
    # MFCCs
    mfcc = librosa.feature.mfcc(S=mel_db, n_mfcc=13)
    delta_mfcc = safe_delta(mfcc)
    
    # Spectral Shape (1D arrays over time)
    cent = librosa.feature.spectral_centroid(S=S, sr=sr)[0]
    rolloff = librosa.feature.spectral_rolloff(S=S, sr=sr)[0]
    bw = librosa.feature.spectral_bandwidth(S=S, sr=sr)[0]
    flatness = librosa.feature.spectral_flatness(y=y)[0]
    
    # Temporal Domain
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    flux = librosa.onset.onset_strength(y=y, sr=sr)
    rms = librosa.feature.rms(y=y)[0]
    
    # Global Summary Stats
    duration_ms = (len(y) / sr) * 1000
    freqs = librosa.fft_frequencies(sr=sr)
    low_e = np.sum(S[np.where(freqs < 250)[0], :])
    high_e = np.sum(S[np.where(freqs > 1000)[0], :])
    ratio = low_e / (high_e + 1e-10)

    # Temporal Phase Slicing (Transient = first 50ms, Decay = remaining)
    transient_samples = int(0.05 * sr)
    transient_frames = librosa.samples_to_frames(transient_samples)
    
    def phase_mean(arr):
        t_phase = np.mean(arr[:transient_frames]) if len(arr[:transient_frames]) > 0 else 0
        d_phase = np.mean(arr[transient_frames:]) if len(arr[transient_frames:]) > 0 else 0
        return t_phase, d_phase

    cent_t, cent_d = phase_mean(cent)
    rms_t, rms_d = phase_mean(rms)
    zcr_t, zcr_d = phase_mean(zcr)

    return {
        # Raw Data for playback/reconstruction
        'y': y,
        'sr': sr,
        
        # 2D Matrices for Web App Plotting
        'P': P,
        'mel_db': mel_db,
        'mfcc': mfcc,
        'delta_mfcc': delta_mfcc,
        'cent': cent,
        'rolloff': rolloff,
        'bw': bw,
        'zcr': zcr,
        'flux': flux,
        
        # 1D Global & Phase Stats for Jupyter Notebooks
        'duration': duration_ms,
        'ratio': ratio,
        'cent_mean_global': np.mean(cent),
        'cent_mean_transient': cent_t,
        'cent_mean_decay': cent_d,
        'rms_transient': rms_t,
        'rms_decay': rms_d,
        'zcr_transient': zcr_t,
        'zcr_decay': zcr_d,
        'flux_max': np.max(flux)
    }