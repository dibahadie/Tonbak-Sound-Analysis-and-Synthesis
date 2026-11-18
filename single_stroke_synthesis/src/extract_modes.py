import numpy as np
import librosa
from scipy.signal import find_peaks, welch
from scipy.stats import linregress

def extract_exact_modes(audio, sr=44100, num_modes=25):
    """
    Deterministically extracts Frequency, Amplitude, and Decay Rate 
    for the dominant modes directly from the acoustic recording.
    """
    # Isolate decay (skip 50ms transient)
    transient_samples = int(0.05 * sr)
    y_decay = audio[transient_samples:]
    
    # 1. Find Frequencies using PSD
    freqs, psd = welch(y_decay, fs=sr, nperseg=4096)
    psd_db = 10 * np.log10(psd / np.max(psd))
    peaks, _ = find_peaks(psd_db, prominence=10, distance=5)
    sorted_indices = np.argsort(psd_db[peaks])[::-1]
    top_peaks = peaks[sorted_indices][:num_modes]
    
    # 2. Extract Decay and Amplitude using STFT
    # Use a high-resolution FFT to isolate frequencies cleanly
    n_fft = 4096
    hop_length = 512
    stft_matrix = np.abs(librosa.stft(audio, n_fft=n_fft, hop_length=hop_length))
    stft_db = librosa.amplitude_to_db(stft_matrix, ref=np.max)
    
    fft_freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    times = librosa.frames_to_time(np.arange(stft_matrix.shape[1]), sr=sr, hop_length=hop_length)
    
    # Only measure the slope after the transient (e.g., from 50ms to 400ms)
    start_frame = int(0.05 * sr / hop_length)
    end_frame = int(0.4 * sr / hop_length)
    time_axis = times[start_frame:end_frame]
    
    extracted_modes = []
    
    for peak_idx in top_peaks:
        target_hz = freqs[peak_idx]
        
        # Find the closest bin in the STFT
        bin_idx = np.argmin(np.abs(fft_freqs - target_hz))
        energy_envelope = stft_db[bin_idx, start_frame:end_frame]
        
        # Linear regression to find the decay slope (dB per second)
        slope, intercept, r_value, p_value, std_err = linregress(time_axis, energy_envelope)
        
        # Convert the dB intercept to a linear amplitude
        linear_amp = 10 ** (intercept / 20.0)
        
        # Convert the slope to a decay multiplier for your synthesizer
        # A steeper negative slope means faster decay (higher multiplier needed)
        decay_mult = max(1.0, abs(slope) * 0.5) 
        
        extracted_modes.append({
            'freq': float(target_hz),
            'amp': float(linear_amp),
            'decay_mult': float(decay_mult),
            'q': 20.0  # Base Q, modulated by decay_mult
        })
        
    return extracted_modes


import numpy as np
import librosa

def extract_exciter_params(audio, sr=44100):
    """
    Analyzes the initial transient of an acoustic drum stroke to extract 
    the attack time, decay time, noise cutoff frequency, and noise Q-factor.
    """
    # 1. Calculate the amplitude envelope using a simple rolling window
    envelope = np.abs(audio)
    
    # Find the start of the sound (threshold at 1% of max volume)
    threshold = 0.01 * np.max(envelope)
    start_idx = np.where(envelope > threshold)[0][0]
    
    # Find the absolute peak of the impact
    peak_idx = np.argmax(envelope)
    
    # Calculate Attack in milliseconds (Time from start to peak)
    attack_ms = max(0.1, (peak_idx - start_idx) / sr * 1000.0)
    
    # Calculate Decay in milliseconds (Time from peak to drop by 90% or -20dB)
    decay_threshold = 0.1 * envelope[peak_idx]
    post_peak_env = envelope[peak_idx:]
    
    # Find where it drops below the threshold, or default to 30ms if it rings heavily
    drop_indices = np.where(post_peak_env < decay_threshold)[0]
    if len(drop_indices) > 0:
        decay_samples = drop_indices[0]
    else:
        decay_samples = int(0.03 * sr)
        
    decay_ms = max(1.0, decay_samples / sr * 1000.0)
    
    # 2. Extract Spectral parameters (Isolate just the first 20ms of the strike)
    transient_end = start_idx + int(0.02 * sr)
    transient_audio = audio[start_idx:transient_end]
    
    # Compute the FFT of the transient
    magnitudes = np.abs(np.fft.rfft(transient_audio))
    freqs = np.fft.rfftfreq(len(transient_audio), 1/sr)
    
    # Calculate Spectral Centroid (The "center of mass" of the noise burst)
    noise_fc = np.sum(freqs * magnitudes) / np.sum(magnitudes)
    
    # Calculate Spectral Bandwidth (The spread of the noise around the centroid)
    bandwidth = np.sqrt(np.sum(magnitudes * (freqs - noise_fc)**2) / np.sum(magnitudes))
    
    # Calculate Q-factor (Center Frequency divided by Bandwidth)
    # Clamp between 0.1 (pure white noise) and 5.0 (highly tonal ping)
    if bandwidth > 0:
        noise_q = np.clip(noise_fc / bandwidth, 0.1, 5.0)
    else:
        noise_q = 0.5
        
    return {
        'attack_ms': round(attack_ms, 2),
        'decay_ms': round(decay_ms, 2),
        'noise_fc': round(noise_fc, 2),
        'noise_q': round(noise_q, 2)
    }

import librosa
import numpy as np
from scipy.stats import linregress

def extract_stroke_constraints(audio, sr=44100):
    """
    Extracts the dynamic Q-clamp and body noise characteristics from an acoustic stroke.
    """
    # 1. Split the audio into Ringing Frequencies (Harmonic) and Friction (Percussive)
    y_harmonic, y_percussive = librosa.effects.hpss(audio)
    
    # 2. Extract Body Noise (The "Haze")
    # Skip the initial 30ms smack to only measure the decaying friction tail
    tail_start = int(0.03 * sr)
    noise_tail = y_percussive[tail_start:]
    
    hop = 512
    noise_env = librosa.feature.rms(y=noise_tail, frame_length=hop, hop_length=hop)[0]
    times = librosa.frames_to_time(np.arange(len(noise_env)), sr=sr, hop_length=hop)
    
    # Linear regression to find the decay slope (dB/sec) of the noise
    noise_db = librosa.amplitude_to_db(noise_env + 1e-9, ref=np.max)
    slope, _, _, _, _ = linregress(times, noise_db)
    
    # Convert dB slope to exponential decay rate for np.exp()
    noise_decay_rate = max(1.0, -slope / 8.686) 
    
    # Calculate noise amplitude relative to the total audio power
    total_rms = np.sqrt(np.mean(audio**2))
    noise_rms = np.sqrt(np.mean(y_percussive[tail_start:tail_start + int(0.05 * sr)]**2))
    noise_amp = min(0.1, noise_rms / (total_rms + 1e-9))
    
    # 3. Extract Dynamic Q-Clamp
    # Find the longest ringing frequency to calculate the absolute max Q
    stft_harm = np.abs(librosa.stft(y_harmonic, n_fft=2048, hop_length=hop))
    stft_db = librosa.amplitude_to_db(stft_harm, ref=np.max)
    
    # Look at the 400ms mark to find frequencies that are still alive
    check_frame = min(int(0.4 * sr / hop), stft_db.shape[1] - 1)
    ringing_bins = np.where(stft_db[:, check_frame] > -40)[0]
    
    max_q = 60.0 # Fallback default
    if len(ringing_bins) > 0:
        freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
        best_bin = ringing_bins[np.argmax(stft_db[ringing_bins, check_frame])]
        target_freq = freqs[best_bin]
        
        # Measure decay of this specific dominant mode
        mode_env = stft_db[best_bin, int(0.05 * sr / hop):]
        m_times = librosa.frames_to_time(np.arange(len(mode_env)), sr=sr, hop_length=hop)
        m_slope, _, _, _, _ = linregress(m_times, mode_env)
        
        if m_slope < 0:
            # T60 = time to decay 60dB
            t60 = -60.0 / m_slope 
            # Convert T60 to Q-factor: Q = (T60 * pi * f) / ln(1000)
            extracted_q = (t60 * np.pi * target_freq) / 6.91
            max_q = np.clip(extracted_q, 20.0, 300.0)

    return {
        'max_q_clamp': float(max_q),
        'body_noise_amp': float(noise_amp),
        'body_noise_decay': float(noise_decay_rate)
    }

import librosa
import numpy as np
from scipy.stats import linregress

def extract_frequency_damping(audio, sr=44100):
    """
    Measures how much faster high frequencies decay compared to low frequencies 
    to dynamically generate a frequency damping scalar.
    """
    hop = 512
    stft = np.abs(librosa.stft(audio, n_fft=2048, hop_length=hop))
    stft_db = librosa.amplitude_to_db(stft, ref=np.max)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    
    # 1. Find the fundamental/lowest dominant frequency (The Pivot)
    # Look at the spectrum slightly after the strike (e.g., 50ms in)
    frame_50ms = int(0.05 * sr / hop)
    spectrum = stft_db[:, frame_50ms]
    
    # Find peaks in the spectrum
    peaks, _ = find_peaks(spectrum, prominence=10)
    if len(peaks) < 2:
        return {'damping_pivot_hz': 300.0, 'damping_exponent': 1.0}
        
    # Get the lowest prominent peak (usually the bass fundamental)
    low_peak_idx = peaks[0]
    pivot_hz = freqs[low_peak_idx]
    
    # Get a high prominent peak to measure the roll-off
    high_peak_idx = peaks[-1] 
    high_hz = freqs[high_peak_idx]
    
    def measure_t60(bin_idx):
        # Measure from 50ms to 200ms
        start_f = int(0.05 * sr / hop)
        end_f = int(0.2 * sr / hop)
        env = stft_db[bin_idx, start_f:end_f]
        times = librosa.frames_to_time(np.arange(len(env)), sr=sr, hop_length=hop)
        
        slope, _, _, _, _ = linregress(times, env)
        if slope >= 0: return 0.01 # Fallback for bad data
        return -60.0 / slope
        
    # 2. Calculate the decay times
    low_t60 = measure_t60(low_peak_idx)
    high_t60 = measure_t60(high_peak_idx)
    
    # 3. Calculate the damping exponent
    # If high frequencies decay twice as fast, the exponent curves the scalar appropriately.
    # We clamp it between 0.5 (gentle roll-off) and 2.0 (aggressive high-frequency choking)
    if high_t60 > 0 and low_t60 > 0 and high_hz > pivot_hz:
        decay_ratio = low_t60 / high_t60
        freq_ratio = high_hz / pivot_hz
        exponent = np.log(decay_ratio) / np.log(freq_ratio)
        exponent = np.clip(exponent, 0.5, 2.0)
    else:
        exponent = 1.0

    return {
        'damping_pivot_hz': float(pivot_hz),
        'damping_exponent': float(exponent)
    }

# Example execution:
# damping_constraints = extract_frequency_damping(original_audio, sr)
# stroke_constraints.update(damping_constraints)