import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import scipy.signal as signal

def plot_acoustic_comparison(original_audio, synth_audio, sr=44100, compare_duration=1.0):
    """
    Generates a visual dashboard comparing original acoustic audio with synthesized audio.
    Includes Waveform envelope, Power Spectral Density, and parallel Spectrograms.
    """
    # Normalize both signals for a fair visual comparison
    if np.max(np.abs(original_audio)) > 0:
        original_audio = original_audio / np.max(np.abs(original_audio))
    if np.max(np.abs(synth_audio)) > 0:
        synth_audio = synth_audio / np.max(np.abs(synth_audio))

    # Trim or pad to the same length
    compare_length = int(sr * compare_duration)
    
    orig_trimmed = original_audio[:compare_length] if len(original_audio) > compare_length else np.pad(original_audio, (0, compare_length - len(original_audio)))
    synth_trimmed = synth_audio[:compare_length] if len(synth_audio) > compare_length else np.pad(synth_audio, (0, compare_length - len(synth_audio)))

    time_axis = np.arange(compare_length) / sr * 1000  # in milliseconds

    # Create the Figure
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2)

    # --- PLOT 1: Waveforms (Time Domain) ---
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(time_axis, orig_trimmed, label='Original Acoustic', color='gray', alpha=0.7)
    ax1.plot(time_axis, synth_trimmed, label='Synthesized (Modal)', color='teal', alpha=0.8)
    ax1.set_title("Waveform Comparison (Envelope and Transient Shape)", fontsize=14)
    ax1.set_ylabel("Amplitude")
    ax1.set_xlabel("Time (ms)")
    ax1.set_xlim(0, 500) # Zoom into the first 500ms
    ax1.legend(loc='upper right')

    # --- PLOT 2: Power Spectral Density (Frequency Domain) ---
    ax2 = fig.add_subplot(gs[1, :])
    f_orig, Pxx_orig = signal.welch(orig_trimmed, sr, nperseg=4096)
    f_synth, Pxx_synth = signal.welch(synth_trimmed, sr, nperseg=4096)

    ax2.plot(f_orig, 10 * np.log10(Pxx_orig), label='Original Spectrum', color='gray')
    ax2.plot(f_synth, 10 * np.log10(Pxx_synth), label='Synthesized Spectrum', color='teal', alpha=0.8)
    ax2.set_title("Spectral Density Overlay (Modal Tuning Accuracy)", fontsize=14)
    ax2.set_ylabel("Power (dB)")
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_xlim(0, 1500) # Focus on the main modal region
    ax2.set_ylim(-100, 0)
    ax2.legend(loc='upper right')

    # --- PLOT 3 & 4: Spectrograms (Time-Frequency Domain) ---
    ax3 = fig.add_subplot(gs[2, 0])
    S_orig = librosa.amplitude_to_db(np.abs(librosa.stft(orig_trimmed, n_fft=2048, hop_length=512)), ref=np.max)
    librosa.display.specshow(S_orig, sr=sr, x_axis='time', y_axis='hz', ax=ax3, cmap='magma', vmin=-80, vmax=0)
    ax3.set_title("Original Spectrogram")
    ax3.set_ylim(0, 2000)

    ax4 = fig.add_subplot(gs[2, 1])
    S_synth = librosa.amplitude_to_db(np.abs(librosa.stft(synth_trimmed, n_fft=2048, hop_length=512)), ref=np.max)
    img = librosa.display.specshow(S_synth, sr=sr, x_axis='time', y_axis='hz', ax=ax4, cmap='magma', vmin=-80, vmax=0)
    ax4.set_title("Synthesized Spectrogram")
    ax4.set_ylim(0, 2000)
    ax4.set_ylabel("") # Hide Y label for the second spectrogram

    # Add a shared colorbar for the spectrograms
    fig.colorbar(img, ax=[ax3, ax4], format="%+2.0f dB", pad=0.01)

    plt.tight_layout()
    plt.show()

# ==============================================================================
# Example Usage Call:
# ==============================================================================
# original_audio, sr = librosa.load('../../single_stroke_analysis/data/raw/hit_low/GIO_MEE_tombak_hit_low.wav', sr=44100)
# plot_acoustic_comparison(original_audio=original_audio, synth_audio=audio_out, sr=sr, compare_duration=1.0)