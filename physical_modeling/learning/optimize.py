import numpy as np
import librosa
import scipy.io.wavfile as wav
from scipy.optimize import differential_evolution
import copy
from config import SR, MEMBRANE, HAMMER, SYSTEM
from synthesizer import TonbakSynthesizer

# 1. Define the bounds of your search space to prevent the "gong" effect
# (Min, Max) for each parameter you want the computer to learn
BOUNDS = [
    (0.01, 0.12),      # [0] hit_r (Center to Rim)
    (0.005, 0.05),     # [1] hammer mass
    (1e4, 1.5e5),      # [2] hammer stiffness
    (1.0, 6.0),        # [3] hammer velocity
    (1e-4, 5e-4),      # [4] d3 (High-freq damping)
    (0.1, 0.8),        # [5] d1 (Low-freq damping)
    (1800.0, 2400.0)   # [6] tension_0 (Tuning)
]

def extract_features(audio_array, sr=SR):
    """Extracts the Log-Mel Spectrogram to capture timbre and decay."""
    # Ensure audio is padded/trimmed to exact length for consistent matrix sizes
    target_length = int(sr * 1.0)
    if len(audio_array) > target_length:
        audio_array = audio_array[:target_length]
    else:
        audio_array = np.pad(audio_array, (0, target_length - len(audio_array)))
        
    mel_spec = librosa.feature.melspectrogram(y=audio_array, sr=sr, n_mels=64, hop_length=256)
    log_mel = librosa.power_to_db(mel_spec, ref=np.max)
    return log_mel

def objective_function(params, target_features):
    """The function the optimizer tries to minimize."""
    hit_r, h_mass, h_stiff, h_vel, d3, d1, tension = params
    
    mem_cfg = copy.deepcopy(MEMBRANE)
    ham_cfg = copy.deepcopy(HAMMER)
    sys_cfg = copy.deepcopy(SYSTEM)
    
    # Inject optimizer parameters
    ham_cfg['mass'] = h_mass
    ham_cfg['stiffness'] = h_stiff
    ham_cfg['velocity_in'] = h_vel
    mem_cfg['d3'] = d3
    mem_cfg['d1'] = d1
    mem_cfg['tension_0'] = tension
    
    # Generate synthetic stroke
    synth = TonbakSynthesizer(
        hit_r=hit_r, hit_phi=0.0, pickup_r=0.08, pickup_phi=0.0,
        membrane_cfg=mem_cfg, hammer_cfg=ham_cfg, system_cfg=sys_cfg
    )
    
    try:
        synth_audio = synth.generate(duration_sec=1.0)
        synth_features = extract_features(synth_audio)
        
        # Calculate Mean Squared Error of the spectrograms
        loss = np.mean((target_features - synth_features) ** 2)
        return loss
    except Exception as e:
        # If the physical model explodes (NaNs due to instability), penalize heavily
        return 1e6

def fit_model_to_audio(target_wav_path):
    print(f"Loading target audio: {target_wav_path}")
    target_y, _ = librosa.load(target_wav_path, sr=SR)
    target_features = extract_features(target_y)
    
    print("Starting Differential Evolution... (This will take a while)")
    
    # The DE algorithm will intelligently search the space, guided by the spectrogram loss
    result = differential_evolution(
        objective_function, 
        bounds=BOUNDS, 
        args=(target_features,), 
        strategy='best1bin', 
        maxiter=30,     # Generations to run
        popsize=15,     # Configurations per generation
        mutation=(0.5, 1.0), 
        recombination=0.7,
        disp=True       # Prints progress to console
    )
    
    print("\n--- Optimization Complete ---")
    print("Best Parameters Found:")
    print(f"Hit Radius: {result.x[0]:.3f}")
    print(f"Hammer Mass: {result.x[1]:.4f}")
    print(f"Hammer Stiffness: {result.x[2]:.1f}")
    print(f"Hammer Velocity: {result.x[3]:.2f}")
    print(f"Damping (d3): {result.x[4]:.6f}")
    print(f"Damping (d1): {result.x[5]:.3f}")
    print(f"Tension: {result.x[6]:.1f}")
    print(f"Final Spectral Loss: {result.fun:.4f}")
    
    return result.x

# Run the fitter
# best_params = fit_model_to_audio("real_tonbak_tom.wav")