import numpy as np
import librosa
import copy
from scipy.optimize import differential_evolution

def optimize_modes(target_audio, initial_modes, synth_instance, sr=44100, target_duration=1.0, max_iter=20, popsize=5):
    """
    Uses Differential Evolution to globally optimize Q, amplitude, and decay_mult 
    while keeping the modal frequencies strictly fixed.
    """
    target_samples = int(sr * target_duration)
    
    # 1. Prepare Target Audio & Features
    if len(target_audio) > target_samples:
        y_target = target_audio[:target_samples]
    else:
        y_target = np.pad(target_audio, (0, target_samples - len(target_audio)))
        
    if np.max(np.abs(y_target)) > 0:
        y_target = y_target / np.max(np.abs(y_target))
        
    mel_target = librosa.feature.melspectrogram(y=y_target, sr=sr, n_mels=128, fmax=2000)
    mel_target_db = librosa.power_to_db(mel_target, ref=np.max)
    mfcc_target = librosa.feature.mfcc(S=mel_target_db, n_mfcc=13)
    
    env_target = librosa.feature.rms(y=y_target, frame_length=2048, hop_length=512)[0]
    if np.max(env_target) > 0:
        env_target = env_target / np.max(env_target)

    # 2. Packing/Unpacking Helpers (Frequencies Excluded)
    def pack_parameters(mode_list):
        return np.array([[m.get('q', 10.0), m.get('amp', 1.0), m.get('decay_mult', 10.0)] for m in mode_list]).flatten()

    def unpack_parameters(flat_params, template_modes):
        new_modes = copy.deepcopy(template_modes)
        for i in range(len(new_modes)):
            new_modes[i]['q'] = flat_params[i*3]
            new_modes[i]['amp'] = flat_params[i*3 + 1]
            new_modes[i]['decay_mult'] = flat_params[i*3 + 2]
        return new_modes

    # 3. Objective Function
    def perceptual_loss(flat_params, template_modes):
        current_modes = unpack_parameters(flat_params, template_modes)
        
        synth_audio = synth_instance.synthesize(
            modes=current_modes, 
            duration=target_duration, 
            attack_ms=1.5, 
            decay_ms=25.0,
            noise_fc=301.0, 
            noise_q=0.5, 
            q_multiplier=1.0, 
            beating_hz=1.8
        )
        
        if len(synth_audio) > target_samples:
            synth_audio = synth_audio[:target_samples]
        else:
            synth_audio = np.pad(synth_audio, (0, target_samples - len(synth_audio)))
            
        if np.max(np.abs(synth_audio)) > 0:
            synth_audio = synth_audio / np.max(np.abs(synth_audio))
            
        mel_synth = librosa.feature.melspectrogram(y=synth_audio, sr=sr, n_mels=128, fmax=2000)
        mel_synth_db = librosa.power_to_db(mel_synth, ref=np.max)
        mfcc_synth = librosa.feature.mfcc(S=mel_synth_db, n_mfcc=13)
        
        env_synth = librosa.feature.rms(y=synth_audio, frame_length=2048, hop_length=512)[0]
        if np.max(env_synth) > 0:
            env_synth = env_synth / np.max(env_synth)

        mel_loss = np.mean((mel_target_db - mel_synth_db) ** 2)
        mfcc_loss = np.mean((mfcc_target - mfcc_synth) ** 2)
        env_loss = np.mean((env_target - env_synth) ** 2)
        
        return (mel_loss * 0.2) + (mfcc_loss * 1.0) + (env_loss * 500.0)

    # 4. Define Boundaries (3 variables per mode)
    bounds = []
    for m in initial_modes:
        bounds.append((0.1, 150.0))  # Q range
        bounds.append((0.0, 10.0))   # Amplitude range
        bounds.append((1.0, 200.0))  # Decay multiplier range

    # 5. Execute Differential Evolution
    print(f"Starting Fixed-Frequency Evolutionary Optimization (maxiter={max_iter}, popsize={popsize})...")
    result = differential_evolution(
        func=perceptual_loss,
        bounds=bounds,
        args=(initial_modes,),
        strategy='best1bin',
        maxiter=max_iter,
        popsize=popsize,
        mutation=(0.5, 1.0),
        recombination=0.7,
        disp=True
    )
    
    # 6. Finalize output
    optimized_modes = unpack_parameters(result.x, initial_modes)
    optimized_audio = synth_instance.synthesize(
        modes=optimized_modes, 
        duration=target_duration, 
        attack_ms=1.5, 
        decay_ms=25.0
    )
    
    print(f"Optimization Complete. Final Perceptual Score: {result.fun:.4f}")
    return optimized_modes, optimized_audio