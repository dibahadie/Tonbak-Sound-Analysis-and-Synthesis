import numpy as np
import librosa
import copy
from scipy.optimize import differential_evolution

def optimize_modes(target_audio, initial_modes, synth_instance, sr=44100, target_duration=1.0, max_iter=50, popsize=10):
    """
    Uses Differential Evolution (MSSL) to globally optimize Q, amplitude, decay_mult, 
    AND the exciter transient, while keeping modal frequencies strictly fixed.
    """
    target_samples = int(sr * target_duration)
    
    # 1. Prepare Target Audio (Strictly crop dead air to fix the Pre-Delay Trap)
    y_target, _ = librosa.effects.trim(target_audio, top_db=30)
    
    if len(y_target) > target_samples:
        y_target = y_target[:target_samples]
    else:
        y_target = np.pad(y_target, (0, target_samples - len(y_target)))
        
    if np.max(np.abs(y_target)) > 0:
        y_target = y_target / np.max(np.abs(y_target))

    # 2. Packing/Unpacking Helpers (Now includes 4 Exciter Variables at the end)
    def pack_parameters(mode_list):
        mode_params = np.array([[m.get('q', 10.0), m.get('amp', 1.0), m.get('decay_mult', 10.0)] for m in mode_list]).flatten()
        # Default Exciter Guess: attack_ms=1.5, decay_ms=25.0, noise_fc=300.0, noise_q=0.5
        exciter_params = np.array([1.5, 25.0, 300.0, 0.5])
        return np.concatenate((mode_params, exciter_params))

    def unpack_parameters(flat_params, template_modes):
        new_modes = copy.deepcopy(template_modes)
        num_mode_vars = len(template_modes) * 3
        
        for i in range(len(new_modes)):
            new_modes[i]['q'] = flat_params[i*3]
            new_modes[i]['amp'] = flat_params[i*3 + 1]
            new_modes[i]['decay_mult'] = flat_params[i*3 + 2]
            
        # Extract exciter variables from the tail of the array
        att = flat_params[num_mode_vars]
        dec = flat_params[num_mode_vars + 1]
        n_fc = flat_params[num_mode_vars + 2]
        n_q = flat_params[num_mode_vars + 3]
        
        return new_modes, att, dec, n_fc, n_q

    # 3. Multi-Scale Spectral Loss
    def mssl_loss(flat_params, template_modes):
        current_modes, att, dec, n_fc, n_q = unpack_parameters(flat_params, template_modes)
        
        synth_audio = synth_instance.synthesize(
            modes=current_modes, 
            duration=target_duration, 
            attack_ms=att, 
            decay_ms=dec,
            noise_fc=n_fc, 
            noise_q=n_q, 
            q_multiplier=1.0, 
            beating_hz=1.8
        )
        
        if len(synth_audio) > target_samples:
            synth_audio = synth_audio[:target_samples]
        else:
            synth_audio = np.pad(synth_audio, (0, target_samples - len(synth_audio)))
            
        if np.max(np.abs(synth_audio)) > 0:
            synth_audio = synth_audio / np.max(np.abs(synth_audio))

        scales = [2048, 1024, 512]
        total_loss = 0.0
        
        for n_fft in scales:
            hop_length = n_fft // 4
            mag_target = np.abs(librosa.stft(y_target, n_fft=n_fft, hop_length=hop_length))
            mag_synth = np.abs(librosa.stft(synth_audio, n_fft=n_fft, hop_length=hop_length))
            
            norm_target = np.linalg.norm(mag_target, ord='fro')
            sc_loss = np.linalg.norm(mag_target - mag_synth, ord='fro') / norm_target if norm_target > 0 else 0.0
            
            log_target = np.log1p(mag_target)
            log_synth = np.log1p(mag_synth)
            log_loss = np.mean(np.abs(log_target - log_synth))
            
            total_loss += (sc_loss + log_loss)
            
        return total_loss

    # 4. Define Boundaries (Modes + Exciter)
    bounds = []
    for m in initial_modes:
        bounds.append((0.1, 150.0))  # Q range
        bounds.append((0.0, 10.0))   # Amplitude range
        bounds.append((1.0, 200.0))  # Decay multiplier range
        
    # Append Exciter Boundaries
    bounds.append((0.1, 15.0))     # Attack ms
    bounds.append((5.0, 150.0))    # Decay ms
    bounds.append((100.0, 1500.0)) # Noise FC
    bounds.append((0.1, 2.0))      # Noise Q

    # 5. Execute Differential Evolution (Popsize increased to 10, tol added)
    print(f"Starting Full-Stroke MSSL Optimization (maxiter={max_iter}, popsize={popsize})...")
    result = differential_evolution(
        func=mssl_loss,
        bounds=bounds,
        args=(initial_modes,),
        strategy='best1bin',
        maxiter=max_iter,
        popsize=popsize,
        mutation=(0.5, 1.0),
        recombination=0.7,
        tol=0.01,
        disp=True
    )
    
    # 6. Finalize output
    opt_modes, opt_att, opt_dec, opt_nfc, opt_nq = unpack_parameters(result.x, initial_modes)
    optimized_audio = synth_instance.synthesize(
        modes=opt_modes, 
        duration=target_duration, 
        attack_ms=opt_att, 
        decay_ms=opt_dec,
        noise_fc=opt_nfc,
        noise_q=opt_nq
    )
    
    print(f"Optimization Complete. Final MSSL Score: {result.fun:.4f}")
    print(f"Fitted Exciter Params -> Attack: {opt_att:.2f}ms, Decay: {opt_dec:.2f}ms, FC: {opt_nfc:.1f}Hz, Q: {opt_nq:.2f}")
    return opt_modes, optimized_audio