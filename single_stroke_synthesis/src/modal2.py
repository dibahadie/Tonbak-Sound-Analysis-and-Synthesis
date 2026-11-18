import numpy as np
import scipy.signal as signal

class ModalSynthesizer:
    def __init__(self, sr=44100):
        self.sr = sr
        
    def generate_exciter(self, duration, attack_ms, decay_ms, noise_fc, noise_q, constraints):
        t = np.arange(int(self.sr * duration)) / self.sr
        raw_noise = np.random.uniform(-1, 1, len(t))
        
        # 1. Broadband Smack
        b_noise, a_noise = signal.iirpeak(noise_fc, noise_q, self.sr)
        friction_gain = constraints.get('friction_gain', 2.0)
        friction = signal.lfilter(b_noise, a_noise, raw_noise) * friction_gain
        
        # 2. Snap
        snap_decay = constraints.get('snap_decay', 500.0)
        snap_gain = constraints.get('snap_gain', 1.5)
        snap = raw_noise * np.exp(-t * snap_decay) * snap_gain 
        
        # 3. Flesh Click
        click = np.zeros_like(t)
        click_width_sec = constraints.get('click_width_sec', 0.002)
        click_gain = constraints.get('click_gain', 3.0)
        click_width = max(1, int(self.sr * click_width_sec))
        click[:click_width] = np.hanning(click_width) * click_gain
        
        # 4. Envelopes
        attack_samples = max(1, int((attack_ms / 1000) * self.sr))
        decay_samples = max(1, int((decay_ms / 1000) * self.sr))
        env = np.zeros_like(t)
        env[:attack_samples] = np.linspace(0, 1, attack_samples)
        end_decay = attack_samples + decay_samples
        
        env_floor = constraints.get('env_floor', 0.0001)
        if end_decay < len(env):
            env[attack_samples:end_decay] = np.geomspace(1, env_floor, decay_samples)
        else:
            env[attack_samples:] = np.geomspace(1, env_floor, len(env) - attack_samples)
            
        # 5. Body Noise Haze
        body_noise_decay = constraints.get('body_noise_decay', 12.0)
        body_noise_amp = constraints.get('body_noise_amp', 0.005)
        body_noise_env = np.exp(-t * body_noise_decay)
        body_noise = raw_noise * body_noise_env * body_noise_amp
            
        return (friction * env) + snap + click + body_noise

    def apply_resonator_bank(self, exciter, modes, constraints, q_multiplier=1.0):
        resonated_audio = np.zeros_like(exciter)
        
        exciter_drive = constraints.get('exciter_drive', 2.0)
        driven_exciter = exciter * exciter_drive 
        
        # Extraction Parameters
        nyquist_margin = constraints.get('nyquist_margin', 0.45)
        max_q_clamp = constraints.get('max_q_clamp', 60.0)
        damping_pivot_hz = constraints.get('damping_pivot_hz', 300.0)
        damping_exponent = constraints.get('damping_exponent', 1.0)
        clamp_floor_scalar = constraints.get('clamp_floor_scalar', 0.05)
        default_base_q = constraints.get('default_base_q', 20.0)
        
        for mode in modes:
            freq = mode['freq']
            if freq > (self.sr * nyquist_margin):
                continue
                
            base_q = mode.get('q', default_base_q)
            decay_mult = mode.get('decay_mult', 1.0)
            raw_q = base_q * decay_mult * q_multiplier
            
            if freq > damping_pivot_hz:
                freq_scalar = (damping_pivot_hz / freq) ** damping_exponent
            else:
                freq_scalar = constraints.get('bass_sustain_multiplier', 1.0)
                                
            dynamic_clamp = max_q_clamp * max(freq_scalar, clamp_floor_scalar)
            final_q = min(raw_q, dynamic_clamp)
            
            bandwidth_compensation = np.sqrt(final_q)
            mapped_amp = mode['amp'] * bandwidth_compensation
            
            b, a = signal.iirpeak(freq, final_q, self.sr)
            resonated_audio += signal.lfilter(b, a, driven_exciter) * mapped_amp
            
        return resonated_audio
        
    def synthesize(self, modes, constraints, duration=1.0, attack_ms=1.5, decay_ms=20.0, noise_fc=350.0, noise_q=0.6, q_multiplier=1.0):
        
        exciter = self.generate_exciter(
            duration=duration, 
            attack_ms=attack_ms, 
            decay_ms=decay_ms, 
            noise_fc=noise_fc, 
            noise_q=noise_q, 
            constraints=constraints
        )
        
        resonated = self.apply_resonator_bank(
            exciter=exciter, 
            modes=modes, 
            constraints=constraints,
            q_multiplier=q_multiplier
        )
        
        dry_mix_ratio = constraints.get('dry_mix_ratio', 0.05)
        saturation_gain = constraints.get('saturation_gain', 1.0)
        
        final_audio = resonated + (exciter * dry_mix_ratio)
        final_audio = np.tanh(final_audio * saturation_gain) 
        
        if np.max(np.abs(final_audio)) > 0:
            final_audio /= np.max(np.abs(final_audio))
            
        return final_audio