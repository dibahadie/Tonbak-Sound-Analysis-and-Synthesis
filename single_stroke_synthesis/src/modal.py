import numpy as np
import scipy.signal as signal

class ModalSynthesizer:
    def __init__(self, sr=44100):
        self.sr = sr
        
    def generate_exciter(self, duration=1.0, attack_ms=0.5, decay_ms=35.0, noise_fc=301.0, noise_q=0.5):
        t = np.arange(int(self.sr * duration)) / self.sr
        raw_noise = np.random.uniform(-1, 1, len(t))
        
        # 1. Broadband Smack
        b_noise, a_noise = signal.iirpeak(noise_fc, noise_q, self.sr)
        friction = signal.lfilter(b_noise, a_noise, raw_noise) * 2.0
        snap = raw_noise * np.exp(-t * (1000 / 2.0)) * 1.5 
        
        # 2. Flesh Click
        click = np.zeros_like(t)
        click_width = int(self.sr * 0.002) 
        click[:click_width] = np.hanning(click_width) * 3.0
        
        # Envelope for friction
        attack_samples = max(1, int((attack_ms / 1000) * self.sr))
        decay_samples = max(1, int((decay_ms / 1000) * self.sr))
        
        env = np.zeros_like(t)
        env[:attack_samples] = np.linspace(0, 1, attack_samples)
        end_decay = attack_samples + decay_samples
        if end_decay < len(env):
            env[attack_samples:end_decay] = np.geomspace(1, 0.0001, decay_samples)
        else:
            env[attack_samples:] = np.geomspace(1, 0.0001, len(env) - attack_samples)
            
        # REMOVED the 400ms body_noise floor completely.
        return (friction * env) + snap + click
        
    def apply_resonator_bank(self, exciter, modes, global_q_multiplier=1.0, beating_hz=1.8):
        output = np.zeros_like(exciter)
        
        for mode in modes:
            fc = mode.get('freq', 100)
            base_q = mode.get('q', 10)
            amp = mode.get('amp', 1.0)
            
            # Independent decay speeds
            mode_q_mult = mode.get('decay_mult', global_q_multiplier)
            Q = max(0.5, base_q * mode_q_mult)
            
            # Main mode
            b1, a1 = signal.iirpeak(fc, Q, self.sr)
            output += signal.lfilter(b1, a1, exciter) * amp
            
            # Beating mode (Acoustic Wobble)
            b2, a2 = signal.iirpeak(fc + beating_hz, max(0.5, Q * 0.9), self.sr)
            output += signal.lfilter(b2, a2, exciter) * (amp * 0.4)
            
        return output
        
    def synthesize(self, modes, duration=1.0, attack_ms=0.5, decay_ms=35.0, noise_fc=301.0, noise_q=0.5, q_multiplier=80.0, beating_hz=1.8):
        exciter = self.generate_exciter(duration, attack_ms, decay_ms, noise_fc, noise_q)
        resonated = self.apply_resonator_bank(exciter, modes, q_multiplier, beating_hz)
        
        # Lowered the dry exciter blend from 0.3 back to 0.1 to keep the background black
        final_audio = resonated + (exciter * 0.1)
        final_audio = np.tanh(final_audio * 2.5) 
        
        if np.max(np.abs(final_audio)) > 0:
            final_audio /= np.max(np.abs(final_audio))
            
        return final_audio