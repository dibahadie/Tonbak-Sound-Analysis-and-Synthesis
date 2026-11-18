import numpy as np
from config import SR, DT 
from core.cavity import generate_modes
from core.exciter import HertzianExciter
from core.modulator import EnergyModulator
from core.resonators import ModalResonatorBank
import scipy.special as sp

class TonbakSynthesizer:
    def __init__(self, hit_r, hit_phi, pickup_r, pickup_phi, membrane_cfg, hammer_cfg, system_cfg):
        self.modes = generate_modes(membrane_cfg, system_cfg, max_n=15, max_m=15)
        self.exciter = HertzianExciter(hammer_cfg, DT)
        self.modulator = EnergyModulator(system_cfg, membrane_cfg) 
        self.resonators = ModalResonatorBank(self.modes, membrane_cfg, DT)
        
        self.K_hit = self.resonators.get_spatial_eigenfunction(hit_r, hit_phi)
        self.K_pickup = self.resonators.get_spatial_eigenfunction(pickup_r, pickup_phi)
        
        # FIX: Store air cavity parameters dynamically from the GUI configuration
        self.air_stiffness = system_cfg['air_stiffness']
        self.air_damping = system_cfg.get('air_damping', 0.05) 
        self.area = np.pi * (membrane_cfg['radius']**2)
        
        self.f_air_coeffs = np.zeros(len(self.modes))
        for i, m in enumerate(self.modes):
            if m['n'] == 0:
                self.f_air_coeffs[i] = (2 * np.pi * membrane_cfg['radius']**2 / m['mu']) * sp.jv(1, m['mu'])
        
    def generate(self, duration_sec=1.0):
        total_samples = int(SR * duration_sec)
        audio_out = np.zeros(total_samples)
        
        z_hit = 0.0 
        z_hit_prev = 0.0  
        z_hat_prev = 0.0  
        
        for i in range(total_samples):
            z_hit_dot = (z_hit - z_hit_prev) / DT
            z_hit_prev = z_hit
            
            force, delta_e = self.exciter.process_sample(z_hit, z_hit_dot)
            
            t_qs = self.modulator.process_sample(delta_e)
            
            z_hat = np.sum((self.resonators.y1 * self.f_air_coeffs) / (self.area * self.resonators.norm_sq))
            z_hat_dot = (z_hat - z_hat_prev) / DT
            z_hat_prev = z_hat
            
            # FIX: Reference the instance variable self.air_damping
            F_air = -(self.air_stiffness * z_hat) - (self.air_damping * z_hat_dot)
            f_air_density = F_air / self.area
            f_air_vector = f_air_density * self.f_air_coeffs
            
            z_pickup, z_hit, y_curr = self.resonators.process_sample(force, t_qs, self.K_hit, self.K_pickup, f_air_vector)
            
            audio_out[i] = z_pickup
            
        audio_out -= np.mean(audio_out)
        max_val = np.max(np.abs(audio_out))
        if max_val > 0:
            audio_out /= max_val
            
        return audio_out