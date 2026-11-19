import numpy as np

class EnergyModulator:
    def __init__(self, system_config, membrane_config):
        self.g = system_config['energy_decay_g']
        
        # 100% PAPER COMPLIANCE: Calculate exact scalar from elasticity and geometry (Eq 8 & 36)
        E = membrane_config['young_modulus']
        h = membrane_config['thickness']
        v = membrane_config['poisson_ratio']
        R = membrane_config['radius']
        T_0 = membrane_config['tension_0']
        
        C_NL = (E * h) / (1.0 - v**2)
        S_0 = np.pi * (R**2)
        
        # Scalar = C_NL / (S_0 * T_0)
        self.exact_scalar = C_NL / (S_0 * T_0)
        self.stored_energy = 0.0
        
    def process_sample(self, delta_e):
        self.stored_energy = delta_e + (self.g * self.stored_energy)
        # Returns absolute tension variation T_qs in N/m
        t_qs = self.exact_scalar * self.stored_energy
        return t_qs