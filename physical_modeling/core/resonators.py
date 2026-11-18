import numpy as np
import scipy.special as sp

class ModalResonatorBank:
    def __init__(self, modes, membrane_config, dt):
        self.modes = modes
        self.dt = dt
        self.R = membrane_config['radius']
        self.T_0 = membrane_config['tension_0']
        
        self.y1 = np.zeros(len(modes))
        self.y2 = np.zeros(len(modes))
        
        # 100% PAPER COMPLIANCE: Track input history for Bilinear numerator zeros
        self.exc1 = np.zeros(len(modes))
        self.exc2 = np.zeros(len(modes))
        
        self.sigma = np.array([m['effective_sigma'] for m in self.modes])
        self.norm_sq = np.array([m['norm_sq'] for m in self.modes])
        self.spatial_freq = np.array([m['spatial_freq'] for m in self.modes])
        self.D_over_sigma = membrane_config['bending_stiffness'] / self.sigma

    def get_spatial_eigenfunction(self, r, phi):
        K = np.zeros(len(self.modes))
        for i, mode in enumerate(self.modes):
            angular_comp = np.cos(mode['n'] * phi)
            radial_comp = sp.jv(mode['n'], mode['mu'] * (r / self.R))
            K[i] = angular_comp * radial_comp
        return K

    def process_sample(self, force, t_qs_absolute, K_hit, K_pickup, f_air=0.0):
        c_mod_sq = (self.T_0 + t_qs_absolute) / self.sigma
        omega_mod = np.sqrt((self.spatial_freq**2) * ((self.D_over_sigma * self.spatial_freq**2) + c_mod_sq))
        
        alpha = np.array([m['alpha'] for m in self.modes])
        
        denominator = 1.0 + (alpha * self.dt) + (omega_mod**2 * (self.dt**2) / 4.0)
        a1 = (2.0 - (omega_mod**2 * (self.dt**2) / 2.0)) / denominator
        a2 = - (1.0 - (alpha * self.dt) + (omega_mod**2 * (self.dt**2) / 4.0)) / denominator
        b0 = (self.dt**2 / 4.0) / denominator
        
        modal_excitation = ((force * K_hit) + f_air) / self.sigma
        
        # 100% PAPER COMPLIANCE: Apply the (1 + 2z^-1 + z^-2) Bilinear zeros
        bilinear_input = b0 * (modal_excitation + (2.0 * self.exc1) + self.exc2)
        
        y_curr = (a1 * self.y1) + (a2 * self.y2) + bilinear_input
        
        # Shift states
        self.y2 = self.y1
        self.y1 = y_curr
        self.exc2 = self.exc1
        self.exc1 = modal_excitation
        
        z_pickup = np.sum((y_curr * K_pickup) / self.norm_sq)
        z_hit = np.sum((y_curr * K_hit) / self.norm_sq)
        
        return z_pickup, z_hit, y_curr