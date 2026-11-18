import scipy.special as sp
import numpy as np

def generate_modes(membrane_config, system_config, max_n=15, max_m=15):
    """Calculates physical modal frequencies, damping, and spatial norms based on exact PDE roots."""
    modes = []
    R = membrane_config['radius']
    sigma = membrane_config['surface_density']
    D = membrane_config['bending_stiffness']
    c_squared = membrane_config['tension_0'] / sigma
    
    for n in range(max_n):
        # Find the first 'max_m' roots of the n-th order Bessel function
        roots = sp.jn_zeros(n, max_m)
        for m_idx, mu in enumerate(roots):
            m = m_idx + 1 
            
            effective_sigma = sigma
            
            if n == 0:
                # Axisymmetric modes compress the shell air dynamically in synthesizer.py
                # No static added stiffness here to prevent double-counting
                delta_n0 = 2.0  # (1 + Kronecker delta) for n=0
            else:
                # Non-axisymmetric modes move air side-to-side (inertive mass load)
                effective_sigma += system_config['air_mass_load']
                delta_n0 = 1.0  # for n>0
                
            # Wave equation angular frequency with bending stiffness (NO static air stiffness)
            spatial_freq = mu / R
            omega_sq = (spatial_freq**2) * ((D / effective_sigma) * (spatial_freq**2) + c_squared)
            omega_0 = np.sqrt(omega_sq)
            
            # Loss factor
            alpha = (membrane_config['d1'] + membrane_config['d3'] * (spatial_freq**2)) / (2 * effective_sigma)
            
            # 100% PAPER COMPLIANCE: Calculate the exact spatial L2 norm mathematically
            bessel_next = sp.jv(n + 1, mu)
            norm_sq = (np.pi * (R**2) / 2.0) * delta_n0 * (bessel_next**2)
            
            modes.append({
                'n': n, 
                'm': m, 
                'mu': mu,
                'omega_0': omega_0,
                'alpha': alpha,
                'effective_sigma': effective_sigma, 
                'norm_sq': norm_sq,
                'spatial_freq': spatial_freq
            })
            
    return modes