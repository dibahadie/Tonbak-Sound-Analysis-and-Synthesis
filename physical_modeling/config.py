import numpy as np

# Global Audio Parameters
SR = 44100
DT = 1.0 / SR

# Membrane Physical Properties (Goat skin approx)
MEMBRANE = {
    'radius': 0.13,              
    'thickness': 0.00035,        
    'density': 1200.0,           
    'young_modulus': 1e3,        
    'poisson_ratio': 0.3,        
    'tension_0': 2000.0,         
    'd1': 0.2,                   # Let the fundamental bass ring beautifully
    'd3': 1.5e-2                 # INCREASED MASSIVELY: Kills the high "metal" ringing instantly
}

# Derived Membrane Properties
MEMBRANE['surface_density'] = MEMBRANE['density'] * MEMBRANE['thickness']
MEMBRANE['bending_stiffness'] = (MEMBRANE['young_modulus'] * MEMBRANE['thickness']**3) / (12 * (1 - MEMBRANE['poisson_ratio']**2))

# Hertzian Hammer Properties (Flesh / Hand)
HAMMER = {
    'mass': 0.04,                
    'velocity_in': 2.0,          
    'stiffness': 2e5,            # Snappy, but still fleshy 
    'dissipation': 0.5,          
    'alpha': 0.7      
}

SYSTEM = {
    'energy_decay_g': 0.9995,    
    'air_mass_load': 0.03,      
    'air_stiffness': 800.0,     # Increased to properly simulate shell pressure 
    'air_damping': 0.05          # LOWERED: Stop suffocating the bass fundamental
}