import os
import numpy as np
import scipy.io.wavfile as wav
import copy
from config import SR, MEMBRANE, HAMMER, SYSTEM
from synthesizer import TonbakSynthesizer

# Define the acoustic bounding boxes for the two main strokes
STROKE_BOUNDS = {
    'tom': {
        'hit_r': (0.01, 0.05),          # Near center
        'hammer_mass': (0.03, 0.06),    # Heavy, fleshy hand
        'hammer_stiffness': (1e4, 3e4), # Soft contact
        'velocity': (2.0, 5.0),         # Dynamic range
        'd3': (1e-4, 5e-4)              # Let some highs breathe, but keep it warm
    },
    'bak': {
        'hit_r': (0.10, 0.125),         # Near the rim
        'hammer_mass': (0.005, 0.015),  # Light fingertip/knuckle
        'hammer_stiffness': (8e4, 2e5), # Very hard contact
        'velocity': (3.0, 7.0),         # Snappy, faster hits
        'd3': (5e-5, 2e-4)              # Keep it bright and clicking
    }
}

def generate_tonbak_dataset(num_samples=100, out_dir="tonbak_dataset"):
    os.makedirs(os.path.join(out_dir, "tom"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "bak"), exist_ok=True)
    
    for stroke_type, bounds in STROKE_BOUNDS.items():
        print(f"Generating {num_samples} {stroke_type} strokes...")
        
        for i in range(num_samples):
            # 1. Randomize parameters within the defined bounds
            mem_cfg = copy.deepcopy(MEMBRANE)
            ham_cfg = copy.deepcopy(HAMMER)
            sys_cfg = copy.deepcopy(SYSTEM)
            
            hit_r = np.random.uniform(*bounds['hit_r'])
            ham_cfg['mass'] = np.random.uniform(*bounds['hammer_mass'])
            ham_cfg['stiffness'] = np.random.uniform(*bounds['hammer_stiffness'])
            ham_cfg['velocity_in'] = np.random.uniform(*bounds['velocity'])
            mem_cfg['d3'] = np.random.uniform(*bounds['d3'])
            
            # Minor randomizations for natural acoustic variance
            mem_cfg['tension_0'] = np.random.uniform(1900, 2300) # Tuning drifts slightly
            ham_cfg['alpha'] = np.random.uniform(1.3, 1.7)       # Hand shape changes slightly
            
            # 2. Initialize Engine
            synth = TonbakSynthesizer(
                hit_r=hit_r, hit_phi=np.random.uniform(0, np.pi/4), # Slight angle variation
                pickup_r=0.08, pickup_phi=0.0,
                membrane_cfg=mem_cfg, hammer_cfg=ham_cfg, system_cfg=sys_cfg
            )
            
            # 3. Generate Audio
            audio_wave = synth.generate(duration_sec=1.0)
            
            # (Optional) Insert convolution reverb or EQ filtering here before saving
            
            # 4. Save to disk
            filename = f"{stroke_type}_{i:04d}.wav"
            filepath = os.path.join(out_dir, stroke_type, filename)
            wav.write(filepath, SR, audio_wave.astype('float32'))

# Run the generator
generate_tonbak_dataset(num_samples=5)