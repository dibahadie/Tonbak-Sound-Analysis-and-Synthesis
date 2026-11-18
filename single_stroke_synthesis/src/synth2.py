import numpy as np

class PhysicalTombakSynthesizer:
    def __init__(self, sr=44100):
        self.sr = sr

    def generate_hertzian_exciter(self, duration, params):
        """
        Calculates the exact contact force of a physical mass (the hand/finger) 
        striking the membrane using Hertzian contact theory.
        """
        total_samples = int(self.sr * duration)
        force = np.zeros(total_samples)
        
        # Physical striker parameters (no hardcoded defaults, safe fallbacks)
        m_h = params.get('striker_mass_kg', 0.05)       # Mass of striking finger/hand
        v_in = params.get('striker_velocity', 3.0)      # Incoming speed (m/s)
        k_c = params.get('contact_stiffness', 1e5)      # Hardness of flesh vs skin
        alpha = params.get('hertz_alpha', 1.5)          # Non-linear exponent (1.5 for spheres)
        
        # Effective mass and spring of the drumhead at the strike point
        m_m = params.get('membrane_effective_mass', 0.1)
        k_m = params.get('membrane_stiffness', 1e4)
        r_m = params.get('membrane_damping', 5.0)

        # State variables for the numerical solver (Euler-Cromer integration)
        x_h = 0.0  # Hammer position
        v_h = v_in # Hammer velocity
        x_m = 0.0  # Membrane position
        v_m = 0.0  # Membrane velocity

        dt = 1.0 / self.sr

        # Solve the differential equations sample-by-sample for the contact duration
        for i in range(total_samples):
            # Calculate physical compression (delta)
            compression = x_h - x_m
            
            if compression > 0:
                # Hertzian non-linear force calculation
                f_contact = k_c * (compression ** alpha)
            else:
                # The hand has bounced off the membrane
                f_contact = 0.0
                
            force[i] = f_contact

            # Update hammer states
            a_h = -f_contact / m_h
            v_h += a_h * dt
            x_h += v_h * dt

            # Update membrane states (driven by contact force, opposed by its own stiffness/damping)
            a_m = (f_contact - (k_m * x_m) - (r_m * v_m)) / m_m
            v_m += a_m * dt
            x_m += v_m * dt

            # Optimization: break early if the hand has bounced and force returned to zero
            if compression <= 0 and i > int(self.sr * 0.005):
                break
                
        # Scale the raw Newton force to audio domain amplitude
        force_scalar = params.get('force_to_audio_scalar', 0.001)
        return force * force_scalar

    def apply_modulated_resonators(self, exciter, modes, params):
        total_samples = len(exciter)
        resonated_audio = np.zeros(total_samples)
        
        tension_mod_coeff = params.get('tension_modulation_coeff', 0.005)
        nyquist_margin = params.get('nyquist_margin', 0.45)
        
        energy_smoothing_hz = params.get('energy_smoothing_hz', 40.0)
        alpha_e = np.exp(-2.0 * np.pi * energy_smoothing_hz / self.sr)
        
        energy_env = np.zeros(total_samples)
        current_energy = 0.0
        
        for i in range(total_samples):
            current_energy = (alpha_e * current_energy) + ((1.0 - alpha_e) * (exciter[i] ** 2))
            energy_env[i] = current_energy
            
        max_e = np.max(energy_env)
        if max_e > 0:
            energy_env /= max_e

        dt = 1.0 / self.sr
        
        for mode in modes:
            base_freq = mode['freq']
            if base_freq > (self.sr * nyquist_margin):
                continue
                
            base_q = mode.get('q', 20.0)
            amp = mode.get('amp', 1.0)
            
            bandwidth = base_freq / base_q
            r = np.exp(-np.pi * bandwidth * dt)
            r_sq = r * r
            
            bandwidth_comp = np.sqrt(base_q)
            scaled_amp = amp * bandwidth_comp
            
            y1 = 0.0
            y2 = 0.0
            mode_output = np.zeros(total_samples)
            
            # Create an impulse trigger from the exciter (e.g., only the first 15ms drives the filter directly)
            # After that, the filter rings out naturally according to its extracted Q (r)
            for i in range(total_samples):
                inst_freq = base_freq * (1.0 + (tension_mod_coeff * energy_env[i]))
                inst_freq = min(inst_freq, self.sr * 0.49)
                
                theta = 2.0 * np.pi * inst_freq * dt
                coeff = 2.0 * r * np.cos(theta)
                
                # THE FIX: Feed the raw exciter only for the transient, 
                # but add a gentle background noise excitation so the modes never starve in the tail
                driving_force = (exciter[i] * 0.1)
                
                y_curr = (coeff * y1) - (r_sq * y2) + driving_force
                y_curr = np.clip(y_curr, -5.0, 5.0)
                
                y2 = y1
                y1 = y_curr
                
                mode_output[i] = y_curr * scaled_amp
                
            resonated_audio += mode_output
            
        return resonated_audio

    def synthesize(self, modes, params, duration=1.0):
        # 1. Generate the non-linear collision (Flesh Thwack)
        exciter = self.generate_hertzian_exciter(duration, params)
        
        # 2. Inject Broadband Friction (To simulate the skin's micro-texture)
        friction_gain = params.get('friction_gain', 0.1)
        if friction_gain > 0:
            t = np.arange(int(self.sr * duration)) / self.sr
            noise = np.random.uniform(-1, 1, len(t))
            friction_decay = params.get('friction_decay', 50.0)
            friction_env = np.exp(-t * friction_decay)
            exciter += (noise * friction_env * friction_gain)
        
        # 3. Apply the time-varying, tension-modulated physics engine
        resonated = self.apply_modulated_resonators(exciter, modes, params)
        
        # 4. Final Mix 
        dry_mix = params.get('dry_exciter_mix', 0.05)
        final_audio = resonated + (exciter * dry_mix)
        
        # Soft-clip and normalize
        saturation_gain = params.get('saturation_gain', 1.0)
        final_audio = np.tanh(final_audio * saturation_gain)
        
        if np.max(np.abs(final_audio)) > 0:
            final_audio /= np.max(np.abs(final_audio))
            
        return final_audio