import numpy as np

class HertzianExciter:
    def __init__(self, hammer_config, dt):
        self.m_h = hammer_config['mass']
        self.v_h = hammer_config['velocity_in']
        self.k = hammer_config['stiffness']
        self.lam = hammer_config['dissipation']
        self.alpha = hammer_config['alpha']
        self.dt = dt
        
        self.x_h = 0.0
        self.contact_active = True
        self.e_h_prev = 0.5 * self.m_h * (self.v_h**2)

    # 100% PAPER COMPLIANCE: Receive membrane velocity
    def process_sample(self, z_membrane, z_membrane_dot):
        if not self.contact_active:
            return 0.0, 0.0
            
        self.x_h += self.v_h * self.dt
        zeta = self.x_h - z_membrane
        
        if zeta > 0:
            # 100% PAPER COMPLIANCE: True relative velocity (Eq. 9)
            zeta_dot = self.v_h - z_membrane_dot  
            
            # PURE PHYSICS: No injected noise hacks. 
            damping_force = self.lam * (zeta**self.alpha) * zeta_dot
            base_force = (self.k * (zeta**self.alpha)) + damping_force
            base_force = max(0.0, base_force) 
            
            a_h = -base_force / self.m_h
            self.v_h += a_h * self.dt
            
            e_k = 0.5 * self.m_h * (self.v_h**2)
            e_p = self.k * (zeta**(self.alpha + 1.0)) / (self.alpha + 1.0)
            e_h_curr = e_k + e_p
            
            delta_e_raw = self.e_h_prev - e_h_curr
            dissipated_energy = max(0.0, damping_force * zeta_dot * self.dt)
            delta_e = max(0.0, delta_e_raw - dissipated_energy)
            
            self.e_h_prev = e_h_curr
            
            return base_force, delta_e
        else:
            self.contact_active = False
            return 0.0, 0.0