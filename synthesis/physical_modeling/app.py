import os
import json
import io
import scipy.io.wavfile as wav
from flask import Flask, render_template, request, send_file, jsonify
from synthesis.physical_modeling.synthesizer import TonbakSynthesizer, SR

app = Flask(__name__)
os.makedirs("configs", exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    
    # Reconstruct configs from slider data
    membrane_cfg = {
        'radius': float(data['radius']),
        'thickness': float(data['thickness']),
        'density': 1200.0,
        'young_modulus': float(data['young_modulus']),
        'poisson_ratio': 0.3,
        'tension_0': float(data['tension_0']),
        'd1': float(data['d1']),
        'd3': float(data['d3'])
    }
    membrane_cfg['surface_density'] = membrane_cfg['density'] * membrane_cfg['thickness']
    membrane_cfg['bending_stiffness'] = (membrane_cfg['young_modulus'] * membrane_cfg['thickness']**3) / (12 * (1 - 0.3**2))

    hammer_cfg = {
        'mass': float(data['hammer_mass']),
        'velocity_in': float(data['hammer_vel']),
        'stiffness': float(data['hammer_stiffness']),
        'dissipation': float(data['hammer_dissipation']),
        'alpha': float(data['hammer_alpha'])  # FIX: Now controlled by the GUI
    }

    system_cfg = {
        'energy_decay_g': float(data['energy_decay_g']),
        'air_mass_load': float(data['air_mass_load']),
        'air_stiffness': float(data['air_stiffness']),
        'air_damping': float(data['air_damping'])
    }

    # Initialize and generate
    synth = TonbakSynthesizer(
        hit_r=float(data['hit_r']), hit_phi=0.0,
        pickup_r=float(data['pickup_r']), pickup_phi=0.0,
        membrane_cfg=membrane_cfg, hammer_cfg=hammer_cfg, system_cfg=system_cfg
    )
    
    audio_wave = synth.generate(duration_sec=1.5)
    
    # Save to in-memory bytes buffer
    wav_io = io.BytesIO()
    wav.write(wav_io, SR, audio_wave.astype('float32'))
    wav_io.seek(0)
    
    return send_file(wav_io, mimetype='audio/wav')

@app.route('/save_config', methods=['POST'])
def save_config():
    data = request.json
    filename = data.pop('config_name', 'default_preset')
    filepath = os.path.join('configs', f"{filename}.json")
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)
        
    return jsonify({"status": "success", "file": filepath})

if __name__ == '__main__':
    app.run(debug=True, port=5000)