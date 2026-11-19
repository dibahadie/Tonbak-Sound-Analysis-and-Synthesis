import os
import io
import base64
import librosa
import librosa.display
import numpy as np
import matplotlib
matplotlib.use('Agg') # Headless mode for web servers
import matplotlib.pyplot as plt
import scipy.io.wavfile
from flask import Flask, render_template_string, request, jsonify, send_file

app = Flask(__name__)

# In-memory storage for the session
SESSION = {
    'y': None,
    'sr': 22050,
    'segments': [],
    'current_idx': 0
}

def safe_delta(feature_matrix):
    """Fixes the librosa crash by dynamically sizing the delta window for very short segments."""
    n_frames = feature_matrix.shape[1]
    if n_frames < 3:
        return np.zeros_like(feature_matrix)
    width = 9 if n_frames >= 9 else (n_frames if n_frames % 2 != 0 else n_frames - 1)
    if width < 3:
        return np.zeros_like(feature_matrix)
    return librosa.feature.delta(feature_matrix, width=width)

def get_plot_as_base64(fig):
    """Converts a matplotlib figure to a base64 string for HTML embedding."""
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", bbox_inches='tight', transparent=True)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

@app.route('/')
def index():
    # A modern, dark-themed web UI with CSS Grid
    html = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Tonbak DSP Analyzer</title>
        <style>
            :root { --bg: #121212; --panel: #1e1e1e; --text: #e0e0e0; --accent: #bb86fc; }
            body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: var(--bg); color: var(--text); margin: 0; padding: 20px; }
            .header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; background: var(--panel); padding: 15px 25px; border-radius: 8px; }
            .controls { display: flex; gap: 15px; align-items: center; }
            input[type="file"] { background: #333; color: white; padding: 5px; border-radius: 4px; }
            button { background: var(--accent); color: #000; border: none; padding: 8px 16px; border-radius: 4px; cursor: pointer; font-weight: bold; }
            button:hover { opacity: 0.8; }
            button:disabled { background: #555; cursor: not-allowed; }
            .nav-controls { display: flex; align-items: center; gap: 20px; }
            .stats-container { display: flex; justify-content: space-between; background: var(--panel); padding: 15px 25px; border-radius: 8px; margin-bottom: 20px; font-size: 1.1em; }
            .stat-box { text-align: center; }
            .stat-val { font-size: 1.5em; font-weight: bold; color: var(--accent); display: block; margin-top: 5px; }
            .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 20px; }
            .card { background: var(--panel); padding: 15px; border-radius: 8px; text-align: center; box-shadow: 0 4px 6px rgba(0,0,0,0.3); }
            img { max-width: 100%; height: auto; border-radius: 4px; }
            #loading { display: none; color: var(--accent); font-weight: bold; }
        </style>
    </head>
    <body>
        <div class="header">
            <h2>Tonbak Segment Analyzer</h2>
            <div class="controls">
                <div>Audio (.wav): <input type="file" id="audioFile" accept=".wav"></div>
                <div>Labels (.lab): <input type="file" id="labFile" accept=".lab,.txt"></div>
                <button onclick="uploadFiles()">Load Data</button>
                <span id="loading">Processing...</span>
            </div>
        </div>

        <div class="header" id="playback-bar" style="display: none;">
            <div class="nav-controls">
                <button onclick="changeSegment(-1)" id="btnPrev">⬅ Previous</button>
                <h3 id="seg-info">Segment 0 / 0</h3>
                <button onclick="changeSegment(1)" id="btnNext">Next ➡</button>
            </div>
            <audio id="audio-player" controls></audio>
        </div>

        <div class="stats-container" id="stats-bar" style="display: none;">
            <div class="stat-box">Label<span class="stat-val" id="stat-label">-</span></div>
            <div class="stat-box">Duration<span class="stat-val" id="stat-duration">-</span></div>
            <div class="stat-box">Low/High Energy<span class="stat-val" id="stat-ratio">-</span></div>
            <div class="stat-box">Mean Centroid<span class="stat-val" id="stat-cent">-</span></div>
            <div class="stat-box">Max Flux<span class="stat-val" id="stat-flux">-</span></div>
        </div>

        <div class="grid" id="plots-grid"></div>

        <script>
            async function uploadFiles() {
                const audio = document.getElementById('audioFile').files[0];
                const lab = document.getElementById('labFile').files[0];
                if (!audio || !lab) { alert("Please select both files."); return; }

                document.getElementById('loading').style.display = 'inline';
                const formData = new FormData();
                formData.append('audio', audio);
                formData.append('label', lab);

                const res = await fetch('/upload', { method: 'POST', body: formData });
                document.getElementById('loading').style.display = 'none';
                
                if (res.ok) {
                    document.getElementById('playback-bar').style.display = 'flex';
                    document.getElementById('stats-bar').style.display = 'flex';
                    loadSegment(0);
                } else {
                    alert("Error loading files.");
                }
            }

            async function changeSegment(dir) {
                const res = await fetch(`/navigate?dir=${dir}`);
                const data = await res.json();
                if (data.success) loadSegment(data.index);
            }

            async function loadSegment(index) {
                document.getElementById('loading').style.display = 'inline';
                const res = await fetch(`/analyze?index=${index}`);
                const data = await res.json();
                document.getElementById('loading').style.display = 'none';

                if (data.error) { alert(data.error); return; }

                // Update UI
                document.getElementById('seg-info').innerText = `Segment ${data.index + 1} / ${data.total}`;
                document.getElementById('stat-label').innerText = data.stats.label;
                document.getElementById('stat-duration').innerText = `${data.stats.duration} ms`;
                document.getElementById('stat-ratio').innerText = data.stats.ratio;
                document.getElementById('stat-cent').innerText = `${data.stats.centroid} Hz`;
                document.getElementById('stat-flux').innerText = data.stats.flux;

                document.getElementById('audio-player').src = `/audio?index=${data.index}&t=${new Date().getTime()}`;

                // Update Plots
                const grid = document.getElementById('plots-grid');
                grid.innerHTML = '';
                for (const [title, img] of Object.entries(data.plots)) {
                    grid.innerHTML += `
                        <div class="card">
                            <h3>${title}</h3>
                            <img src="data:image/png;base64,${img}" />
                        </div>`;
                }
            }
        </script>
    </body>
    </html>
    """
    return render_template_string(html)

@app.route('/upload', methods=['POST'])
def upload():
    audio_file = request.files.get('audio')
    label_file = request.files.get('label')
    
    if audio_file and label_file:
        # Load audio into memory
        SESSION['y'], SESSION['sr'] = librosa.load(audio_file, sr=22050)
        
        # Parse labels
        lines = label_file.read().decode('utf-8').splitlines()
        SESSION['segments'] = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 3:
                SESSION['segments'].append({
                    'start': float(parts[0]),
                    'end': float(parts[1]),
                    'label': parts[2]
                })
        SESSION['current_idx'] = 0
        return jsonify({"success": True, "total": len(SESSION['segments'])})
    return jsonify({"success": False}), 400

@app.route('/navigate')
def navigate():
    dir = int(request.args.get('dir', 0))
    new_idx = SESSION['current_idx'] + dir
    if 0 <= new_idx < len(SESSION['segments']):
        SESSION['current_idx'] = new_idx
        return jsonify({"success": True, "index": new_idx})
    return jsonify({"success": False})

@app.route('/audio')
def get_audio():
    idx = int(request.args.get('index', 0))
    seg = SESSION['segments'][idx]
    start_sample = int(seg['start'] * SESSION['sr'])
    end_sample = int(seg['end'] * SESSION['sr'])
    y_seg = SESSION['y'][start_sample:end_sample]
    
    buf = io.BytesIO()
    scipy.io.wavfile.write(buf, SESSION['sr'], (y_seg * 32767).astype(np.int16))
    buf.seek(0)
    return send_file(buf, mimetype='audio/wav')

@app.route('/analyze')
def analyze():
    idx = int(request.args.get('index', 0))
    seg = SESSION['segments'][idx]
    start_sample = int(seg['start'] * SESSION['sr'])
    end_sample = int(seg['end'] * SESSION['sr'])
    y_seg = SESSION['y'][start_sample:end_sample]
    
    if len(y_seg) == 0:
        return jsonify({"error": "Empty segment"})

    # --- DSP Calculations ---
    # Global setup
    plt.style.use('dark_background')
    plots = {}
    
    # Base STFT
    S = np.abs(librosa.stft(y_seg))
    
    # 1. HPSS
    H, P = librosa.decompose.hpss(S)
    fig, ax = plt.subplots(figsize=(6, 3))
    librosa.display.specshow(librosa.amplitude_to_db(P, ref=np.max), y_axis='log', x_axis='time', sr=SESSION['sr'], ax=ax)
    plots['HPSS (Percussive Transients)'] = get_plot_as_base64(fig)
    
    # 2. Mel-Spectrogram
    mel = librosa.feature.melspectrogram(y=y_seg, sr=SESSION['sr'])
    fig, ax = plt.subplots(figsize=(6, 3))
    librosa.display.specshow(librosa.power_to_db(mel, ref=np.max), y_axis='mel', x_axis='time', sr=SESSION['sr'], ax=ax)
    plots['Mel-Spectrogram'] = get_plot_as_base64(fig)

    # 3. MFCCs & Deltas (With fix for short segments)
    mfcc = librosa.feature.mfcc(S=librosa.power_to_db(mel), n_mfcc=13)
    delta_mfcc = safe_delta(mfcc) # <-- BUG FIXED HERE
    
    fig, ax = plt.subplots(figsize=(6, 3))
    librosa.display.specshow(mfcc, x_axis='time', sr=SESSION['sr'], ax=ax)
    plots['MFCCs'] = get_plot_as_base64(fig)
    
    fig, ax = plt.subplots(figsize=(6, 3))
    librosa.display.specshow(delta_mfcc, x_axis='time', sr=SESSION['sr'], ax=ax)
    plots['Delta MFCCs (Velocity)'] = get_plot_as_base64(fig)

    # 4. Spectral Shape (Centroid, Rolloff, Spread)
    cent = librosa.feature.spectral_centroid(S=S, sr=SESSION['sr'])[0]
    rolloff = librosa.feature.spectral_rolloff(S=S, sr=SESSION['sr'])[0]
    bw = librosa.feature.spectral_bandwidth(S=S, sr=SESSION['sr'])[0]
    
    fig, ax = plt.subplots(figsize=(6, 3))
    frames = range(len(cent))
    t = librosa.frames_to_time(frames, sr=SESSION['sr'])
    ax.plot(t, cent, label='Centroid', color='#bb86fc')
    ax.plot(t, rolloff, label='Rolloff (85%)', color='#cf6679')
    ax.fill_between(t, np.maximum(0, cent - bw), cent + bw, alpha=0.3, color='#bb86fc')
    ax.legend(loc='upper right')
    plots['Spectral Shape'] = get_plot_as_base64(fig)

    # 5. Time Domain (ZCR & Flux)
    zcr = librosa.feature.zero_crossing_rate(y_seg)[0]
    flux = librosa.onset.onset_strength(y=y_seg, sr=SESSION['sr'])
    
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(t, zcr / (np.max(zcr) or 1), label='ZCR (Norm)', color='#03dac6')
    ax.plot(t, flux / (np.max(flux) or 1), label='Flux (Norm)', color='#ffeb3b')
    ax.legend(loc='upper right')
    plots['ZCR & Spectral Flux'] = get_plot_as_base64(fig)

    # Calculate Summary Stats
    duration = (seg['end'] - seg['start']) * 1000
    freqs = librosa.fft_frequencies(sr=SESSION['sr'])
    low_e = np.sum(S[np.where(freqs < 250)[0], :])
    high_e = np.sum(S[np.where(freqs > 1000)[0], :])
    ratio = low_e / (high_e + 1e-10)

    stats = {
        'label': seg['label'],
        'duration': f"{duration:.1f}",
        'ratio': f"{ratio:.2f}",
        'centroid': f"{np.mean(cent):.0f}",
        'flux': f"{np.max(flux):.2f}"
    }

    return jsonify({
        "index": idx,
        "total": len(SESSION['segments']),
        "stats": stats,
        "plots": plots
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)