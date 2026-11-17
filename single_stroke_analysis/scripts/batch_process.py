import os
import glob
import numpy as np
import sys

# Ensure src/ is accessible
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preprocess import load_and_preprocess
from src.extract import extract_features

def process_all():
    data_dir = os.path.join("data", "raw")
    wav_files = glob.glob(os.path.join(data_dir, "**", "*.wav"), recursive=True)
    
    if not wav_files:
        print(f"No .wav files found in {data_dir}")
        return

    print(f"Found {len(wav_files)} files. Starting batch processing...")
    
    for wav_path in wav_files:
        print(f"Processing: {wav_path}")
        
        # 1. Preprocess
        y, sr = load_and_preprocess(wav_path, target_sr=44100)
        
        # 2. Extract Features
        features = extract_features(y, sr)
        
        # 3. Save as compressed NumPy dictionary array
        out_path = wav_path.replace('.wav', '.npz')
        np.savez_compressed(out_path, **features)
        
    print("Batch processing complete.")

if __name__ == "__main__":
    process_all()