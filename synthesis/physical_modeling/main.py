import scipy.io.wavfile as wav
from synthesis.physical_modeling.synthesizer import TonbakSynthesizer, SR

def main():
    print("Initializing Physical Tonbak Model...")
    
    # Hit exactly in the center (r=0), mic placed halfway to the edge
    synth = TonbakSynthesizer(
        hit_r=0.03, hit_phi=0.0, 
        pickup_r=0.07, pickup_phi=0.0
    )
    
    print("Synthesizing 1.5 seconds of non-linear audio...")
    audio_wave = synth.generate(duration_sec=1.5)
    
    # Save the output
    wav.write("tonbak_physical_strike.wav", SR, audio_wave.astype('float32'))
    print("Synthesis complete. Saved to tonbak_physical_strike.wav")

if __name__ == "__main__":
    main()