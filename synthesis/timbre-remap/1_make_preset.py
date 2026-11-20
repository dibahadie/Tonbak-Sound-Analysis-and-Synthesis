import argparse
from pathlib import Path
from timbreremap.synth import Snare808


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a base Snare808 preset JSON for timbre remapping.")
    parser.add_argument("--out", type=Path, default=Path("synthesis/timbre_remap/presets/tonbak_808.json"))
    args = parser.parse_args()

    patch = {
        ("osc1", "freq"): 140.0,       # Low body resonance (Hz)
        ("osc1", "mod"): 0.25,
        ("osc2", "freq"): 290.0,       # Upper membrane resonance (Hz)
        ("osc2", "mod"): 0.15,
        ("freq_env", "decay"): 35.0,   # Pitch envelope decay (ms)
        ("osc1_env", "decay"): 220.0,  # Body decay (ms)
        ("osc2_env", "decay"): 140.0,  # Upper mode decay (ms)
        ("noise_env", "decay"): 90.0,  # Transient/skin noise decay (ms)
        ("noise_filter", "freq"): 1800.0,
        ("noise_filter", "q"): 1.2,
        ("osc1_gain", "gain"): -3.0,
        ("osc2_gain", "gain"): -6.0,
        ("noise_gain", "gain"): -12.0,
        ("tanh", "in_gain"): 3.0,
    }

    # Damp oscillator frequency modulations by 1e-3
    damping = {
        ("osc1", "freq"): 1e-3,
        ("osc2", "freq"): 1e-3,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    Snare808.save_params_json(str(args.out), patch, damping)
    print(f"[OK] Saved preset -> {args.out}")


if __name__ == "__main__":
    main()