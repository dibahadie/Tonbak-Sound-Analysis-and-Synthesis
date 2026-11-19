import argparse
import json
from pathlib import Path

import torch
import torchaudio
from einops import rearrange
from tqdm import tqdm

from utils.modal_analysis import CQTModalAnalysis


def extract_modal_features(
    work_dir: Path,
    sample_rate: int,
    num_modes: int,
    hop_length: int,
    fmin: float,
    bins_per_octave: int,
    n_bins: int,
    min_length: int,
    threshold: float,
    diff_threshold: float,
) -> None:
    data_dir = work_dir / "data"
    meta_path = data_dir / "metadata.json"
    feature_dir = data_dir / "features"
    feature_dir.mkdir(parents=True, exist_ok=True)

    with meta_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)

    modal = CQTModalAnalysis(
        sample_rate=sample_rate,
        hop_length=hop_length,
        fmin=fmin,
        n_bins=n_bins,
        bins_per_octave=bins_per_octave,
        min_length=min_length,
        num_modes=num_modes,
        threshold=threshold,
        diff_threshold=diff_threshold,
    )

    keys_to_remove = []

    for key, item in tqdm(metadata.items(), desc="Extracting CQT modal features"):
        audio_file = data_dir / item["filename"]
        modal_file = feature_dir / audio_file.name.replace(".wav", ".pt")

        if not modal_file.exists():
            waveform, sr = torchaudio.load(audio_file)
            assert sr == sample_rate, f"Expected {sample_rate} Hz, got {sr} Hz"
            try:
                modal_freqs, modal_amps, modal_phases = modal(waveform)
            except RuntimeError:
                print(f"[SKIP] Modal extraction failed for {audio_file.name}")
                keys_to_remove.append(key)
                continue

            # Convert Hz to angular frequency (rad/sample) as in ModalDataModule
            modal_freqs = 2.0 * torch.pi * modal_freqs / sample_rate
            modal_tensor = torch.stack([modal_freqs, modal_amps, modal_phases])
            modal_tensor = rearrange(modal_tensor, "s 1 m f -> s m f")
            torch.save(modal_tensor, modal_file)

        metadata[key]["feature_file"] = str(modal_file.relative_to(data_dir))

    for key in keys_to_remove:
        metadata.pop(key)

    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Done. Extracted modal features for {len(metadata)} files ({len(keys_to_remove)} failed).")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract CQT modal features using official CQTModalAnalysis.")
    parser.add_argument("--work_dir", type=Path, default=Path("synthesis/drumblender"))
    parser.add_argument("--sample_rate", type=int, default=48000)
    parser.add_argument("--num_modes", type=int, default=64, help="N_S from paper (default: 64).")
    parser.add_argument("--hop_length", type=int, default=256, help="CQT hop size from paper (default: 256).")
    parser.add_argument("--fmin", type=float, default=20.0, help="Min CQT frequency in Hz (default: 20.0).")
    parser.add_argument("--bins_per_octave", type=int, default=24, help="CQT bins/octave from paper (default: 24).")
    parser.add_argument("--n_bins", type=int, default=240, help="24 bins/oct * 10 octaves = 240.")
    parser.add_argument("--min_length", type=int, default=10)
    parser.add_argument("--threshold", type=float, default=-80.0)
    parser.add_argument("--diff_threshold", type=float, default=5.0)
    args = parser.parse_args()

    extract_modal_features(
        work_dir=args.work_dir,
        sample_rate=args.sample_rate,
        num_modes=args.num_modes,
        hop_length=args.hop_length,
        fmin=args.fmin,
        bins_per_octave=args.bins_per_octave,
        n_bins=args.n_bins,
        min_length=args.min_length,
        threshold=args.threshold,
        diff_threshold=args.diff_threshold,
    )


if __name__ == "__main__":
    main()