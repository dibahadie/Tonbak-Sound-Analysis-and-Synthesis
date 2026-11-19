import argparse
from pathlib import Path
import numpy as np
import librosa

AUDIO_EXTS = {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".aiff", ".aif"}


def extract_log_f0_stats(
    dataset_root: Path,
    sr: int = 22050,
    fmin: float = 50.0,
    fmax: float = 500.0,
    hop_length: int = 256,
) -> None:
    audio_dir = dataset_root / "audio"
    mean_dir = dataset_root / "prosodic" / "mean_log_f0"
    std_dir = dataset_root / "prosodic" / "std_log_f0"

    if not audio_dir.exists():
        raise FileNotFoundError(f"Audio directory not found: {audio_dir}")

    audio_files = sorted(
        p for p in audio_dir.iterdir() if p.is_file() and p.suffix.lower() in AUDIO_EXTS
    )
    if not audio_files:
        raise FileNotFoundError(f"No supported audio files found in {audio_dir}")

    mean_dir.mkdir(parents=True, exist_ok=True)
    std_dir.mkdir(parents=True, exist_ok=True)

    processed = 0

    for audio_path in audio_files:
        y, _ = librosa.load(audio_path, sr=sr, mono=True)
        if len(y) == 0:
            print(f"[SKIP] Empty audio file: {audio_path.name}")
            continue

        f0, voiced_flag, _ = librosa.pyin(
            y,
            fmin=fmin,
            fmax=fmax,
            sr=sr,
            hop_length=hop_length,
        )

        valid = voiced_flag & np.isfinite(f0) & (f0 > 0)
        voiced_f0 = f0[valid]

        if len(voiced_f0) == 0:
            print(f"[WARN] No voiced frames detected in {audio_path.name}")
            continue

        log_f0 = np.log(voiced_f0)
        mean_log_f0 = float(np.mean(log_f0))
        std_log_f0 = float(np.std(log_f0))

        mean_path = mean_dir / f"{audio_path.stem}.txt"
        std_path = std_dir / f"{audio_path.stem}.txt"

        mean_path.write_text(f"{mean_log_f0:.6f}\n", encoding="utf-8")
        std_path.write_text(f"{std_log_f0:.6f}\n", encoding="utf-8")

        processed += 1
        print(
            f"[OK] {audio_path.name} ({len(voiced_f0)} voiced frames) "
            f"-> mean={mean_log_f0:.6f}, std={std_log_f0:.6f}"
        )

    print(f"\nDone. Processed {processed}/{len(audio_files)} files.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute per-file mean and std of log F0 and save to prosodic/."
    )
    parser.add_argument(
        "--dataset_root",
        type=Path,
        required=True,
        help="Path to dataset root containing 'audio/'.",
    )
    parser.add_argument(
        "--sr",
        type=int,
        default=22050,
        help="Target sample rate for audio loading (default: 22050).",
    )
    parser.add_argument(
        "--fmin",
        type=float,
        default=50.0,
        help="Minimum F0 frequency in Hz (default: 50.0).",
    )
    parser.add_argument(
        "--fmax",
        type=float,
        default=500.0,
        help="Maximum F0 frequency in Hz (default: 500.0).",
    )
    parser.add_argument(
        "--hop_length",
        type=int,
        default=256,
        help="Hop length in samples for F0 extraction (default: 256).",
    )
    args = parser.parse_args()
    extract_log_f0_stats(
        args.dataset_root,
        sr=args.sr,
        fmin=args.fmin,
        fmax=args.fmax,
        hop_length=args.hop_length,
    )


if __name__ == "__main__":
    main()