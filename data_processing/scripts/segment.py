import argparse
from pathlib import Path
import soundfile as sf

AUDIO_EXTS = {".wav", ".flac", ".ogg", ".aiff", ".aif"}


def parse_lab_file(lab_path: Path) -> list[tuple[float, float]]:
    """Read start and end timestamps (in seconds) from a .lab file."""
    segments = []
    with lab_path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            start, end = float(parts[0]), float(parts[1])
            if end > start:
                segments.append((start, end))
    return segments


def extract_one_shots(dataset_root: Path, seq_folder: str) -> None:
    audio_dir = dataset_root / "audio"
    lab_dir = dataset_root / "seq" / seq_folder
    out_dir = "extracted_one_shots/audio"

    if not audio_dir.exists():
        raise FileNotFoundError(f"Audio directory not found: {audio_dir}")
    if not lab_dir.exists():
        raise FileNotFoundError(f"Segmentation directory not found: {lab_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)

    audio_files = sorted(
        p for p in audio_dir.iterdir() if p.is_file() and p.suffix.lower() in AUDIO_EXTS
    )
    if not audio_files:
        raise FileNotFoundError(f"No supported audio files found in {audio_dir}")

    total_extracted = 0

    for audio_path in audio_files:
        lab_path = lab_dir / f"{audio_path.stem}.lab"
        if not lab_path.exists():
            print(f"[SKIP] No matching .lab file for {audio_path.name}")
            continue

        y, sr = sf.read(audio_path)
        segments = parse_lab_file(lab_path)
        count = 0

        for idx, (start, end) in enumerate(segments, start=1):
            s_idx = max(0, int(round(start * sr)))
            e_idx = min(len(y), int(round(end * sr)))
            seg_audio = y[s_idx:e_idx]

            if len(seg_audio) == 0:
                continue

            out_path = out_dir / f"{audio_path.stem}_{idx:04d}.wav"
            sf.write(out_path, seg_audio, sr)
            count += 1

        total_extracted += count
        print(f"[OK] {audio_path.name} -> extracted {count} segments")

    print(f"\nDone. Saved {total_extracted} one-shots to {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Segment audio files into one-shots based on .lab files."
    )
    parser.add_argument(
        "--dataset_root",
        type=Path,
        required=True,
        help="Path to dataset root containing 'audio/' and 'seq/'.",
    )
    parser.add_argument(
        "--seq_folder",
        type=str,
        default="manual_transcription",
        help="Name of the folder inside 'seq/' containing the .lab files (e.g., 'lab_seg').",
    )
    args = parser.parse_args()
    extract_one_shots(args.dataset_root, args.seq_folder)


if __name__ == "__main__":
    main()