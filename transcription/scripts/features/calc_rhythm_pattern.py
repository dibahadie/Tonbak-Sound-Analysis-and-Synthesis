import argparse
from pathlib import Path
import soundfile as sf

AUDIO_EXTS = {".wav", ".flac", ".ogg", ".aiff", ".aif"}


def find_audio_file(audio_dir: Path, stem: str) -> Path | None:
    """Find a matching audio file in audio_dir for a given .lab stem."""
    for ext in AUDIO_EXTS:
        candidate = audio_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def parse_lab_segments(lab_path: Path) -> list[tuple[float, float]]:
    """Read (start, end) timestamps from a .lab file."""
    segments = []
    with lab_path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            start, end = float(parts[0]), float(parts[1])
            segments.append((start, end))
    segments.sort(key=lambda x: x[0])
    return segments


def extract_rhythm(dataset_root: Path, seq_folder: str) -> None:
    audio_dir = dataset_root / "audio"
    in_dir = dataset_root / "seq" / seq_folder
    out_dir = dataset_root / "seq" / "lab_rhythm"

    if not in_dir.exists():
        raise FileNotFoundError(f"Input sequence folder not found: {in_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)
    lab_files = sorted(in_dir.glob("*.lab"))

    if not lab_files:
        raise FileNotFoundError(f"No .lab files found in {in_dir}")

    for lab_path in lab_files:
        segments = parse_lab_segments(lab_path)
        audio_path = find_audio_file(audio_dir, lab_path.stem)

        if audio_path is not None:
            track_length = sf.info(audio_path).duration
        elif segments:
            track_length = max(end for _, end in segments)
            print(f"[WARN] No audio file for {lab_path.name}; using last segment end ({track_length:.6f}s) as track_length.")
        else:
            print(f"[SKIP] {lab_path.name} is empty and has no matching audio file.")
            continue

        out_path = out_dir / lab_path.name
        with out_path.open("w", encoding="utf-8") as f:
            f.write("0 start\n")
            for start, _ in segments:
                f.write(f"{start:.6f} hit\n")
            f.write(f"{track_length:.6f} end\n")

        print(f"[OK] {lab_path.name} -> {out_path} ({len(segments)} hits, {track_length:.2f}s)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract rhythm onset timestamps from .lab files into seq/lab_rhythm."
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
        help="Name of the folder inside 'seq/' containing the input .lab files (e.g., 'lab_seg').",
    )
    args = parser.parse_args()
    extract_rhythm(args.dataset_root, args.seq_folder)


if __name__ == "__main__":
    main()