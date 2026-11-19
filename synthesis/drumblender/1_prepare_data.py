import argparse
import json
from pathlib import Path

from tqdm import tqdm

from utils.audio import preprocess_audio_file
from utils.data import str2int

AUDIO_EXTS = {".wav", ".flac", ".ogg", ".mp3", ".aiff", ".aif"}


def prepare_data(
    source_dir: Path,
    work_dir: Path,
    sample_rate: int,
    duration: float,
    silence_threshold_db: float,
    group_delimiter: str,
    sample_type: str,
) -> None:
    data_dir = work_dir / "data"
    audio_dir = data_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    num_samples = int(round(sample_rate * duration))
    source_files = sorted(
        p for p in source_dir.rglob("*") if p.is_file() and p.suffix.lower() in AUDIO_EXTS
    )
    if not source_files:
        raise FileNotFoundError(f"No audio files found in {source_dir}")

    metadata = {}
    skipped = 0

    for src in tqdm(source_files, desc="Preprocessing audio"):
        rel_str = str(src.relative_to(source_dir))
        file_id = str(str2int(rel_str))
        out_file = audio_dir / f"{file_id}.wav"

        try:
            preprocess_audio_file(
                input_file=src,
                output_file=out_file,
                sample_rate=sample_rate,
                num_samples=num_samples,
                mono=True,
                remove_start_silence=True,
                silence_threshold_db=silence_threshold_db,
            )
        except ValueError as e:
            skipped += 1
            print(f"[SKIP] {src.name}: {e}")
            continue

        # Group by recording prefix so hits from the same recording stay in one split
        stem = src.stem
        pack_key = stem.rsplit(group_delimiter, 1)[0] if group_delimiter in stem else stem

        metadata[file_id] = {
            "filename": str(out_file.relative_to(data_dir)),
            "original_name": rel_str,
            "type": sample_type,
            "sample_pack_key": pack_key,
        }

    meta_path = data_dir / "metadata.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Done. Saved {len(metadata)} files ({skipped} skipped) -> {meta_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess drum one-shots for DrumBlender.")
    parser.add_argument("--source_dir", type=Path, required=True, help="Source folder of drum one-shots.")
    parser.add_argument("--work_dir", type=Path, default=Path("synthesis/drumblender"))
    parser.add_argument("--sample_rate", type=int, default=48000)
    parser.add_argument("--duration", type=float, default=2.0)
    parser.add_argument("--silence_threshold_db", type=float, default=-60.0)
    parser.add_argument("--group_delimiter", type=str, default="_")
    parser.add_argument("--sample_type", type=str, default="acoustic")
    args = parser.parse_args()

    prepare_data(
        source_dir=args.source_dir,
        work_dir=args.work_dir,
        sample_rate=args.sample_rate,
        duration=args.duration,
        silence_threshold_db=args.silence_threshold_db,
        group_delimiter=args.group_delimiter,
        sample_type=args.sample_type,
    )


if __name__ == "__main__":
    main()