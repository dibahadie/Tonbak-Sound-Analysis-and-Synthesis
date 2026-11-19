import argparse
import csv
import random
import shutil
from collections import defaultdict
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

AUDIO_EXTS = {".wav", ".flac", ".ogg", ".mp3", ".aiff", ".aif"}


def trim_leading_silence(y: np.ndarray, top_db: float = 40.0) -> np.ndarray:
    """Remove leading silence while preserving the decay tail."""
    _, idx = librosa.effects.trim(y, top_db=top_db)
    return y[idx[0]:]


def fix_length(y: np.ndarray, target_len: int) -> np.ndarray:
    """Trim or zero-pad audio to exact target sample length."""
    if len(y) >= target_len:
        return y[:target_len]
    return np.pad(y, (0, target_len - len(y)))


def get_group_key(path: Path, delimiter: str) -> str:
    """Group files by recording prefix (e.g., 'rec01_0001.wav' -> 'rec01')."""
    return path.stem.rsplit(delimiter, 1)[0] if delimiter in path.stem else path.stem


def prepare_data(
    source_dir: Path,
    work_dir: Path,
    sr: int,
    duration: float,
    top_db: float,
    group_delimiter: str,
    seed: int,
) -> None:
    raw_dir = work_dir / "data" / "raw"
    proc_dir = work_dir / "data" / "processed"
    manifest_path = work_dir / "data" / "manifest.csv"

    raw_dir.mkdir(parents=True, exist_ok=True)
    proc_dir.mkdir(parents=True, exist_ok=True)

    source_files = sorted(
        p for p in source_dir.rglob("*") if p.is_file() and p.suffix.lower() in AUDIO_EXTS
    )
    if not source_files:
        raise FileNotFoundError(f"No audio files found in {source_dir}")

    target_len = int(round(sr * duration))
    groups: dict[str, list[dict]] = defaultdict(list)

    for src in source_files:
        rel_name = src.relative_to(source_dir).as_posix().replace("/", "_")
        raw_dst = raw_dir / rel_name
        shutil.copy2(src, raw_dst)

        y, _ = librosa.load(raw_dst, sr=sr, mono=True)
        if len(y) == 0:
            continue

        y = trim_leading_silence(y, top_db=top_db)
        peak = np.max(np.abs(y))
        if peak > 0:
            y = y / peak * 0.99
        y = fix_length(y, target_len)

        proc_dst = proc_dir / f"{Path(rel_name).stem}.wav"
        sf.write(proc_dst, y, sr, subtype="PCM_16")

        group_key = get_group_key(proc_dst, group_delimiter)
        groups[group_key].append({
            "id": proc_dst.stem,
            "audio_path": str(proc_dst.relative_to(work_dir)),
            "group": group_key,
        })

    # Leak-free 80 / 10 / 10 split by group
    group_keys = sorted(groups.keys())
    rng = random.Random(seed)
    rng.shuffle(group_keys)

    n_groups = len(group_keys)
    n_train = max(1, int(round(0.8 * n_groups)))
    n_val = max(1, int(round(0.1 * n_groups))) if n_groups >= 3 else 0

    split_map = {}
    for i, gk in enumerate(group_keys):
        if i < n_train:
            split_map[gk] = "train"
        elif i < n_train + n_val:
            split_map[gk] = "val"
        else:
            split_map[gk] = "test"

    rows = []
    for gk, items in groups.items():
        for item in items:
            item["split"] = split_map[gk]
            rows.append(item)

    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "audio_path", "group", "split"])
        writer.writeheader()
        writer.writerows(rows)

    counts = {s: sum(1 for r in rows if r["split"] == s) for s in ("train", "val", "test")}
    print(f"Prepared {len(rows)} files -> {manifest_path}")
    print(f"Split counts: {counts}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Copy, preprocess, and split drum one-shots.")
    parser.add_argument("--source_dir", type=Path, required=True, help="Source folder of drum audio files.")
    parser.add_argument("--work_dir", type=Path, default=Path("synthesis/drumblender"), help="DrumBlender workspace root.")
    parser.add_argument("--sr", type=int, default=48000, help="Target sample rate (default: 48000).")
    parser.add_argument("--duration", type=float, default=2.0, help="Fixed clip length in seconds (default: 2.0).")
    parser.add_argument("--top_db", type=float, default=40.0, help="Silence threshold in dB for leading trim.")
    parser.add_argument("--group_delimiter", type=str, default="_", help="Delimiter separating recording ID from segment ID.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible splits.")
    args = parser.parse_args()

    prepare_data(
        source_dir=args.source_dir,
        work_dir=args.work_dir,
        sr=args.sr,
        duration=args.duration,
        top_db=args.top_db,
        group_delimiter=args.group_delimiter,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()