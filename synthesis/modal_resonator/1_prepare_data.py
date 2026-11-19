import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from tqdm import tqdm

AUDIO_EXTS = {".wav", ".flac", ".ogg", ".mp3", ".aiff", ".aif"}
FEATURE_NAMES = ["log_f0", "log_centroid", "flatness", "tail_ratio", "log_rms"]


def trim_leading_silence(y: np.ndarray, top_db: float = 40.0) -> np.ndarray:
    _, idx = librosa.effects.trim(y, top_db=top_db)
    return y[idx[0]:]


def fix_length(y: np.ndarray, target_len: int) -> np.ndarray:
    if len(y) >= target_len:
        return y[:target_len]
    return np.pad(y, (0, target_len - len(y)))


def extract_controls(y: np.ndarray, sr: int) -> dict[str, float]:
    # 1. Pitch (median log-F0 from YIN)
    f0 = librosa.yin(y, fmin=50.0, fmax=500.0, sr=sr, frame_length=1024, hop_length=256)
    f0_valid = f0[np.isfinite(f0) & (f0 > 0)]
    log_f0 = float(np.log(np.median(f0_valid))) if len(f0_valid) > 0 else float(np.log(150.0))

    # 2. Brightness (log spectral centroid)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=1024, hop_length=256)[0]
    log_centroid = float(np.log(np.mean(centroid) + 1e-6))

    # 3. Noisiness (spectral flatness)
    flatness = float(np.mean(librosa.feature.spectral_flatness(y=y, n_fft=1024, hop_length=256)))

    # 4. Decay (energy after 30 ms divided by total energy)
    attack_samples = min(int(0.03 * sr), len(y))
    total_energy = float(np.sum(y**2)) + 1e-10
    tail_energy = float(np.sum(y[attack_samples:] ** 2))
    tail_ratio = tail_energy / total_energy

    # 5. Dynamics (log RMS)
    log_rms = float(np.log(np.sqrt(np.mean(y**2)) + 1e-7))

    return {
        "log_f0": log_f0,
        "log_centroid": log_centroid,
        "flatness": flatness,
        "tail_ratio": tail_ratio,
        "log_rms": log_rms,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare real tonbak one-shots with continuous control features.")
    parser.add_argument("--source_dir", type=Path, required=True, help="Folder of tonbak one-shot audio files.")
    parser.add_argument("--work_dir", type=Path, default=Path("synthesis/modal_resonator"))
    parser.add_argument("--sample_rate", type=int, default=32000)
    parser.add_argument("--audio_length", type=float, default=0.2, help="Clip length in seconds (default: 0.2).")
    parser.add_argument("--top_db", type=float, default=40.0)
    parser.add_argument("--group_delimiter", type=str, default="_")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_dir = args.work_dir / "data"
    audio_dir = data_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    source_files = sorted(
        p for p in args.source_dir.rglob("*") if p.is_file() and p.suffix.lower() in AUDIO_EXTS
    )
    if not source_files:
        raise FileNotFoundError(f"No audio files found in {args.source_dir}")

    target_len = int(round(args.sample_rate * args.audio_length))
    groups: dict[str, list[dict]] = defaultdict(list)

    for idx, src in enumerate(tqdm(source_files, desc="Extracting controls")):
        y, _ = librosa.load(src, sr=args.sample_rate, mono=True)
        if len(y) == 0:
            continue

        y = trim_leading_silence(y, top_db=args.top_db)
        y = fix_length(y, target_len)
        controls = extract_controls(y, args.sample_rate)

        out_wav = audio_dir / f"{idx:05d}_{src.stem}.wav"
        sf.write(out_wav, np.clip(y, -1.0, 1.0), args.sample_rate, subtype="PCM_16")

        group_key = src.stem.rsplit(args.group_delimiter, 1)[0] if args.group_delimiter in src.stem else src.stem
        groups[group_key].append({
            "audio_path": str(out_wav.relative_to(data_dir)),
            "group": group_key,
            "raw_controls": controls,
        })

    # Leak-free 80 / 10 / 10 split by group
    group_keys = sorted(groups.keys())
    rng = random.Random(args.seed)
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

    items = []
    for gk, group_items in groups.items():
        for item in group_items:
            item["split"] = split_map[gk]
            items.append(item)

    # Fit min/max normalization on train split only
    train_items = [it for it in items if it["split"] == "train"]
    stats = {}
    for name in FEATURE_NAMES:
        vals = [it["raw_controls"][name] for it in train_items]
        lo, hi = float(np.min(vals)), float(np.max(vals))
        stats[name] = {"min": lo, "max": hi if hi > lo else lo + 1e-6}

    for it in items:
        it["controls"] = [
            float(np.clip((it["raw_controls"][name] - stats[name]["min"]) / (stats[name]["max"] - stats[name]["min"]), 0.0, 1.0))
            for name in FEATURE_NAMES
        ]

    (data_dir / "feature_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    (data_dir / "manifest.json").write_text(json.dumps(items, indent=2), encoding="utf-8")

    counts = {s: sum(1 for it in items if it["split"] == s) for s in ("train", "val", "test")}
    print(f"Prepared {len(items)} tonbak strokes -> {data_dir / 'manifest.json'}")
    print(f"Split counts: {counts}")


if __name__ == "__main__":
    main()