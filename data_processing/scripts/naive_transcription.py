import argparse
from pathlib import Path
import numpy as np
import librosa

AUDIO_EXTS = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}


def extract_feature(y: np.ndarray, sr: int, n_mfcc: int = 20) -> np.ndarray:
    """Extract a fixed-length feature vector (MFCC + delta + spectral stats) from an audio clip."""
    # Ensure minimum length for FFT
    min_len = 512
    if len(y) < min_len:
        y = np.pad(y, (0, min_len - len(y)))

    # Normalize amplitude so volume differences don't skew similarity
    peak = np.max(np.abs(y))
    if peak > 0:
        y = y / peak

    n_fft = min(1024, len(y))
    hop_length = max(128, n_fft // 4)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
    zcr = librosa.feature.zero_crossing_rate(y=y, frame_length=n_fft, hop_length=hop_length)

    feat = np.concatenate([
        np.mean(mfcc, axis=1),
        np.std(mfcc, axis=1),
        [np.mean(centroid), np.std(centroid), np.mean(zcr), np.std(zcr)],
    ])

    # L2 normalize for cosine similarity
    norm = np.linalg.norm(feat)
    return feat / norm if norm > 0 else feat


def load_oneshots(oneshots_dir: Path, sr: int) -> tuple[list[str], np.ndarray]:
    """
    Load one-shot audio files and extract their features.
    Supports both:
      - Flat layout: oneshots_dir/tom.wav -> label 'tom'
      - Subfolder layout: oneshots_dir/tom/tom_01.wav -> label 'tom'
    """
    labels = []
    features = []

    for path in sorted(oneshots_dir.rglob("*")):
        if path.suffix.lower() not in AUDIO_EXTS or not path.is_file():
            continue

        # Use parent folder name as label if nested, otherwise file stem
        label = path.parent.name if path.parent != oneshots_dir else path.stem
        y, _ = librosa.load(path, sr=sr, mono=True)
        if len(y) == 0:
            continue

        labels.append(label)
        features.append(extract_feature(y, sr))

    if not features:
        raise ValueError(f"No valid audio files found in one-shots directory: {oneshots_dir}")

    return labels, np.vstack(features)


def parse_lab_file(lab_path: Path) -> list[tuple[float, float]]:
    """Read start and end timestamps from a .lab file."""
    segments = []
    with lab_path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            start, end = float(parts[0]), float(parts[1])
            segments.append((start, end))
    return segments


def write_lab_file(out_path: Path, transcriptions: list[tuple[float, float, str]]) -> None:
    """Write (start, end, label) rows in standard tab-separated .lab format."""
    with out_path.open("w", encoding="utf-8") as f:
        for start, end, label in transcriptions:
            f.write(f"{start:.6f}\t{end:.6f}\t{label}\n")


def transcribe_dataset(dataset_root: Path, oneshots_dir: Path, sr: int = 22050) -> None:
    audio_dir = dataset_root / "audio"
    lab_seg_dir = dataset_root / "seq" / "lab_seg"
    out_dir = dataset_root / "seq" / "naive_transcription"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading one-shots from {oneshots_dir}...")
    oneshot_labels, oneshot_matrix = load_oneshots(oneshots_dir, sr=sr)
    print(f"Loaded {len(oneshot_labels)} one-shot templates across {len(set(oneshot_labels))} stroke classes.")

    audio_files = sorted(p for p in audio_dir.iterdir() if p.is_file() and p.suffix.lower() in AUDIO_EXTS)
    if not audio_files:
        raise FileNotFoundError(f"No audio files found in {audio_dir}")

    for audio_path in audio_files:
        lab_path = lab_seg_dir / f"{audio_path.stem}.lab"
        if not lab_path.exists():
            print(f"[SKIP] No matching .lab file found for {audio_path.name}")
            continue

        y, _ = librosa.load(audio_path, sr=sr, mono=True)
        segments = parse_lab_file(lab_path)
        results = []

        for start, end in segments:
            s_idx = max(0, int(round(start * sr)))
            e_idx = min(len(y), int(round(end * sr)))
            seg_audio = y[s_idx:e_idx]

            if len(seg_audio) == 0:
                best_label = "UNK"
            else:
                seg_feat = extract_feature(seg_audio, sr)
                # Cosine similarity (vectors are already L2-normalized)
                similarities = oneshot_matrix @ seg_feat
                best_label = oneshot_labels[int(np.argmax(similarities))]

            results.append((start, end, best_label))

        out_path = out_dir / f"{audio_path.stem}.lab"
        write_lab_file(out_path, results)
        print(f"[OK] {audio_path.name} -> {out_path} ({len(results)} segments)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Naive tonbak stroke transcription using one-shot template matching."
    )
    parser.add_argument(
        "--dataset_root",
        type=Path,
        required=True,
        help="Path to dataset root containing 'audio/' and 'seq/lab_seg/'.",
    )
    parser.add_argument(
        "--oneshots_dir",
        type=Path,
        required=True,
        help="Path to folder containing one-shot stroke audio files (flat or grouped in subfolders).",
    )
    parser.add_argument(
        "--sr",
        type=int,
        default=22050,
        help="Target sample rate for audio loading (default: 22050).",
    )
    args = parser.parse_args()
    transcribe_dataset(args.dataset_root, args.oneshots_dir, sr=args.sr)


if __name__ == "__main__":
    main()