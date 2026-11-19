import argparse
import csv
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch


def extract_and_synthesize_sines(
    y: np.ndarray,
    sr: int,
    n_sines: int = 64,
    hop_length: int = 256,
    fmin: float = 20.0,
    bins_per_octave: int = 24,
    n_octaves: int = 10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract top-N sinusoidal tracks from CQT and synthesize S_n."""
    n_bins = bins_per_octave * n_octaves
    # Cap n_bins if fmax would exceed Nyquist
    max_octaves = np.log2((sr / 2.0) / fmin)
    n_bins = min(n_bins, int(np.floor(max_octaves * bins_per_octave)))

    cqt = librosa.cqt(
        y,
        sr=sr,
        hop_length=hop_length,
        fmin=fmin,
        n_bins=n_bins,
        bins_per_octave=bins_per_octave,
    )
    mag = np.abs(cqt)
    phase = np.angle(cqt)
    freqs_hz = librosa.cqt_frequencies(n_bins=n_bins, fmin=fmin, bins_per_octave=bins_per_octave)

    n_frames = mag.shape[1]
    amps = np.zeros((n_sines, n_frames), dtype=np.float32)
    freqs = np.zeros((n_sines, n_frames), dtype=np.float32)
    init_phases = np.zeros((n_sines,), dtype=np.float32)

    # Pick top-N local spectral peaks per frame
    for t in range(n_frames):
        col = mag[:, t]
        peaks = np.where((col[1:-1] > col[:-2]) & (col[1:-1] > col[2:]))[0] + 1
        if len(peaks) == 0:
            continue
        top_idx = peaks[np.argsort(col[peaks])[::-1][:n_sines]]
        k = len(top_idx)
        amps[:k, t] = col[top_idx]
        freqs[:k, t] = freqs_hz[top_idx]
        if t == 0:
            init_phases[:k] = phase[top_idx, 0]

    # Resynthesize S_n at sample rate via linear interpolation
    n_samples = len(y)
    frame_times = np.linspace(0, n_samples - 1, n_frames)
    sample_idx = np.arange(n_samples, dtype=np.float32)

    s_out = np.zeros(n_samples, dtype=np.float32)
    for j in range(n_sines):
        if np.max(amps[j]) == 0:
            continue
        a_n = np.interp(sample_idx, frame_times, amps[j])
        f_n = np.interp(sample_idx, frame_times, freqs[j])
        inst_phase = 2.0 * np.pi * np.cumsum(f_n) / sr + init_phases[j]
        s_out += a_n * np.sin(inst_phase)

    # Match RMS scale roughly to input to stabilize TCN input
    y_rms = np.sqrt(np.mean(y**2)) + 1e-8
    s_rms = np.sqrt(np.mean(s_out**2)) + 1e-8
    s_out = s_out * (y_rms / s_rms)

    return s_out.astype(np.float32), amps, freqs, init_phases


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract and cache CQT sinusoidal tracks.")
    parser.add_argument("--work_dir", type=Path, default=Path("synthesis/drumblender"), help="DrumBlender workspace root.")
    parser.add_argument("--n_sines", type=int, default=64, help="Number of sinusoids N_S (default: 64).")
    parser.add_argument("--hop_length", type=int, default=256, help="CQT hop size in samples (default: 256).")
    parser.add_argument("--fmin", type=float, default=20.0, help="Minimum CQT frequency in Hz (default: 20.0).")
    parser.add_argument("--bins_per_octave", type=int, default=24, help="CQT bins per octave (default: 24).")
    parser.add_argument("--n_octaves", type=int, default=10, help="CQT octave span (default: 10).")
    args = parser.parse_args()

    manifest_path = args.work_dir / "data" / "manifest.csv"
    sines_dir = args.work_dir / "data" / "sines"
    sines_dir.mkdir(parents=True, exist_ok=True)

    with manifest_path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    for idx, row in enumerate(rows, start=1):
        audio_path = args.work_dir / row["audio_path"]
        out_path = sines_dir / f"{row['id']}.pt"
        if out_path.exists():
            continue

        y, sr = sf.read(audio_path, dtype="float32")
        s_out, amps, freqs, init_phases = extract_and_synthesize_sines(
            y,
            sr=sr,
            n_sines=args.n_sines,
            hop_length=args.hop_length,
            fmin=args.fmin,
            bins_per_octave=args.bins_per_octave,
            n_octaves=args.n_octaves,
        )
        torch.save(
            {
                "sines_audio": torch.from_numpy(s_out),
                "amps": torch.from_numpy(amps),
                "freqs": torch.from_numpy(freqs),
                "init_phases": torch.from_numpy(init_phases),
            },
            out_path,
        )
        if idx % 25 == 0 or idx == len(rows):
            print(f"[{idx}/{len(rows)}] Extracted sines -> {out_path.name}")


if __name__ == "__main__":
    main()