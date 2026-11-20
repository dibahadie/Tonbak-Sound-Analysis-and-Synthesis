# synthesis/brave/concat_hits.py
import argparse
from pathlib import Path
import numpy as np
import soundfile as sf
import torchaudio


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, default=Path("./concat_wavs"))
    parser.add_argument("--sr", type=int, default=44100)
    parser.add_argument("--gap_sec", type=float, default=0.25)
    parser.add_argument("--chunk_sec", type=float, default=60.0)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    gap = np.zeros(int(args.gap_sec * args.sr), dtype=np.float32)
    target_len = int(args.chunk_sec * args.sr)

    buf, buf_len, file_idx = [], 0, 0
    for f in sorted(args.input_dir.glob("*.wav")):
        x, sr = torchaudio.load(f)
        if x.shape[0] > 1:
            x = x.mean(dim=0, keepdim=True)
        if sr != args.sr:
            x = torchaudio.transforms.Resample(sr, args.sr)(x)
        arr = x.squeeze(0).numpy().astype(np.float32)

        buf.extend([arr, gap])
        buf_len += len(arr) + len(gap)

        if buf_len >= target_len:
            sf.write(args.out_dir / f"track_{file_idx:04d}.wav", np.concatenate(buf), args.sr)
            buf, buf_len, file_idx = [], 0, file_idx + 1

    if buf_len >= 65536:
        sf.write(args.out_dir / f"track_{file_idx:04d}.wav", np.concatenate(buf), args.sr)
        file_idx += 1

    print(f"[OK] Wrote {file_idx} concatenated tracks -> {args.out_dir}")


if __name__ == "__main__":
    main()