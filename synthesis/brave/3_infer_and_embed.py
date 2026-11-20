import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description="Run BRAVE TorchScript inference and extract 128D latent embeddings.")
    parser.add_argument("--ts_model", type=Path, required=True, help="Exported streaming .ts model (from `rave export --streaming`).")
    parser.add_argument("--input_dir", type=Path, required=True, help="Directory of input .wav files.")
    parser.add_argument("--out_dir", type=Path, default=Path("synthesis/brave/infer_out"))
    parser.add_argument("--sample_rate", type=int, default=44100)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.jit.load(str(args.ts_model), map_location=device).eval()

    wav_files = sorted(args.input_dir.glob("*.wav"))
    assert len(wav_files) > 0, f"No .wav files found in {args.input_dir}"

    audio_out = args.out_dir / "audio"
    audio_out.mkdir(parents=True, exist_ok=True)

    embeddings = []
    paths = []

    for f in wav_files:
        x, sr = torchaudio.load(f)
        if x.shape[0] > 1:
            x = x.mean(dim=0, keepdim=True)
        if sr != args.sample_rate:
            x = torchaudio.transforms.Resample(sr, args.sample_rate)(x)

        # Pad to multiple of compression ratio C_r = 128
        rem = x.shape[-1] % 128
        if rem != 0:
            x = torch.nn.functional.pad(x, (0, 128 - rem))

        x_in = x.unsqueeze(0).to(device)  # (1, 1, T)
        z = model.encode(x_in)            # (1, 128, T // 128)
        y = model.decode(z)               # (1, 1, T)

        sf.write(audio_out / f.name, y.squeeze().cpu().numpy(), args.sample_rate)

        # Temporal mean + std over latent trajectory for stroke clustering
        z_np = z.squeeze(0).cpu().numpy()
        emb = np.concatenate([z_np.mean(axis=-1), z_np.std(axis=-1)], axis=0)
        embeddings.append(emb)
        paths.append(str(f))

    np.save(args.out_dir / "embeddings.npy", np.stack(embeddings, axis=0))
    (args.out_dir / "paths.json").write_text(json.dumps(paths, indent=2), encoding="utf-8")
    print(f"[OK] Processed {len(wav_files)} files -> {args.out_dir}")


if __name__ == "__main__":
    main()