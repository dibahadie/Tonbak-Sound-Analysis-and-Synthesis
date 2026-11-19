import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from neuralresonator.dsp import apply_filter
from neuralresonator.models import CoefficientsFC


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description="Synthesize tonbak stroke from 5 continuous control knobs.")
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument(
        "--controls",
        type=float,
        nargs=5,
        default=[0.2, 0.2, 0.2, 0.7, 0.8],
        help="5 normalized knobs in [0, 1]: pitch brightness noisiness decay velocity.",
    )
    parser.add_argument("--out_wav", type=Path, default=Path("synthesis/modal_resonator/tonbak_out.wav"))
    parser.add_argument("--duration", type=float, default=0.5)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=True)

    model = CoefficientsFC(
        input_size=5,
        n_parallel=ckpt["n_parallel"],
        n_biquads=ckpt["n_biquads"],
        use_zp_init=True,
        initial_pole_mag=1.5,
        initial_zero_mag=0.0,
        hidden_sizes=ckpt["hidden_sizes"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    c = torch.tensor([args.controls], dtype=torch.float32, device=device)
    ba = model(c)
    b = ba[0, ..., :3].cpu().numpy().astype(np.float64)
    a = ba[0, ..., 3:].cpu().numpy().astype(np.float64)

    sr = ckpt["sample_rate"]
    impulse = np.zeros(int(round(sr * args.duration)), dtype=np.float64)
    impulse[0] = 1.0

    y = apply_filter(impulse, a=a, b=b)
    peak = np.max(np.abs(y))
    if peak > 0:
        y = y / peak * 0.95

    args.out_wav.parent.mkdir(parents=True, exist_ok=True)
    sf.write(args.out_wav, y.astype(np.float32), sr)
    print(f"[OK] Rendered -> {args.out_wav}")


if __name__ == "__main__":
    main()