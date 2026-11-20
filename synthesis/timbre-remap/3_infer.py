import argparse
from pathlib import Path

import soundfile as sf
import torch

from features_util import build_full_feature_extractor, build_mapper
from timbreremap.loss import FeatureDifferenceLoss
from timbreremap.synth import Snare808
from timbreremap.tasks import TimbreRemappingTask


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description="Synthesize timbre-remapped audio from 3D onset controls.")
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--preset", type=Path, default=Path("synthesis/timbre_remap/presets/tonbak_808.json"))
    parser.add_argument("--arch", type=str, default="mlp", choices=["linear", "mlp", "mlp_lrg"])
    parser.add_argument("--controls", type=float, nargs=3, default=[0.5, 0.5, 0.5], help="Normalized [rms, centroid, flatness] in [0, 1].")
    parser.add_argument("--sample_rate", type=int, default=44100)
    parser.add_argument("--out_wav", type=Path, default=Path("synthesis/timbre_remap/remapped_out.wav"))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    synth = Snare808(sample_rate=args.sample_rate, num_samples=args.sample_rate, buffer_noise=False)
    mapper = build_mapper(args.arch, in_size=3, out_size=synth.get_num_params())
    task = TimbreRemappingTask(
        model=mapper,
        synth=synth,
        feature=build_full_feature_extractor(args.sample_rate),
        loss_fn=FeatureDifferenceLoss(),
        preset=str(args.preset),
    )

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    task.load_state_dict(ckpt["state_dict"], strict=False)
    task = task.to(device).eval()

    x_ctrl = torch.tensor([args.controls], dtype=torch.float32, device=device)
    y = task(x_ctrl).squeeze(0).cpu().numpy()

    args.out_wav.parent.mkdir(parents=True, exist_ok=True)
    sf.write(args.out_wav, y, args.sample_rate)
    print(f"[OK] Saved remapped audio -> {args.out_wav}")


if __name__ == "__main__":
    main()