import argparse
import json
import random
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import DataLoader, Dataset

from neuralresonator.dsp import biquad_freqz
from neuralresonator.models import CoefficientsFC
from neuralresonator.utilities import MelScaleLoss


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class TonbakControlDataset(Dataset):
    def __init__(self, data_dir: Path, split: str):
        self.data_dir = data_dir
        manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
        self.items = [it for it in manifest if it["split"] == split]
        if not self.items:
            raise ValueError(f"No samples found for split '{split}'")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        it = self.items[idx]
        y, _ = sf.read(self.data_dir / it["audio_path"], dtype="float32")
        controls = torch.tensor(it["controls"], dtype=torch.float32)
        return controls, torch.from_numpy(y)


@torch.no_grad()
def evaluate(model: CoefficientsFC, loader: DataLoader, criterion: MelScaleLoss, device: torch.device) -> float:
    model.eval()
    total = 0.0
    for controls, audio in loader:
        controls, audio = controls.to(device), audio.to(device)
        ba = model(controls)
        b, a = ba[..., :3], ba[..., 3:]
        p_mag = biquad_freqz(b, a, audio.shape[-1]).prod(dim=-2).sum(dim=-2).abs()
        t_mag = torch.fft.rfft(audio.float().clamp(-1, 1)).abs()
        total += criterion(p_mag, t_mag).item() * audio.size(0)
    return total / len(loader.dataset)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Direct Control-to-Resonator on real tonbak strokes.")
    parser.add_argument("--work_dir", type=Path, default=Path("synthesis/modal_resonator"))
    parser.add_argument("--run_name", type=str, default="tonbak_direct_32x2")
    parser.add_argument("--n_parallel", type=int, default=32)
    parser.add_argument("--n_biquads", type=int, default=2)
    parser.add_argument("--hidden_sizes", type=int, nargs="+", default=[512, 512, 512])
    parser.add_argument("--sample_rate", type=int, default=32000)
    parser.add_argument("--audio_length", type=float, default=0.2)
    parser.add_argument("--n_mels", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max_epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = args.work_dir / "data"
    run_dir = args.work_dir / "runs" / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    train_loader = DataLoader(TonbakControlDataset(data_dir, "train"), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(TonbakControlDataset(data_dir, "val"), batch_size=args.batch_size, shuffle=False)

    model = CoefficientsFC(
        input_size=5,
        n_parallel=args.n_parallel,
        n_biquads=args.n_biquads,
        use_zp_init=True,
        initial_pole_mag=1.5,
        initial_zero_mag=0.0,
        hidden_sizes=args.hidden_sizes,
    ).to(device)

    n_samples = int(round(args.audio_length * args.sample_rate))
    criterion = MelScaleLoss(
        n_mels=args.n_mels,
        sample_rate=args.sample_rate,
        n_fft=n_samples // 2 + 1,
        lin_l1=0.0,
        lin_l2=1.0,
        log_l1=0.0,
        log_l2=0.2,
        f_min=0.0,
        f_max=float(args.sample_rate // 2),
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)

    best_val = float("inf")
    no_improve = 0

    for epoch in range(1, args.max_epochs + 1):
        model.train()
        train_loss = 0.0
        for controls, audio in train_loader:
            controls, audio = controls.to(device), audio.to(device)
            optimizer.zero_grad()

            ba = model(controls)
            b, a = ba[..., :3], ba[..., 3:]
            p_mag = biquad_freqz(b, a, audio.shape[-1]).prod(dim=-2).sum(dim=-2).abs()
            t_mag = torch.fft.rfft(audio.float().clamp(-1, 1)).abs()

            loss = criterion(p_mag, t_mag)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * audio.size(0)

        train_loss /= len(train_loader.dataset)
        val_loss = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        print(f"Epoch {epoch:03d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | lr={optimizer.param_groups[0]['lr']:.2e}")

        if val_loss < best_val:
            best_val = val_loss
            no_improve = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "n_parallel": args.n_parallel,
                    "n_biquads": args.n_biquads,
                    "hidden_sizes": args.hidden_sizes,
                    "sample_rate": args.sample_rate,
                },
                run_dir / "best.pt",
            )
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"Early stopping at epoch {epoch} (best val_loss={best_val:.4f}).")
                break


if __name__ == "__main__":
    main()