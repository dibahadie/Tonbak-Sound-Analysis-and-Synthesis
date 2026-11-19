import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import DataLoader, Dataset

from model import DrumBlender, MultiResolutionSTFTLoss


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class DrumBlenderDataset(Dataset):
    def __init__(self, work_dir: Path, split: str):
        self.work_dir = work_dir
        manifest_path = work_dir / "data" / "manifest.csv"
        with manifest_path.open("r", encoding="utf-8") as f:
            self.items = [r for r in csv.DictReader(f) if r["split"] == split]
        if not self.items:
            raise ValueError(f"No samples found for split '{split}' in {manifest_path}")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.items[idx]
        y, _ = sf.read(self.work_dir / row["audio_path"], dtype="float32")
        sines_data = torch.load(self.work_dir / "data" / "sines" / f"{row['id']}.pt", weights_only=True)
        y_t = torch.from_numpy(y).unsqueeze(0)
        s_t = sines_data["sines_audio"].unsqueeze(0)
        return y_t, s_t


def evaluate(model: DrumBlender, loader: DataLoader, criterion: MultiResolutionSTFTLoss, device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for y, sines in loader:
            y, sines = y.to(device), sines.to(device)
            y_hat, _ = model(y, sines)
            total_loss += criterion(y_hat, y).item() * y.size(0)
    return total_loss / len(loader.dataset)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train DrumBlender on preprocessed tonbak one-shots.")
    parser.add_argument("--work_dir", type=Path, default=Path("synthesis/drumblender"))
    parser.add_argument("--run_name", type=str, default="tonbak_ts_n")
    parser.add_argument("--mixing", type=str, default="T(S)+N", choices=["T(S)+N", "T(S)+S+N", "T(S+N)", "T(S)"])
    parser.add_argument("--batch_size", type=int, default=12)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max_epochs", type=int, default=200)
    parser.add_argument("--lr_patience", type=int, default=20)
    parser.add_argument("--early_stop_patience", type=int, default=40)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_dir = args.work_dir / "runs" / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps({k: str(v) for k, v in vars(args).items()}, indent=2))

    train_ds = DrumBlenderDataset(args.work_dir, "train")
    val_ds = DrumBlenderDataset(args.work_dir, "val")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = DrumBlender(mixing=args.mixing).to(device)
    criterion = MultiResolutionSTFTLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=args.lr_patience
    )

    best_val = float("inf")
    no_improve = 0

    for epoch in range(1, args.max_epochs + 1):
        model.train()
        train_loss = 0.0
        for y, sines in train_loader:
            y, sines = y.to(device), sines.to(device)
            optimizer.zero_grad()
            y_hat, _ = model(y, sines)
            loss = criterion(y_hat, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * y.size(0)

        train_loss /= len(train_loader.dataset)
        val_loss = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        lr_now = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch:03d} | train_mss={train_loss:.4f} | val_mss={val_loss:.4f} | lr={lr_now:.2e}")

        if val_loss < best_val:
            best_val = val_loss
            no_improve = 0
            torch.save(
                {"epoch": epoch, "model_state_dict": model.state_dict(), "val_loss": val_loss, "mixing": args.mixing},
                run_dir / "best.pt",
            )
        else:
            no_improve += 1
            if no_improve >= args.early_stop_patience:
                print(f"Early stopping at epoch {epoch} (best val_mss={best_val:.4f}).")
                break


if __name__ == "__main__":
    main()