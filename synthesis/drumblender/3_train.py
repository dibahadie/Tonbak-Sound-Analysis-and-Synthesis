import argparse
from pathlib import Path

import auraloss
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from torch.utils.data import DataLoader

from drumblender.data.audio import AudioWithParametersDataset
from drumblender.models.encoders import ModalAmpParameters
from drumblender.models.soundstream import SoundStreamAttentionEncoder, SoundStreamEncoder
from drumblender.models.tcn import TCN
from drumblender.synths import ModalSynth, NoiseGenerator
from drumblender.tasks import DrumBlender
from drumblender.utils.metrics import LogSpectralDistance, SpectralFluxOnsetError


class DrumBlenderWithOptim(DrumBlender):
    """Adds the official Adam + ReduceLROnPlateau optimizer configuration."""
    def __init__(self, *args, lr: float = 1e-4, lr_patience: int = 20, **kwargs):
        super().__init__(*args, **kwargs)
        self.lr = lr
        self.lr_patience = lr_patience

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=self.lr_patience
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "validation/loss"},
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train official DrumBlender model on tonbak one-shots.")
    parser.add_argument("--work_dir", type=Path, default=Path("synthesis/drumblender"))
    parser.add_argument("--run_name", type=str, default="tonbak_ts_n")
    parser.add_argument("--sample_rate", type=int, default=48000)
    parser.add_argument("--duration", type=float, default=2.0)
    parser.add_argument("--num_modes", type=int, default=64)
    parser.add_argument("--split_strategy", type=str, default="sample_pack", choices=["sample_pack", "random"])
    parser.add_argument("--mixing", type=str, default="T(S)+N", choices=["T(S)+N", "T(S)+S+N", "T(S+N)", "T(S)"])
    parser.add_argument("--batch_size", type=int, default=12)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max_epochs", type=int, default=-1)
    parser.add_argument("--max_hours", type=int, default=48)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=396818285)
    args = parser.parse_args()

    pl.seed_everything(args.seed, workers=True)
    data_dir = args.work_dir / "data"
    num_samples = int(round(args.sample_rate * args.duration))

    ds_kwargs = dict(
        data_dir=data_dir,
        meta_file="metadata.json",
        sample_rate=args.sample_rate,
        num_samples=num_samples,
        parameter_key="feature_file",
        expected_num_modes=args.num_modes,
        split_strategy=args.split_strategy,
        seed=args.seed,
    )
    train_loader = DataLoader(
        AudioWithParametersDataset(split="train", **ds_kwargs),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        AudioWithParametersDataset(split="val", **ds_kwargs),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    test_loader = DataLoader(
        AudioWithParametersDataset(split="test", **ds_kwargs),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    use_noise = args.mixing != "T(S)"
    transient_parallel = args.mixing == "T(S)+S+N"
    transient_takes_noise = args.mixing == "T(S+N)"

    loss_fn = auraloss.freq.MultiResolutionSTFTLoss(
        fft_sizes=[1024, 2048, 512],
        hop_sizes=[120, 240, 50],
        win_lengths=[600, 1200, 240],
    )

    model = DrumBlenderWithOptim(
        modal_synth=ModalSynth(),
        modal_autoencoder=ModalAmpParameters(num_modes=args.num_modes, ones=True),
        loss_fn=loss_fn,
        noise_synth=NoiseGenerator(window_size=256) if use_noise else None,
        transient_synth=TCN(
            in_channels=1,
            hidden_channels=32,
            out_channels=1,
            dilation_base=2,
            num_layers=8,
            kernel_size=13,
            causal=True,
            activation="GELU",
            film_conditioning=True,
            film_embedding_size=128,
            film_batch_norm=True,
        ),
        noise_autoencoder=(
            SoundStreamEncoder(
                input_channels=1,
                hidden_channels=16,
                output_channels=128,
                kernel_size=7,
                strides=(2, 2, 4, 8),
                causal=False,
                transpose_output=True,
            )
            if use_noise
            else None
        ),
        transient_autoencoder=SoundStreamAttentionEncoder(
            input_channels=1,
            hidden_channels=16,
            output_channels=128,
            kernel_size=7,
            strides=(2, 2, 4, 8),
            causal=False,
            transpose_output=False,
        ),
        transient_parallel=transient_parallel,
        transient_takes_noise=transient_takes_noise,
        noise_autoencoder_accepts_audio=True,
        transient_autoencoder_accepts_audio=True,
        float32_matmul_precision="medium",
        test_metrics=torch.nn.ModuleDict({
            "lsd": LogSpectralDistance(),
            "sf": SpectralFluxOnsetError(),
        }),
        lr=args.lr,
        lr_patience=20,
    )

    run_dir = args.work_dir / "runs" / args.run_name
    callbacks = [
        ModelCheckpoint(dirpath=run_dir, monitor="validation/loss", mode="min", save_top_k=1, filename="best"),
        EarlyStopping(monitor="validation/loss", mode="min", patience=40, check_finite=True),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    trainer = pl.Trainer(
        default_root_dir=run_dir,
        max_epochs=args.max_epochs,
        max_time={"hours": args.max_hours},
        callbacks=callbacks,
        accelerator="auto",
        devices=1,
        deterministic=True,
    )
    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, test_loader, ckpt_path="best")


if __name__ == "__main__":
    main()