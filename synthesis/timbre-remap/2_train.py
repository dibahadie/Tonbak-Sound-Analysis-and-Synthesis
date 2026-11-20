import argparse
from pathlib import Path

import lightning as L
import torch
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint

from features_util import (
    build_full_feature_extractor,
    build_mapper,
    build_onset_feature_extractor,
)
from timbreremap.data import OnsetFeatureDataModule
from timbreremap.loss import FeatureDifferenceLoss
from timbreremap.synth import Snare808
from timbreremap.tasks import TimbreRemappingTask


class TimbreRemappingTaskWithOptim(TimbreRemappingTask):
    """Adds Adam + ReduceLROnPlateau(factor=0.5, patience=20)."""

    def __init__(self, *args, lr: float = 5e-4, lr_patience: int = 20, **kwargs):
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
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val/loss"},
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Real-Time Timbre Remapping model.")
    parser.add_argument("--audio_path", type=Path, required=True, help="Directory of .wav files or a single performance .wav.")
    parser.add_argument("--preset", type=Path, default=Path("synthesis/timbre_remap/presets/tonbak_808.json"))
    parser.add_argument("--work_dir", type=Path, default=Path("synthesis/timbre_remap"))
    parser.add_argument("--run_name", type=str, default="tonbak_mlp_256")
    parser.add_argument("--arch", type=str, default="mlp", choices=["linear", "mlp", "mlp_lrg"])
    parser.add_argument("--onset_window", type=int, default=256, choices=[256, 2048])
    parser.add_argument("--sample_rate", type=int, default=44100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--max_epochs", type=int, default=250)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--test_split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=3608013887)
    args = parser.parse_args()

    L.seed_everything(args.seed, workers=True)

    full_feature = build_full_feature_extractor(args.sample_rate)
    onset_feature = build_onset_feature_extractor(args.sample_rate, window=args.onset_window)

    datamodule = OnsetFeatureDataModule(
        audio_path=args.audio_path,
        feature=full_feature,
        onset_feature=onset_feature,
        sample_rate=args.sample_rate,
        batch_size=args.batch_size,
        return_norm=True,
        val_split=args.val_split,
        test_split=args.test_split,
        data_seed=args.seed,
    )

    synth = Snare808(
        sample_rate=args.sample_rate,
        num_samples=args.sample_rate,
        buffer_noise=True,
        buffer_size=args.sample_rate,
    )
    mapper = build_mapper(args.arch, in_size=3, out_size=synth.get_num_params())

    task = TimbreRemappingTaskWithOptim(
        model=mapper,
        synth=synth,
        feature=full_feature,
        loss_fn=FeatureDifferenceLoss(),
        preset=str(args.preset),
        lr=args.lr,
        lr_patience=20,
    )

    run_dir = args.work_dir / "runs" / args.run_name
    callbacks = [
        ModelCheckpoint(dirpath=run_dir, monitor="val/loss", mode="min", save_top_k=1, filename="best"),
        LearningRateMonitor(logging_interval="step"),
    ]

    trainer = L.Trainer(
        default_root_dir=run_dir,
        max_epochs=args.max_epochs,
        callbacks=callbacks,
        accelerator="auto",
        devices=1,
    )
    trainer.fit(task, datamodule=datamodule)
    trainer.test(task, datamodule=datamodule, ckpt_path="best")


if __name__ == "__main__":
    main()