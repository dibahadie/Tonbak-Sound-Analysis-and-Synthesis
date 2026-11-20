import torch
from timbreremap.feature import (
    CascadingFrameExtactor,
    FeatureCollection,
    Loudness,
    RMS,
    SpectralCentroid,
    SpectralFlatness,
    TemporalCentroid,
)
from timbreremap.model import LinearMapping, MLP


def build_full_feature_extractor(sample_rate: int = 44100) -> torch.nn.Module:
    """7D feature space matching onset_feature_full.yaml."""
    cascading = CascadingFrameExtactor(
        extractors=[
            Loudness(sample_rate=sample_rate),
            SpectralCentroid(
                sample_rate=sample_rate,
                window="flat_top",
                compress=True,
                floor=1e-4,
                scaling="kazazis",
            ),
            SpectralFlatness(),
        ],
        num_frames=[2, 64],
        frame_size=2048,
        hop_size=512,
    )

    tc_frame_size = int(round(0.125 * sample_rate))  # 5512 at 44100 Hz
    tc = CascadingFrameExtactor(
        extractors=[
            TemporalCentroid(sample_rate=sample_rate, scaling="schlauch"),
        ],
        num_frames=[1],
        frame_size=tc_frame_size,
        hop_size=tc_frame_size,
    )

    return FeatureCollection([cascading, tc])


def build_onset_feature_extractor(sample_rate: int = 44100, window: int = 256) -> torch.nn.Module:
    """3D short-term onset features matching onset_features.yaml."""
    return CascadingFrameExtactor(
        extractors=[
            RMS(db=True),
            SpectralCentroid(sample_rate=sample_rate),
            SpectralFlatness(),
        ],
        num_frames=[1],
        frame_size=window,
        hop_size=window,
    )


def build_mapper(arch: str, in_size: int = 3, out_size: int = 14) -> torch.nn.Module:
    """Mapping networks matching linear_mapper.yaml, mlp_mapper.yaml, and mlp_mapper_lrg.yaml."""
    if arch == "linear":
        return LinearMapping(
            in_size=in_size,
            out_size=out_size,
            bias=False,
            input_bias=0.0,
            clamp=True,
            init_std=1e-6,
        )
    if arch == "mlp":
        return MLP(
            in_size=in_size,
            hidden_size=32,
            out_size=out_size,
            num_layers=1,
            activation=torch.nn.ReLU(),
            input_bias=0.0,
            layer_norm=True,
            init_std=0.001,
            scale_output=True,
        )
    if arch == "mlp_lrg":
        return MLP(
            in_size=in_size,
            hidden_size=64,
            out_size=out_size,
            num_layers=3,
            activation=torch.nn.ReLU(),
            input_bias=0.0,
            layer_norm=True,
            init_std=0.001,
            scale_output=True,
        )
    raise ValueError(f"Unknown architecture: {arch}")