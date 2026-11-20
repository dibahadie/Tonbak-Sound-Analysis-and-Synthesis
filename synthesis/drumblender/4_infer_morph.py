import argparse
import json
from pathlib import Path

import torch
import torchaudio
from einops import rearrange

from models.encoders import ModalAmpParameters
from models.soundstream import SoundStreamAttentionEncoder, SoundStreamEncoder
from models.tcn import TCN
from synths.modal import modal_synth, ModalSynth
from synths.noise import NoiseGenerator
from tasks.drumblender import DrumBlender


def load_drumblender(ckpt_path: Path, mixing: str, num_modes: int, device: torch.device) -> DrumBlender:
    use_noise = mixing != "T(S)"
    transient_parallel = mixing == "T(S)+S+N"
    transient_takes_noise = mixing == "T(S+N)"

    model = DrumBlender(
        modal_synth=ModalSynth(),
        modal_autoencoder=ModalAmpParameters(num_modes=num_modes, ones=True),
        loss_fn=torch.nn.Identity(),  # unused at inference
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
    )

    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt.get("state_dict", ckpt)
    # Ignore metric buffers saved during training
    filtered = {k: v for k, v in state_dict.items() if not k.startswith("metrics.")}
    model.load_state_dict(filtered, strict=True)
    return model.to(device).eval()


def load_sample(data_dir: Path, sample_id: str, num_modes: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    with (data_dir / "metadata.json").open("r", encoding="utf-8") as f:
        metadata = json.load(f)

    if sample_id not in metadata:
        raise KeyError(f"Sample ID '{sample_id}' not found in {data_dir / 'metadata.json'}")

    item = metadata[sample_id]
    waveform, _ = torchaudio.load(data_dir / item["filename"])  # (1, L)
    modal = torch.load(data_dir / item["feature_file"], map_location="cpu")  # (3, M, T)

    if modal.shape[1] < num_modes:
        pad = torch.zeros((modal.shape[0], num_modes - modal.shape[1], modal.shape[2]))
        modal = torch.cat([modal, pad], dim=1)

    return waveform.unsqueeze(0).to(device), modal.unsqueeze(0).to(device)


@torch.no_grad()
def encode_stroke(model: DrumBlender, audio: torch.Tensor, modal: torch.Tensor) -> dict:
    modal_params, _ = model.modal_autoencoder(None, modal)
    noise_params = model.noise_autoencoder(audio) if model.noise_autoencoder is not None else None
    z = model.transient_autoencoder(audio)
    return {"modal": modal_params, "noise": noise_params, "z": z}


@torch.no_grad()
def synthesize_morph(
    model: DrumBlender,
    enc_a: dict,
    enc_b: dict,
    alpha: float,
    num_samples: int,
    pitch_semitones: float = 0.0,
) -> torch.Tensor:
    # Pitch-scale angular frequencies (channel 0 of modal params)
    pitch_ratio = 2.0 ** (pitch_semitones / 12.0)
    modal_a = enc_a["modal"].clone()
    modal_b = enc_b["modal"].clone()
    modal_a[:, 0] *= pitch_ratio
    modal_b[:, 0] *= pitch_ratio

    # Crossfade synthesized sinusoids cleanly (avoids phase/frequency collisions between unrelated modes)
    s_a = model.modal_synth(modal_a, num_samples)
    s_b = model.modal_synth(modal_b, num_samples)
    y_hat = (1.0 - alpha) * s_a + alpha * s_b

    # Interpolate noise filter envelopes
    noise = None
    if model.noise_synth is not None:
        noise_params = (1.0 - alpha) * enc_a["noise"] + alpha * enc_b["noise"]
        noise = model.noise_synth(noise_params, num_samples)
        noise = rearrange(noise, "b n -> b () n")
        if model.transient_takes_noise:
            y_hat = y_hat + noise

    # Interpolate transient embedding z
    z = (1.0 - alpha) * enc_a["z"] + alpha * enc_b["z"]
    transient = model.transient_synth(y_hat, z)
    y_hat = (y_hat + transient) if model.transient_parallel else transient

    if noise is not None and not model.transient_takes_noise:
        y_hat = y_hat + noise

    return y_hat.squeeze(0).cpu()  # (1, L)


def main() -> None:
    parser = argparse.ArgumentParser(description="Morph between two tonbak strokes with trained DrumBlender.")
    parser.add_argument("--work_dir", type=Path, default=Path("."))
    parser.add_argument("--ckpt", type=Path, required=True, help="Path to trained .ckpt file.")
    parser.add_argument("--id_a", type=str, required=True, help="Metadata ID for Stroke A (e.g., tom).")
    parser.add_argument("--id_b", type=str, required=True, help="Metadata ID for Stroke B (e.g., bak).")
    parser.add_argument("--out_dir", type=Path, default=Path("synthesis/drumblender/morph_out"))
    parser.add_argument("--mixing", type=str, default="T(S)+N", choices=["T(S)+N", "T(S)+S+N", "T(S+N)", "T(S)"])
    parser.add_argument("--steps", type=int, default=5, help="Number of morph steps from 0.0 to 1.0.")
    parser.add_argument("--pitch_semitones", type=float, default=0.0, help="Optional pitch shift in semitones.")
    parser.add_argument("--sample_rate", type=int, default=48000)
    parser.add_argument("--duration", type=float, default=2.0)
    parser.add_argument("--num_modes", type=int, default=64)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = args.work_dir / "data"
    num_samples = int(round(args.sample_rate * args.duration))
    args.out_dir.mkdir(parents=True, exist_ok=True)

    model = load_drumblender(args.ckpt, args.mixing, args.num_modes, device)
    audio_a, modal_a = load_sample(data_dir, args.id_a, args.num_modes, device)
    audio_b, modal_b = load_sample(data_dir, args.id_b, args.num_modes, device)

    enc_a = encode_stroke(model, audio_a, modal_a)
    enc_b = encode_stroke(model, audio_b, modal_b)

    alphas = torch.linspace(0.0, 1.0, args.steps).tolist()
    for alpha in alphas:
        y_morph = synthesize_morph(model, enc_a, enc_b, alpha, num_samples, args.pitch_semitones)
        out_path = args.out_dir / f"morph_{args.id_a}_to_{args.id_b}_a{alpha:.2f}.wav"
        torchaudio.save(out_path, y_morph, args.sample_rate)
        print(f"[OK] Saved {out_path}")


if __name__ == "__main__":
    main()