import torch
import torch.nn as nn
import torch.nn.functional as F

from models.soundstream import SoundStreamAttentionEncoder, SoundStreamEncoder
from models.tcn import TCN


class FilteredNoiseGenerator(nn.Module):
    """Linear-band filtered noise generator with overlap-add Hanning windows."""
    def __init__(self, n_bands: int = 128, hop_size: int = 128):
        super().__init__()
        self.n_bands = n_bands
        self.hop_size = hop_size
        self.win_len = 2 * hop_size
        self.register_buffer("window", torch.hann_window(self.win_len))

    def forward(self, mags: torch.Tensor, target_len: int) -> torch.Tensor:
        # mags: (B, n_bands, n_frames)
        B, _, n_frames = mags.shape
        mags = F.softplus(mags)

        # Pad 1 bin for Nyquist so irfft has n_bands + 1 = 129 bins -> 256-sample IR
        mags_padded = F.pad(mags.transpose(1, 2), (0, 1))
        ir = torch.fft.irfft(mags_padded, n=self.win_len, dim=-1)
        ir = torch.roll(ir, shifts=self.win_len // 2, dims=-1) * self.window

        noise = torch.rand(B, n_frames, self.win_len, device=mags.device) * 2.0 - 1.0
        N_fft = torch.fft.rfft(noise, n=self.win_len * 2, dim=-1)
        H_fft = torch.fft.rfft(ir, n=self.win_len * 2, dim=-1)
        filtered = torch.fft.irfft(N_fft * H_fft, n=self.win_len * 2, dim=-1)[..., : self.win_len]
        filtered = filtered * self.window

        out_len = (n_frames - 1) * self.hop_size + self.win_len
        folded = F.fold(
            filtered.transpose(1, 2),
            output_size=(1, out_len),
            kernel_size=(1, self.win_len),
            stride=(1, self.hop_size),
        ).squeeze(2)

        if out_len >= target_len:
            return folded[..., :target_len]
        return F.pad(folded, (0, target_len - out_len))


class DrumBlender(nn.Module):
    def __init__(
        self,
        n_noise_bands: int = 128,
        noise_hop: int = 128,
        z_dim: int = 128,
        encoder_hidden: int = 16,
        tcn_layers: int = 8,
        tcn_channels: int = 32,
        tcn_kernel_size: int = 13,
        mixing: str = "T(S)+N",
    ):
        super().__init__()
        self.mixing = mixing

        # Strides product = 2*4*4*4 = 128 to match noise_hop=128
        self.noise_encoder = SoundStreamEncoder(
            input_channels=1,
            hidden_channels=encoder_hidden,
            output_channels=n_noise_bands,
            kernel_size=7,
            strides=(2, 4, 4, 4),
            causal=False,
        )
        self.noise_gen = FilteredNoiseGenerator(n_bands=n_noise_bands, hop_size=noise_hop)

        self.transient_encoder = SoundStreamAttentionEncoder(
            input_channels=1,
            hidden_channels=encoder_hidden,
            output_channels=z_dim,
            kernel_size=7,
            strides=(2, 4, 4, 4),
            causal=False,
        )

        # Matches the official FiLM-TCN diagram exactly
        self.tcn = TCN(
            in_channels=1,
            hidden_channels=tcn_channels,
            out_channels=1,
            dilation_base=2,
            num_layers=tcn_layers,
            kernel_size=tcn_kernel_size,
            causal=True,
            activation="GELU",
            film_conditioning=True,
            film_embedding_size=z_dim,
            film_batch_norm=True,
        )

    def forward(self, y: torch.Tensor, sines: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        L = y.shape[-1]
        noise_mags = self.noise_encoder(y)
        noise = self.noise_gen(noise_mags, target_len=L)

        z = self.transient_encoder(y)  # (B, 128)

        if self.mixing == "T(S)+N":
            y_hat = self.tcn(sines, z) + noise
        elif self.mixing == "T(S)+S+N":
            y_hat = self.tcn(sines, z) + sines + noise
        elif self.mixing == "T(S+N)":
            y_hat = self.tcn(sines + noise, z)
        elif self.mixing == "T(S)":
            y_hat = self.tcn(sines, z)
        else:
            raise ValueError(f"Unknown mixing strategy: {self.mixing}")

        return y_hat, z


class MultiResolutionSTFTLoss(nn.Module):
    """Weighted sum of spectral convergence and log-magnitude spectral difference."""
    def __init__(self, fft_sizes=(1024, 2048, 512), hop_sizes=(120, 240, 50), win_lengths=(600, 1200, 240)):
        super().__init__()
        self.configs = list(zip(fft_sizes, hop_sizes, win_lengths))

    def forward(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        y_hat = y_hat.squeeze(1)
        y = y.squeeze(1)
        total_loss = 0.0
        for n_fft, hop, win in self.configs:
            window = torch.hann_window(win, device=y.device)
            S_hat = torch.abs(torch.stft(y_hat, n_fft, hop, win, window=window, return_complex=True)) + 1e-7
            S_true = torch.abs(torch.stft(y, n_fft, hop, win, window=window, return_complex=True)) + 1e-7
            sc_loss = torch.norm(S_true - S_hat, p="fro") / (torch.norm(S_true, p="fro") + 1e-7)
            mag_loss = F.l1_loss(torch.log(S_true), torch.log(S_hat))
            total_loss = total_loss + sc_loss + mag_loss
        return total_loss / len(self.configs)