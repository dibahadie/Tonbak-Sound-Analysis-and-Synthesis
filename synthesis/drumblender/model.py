import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualUnit(nn.Module):
    def __init__(self, channels: int, dilation: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.ELU(),
            nn.Conv1d(channels, channels, kernel_size=7, padding=3 * dilation, dilation=dilation),
            nn.ELU(),
            nn.Conv1d(channels, channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class SoundStreamEncoder(nn.Module):
    """Non-causal 1D conv encoder with tripling dilation residual units and strided downsampling."""
    def __init__(self, out_channels: int, strides: tuple[int, ...] = (2, 4, 4, 4), base_channels: int = 32):
        super().__init__()
        layers = [nn.Conv1d(1, base_channels, kernel_size=7, padding=3)]
        ch = base_channels
        for s in strides:
            layers.extend([
                ResidualUnit(ch, dilation=1),
                ResidualUnit(ch, dilation=3),
                ResidualUnit(ch, dilation=9),
                nn.ELU(),
                nn.Conv1d(ch, ch * 2, kernel_size=2 * s, stride=s, padding=s // 2),
            ])
            ch *= 2
        layers.extend([nn.ELU(), nn.Conv1d(ch, out_channels, kernel_size=3, padding=1)])
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AttentionPooling(nn.Module):
    """Aggregates frame-wise transient features into a single latent vector z."""
    def __init__(self, dim: int):
        super().__init__()
        self.score = nn.Linear(dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, D, T) -> (B, T, D)
        xt = x.transpose(1, 2)
        weights = F.softmax(self.score(xt), dim=1)
        return torch.sum(xt * weights, dim=1)  # (B, D)


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
        mags = F.softplus(mags)  # ensure non-negative filter gains

        # Construct real-valued IR via irfft
        ir = torch.fft.irfft(mags.transpose(1, 2), n=self.win_len, dim=-1)  # (B, n_frames, win_len)
        ir = torch.roll(ir, shifts=self.win_len // 2, dims=-1) * self.window

        noise = torch.rand(B, n_frames, self.win_len, device=mags.device) * 2.0 - 1.0
        # Frame-wise circular convolution via FFT
        N_fft = torch.fft.rfft(noise, n=self.win_len * 2, dim=-1)
        H_fft = torch.fft.rfft(ir, n=self.win_len * 2, dim=-1)
        filtered = torch.fft.irfft(N_fft * H_fft, n=self.win_len * 2, dim=-1)[..., : self.win_len]
        filtered = filtered * self.window

        # Overlap-add via fold
        out_len = (n_frames - 1) * self.hop_size + self.win_len
        folded = F.fold(
            filtered.transpose(1, 2),
            output_size=(1, out_len),
            kernel_size=(1, self.win_len),
            stride=(1, self.hop_size),
        ).squeeze(2)  # (B, 1, out_len)

        if out_len >= target_len:
            return folded[..., :target_len]
        return F.pad(folded, (0, target_len - out_len))


class FiLMTCNBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dilation: int, z_dim: int):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=dilation, dilation=dilation)
        self.film = nn.Sequential(nn.Linear(z_dim, out_ch * 2), nn.PReLU(), nn.Linear(out_ch * 2, out_ch * 2))
        self.act = nn.PReLU()
        self.res = nn.Conv1d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        h = self.conv(x)
        gamma, beta = self.film(z).chunk(2, dim=-1)
        h = h * (1.0 + gamma.unsqueeze(-1)) + beta.unsqueeze(-1)
        return self.act(h) + self.res(x)


class FiLMTCN(nn.Module):
    """8-block FiLM-modulated TCN for transient reconstruction."""
    def __init__(self, n_blocks: int = 8, channels: int = 32, z_dim: int = 128):
        super().__init__()
        self.blocks = nn.ModuleList([
            FiLMTCNBlock(1 if i == 0 else channels, channels, dilation=2**i, z_dim=z_dim)
            for i in range(n_blocks)
        ])
        self.out_conv = nn.Conv1d(channels, 1, kernel_size=1)

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        h = x
        for block in self.blocks:
            h = block(h, z)
        return self.out_conv(h)


class DrumBlender(nn.Module):
    def __init__(
        self,
        n_noise_bands: int = 128,
        noise_hop: int = 128,
        z_dim: int = 128,
        tcn_blocks: int = 8,
        tcn_channels: int = 32,
        mixing: str = "T(S)+N",
    ):
        super().__init__()
        self.mixing = mixing
        # Strides product = 2*4*4*4 = 128 -> matches noise_hop=128
        self.noise_encoder = SoundStreamEncoder(out_channels=n_noise_bands, strides=(2, 4, 4, 4))
        self.noise_gen = FilteredNoiseGenerator(n_bands=n_noise_bands, hop_size=noise_hop)

        self.transient_encoder = SoundStreamEncoder(out_channels=z_dim, strides=(2, 4, 4, 4))
        self.attn_pool = AttentionPooling(z_dim)
        self.tcn = FiLMTCN(n_blocks=tcn_blocks, channels=tcn_channels, z_dim=z_dim)

    def forward(self, y: torch.Tensor, sines: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # y, sines: (B, 1, L)
        L = y.shape[-1]
        noise_mags = self.noise_encoder(y)
        noise = self.noise_gen(noise_mags, target_len=L)

        z = self.attn_pool(self.transient_encoder(y))

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