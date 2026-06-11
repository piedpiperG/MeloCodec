import math
from collections.abc import Sequence

import torch
from torch import nn
from torch.nn.utils import weight_norm


def WNConv1d(*args, **kwargs):
    return weight_norm(nn.Conv1d(*args, **kwargs))


def WNConvTranspose1d(*args, **kwargs):
    return weight_norm(nn.ConvTranspose1d(*args, **kwargs))


class Snake1d(nn.Module):
    """Periodic activation used by DAC-style audio codecs."""

    def __init__(self, channels: int):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, channels, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + (self.alpha + 1e-9).reciprocal() * torch.sin(self.alpha * x).pow(2)


class ResidualUnit(nn.Module):
    def __init__(self, channels: int, dilation: int = 1):
        super().__init__()
        padding = ((7 - 1) * dilation) // 2
        self.block = nn.Sequential(
            Snake1d(channels),
            WNConv1d(channels, channels, kernel_size=7, dilation=dilation, padding=padding),
            Snake1d(channels),
            WNConv1d(channels, channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.block(x)
        if x.shape[-1] != y.shape[-1]:
            length = min(x.shape[-1], y.shape[-1])
            x = x[..., :length]
            y = y[..., :length]
        return x + y


class EncoderBlock(nn.Module):
    def __init__(self, channels: int, stride: int):
        super().__init__()
        in_channels = channels // 2
        self.block = nn.Sequential(
            ResidualUnit(in_channels, dilation=1),
            ResidualUnit(in_channels, dilation=3),
            ResidualUnit(in_channels, dilation=9),
            Snake1d(in_channels),
            WNConv1d(
                in_channels,
                channels,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Encoder(nn.Module):
    def __init__(
        self,
        channels: int = 64,
        rates: Sequence[int] = (3, 4, 7, 7),
        latent_dim: int | None = None,
    ):
        super().__init__()
        if latent_dim is None:
            latent_dim = channels * (2 ** len(rates))

        layers: list[nn.Module] = [WNConv1d(1, channels, kernel_size=7, padding=3)]
        current = channels
        for rate in rates:
            current *= 2
            layers.append(EncoderBlock(current, stride=int(rate)))
        layers.extend([Snake1d(current), WNConv1d(current, latent_dim, kernel_size=3, padding=1)])

        self.block = nn.Sequential(*layers)
        self.output_dim = latent_dim
        self.hop_length = math.prod(int(rate) for rate in rates)

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        return self.block(wav)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int):
        super().__init__()
        self.block = nn.Sequential(
            Snake1d(in_channels),
            WNConvTranspose1d(
                in_channels,
                out_channels,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
            ),
            ResidualUnit(out_channels, dilation=1),
            ResidualUnit(out_channels, dilation=3),
            ResidualUnit(out_channels, dilation=9),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Decoder(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        channels: int = 1536,
        rates: Sequence[int] = (7, 7, 4, 3),
        out_channels: int = 1,
    ):
        super().__init__()
        layers: list[nn.Module] = [WNConv1d(latent_dim, channels, kernel_size=7, padding=3)]
        current = channels
        for rate in rates:
            next_channels = current // 2
            layers.append(DecoderBlock(current, next_channels, stride=int(rate)))
            current = next_channels
        layers.extend([Snake1d(current), WNConv1d(current, out_channels, kernel_size=7, padding=3), nn.Tanh()])
        self.block = nn.Sequential(*layers)

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        return self.block(latents)


def init_weights(module: nn.Module) -> None:
    if isinstance(module, (nn.Conv1d, nn.ConvTranspose1d)):
        nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
