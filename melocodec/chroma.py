from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn

from .layers import ResidualUnit, Snake1d, init_weights
from .quantize import ResidualVectorQuantize


def _make_chroma_filterbank(
    sample_rate: int,
    n_fft: int,
    n_chroma: int,
    fmin: float = 32.7,
    fmax: float = 4186.0,
) -> torch.Tensor:
    freqs = torch.linspace(0, sample_rate / 2, n_fft // 2 + 1)
    fb = torch.zeros(n_chroma, freqs.numel())
    valid = (freqs >= fmin) & (freqs <= fmax)
    valid_freqs = freqs[valid].clamp_min(1e-6)
    midi = 69.0 + 12.0 * torch.log2(valid_freqs / 440.0)
    classes = torch.remainder(torch.round(midi).long(), n_chroma)
    fb[classes, valid.nonzero(as_tuple=True)[0]] = 1.0
    return fb / fb.sum(dim=1, keepdim=True).clamp_min(1.0)


class ChromaEncoder(nn.Module):
    def __init__(
        self,
        chroma_dim: int = 12,
        hidden_dims: Sequence[int] = (64, 128, 256),
        strides: Sequence[int] = (1, 2, 2),
    ):
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = chroma_dim
        for index, (hidden_dim, stride) in enumerate(zip(hidden_dims, strides, strict=True)):
            layers.extend(
                [
                    nn.Conv1d(in_dim, hidden_dim, kernel_size=3, stride=int(stride), padding=1),
                    nn.BatchNorm1d(hidden_dim),
                    Snake1d(hidden_dim),
                ]
            )
            if index > 0:
                layers.extend([ResidualUnit(hidden_dim, dilation=1), ResidualUnit(hidden_dim, dilation=3)])
            in_dim = hidden_dim
        self.block = nn.Sequential(*layers)
        self.output_dim = int(hidden_dims[-1])

    def forward(self, chroma: torch.Tensor) -> torch.Tensor:
        return self.block(chroma)


class ChromaDecoder(nn.Module):
    def __init__(
        self,
        input_dim: int = 256,
        hidden_dims: Sequence[int] = (256, 128, 64),
        output_dim: int = 12,
        strides: Sequence[int] = (2, 2, 1),
    ):
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = input_dim
        for index, (hidden_dim, stride) in enumerate(zip(hidden_dims, strides, strict=True)):
            if index < len(hidden_dims) - 1:
                layers.extend([ResidualUnit(in_dim, dilation=3), ResidualUnit(in_dim, dilation=1)])
            layers.append(
                nn.ConvTranspose1d(
                    in_dim,
                    hidden_dim,
                    kernel_size=3,
                    stride=int(stride),
                    padding=1,
                    output_padding=max(int(stride) - 1, 0),
                )
            )
            if index < len(hidden_dims) - 1:
                layers.extend([nn.BatchNorm1d(hidden_dim), Snake1d(hidden_dim)])
            in_dim = hidden_dim
        layers.extend([nn.Conv1d(in_dim, output_dim, kernel_size=3, padding=1), nn.Sigmoid()])
        self.block = nn.Sequential(*layers)

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        return self.block(latents)


class ChromaCodec(nn.Module):
    """Discrete chromagram codec used as MeloCodec's melody tokenizer."""

    def __init__(
        self,
        sample_rate: int = 44100,
        chroma_dim: int = 12,
        chroma_hop_length: int = 147,
        chroma_n_fft: int = 2048,
        encoder_dims: Sequence[int] = (64, 128, 256),
        decoder_dims: Sequence[int] = (256, 128, 64),
        encoder_strides: Sequence[int] = (1, 2, 2),
        decoder_strides: Sequence[int] = (2, 2, 1),
        n_codebooks: int = 1,
        codebook_size: int = 512,
        codebook_dim: int = 32,
    ):
        super().__init__()
        self.sample_rate = int(sample_rate)
        self.chroma_dim = int(chroma_dim)
        self.chroma_hop_length = int(chroma_hop_length)
        self.chroma_n_fft = int(chroma_n_fft)
        self.encoder = ChromaEncoder(chroma_dim, encoder_dims, encoder_strides)
        self.quantizer = ResidualVectorQuantize(
            input_dim=self.encoder.output_dim,
            n_codebooks=n_codebooks,
            codebook_size=codebook_size,
            codebook_dim=codebook_dim,
        )
        self.decoder = ChromaDecoder(self.encoder.output_dim, decoder_dims, chroma_dim, decoder_strides)
        self.register_buffer("window", torch.hann_window(self.chroma_n_fft), persistent=False)
        self.register_buffer(
            "chroma_filterbank",
            _make_chroma_filterbank(self.sample_rate, self.chroma_n_fft, self.chroma_dim),
            persistent=False,
        )
        self.apply(init_weights)

    @torch.no_grad()
    def extract_chroma(self, wav: torch.Tensor) -> torch.Tensor:
        if wav.dim() != 3:
            raise ValueError(f"Expected wav with shape [B, C, T], got {tuple(wav.shape)}")
        mono = wav[:, :1, :].squeeze(1).float()
        spec = torch.stft(
            mono,
            n_fft=self.chroma_n_fft,
            hop_length=self.chroma_hop_length,
            win_length=self.chroma_n_fft,
            window=self.window.to(mono.device),
            center=True,
            return_complex=True,
        ).abs()
        chroma = torch.einsum("cf,bft->bct", self.chroma_filterbank.to(spec.device), spec)
        chroma = chroma / chroma.amax(dim=1, keepdim=True).clamp_min(1e-8)
        return chroma.clamp(0.0, 1.0)

    def encode_chroma(self, chroma: torch.Tensor) -> dict[str, torch.Tensor]:
        z = self.encoder(chroma)
        z_q, codes, latents, commitment_loss, codebook_loss = self.quantizer(z)
        return {
            "z": z,
            "z_q": z_q,
            "codes": codes,
            "tokens": codes.squeeze(1),
            "latents": latents,
            "commitment_loss": commitment_loss,
            "codebook_loss": codebook_loss,
        }

    def encode(self, wav: torch.Tensor, return_chroma: bool = True) -> dict[str, torch.Tensor]:
        chroma = self.extract_chroma(wav)
        output = self.encode_chroma(chroma)
        if return_chroma:
            output["chroma"] = chroma
        return output

    def decode(self, z_q: torch.Tensor) -> torch.Tensor:
        return self.decoder(z_q)

    def decode_from_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.dim() == 2:
            tokens = tokens.unsqueeze(1)
        z_q, _, _ = self.quantizer.from_codes(tokens)
        return self.decode(z_q)

    def forward(self, wav: torch.Tensor) -> dict[str, torch.Tensor]:
        encoded = self.encode(wav, return_chroma=True)
        chroma_recon = self.decode(encoded["z_q"])
        min_len = min(encoded["chroma"].shape[-1], chroma_recon.shape[-1])
        return {
            **encoded,
            "chroma_recon": chroma_recon[..., :min_len],
            "chroma": encoded["chroma"][..., :min_len],
            "recon_loss": F.mse_loss(encoded["chroma"][..., :min_len], chroma_recon[..., :min_len]),
        }
