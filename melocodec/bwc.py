import math
from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn

from .chroma import ChromaCodec
from .layers import Decoder, Encoder, init_weights
from .quantize import ResidualVectorQuantize


class BWC(nn.Module):
    """MeloCodec architecture as implemented in code.

    BWC stands for *Bandwidth-efficient With Chroma*. It is the code-level name
    for the MeloCodec model described in the paper and project page.
    """

    def __init__(
        self,
        encoder_dim: int = 64,
        encoder_rates: Sequence[int] = (3, 4, 7, 7),
        latent_dim: int | None = None,
        decoder_dim: int = 1536,
        decoder_rates: Sequence[int] = (7, 7, 4, 3),
        n_codebooks: int = 9,
        codebook_size: int = 1024,
        codebook_dim: int = 8,
        quantizer_dropout: float = 0.0,
        sample_rate: int = 44100,
        melody_codec: ChromaCodec | None = None,
        freeze_melody_encoder: bool = True,
        chroma_latent_dim: int | None = None,
    ):
        super().__init__()
        self.sample_rate = int(sample_rate)
        self.encoder_rates = tuple(int(rate) for rate in encoder_rates)
        self.decoder_rates = tuple(int(rate) for rate in decoder_rates)
        self.hop_length = math.prod(self.encoder_rates)

        if latent_dim is None:
            latent_dim = encoder_dim * (2 ** len(self.encoder_rates))
        self.latent_dim = int(latent_dim)

        self.acoustic_encoder = Encoder(encoder_dim, self.encoder_rates, self.latent_dim)
        self.acoustic_decoder = Decoder(self.latent_dim, decoder_dim, self.decoder_rates)
        self.melody_codec = melody_codec or ChromaCodec(sample_rate=sample_rate)
        melody_dim = int(chroma_latent_dim or self.melody_codec.encoder.output_dim)

        if freeze_melody_encoder:
            for param in self.melody_codec.encoder.parameters():
                param.requires_grad = False
            for param in self.melody_codec.quantizer.parameters():
                param.requires_grad = False

        self.melody_to_fusion = nn.Conv1d(self.melody_codec.encoder.output_dim, melody_dim, kernel_size=1)
        fusion_dim = self.latent_dim + melody_dim
        self.fusion = nn.Conv1d(fusion_dim, fusion_dim, kernel_size=1)
        self.quantizer = ResidualVectorQuantize(
            input_dim=fusion_dim,
            n_codebooks=n_codebooks,
            codebook_size=codebook_size,
            codebook_dim=codebook_dim,
            quantizer_dropout=quantizer_dropout,
        )
        self.to_acoustic = nn.Conv1d(fusion_dim, self.latent_dim, kernel_size=1)
        self.to_melody = nn.Conv1d(fusion_dim, self.melody_codec.encoder.output_dim, kernel_size=1)
        self.n_codebooks = int(n_codebooks)
        self.codebook_size = int(codebook_size)
        self.apply(init_weights)

    def preprocess(self, wav: torch.Tensor) -> tuple[torch.Tensor, int]:
        if wav.dim() == 2:
            wav = wav.unsqueeze(1)
        if wav.dim() != 3:
            raise ValueError(f"Expected wav with shape [B, C, T] or [B, T], got {tuple(wav.shape)}")
        length = wav.shape[-1]
        right_pad = math.ceil(length / self.hop_length) * self.hop_length - length
        if right_pad:
            wav = F.pad(wav, (0, right_pad))
        return wav, length

    @staticmethod
    def _match_time(x: torch.Tensor, target_steps: int) -> torch.Tensor:
        if x.shape[-1] == target_steps:
            return x
        return F.interpolate(x, size=target_steps, mode="linear", align_corners=False)

    @staticmethod
    def _scale_melody(acoustic: torch.Tensor, melody: torch.Tensor) -> torch.Tensor:
        acoustic_std = acoustic.std(dim=(1, 2), keepdim=True).clamp_min(1e-8)
        melody_std = melody.std(dim=(1, 2), keepdim=True).clamp_min(1e-8)
        scale = (0.5 * acoustic_std / melody_std).clamp(0.02, 1.0)
        return melody * scale

    def encode(
        self,
        wav: torch.Tensor,
        n_quantizers: int | None = None,
        return_chroma: bool = True,
    ) -> dict[str, torch.Tensor]:
        wav, _ = self.preprocess(wav)
        acoustic = self.acoustic_encoder(wav)

        with torch.no_grad():
            melody = self.melody_codec.encode(wav, return_chroma=return_chroma)
            melody_latent = melody["z_q"]

        melody_latent = self.melody_to_fusion(melody_latent)
        melody_latent = self._match_time(melody_latent, acoustic.shape[-1])
        melody_latent = self._scale_melody(acoustic, melody_latent)

        fused = torch.cat([acoustic, melody_latent], dim=1)
        fused = self.fusion(fused)
        z_q, codes, latents, commitment_loss, codebook_loss = self.quantizer(fused, n_quantizers=n_quantizers)
        output = {
            "z_q": z_q,
            "codes": codes,
            "latents": latents,
            "commitment_loss": commitment_loss,
            "codebook_loss": codebook_loss,
            "melody_tokens": melody["tokens"],
            "melody_z_q": melody["z_q"],
        }
        if return_chroma:
            output["chroma"] = melody["chroma"]
        return output

    def decode(self, z_q: torch.Tensor) -> dict[str, torch.Tensor]:
        acoustic = self.to_acoustic(z_q)
        melody_pred = self.to_melody(z_q)
        return {
            "audio": self.acoustic_decoder(acoustic),
            "chroma_recon": self.melody_codec.decode(melody_pred),
            "melody_pred": melody_pred,
        }

    def forward(self, wav: torch.Tensor, n_quantizers: int | None = None) -> dict[str, torch.Tensor]:
        _, length = self.preprocess(wav)
        encoded = self.encode(wav, n_quantizers=n_quantizers, return_chroma=True)
        decoded = self.decode(encoded["z_q"])
        audio = decoded["audio"][..., :length]

        chroma = encoded["chroma"]
        chroma_recon = decoded["chroma_recon"]
        min_chroma_len = min(chroma.shape[-1], chroma_recon.shape[-1])
        chroma = chroma[..., :min_chroma_len]
        chroma_recon = chroma_recon[..., :min_chroma_len]

        return {
            "audio": audio,
            "codes": encoded["codes"],
            "latents": encoded["latents"],
            "chroma": chroma,
            "chroma_recon": chroma_recon,
            "vq/commitment_loss": encoded["commitment_loss"],
            "vq/codebook_loss": encoded["codebook_loss"],
            "chroma/recon_loss": F.mse_loss(chroma, chroma_recon),
        }


MeloCodec = BWC
