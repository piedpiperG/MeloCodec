from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn

from .layers import WNConv1d


class VectorQuantize(nn.Module):
    """Factorized vector quantizer used by residual codec stacks."""

    def __init__(self, input_dim: int, codebook_size: int, codebook_dim: int):
        super().__init__()
        self.codebook_size = int(codebook_size)
        self.codebook_dim = int(codebook_dim)
        self.in_proj = WNConv1d(input_dim, codebook_dim, kernel_size=1)
        self.out_proj = WNConv1d(codebook_dim, input_dim, kernel_size=1)
        self.codebook = nn.Embedding(codebook_size, codebook_dim)

    def decode_latents(self, latents: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, dim, steps = latents.shape
        flat = latents.permute(0, 2, 1).reshape(batch * steps, dim)
        flat = F.normalize(flat, dim=-1)
        codebook = F.normalize(self.codebook.weight, dim=-1)
        indices = torch.argmax(flat @ codebook.t(), dim=-1).view(batch, steps)
        quantized = self.decode_code(indices)
        return quantized, indices

    def decode_code(self, indices: torch.Tensor) -> torch.Tensor:
        return F.embedding(indices, self.codebook.weight).transpose(1, 2)

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        projected = self.in_proj(z)
        quantized, indices = self.decode_latents(projected)
        commitment_loss = F.mse_loss(projected, quantized.detach(), reduction="none").mean(dim=(1, 2))
        codebook_loss = F.mse_loss(quantized, projected.detach(), reduction="none").mean(dim=(1, 2))
        quantized = projected + (quantized - projected).detach()
        return self.out_proj(quantized), commitment_loss, codebook_loss, indices, projected


class ResidualVectorQuantize(nn.Module):
    """Residual vector quantizer.

    ``n_quantizers`` controls bitrate at inference time. For example, BWC uses
    fewer codebooks for low-bitrate reconstruction and more codebooks for
    high-bitrate reconstruction.
    """

    def __init__(
        self,
        input_dim: int,
        n_codebooks: int = 9,
        codebook_size: int = 1024,
        codebook_dim: int | Sequence[int] = 8,
        quantizer_dropout: float = 0.0,
    ):
        super().__init__()
        if isinstance(codebook_dim, int):
            dims = [codebook_dim] * n_codebooks
        else:
            dims = list(codebook_dim)
            if len(dims) != n_codebooks:
                raise ValueError("codebook_dim sequence length must match n_codebooks")

        self.n_codebooks = int(n_codebooks)
        self.codebook_size = int(codebook_size)
        self.codebook_dim = dims
        self.quantizer_dropout = float(quantizer_dropout)
        self.quantizers = nn.ModuleList(
            [VectorQuantize(input_dim, codebook_size, dims[i]) for i in range(n_codebooks)]
        )

    def _active_quantizers(self, z: torch.Tensor, n_quantizers: int | None) -> torch.Tensor:
        if n_quantizers is None:
            n_quantizers = self.n_codebooks
        if not self.training or self.quantizer_dropout <= 0:
            return torch.full((z.shape[0],), int(n_quantizers), device=z.device, dtype=torch.long)

        active = torch.full((z.shape[0],), self.n_codebooks, device=z.device, dtype=torch.long)
        n_dropout = int(z.shape[0] * self.quantizer_dropout)
        if n_dropout > 0:
            active[:n_dropout] = torch.randint(1, self.n_codebooks + 1, (n_dropout,), device=z.device)
        return active

    def forward(
        self,
        z: torch.Tensor,
        n_quantizers: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        active = self._active_quantizers(z, n_quantizers)
        quantized = torch.zeros_like(z)
        residual = z
        commitment_loss = z.new_zeros(())
        codebook_loss = z.new_zeros(())
        codes: list[torch.Tensor] = []
        latents: list[torch.Tensor] = []

        max_active = int(active.max().item())
        for idx, quantizer in enumerate(self.quantizers[:max_active]):
            q_i, commitment_i, codebook_i, codes_i, latents_i = quantizer(residual)
            mask = (active > idx).to(z.dtype).view(-1, 1, 1)
            quantized = quantized + q_i * mask
            residual = residual - q_i
            commitment_loss = commitment_loss + (commitment_i * mask.flatten()).mean()
            codebook_loss = codebook_loss + (codebook_i * mask.flatten()).mean()
            codes.append(codes_i)
            latents.append(latents_i)

        return quantized, torch.stack(codes, dim=1), torch.cat(latents, dim=1), commitment_loss, codebook_loss

    def from_codes(self, codes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        quantized = None
        projected = []
        for idx in range(codes.shape[1]):
            z_i = self.quantizers[idx].decode_code(codes[:, idx, :])
            projected.append(z_i)
            q_i = self.quantizers[idx].out_proj(z_i)
            quantized = q_i if quantized is None else quantized + q_i
        return quantized, torch.cat(projected, dim=1), codes
