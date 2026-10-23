"""Latent physiological pattern encoder.

One pass over a sixty-second window produces three things the rest of the system consumes: the
sixteen-dimensional latent ``z``, the three axes ``s`` and the recurrent summary ``h``. The
decoder, the one-step head and the intervention head exist only so the encoder can be trained
without episode labels; they are not used after the encoder is frozen.

Ref: Sec. 4.2, Fig. 2 (LPPE, z in R^16 with sparse readout, s in R^3), Sec. 1 (three axes).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import torch
from torch import Tensor, nn

from respire.models.attention import CrossChannelAttention
from respire.models.ssm import SelectiveStack
from respire.settings import EncoderConfig


class SparseReadout(nn.Module):
    """Top-k sparse coding of a pooled feature into the latent.

    Only the k highest-scoring atoms contribute, so the latent is a combination of a few
    physiological patterns rather than a dense mixture.
    """

    def __init__(self, d_in: int, n_atoms: int, latent_dim: int, topk: int) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.topk = topk
        self.score = nn.Linear(d_in, n_atoms)
        self.value = nn.Linear(d_in, n_atoms)
        self.dictionary = nn.Parameter(torch.randn(n_atoms, latent_dim) * 0.02)

    def forward(self, features: Tensor) -> tuple[Tensor, Tensor]:
        scores = self.score(features)
        k = min(self.topk, scores.shape[-1])
        threshold = scores.topk(k, dim=-1).values[..., -1:]
        mask = (scores >= threshold).to(features.dtype)
        gate = torch.sigmoid(scores)
        coefficients = torch.nn.functional.softplus(self.value(features)) * gate * mask
        latent = coefficients @ self.dictionary
        return latent, mask


@dataclass(frozen=True)
class EncoderOutput:
    """Everything one forward pass exposes."""

    latent: Tensor
    axes: Tensor
    hidden: Tensor
    reconstruction: Tensor
    sparsity: Tensor


class LatentEncoder(nn.Module):
    """Selective stack, cross-channel attention and a sparse readout."""

    def __init__(self, cfg: EncoderConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.input_proj = nn.Linear(cfg.n_channels + cfg.action_channels, cfg.d_model)
        self.stack = SelectiveStack(cfg.d_model, cfg.d_state, cfg.d_conv, cfg.expand, cfg.n_blocks)
        self.channel_attention = CrossChannelAttention(
            cfg.n_channels, cfg.window_steps, cfg.d_model, cfg.n_heads
        )
        self.fuse = nn.Linear(3 * cfg.d_model, cfg.d_model)
        self.readout = SparseReadout(cfg.d_model, 32, cfg.latent_dim, cfg.sparse_topk)
        self.axis_head = nn.Linear(cfg.d_model, cfg.axis_dim)
        self.hidden_head = nn.Linear(cfg.d_model, cfg.hidden_dim)
        self.decoder = nn.Linear(cfg.latent_dim, cfg.n_channels)
        self.one_step = nn.Linear(cfg.latent_dim + cfg.hidden_dim + cfg.action_channels, 2)
        self.intervention = nn.Sequential(
            nn.Linear(cfg.latent_dim + cfg.action_channels, cfg.d_model),
            nn.SiLU(),
            nn.Linear(cfg.d_model, 1),
        )

    def forward(self, window: Tensor, actions: Tensor) -> EncoderOutput:
        if window.ndim != 3:
            raise ValueError("window must be [batch, steps, channels]")
        if actions.ndim != 3:
            raise ValueError("actions must be [batch, steps, action_dim]")
        joined = torch.cat([window, actions], dim=-1)
        embedded = self.input_proj(joined)
        last, pooled = self.stack(embedded)
        attended, _ = self.channel_attention(window)
        mixed = cast(Tensor, self.fuse(torch.cat([last, pooled, attended], dim=-1)))
        fused = torch.nn.functional.silu(mixed)
        latent, mask = self.readout(fused)
        return EncoderOutput(
            latent=latent,
            axes=cast(Tensor, self.axis_head(fused)),
            hidden=cast(Tensor, self.hidden_head(fused)),
            reconstruction=cast(Tensor, self.decoder(latent)),
            sparsity=mask.mean(dim=-1),
        )

    def predict_one_step(self, latent: Tensor, hidden: Tensor, action: Tensor) -> Tensor:
        """One-step change of pressure and depth; the recurrent summary enters here so it trains."""
        return cast(Tensor, self.one_step(torch.cat([latent, hidden, action], dim=-1)))

    def predict_delta_depth(self, latent: Tensor, action: Tensor) -> Tensor:
        return cast(Tensor, self.intervention(torch.cat([latent, action], dim=-1))).squeeze(-1)

    def encode(self, window: Tensor, actions: Tensor) -> EncoderOutput:
        with torch.no_grad():
            return self.forward(window, actions)


class EncoderStandardiser:
    """Channel-wise standardisation fitted on the development split.

    Kept beside the encoder so a window recorded at another site is scaled the way the encoder
    expects; the statistics are buffers, not learned parameters.
    """

    def __init__(self, n_channels: int) -> None:
        self.mean = torch.zeros(n_channels)
        self.std = torch.ones(n_channels)

    def fit(self, windows: Tensor) -> None:
        flat = windows.reshape(-1, windows.shape[-1])
        self.mean = flat.mean(dim=0)
        self.std = flat.std(dim=0).clamp(min=1e-3)

    def transform(self, windows: Tensor) -> Tensor:
        return (windows - self.mean) / self.std

    def inverse(self, standardised: Tensor) -> Tensor:
        return standardised * self.std + self.mean

    def state(self) -> dict[str, list[float]]:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_state(cls, state: dict[str, object]) -> EncoderStandardiser:
        mean = torch.tensor(state["mean"], dtype=torch.float32)
        scale = cls(mean.numel())
        scale.mean = mean
        scale.std = torch.tensor(state["std"], dtype=torch.float32)
        return scale
