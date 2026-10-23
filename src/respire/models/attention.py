"""Cross-channel attention over the twelve monitored channels.

Each channel contributes one token built from its own sixty-second trajectory, so the block can
relate, for instance, the pressure trend to the infusion history without mixing channels into a
single vector first.

Ref: Sec. 4.2, Fig. 2 (cross-channel attention after the selective blocks).
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class CrossChannelAttention(nn.Module):
    """Self-attention across channels, pooled to one interaction vector."""

    def __init__(self, n_channels: int, window: int, d_model: int, n_heads: int) -> None:
        super().__init__()
        self.n_channels = n_channels
        self.embed = nn.Linear(window, d_model)
        self.channel_embedding = nn.Parameter(torch.zeros(1, n_channels, d_model))
        self.attention = nn.MultiheadAttention(d_model, n_heads, batch_first=True, dropout=0.0)
        self.norm = nn.LayerNorm(d_model)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, window: Tensor) -> tuple[Tensor, Tensor]:
        """``window`` is ``[batch, steps, channels]``; returns the pooled vector and the weights."""
        series = window.transpose(1, 2)
        tokens = self.embed(series) + self.channel_embedding
        attended, weights = self.attention(tokens, tokens, tokens, need_weights=True)
        fused = self.norm(tokens + attended)
        pooled = self.proj(fused.mean(dim=1))
        return pooled, weights.mean(dim=1)
