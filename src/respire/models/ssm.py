"""Selective state-space block.

The recurrence is input-dependent: the step size, the input matrix and the readout matrix are all
projected from the current sample, so the block can hold a channel's history when the surgical
context is stable and refresh it when an infusion changes. Six of these blocks form the temporal
backbone of the encoder.

Ref: Sec. 4.2, Fig. 2 (six selective SSM blocks over a 60-s window).
"""

from __future__ import annotations

import math
from typing import cast

import torch
from torch import Tensor, nn

from respire.utils.numerics import softplus_inverse


def selective_scan(
    inputs: Tensor,
    delta: Tensor,
    a_matrix: Tensor,
    b_matrix: Tensor,
    c_matrix: Tensor,
    d_vector: Tensor,
) -> Tensor:
    """Discretised scan ``h_t = exp(dt A) h_{t-1} + dt B u_t``, read out by ``C``.

    ``inputs`` and ``delta`` are ``[batch, steps, inner]``; the state matrices are
    ``[batch, steps, inner, state]``; the skip term ``d_vector`` is ``[inner]``.
    """
    batch, steps, inner = inputs.shape
    state_dim = a_matrix.shape[-1]
    state = inputs.new_zeros(batch, inner, state_dim)
    outputs = []
    for index in range(steps):
        decay = torch.exp(delta[:, index].unsqueeze(-1) * a_matrix[:, index])
        drive = delta[:, index].unsqueeze(-1) * b_matrix[:, index] * inputs[:, index].unsqueeze(-1)
        state = decay * state + drive
        read = (state * c_matrix[:, index]).sum(dim=-1) + d_vector * inputs[:, index]
        outputs.append(read)
    return torch.stack(outputs, dim=1)


class SelectiveSSMBlock(nn.Module):
    """Depthwise convolution, gated selective scan and output projection."""

    def __init__(self, d_model: int, d_state: int, d_conv: int, expand: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.inner = d_model * expand
        self.dt_rank = max(d_model // 16, 1)
        self.in_proj = nn.Linear(d_model, 2 * self.inner)
        self.conv = nn.Conv1d(
            self.inner, self.inner, kernel_size=d_conv, groups=self.inner, padding=d_conv - 1
        )
        self.x_proj = nn.Linear(self.inner, self.dt_rank + 2 * d_state)
        self.dt_proj = nn.Linear(self.dt_rank, self.inner)
        self.a_log = nn.Parameter(
            torch.log(torch.arange(1, d_state + 1, dtype=torch.float32)).repeat(self.inner, 1)
        )
        self.d_vector = nn.Parameter(torch.ones(self.inner))
        self.out_proj = nn.Linear(self.inner, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dt_bias = nn.Parameter(torch.full((self.inner,), softplus_inverse(0.05)))

    def forward(self, hidden: Tensor) -> Tensor:
        residual = hidden
        projected = self.in_proj(self.norm(hidden))
        value, gate = projected.chunk(2, dim=-1)
        convolved = self.conv(value.transpose(1, 2)).transpose(1, 2)[:, : hidden.shape[1]]
        convolved = torch.nn.functional.silu(convolved)

        delta_raw, b_matrix, c_matrix = self.x_proj(convolved).split(
            [self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        delta = torch.nn.functional.softplus(self.dt_proj(delta_raw) + self.dt_bias)
        a_matrix = -torch.exp(self.a_log)
        a_expanded = (
            a_matrix.unsqueeze(0)
            .unsqueeze(0)
            .expand(hidden.shape[0], hidden.shape[1], self.inner, self.d_state)
        )
        b_expanded = b_matrix.unsqueeze(2).expand(-1, -1, self.inner, -1)
        c_expanded = c_matrix.unsqueeze(2).expand(-1, -1, self.inner, -1)
        scanned = selective_scan(
            convolved, delta, a_expanded, b_expanded, c_expanded, self.d_vector
        )
        gated = scanned * torch.nn.functional.silu(gate)
        return residual + cast(Tensor, self.out_proj(gated))


class SelectiveStack(nn.Module):
    """A stack of blocks with a final norm, returning the last step and the pooled state."""

    def __init__(self, d_model: int, d_state: int, d_conv: int, expand: int, n_blocks: int) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [SelectiveSSMBlock(d_model, d_state, d_conv, expand) for _ in range(n_blocks)]
        )
        self.norm = nn.LayerNorm(d_model)

    @property
    def n_blocks(self) -> int:
        return len(self.blocks)

    def forward(self, hidden: Tensor) -> tuple[Tensor, Tensor]:
        current = hidden
        for block in self.blocks:
            current = block(current)
        current = self.norm(current)
        return current[:, -1], current.mean(dim=1)


def parameter_count(module: nn.Module) -> int:
    return sum(int(math.prod(p.shape)) for p in module.parameters() if p.requires_grad)
