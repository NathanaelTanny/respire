"""Deterministic policy head.

The replacement used by the flow-head ablation: a single regression to the action instead of a
density transported from noise. It keeps the same call surface as the flow so the rest of the
controller is unchanged, and its log-density is the constant zero, which is precisely why removing
the head also removes the trust region's meaning.

Ref: Table 2 (w/o flow head), Eq. (4) (the KL term needs a density).
"""

from __future__ import annotations

import math
from typing import cast

import torch
from torch import Tensor, nn

from respire.models.flow import FlowSample
from respire.settings import PolicyConfig


class DeterministicHead(nn.Module):
    """Mean-action head with a flow-shaped call surface."""

    def __init__(self, cfg: PolicyConfig, cond_dim: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.cond_dim = cond_dim
        self.kind = "deterministic"
        width = cond_dim
        layers: list[nn.Module] = [nn.Linear(width, cfg.hidden), nn.SiLU()]
        for _ in range(cfg.depth - 1):
            layers.extend([nn.Linear(cfg.hidden, cfg.hidden), nn.SiLU()])
        layers.append(nn.Linear(cfg.hidden, cfg.action_dim))
        self.net = nn.Sequential(*layers)

    def mean_action(self, condition: Tensor) -> Tensor:
        return cast(Tensor, self.net(condition))

    def sample(self, condition: Tensor, generator: torch.Generator | None = None) -> FlowSample:
        action = self.mean_action(condition)
        zeros = torch.zeros(condition.shape[0])
        return FlowSample(action=action, log_prob=zeros, divergence_integral=zeros)

    def sample_pathwise(self, condition: Tensor) -> Tensor:
        return self.mean_action(condition)

    def log_prob(self, action: Tensor, condition: Tensor) -> Tensor:
        return torch.zeros(condition.shape[0])

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def deterministic_regulariser(prediction: Tensor, target: Tensor) -> Tensor:
    """Mean squared action error, the term that replaces flow matching for this head."""
    return ((prediction - target) ** 2).mean()


def gaussian_log_density(value: Tensor, scale: float) -> Tensor:
    """Reference density used when a deterministic head is compared with a flow."""
    return -0.5 * (value / scale) ** 2 - math.log(scale) - 0.5 * math.log(2.0 * math.pi)
