"""Twin action-value critics over the latent condition.

Two heads per critic and two critics, because the constrained step needs an uncertainty read on
the cost estimates and the offline comparators need the same object. The critic is trained on
admitted episodes only; the certificate, not the critic, is what bounds its error.

All four target columns are regressed in standardised units so a return and three costs can share
one loss; the actor step unscales the cost columns before comparing them with the budgets, which is
what keeps a multiplier interpretable as a price per minute.

Ref: Eq. (2) (costs), Eq. (4) (constrained objective), Sec. 3.2 (returns of episodes).
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class TwinCritic(nn.Module):
    """Two independent Q functions with a shared input embedding."""

    def __init__(self, input_dim: int, output_dim: int, hidden: int = 128) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.register_buffer("target_mean", torch.zeros(output_dim))
        self.register_buffer("target_std", torch.ones(output_dim))
        self.embed = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
        )
        self.heads = nn.ModuleList([nn.Linear(hidden, output_dim) for _ in range(2)])

    def set_target_scaler(self, targets: Tensor) -> None:
        """Freeze the column statistics of the fitting pool."""
        self.target_mean = targets.mean(dim=0)
        self.target_std = targets.std(dim=0).clamp(min=1e-3)

    def scale_targets(self, targets: Tensor) -> Tensor:
        return (targets - self.target_mean) / self.target_std

    def unscale_targets(self, values: Tensor) -> Tensor:
        return values * self.target_std + self.target_mean

    def forward(self, features: Tensor) -> tuple[Tensor, Tensor]:
        embedded = self.embed(features)
        return self.heads[0](embedded), self.heads[1](embedded)

    def conservative(self, features: Tensor) -> Tensor:
        """Lower of the two heads, the value the constrained step acts on."""
        first, second = self.forward(features)
        return torch.minimum(first, second)

    def disagreement(self, features: Tensor) -> Tensor:
        first, second = self.forward(features)
        return (first - second).abs()

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
