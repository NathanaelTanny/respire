"""Dual variables for the three safety costs and for the trust region.

Each multiplier rises while its constraint is violated and decays while it is slack, projected onto
the non-negative orthant. The budgets are the ones the article fixes: ten minutes of hypotension,
one minute of burst suppression, and the ninetieth percentile of the cohort's rate variability.

Ref: Eq. (2) (costs), Eq. (4) (constraints and the KL term), Sec. 3.1 (budgets d1, d2, d3).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class DualAscent:
    """Projected gradient ascent on four multipliers."""

    learning_rate: float = 0.05
    initial: float = 0.1
    multipliers: Tensor | None = None
    history: list[list[float]] | None = None

    def __post_init__(self) -> None:
        if self.multipliers is None:
            self.multipliers = torch.full((4,), self.initial)
        if self.history is None:
            self.history = []

    def update(self, violations: Tensor) -> Tensor:
        """``violations`` are measured minus budget, one entry per constraint."""
        if self.multipliers is None:
            raise RuntimeError("multipliers are not initialised")
        if violations.numel() != self.multipliers.numel():
            raise ValueError("expected one violation per multiplier")
        with torch.no_grad():
            self.multipliers = torch.clamp(
                self.multipliers + self.learning_rate * violations, min=0.0, max=100.0
            )
        if self.history is not None:
            self.history.append([float(v) for v in self.multipliers])
        return self.multipliers.clone()

    @property
    def values(self) -> Tensor:
        if self.multipliers is None:
            raise RuntimeError("multipliers are not initialised")
        return self.multipliers

    def penalties(self) -> Tensor:
        """The differentiable part: multipliers for the three costs, then the trust-region one."""
        return self.values


def cost_penalty(cost_estimates: Tensor, budgets: Tensor, multipliers: Tensor) -> Tensor:
    """Sum over constraints of lambda_j times the estimated cost relative to its budget."""
    excess = cost_estimates / torch.clamp(budgets, min=1e-6) - 1.0
    return (multipliers[: cost_estimates.shape[-1]] * excess).sum(dim=-1).mean()


def trust_region_penalty(divergence: Tensor, budget: float, multiplier: Tensor) -> Tensor:
    """Multiplier times the KL excess, the term the policy step minimises."""
    return multiplier * torch.clamp(divergence - budget, min=0.0)


def violations(cost_means: Tensor, budgets: Tensor, divergence: float, kl_budget: float) -> Tensor:
    """Measured cost relative to budget, one entry per constraint plus the trust region.

    The costs are normalised by their budgets before the ascent, so a multiplier of one means "one
    budget over"; in the primal step the same normalisation is applied. Multipliers in raw units
    would otherwise differ by three orders of magnitude across the three costs and the trust region,
    and a single learning rate could not serve all four.
    """
    return torch.cat(
        [
            cost_means / torch.clamp(budgets, min=1e-6) - 1.0,
            torch.tensor([divergence / max(kl_budget, 1e-6) - 1.0]),
        ]
    )
