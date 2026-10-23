"""Shared network blocks for the comparison arms.

Each comparator owns its own update rule; only the function approximators are shared, so a
difference between two rows of the comparison table cannot come from a difference in width or
initialisation.

Ref: Table 1 (the thirteen comparison arms).
"""

from __future__ import annotations

from typing import cast

import torch
from torch import Tensor, nn

ACTION_DIM = 2


class GaussianActor(nn.Module):
    """Diagonal Gaussian policy with a state-independent log standard deviation."""

    def __init__(self, state_dim: int, hidden: int = 128, action_dim: int = ACTION_DIM) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
        )
        self.mean = nn.Linear(hidden, action_dim)
        self.log_std = nn.Parameter(torch.full((action_dim,), -0.5))

    def forward(self, state: Tensor) -> tuple[Tensor, Tensor]:
        hidden = cast(Tensor, self.body(state))
        return cast(Tensor, self.mean(hidden)), self.log_std.exp().expand(hidden.shape[0], -1)

    def sample(
        self, state: Tensor, generator: torch.Generator | None = None
    ) -> tuple[Tensor, Tensor]:
        mean, std = self.forward(state)
        noise = torch.randn(mean.shape, generator=generator)
        return mean + std * noise, mean

    def log_prob(self, state: Tensor, action: Tensor) -> Tensor:
        mean, std = self.forward(state)
        normalised = (action - mean) / std
        return (
            -0.5 * (normalised**2).sum(dim=-1)
            - std.log().sum(dim=-1)
            - 0.5 * ACTION_DIM * torch.log(torch.tensor(2.0 * torch.pi))
        )


class DeterministicActor(nn.Module):
    """Tanh-squashed deterministic policy on the normalised action box."""

    def __init__(self, state_dim: int, hidden: int = 128, action_dim: int = ACTION_DIM) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, action_dim),
        )

    def forward(self, state: Tensor) -> Tensor:
        return torch.tanh(cast(Tensor, self.net(state)))


class TwinQ(nn.Module):
    """Two Q functions on (state, action)."""

    def __init__(self, state_dim: int, action_dim: int = ACTION_DIM, hidden: int = 128) -> None:
        super().__init__()
        self.first = _q_net(state_dim + action_dim, hidden)
        self.second = _q_net(state_dim + action_dim, hidden)

    def forward(self, state: Tensor, action: Tensor) -> tuple[Tensor, Tensor]:
        joined = torch.cat([state, action], dim=-1)
        return self.first(joined).squeeze(-1), self.second(joined).squeeze(-1)

    def minimum(self, state: Tensor, action: Tensor) -> Tensor:
        first, second = self.forward(state, action)
        return torch.minimum(first, second)


def _q_net(width: int, hidden: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(width, hidden),
        nn.SiLU(),
        nn.Linear(hidden, hidden),
        nn.SiLU(),
        nn.Linear(hidden, 1),
    )


class ValueNet(nn.Module):
    """State value used by the policy-gradient arms."""

    def __init__(self, state_dim: int, hidden: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, state: Tensor) -> Tensor:
        return cast(Tensor, self.net(state)).squeeze(-1)


class CostCritic(nn.Module):
    """Per-constraint cost critic for the Lagrangian arms."""

    def __init__(self, state_dim: int, n_costs: int = 3, hidden: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, n_costs),
        )

    def forward(self, state: Tensor, action: Tensor) -> Tensor:
        return cast(Tensor, self.net(torch.cat([state, action], dim=-1)))
