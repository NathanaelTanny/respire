"""Recurrent state-space world model with imagined actor-critic.

A categorical latent on a deterministic recurrent path, trained by the world-model objective on
replayed transitions, with the actor and the value trained inside imagined rollouts of the prior.
This is the unconstrained arm with the best depth accuracy in the comparison, and the reason the
paper's argument is about the tail rather than the average.

Ref: Sec. 2.1-2.2 (world-model control, DreamerV3 row of Table 1).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor, nn

from respire.comparators.base import (
    TransitionBatch,
    denormalise_action,
    flatten_window,
)
from respire.settings import TheatreConfig
from respire.theatre.environment import Observation
from respire.utils.autograd import backward

HIDDEN = 64
CLASSES = 8
GROUPS = 6
LATENT = CLASSES * GROUPS
FEATURES = HIDDEN + LATENT


def categorical_sample(logits: Tensor) -> Tensor:
    """One-hot sample with the softmax gradient of the standard categorical relaxation."""
    shaped = logits.reshape(-1, GROUPS, CLASSES)
    probabilities = torch.softmax(shaped, dim=-1)
    flat = probabilities.reshape(-1, CLASSES)
    index = torch.multinomial(flat, 1).squeeze(-1)
    one_hot = torch.nn.functional.one_hot(index, CLASSES).to(probabilities.dtype)
    sample = one_hot.reshape(-1, GROUPS, CLASSES) + probabilities - probabilities.detach()
    return sample.reshape(logits.shape[0], LATENT)


def categorical_kl(posterior_logits: Tensor, prior_logits: Tensor) -> Tensor:
    """Exact KL between two factorised categorical distributions."""
    post = torch.softmax(posterior_logits.reshape(-1, GROUPS, CLASSES), dim=-1)
    prior = torch.softmax(prior_logits.reshape(-1, GROUPS, CLASSES), dim=-1)
    return (post * (torch.log(post + 1e-9) - torch.log(prior + 1e-9))).sum(dim=-1).mean()


class RSSM(nn.Module):
    """Deterministic recurrent path with categorical stochastic variables."""

    def __init__(self, state_dim: int) -> None:
        super().__init__()
        self.embed = nn.Sequential(nn.Linear(state_dim, HIDDEN), nn.SiLU())
        self.gru = nn.GRUCell(HIDDEN + LATENT, HIDDEN)
        self.prior_head = nn.Linear(HIDDEN, LATENT)
        self.posterior_head = nn.Linear(HIDDEN + HIDDEN, LATENT)
        self.decoder = nn.Sequential(
            nn.Linear(FEATURES, HIDDEN * 2),
            nn.SiLU(),
            nn.Linear(HIDDEN * 2, state_dim),
        )
        self.reward_head = nn.Linear(FEATURES, 1)

    def initial(self, batch: int) -> tuple[Tensor, Tensor]:
        return (torch.zeros(batch, HIDDEN), torch.zeros(batch, LATENT))

    def observe(
        self, state: Tensor, hidden: Tensor, sample: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        next_hidden = self.gru(torch.cat([hidden, sample], dim=-1), hidden)
        logits = self.posterior_head(torch.cat([next_hidden, self.embed(state)], dim=-1))
        return categorical_sample(logits), next_hidden, logits

    def imagine(self, hidden: Tensor, sample: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        next_hidden = self.gru(torch.cat([hidden, sample], dim=-1), hidden)
        logits = self.prior_head(next_hidden)
        return categorical_sample(logits), next_hidden, logits

    def features(self, hidden: Tensor, sample: Tensor) -> Tensor:
        return torch.cat([hidden, sample], dim=-1)


@dataclass
class DreamerAgent:
    """World model, imagined actor and value."""

    theatre: TheatreConfig
    state_dim: int = 146
    horizon: int = 8
    gamma: float = 0.995
    batch: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.world = RSSM(self.state_dim)
        self.actor = nn.Sequential(nn.Linear(FEATURES, HIDDEN), nn.SiLU(), nn.Linear(HIDDEN, 2))
        self.value = nn.Sequential(nn.Linear(FEATURES, HIDDEN), nn.SiLU(), nn.Linear(HIDDEN, 1))
        self.log_std = nn.Parameter(torch.full((2,), -1.0))
        self.world_optimiser = torch.optim.Adam(self.world.parameters(), lr=3e-4)
        self.actor_optimiser = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.value.parameters()) + [self.log_std],
            lr=3e-4,
        )
        self._generator = torch.Generator(device="cpu")
        self._hidden = torch.zeros(0, HIDDEN)
        self._sample = torch.zeros(0, LATENT)

    def reset(self, batch: int) -> None:
        self._hidden, self._sample = self.world.initial(batch)

    def act(self, observation: Observation, explore: bool = False) -> Tensor:
        state = flatten_window(observation)
        with torch.no_grad():
            posterior, hidden, _ = self.world.observe(state, self._hidden, self._sample)
            self._sample = posterior
            self._hidden = hidden
            mean = self.actor(self.world.features(hidden, posterior))
        return denormalise_action(
            torch.tanh(mean), self.theatre.propofol_max, self.theatre.remifentanil_max
        )

    def update(self, batch: TransitionBatch) -> dict[str, float]:
        hidden, sample = self.world.initial(batch.state.shape[0])
        posterior, hidden, posterior_logits = self.world.observe(batch.state, hidden, sample)
        features = self.world.features(hidden, posterior)
        reconstruction = self.world.decoder(features)
        prior_logits = self.world.prior_head(hidden)
        world_loss = ((reconstruction - batch.state) ** 2).mean()
        world_loss = world_loss + 0.1 * categorical_kl(posterior_logits, prior_logits)
        reward_prediction = self.world.reward_head(features).squeeze(-1)
        reward_loss = ((reward_prediction - batch.reward) ** 2).mean()
        self.world_optimiser.zero_grad()
        backward(world_loss + reward_loss)
        self.world_optimiser.step()

        roll_hidden = hidden.detach()
        roll_sample = posterior.detach()
        rewards: list[Tensor] = []
        values: list[Tensor] = []
        log_probs: list[Tensor] = []
        for _ in range(self.horizon):
            roll_features = self.world.features(roll_hidden, roll_sample)
            mean = self.actor(roll_features)
            std = self.log_std.exp().expand_as(mean)
            noise = torch.randn(mean.shape, generator=self._generator)
            action = torch.tanh(mean + std * noise)
            log_prob = -0.5 * (((action - mean) / std) ** 2).sum(dim=-1) - std.log().sum(dim=-1)
            log_probs.append(log_prob)
            rewards.append(self.world.reward_head(roll_features).squeeze(-1))
            values.append(self.value(roll_features).squeeze(-1))
            roll_sample, roll_hidden, _ = self.world.imagine(roll_hidden, roll_sample)

        returns: list[Tensor] = []
        running = values[-1].detach()
        for reward in reversed(rewards):
            running = reward.detach() + self.gamma * running
            returns.append(running)
        returns.reverse()
        stacked_returns = torch.stack(returns, dim=0)
        stacked_values = torch.stack(values, dim=0)
        stacked_log_probs = torch.stack(log_probs, dim=0)
        advantage = (stacked_returns - stacked_values).detach()
        actor_loss = -(advantage * stacked_log_probs).mean()
        value_loss = ((stacked_values - stacked_returns) ** 2).mean()
        self.actor_optimiser.zero_grad()
        backward(actor_loss + 0.5 * value_loss)
        self.actor_optimiser.step()
        return {
            "world_loss": float(world_loss.detach()),
            "reward_loss": float(reward_loss.detach()),
            "actor_loss": float(actor_loss.detach()),
            "value_loss": float(value_loss.detach()),
        }
