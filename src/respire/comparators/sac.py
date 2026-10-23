"""Soft actor-critic on the flat observation.

Maximum-entropy actor-critic with twin critics, a temperature that is tuned against the target
entropy, and soft target updates. The only arm in the comparison with an explicit exploration
bonus, which is why it buys depth at the expense of rate variability.

Ref: Table 1 (SAC).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor

from respire.comparators.base import (
    TransitionBatch,
    flatten_window,
    normalise_action,
    pump_action_from_mean,
)
from respire.comparators.nets import GaussianActor, TwinQ
from respire.settings import TheatreConfig
from respire.theatre.environment import Observation
from respire.utils.autograd import backward


@dataclass
class SACAgent:
    """Entropy-regularised actor-critic."""

    theatre: TheatreConfig
    state_dim: int = 146
    gamma: float = 0.995
    tau: float = 0.005
    target_entropy: float = -2.0
    batch: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.actor = GaussianActor(self.state_dim)
        self.critic = TwinQ(self.state_dim)
        self.critic_target = TwinQ(self.state_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.actor_optimiser = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_optimiser = torch.optim.Adam(self.critic.parameters(), lr=3e-4)
        self.log_alpha = torch.zeros(1, requires_grad=True)
        self.alpha_optimiser = torch.optim.Adam([self.log_alpha], lr=3e-4)
        self._generator = torch.Generator(device="cpu")

    def reset(self, batch: int) -> None:
        self.batch = batch

    def act(self, observation: Observation, explore: bool = False) -> Tensor:
        state = flatten_window(observation)
        with torch.no_grad():
            mean, std = self.actor(state)
            if explore:
                mean = mean + std * torch.randn(mean.shape, generator=self._generator)
        return pump_action_from_mean(mean, self.theatre.propofol_max, self.theatre.remifentanil_max)

    def update(self, batch: TransitionBatch) -> dict[str, float]:
        action = normalise_action(
            batch.action, self.theatre.propofol_max, self.theatre.remifentanil_max
        )
        with torch.no_grad():
            next_mean, next_std = self.actor(batch.next_state)
            next_sample = torch.tanh(
                next_mean + next_std * torch.randn(next_mean.shape, generator=self._generator)
            )
            next_log_prob = -0.5 * (((next_sample - next_mean) / next_std) ** 2).sum(
                dim=-1
            ) - next_std.log().sum(dim=-1)
            target_q = self.critic_target.minimum(batch.next_state, next_sample)
            target = batch.reward + self.gamma * (1.0 - batch.done) * (
                target_q - self.log_alpha.exp().detach() * next_log_prob
            )
        first, second = self.critic(batch.state, action)
        critic_loss = ((first - target) ** 2).mean() + ((second - target) ** 2).mean()
        self.critic_optimiser.zero_grad()
        backward(critic_loss)
        self.critic_optimiser.step()

        mean, std = self.actor(batch.state)
        sample = torch.tanh(mean + std * torch.randn(mean.shape, generator=self._generator))
        log_prob = -0.5 * (((sample - mean) / std) ** 2).sum(dim=-1) - std.log().sum(dim=-1)
        actor_loss = (
            self.log_alpha.exp().detach() * log_prob - self.critic.minimum(batch.state, sample)
        ).mean()
        self.actor_optimiser.zero_grad()
        backward(actor_loss)
        self.actor_optimiser.step()

        alpha_loss = -(self.log_alpha * (log_prob.detach() + self.target_entropy)).mean()
        self.alpha_optimiser.zero_grad()
        backward(alpha_loss)
        self.alpha_optimiser.step()

        with torch.no_grad():
            for target_parameter, parameter in zip(
                self.critic_target.parameters(), self.critic.parameters(), strict=True
            ):
                target_parameter.mul_(1.0 - self.tau).add_(self.tau * parameter)
        return {
            "critic_loss": float(critic_loss.detach()),
            "actor_loss": float(actor_loss.detach()),
            "alpha": float(self.log_alpha.exp().detach()),
        }
