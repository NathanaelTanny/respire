"""Offline deterministic actor-critic with a behaviour-cloning regulariser.

The actor maximises the twin critic and is pulled back towards the logged action by a squared
penalty whose weight is normalised by the critic's own scale, which is what keeps the arm stable
when the dataset is narrow.

Ref: Table 1 (TD3+BC).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor

from respire.comparators.base import (
    TransitionBatch,
    flatten_window,
    normalise_action,
    pump_action,
)
from respire.comparators.nets import DeterministicActor, TwinQ
from respire.settings import TheatreConfig
from respire.theatre.environment import Observation
from respire.utils.autograd import backward


@dataclass
class TD3BCAgent:
    """Twin critic, lagged target networks and a cloning penalty on the actor."""

    theatre: TheatreConfig
    state_dim: int = 146
    alpha: float = 2.5
    gamma: float = 0.995
    tau: float = 0.005
    batch: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.actor = DeterministicActor(self.state_dim)
        self.actor_target = DeterministicActor(self.state_dim)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic = TwinQ(self.state_dim)
        self.critic_target = TwinQ(self.state_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.actor_optimiser = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_optimiser = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

    def reset(self, batch: int) -> None:
        self.batch = batch

    def act(self, observation: Observation, explore: bool = False) -> Tensor:
        state = flatten_window(observation)
        with torch.no_grad():
            normalised = self.actor(state)
        return pump_action(normalised, self.theatre.propofol_max, self.theatre.remifentanil_max)

    def update(self, batch: TransitionBatch) -> dict[str, float]:
        action = normalise_action(
            batch.action, self.theatre.propofol_max, self.theatre.remifentanil_max
        )
        with torch.no_grad():
            next_action = self.actor_target(batch.next_state)
            target_q = self.critic_target.minimum(batch.next_state, next_action)
            target = batch.reward + self.gamma * (1.0 - batch.done) * target_q
        first, second = self.critic(batch.state, action)
        critic_loss = ((first - target) ** 2).mean() + ((second - target) ** 2).mean()
        self.critic_optimiser.zero_grad()
        backward(critic_loss)
        self.critic_optimiser.step()

        proposed = self.actor(batch.state)
        q_value = self.critic.minimum(batch.state, proposed).mean()
        clone = ((proposed - action) ** 2).mean()
        weight = self.alpha / (self.critic.minimum(batch.state, action).abs().mean() + 1e-6)
        actor_loss = -q_value + weight * clone
        self.actor_optimiser.zero_grad()
        backward(actor_loss)
        self.actor_optimiser.step()

        with torch.no_grad():
            for target_parameter, parameter in zip(
                self.critic_target.parameters(), self.critic.parameters(), strict=True
            ):
                target_parameter.mul_(1.0 - self.tau).add_(self.tau * parameter)
            for target_parameter, parameter in zip(
                self.actor_target.parameters(), self.actor.parameters(), strict=True
            ):
                target_parameter.mul_(1.0 - self.tau).add_(self.tau * parameter)
        return {
            "critic_loss": float(critic_loss.detach()),
            "actor_loss": float(actor_loss.detach()),
            "clone_penalty": float(clone.detach()),
        }
