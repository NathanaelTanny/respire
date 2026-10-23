"""Implicit Q-learning with advantage-weighted regression.

No policy action is ever evaluated by the critic. The value is fitted to an expectile of the
critic's own estimate at the logged actions, and the actor is then regressed towards the logged
action with a weight that grows with the advantage, which keeps the update inside the data.

Ref: Table 1 (IQL).
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
from respire.comparators.nets import DeterministicActor, TwinQ, ValueNet
from respire.settings import TheatreConfig
from respire.theatre.environment import Observation
from respire.utils.autograd import backward


@dataclass
class IQLAgent:
    """Expectile value, twin critic and advantage-weighted actor."""

    theatre: TheatreConfig
    state_dim: int = 146
    gamma: float = 0.995
    tau: float = 0.7
    beta: float = 3.0
    batch: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.actor = DeterministicActor(self.state_dim)
        self.value = ValueNet(self.state_dim)
        self.critic = TwinQ(self.state_dim)
        self.value_optimiser = torch.optim.Adam(self.value.parameters(), lr=3e-4)
        self.critic_optimiser = torch.optim.Adam(self.critic.parameters(), lr=3e-4)
        self.actor_optimiser = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

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
            next_value = self.value(batch.next_state)
            target = batch.reward + self.gamma * (1.0 - batch.done) * next_value
        value_prediction = self.value(batch.state)
        difference = target - value_prediction
        expectile_weight = torch.where(
            difference > 0.0,
            torch.full_like(difference, self.tau),
            torch.full_like(difference, 1.0 - self.tau),
        )
        value_loss = (expectile_weight * difference**2).mean()
        self.value_optimiser.zero_grad()
        backward(value_loss)
        self.value_optimiser.step()

        first, second = self.critic(batch.state, action)
        critic_loss = ((first - target) ** 2).mean() + ((second - target) ** 2).mean()
        self.critic_optimiser.zero_grad()
        backward(critic_loss)
        self.critic_optimiser.step()

        with torch.no_grad():
            q_value = self.critic.minimum(batch.state, action)
            advantage = q_value - self.value(batch.state)
            weights = torch.exp(self.beta * advantage).clamp(max=100.0)
        proposed = self.actor(batch.state)
        actor_loss = (weights * ((proposed - action) ** 2).sum(dim=-1)).mean()
        self.actor_optimiser.zero_grad()
        backward(actor_loss)
        self.actor_optimiser.step()
        return {
            "value_loss": float(value_loss.detach()),
            "critic_loss": float(critic_loss.detach()),
            "actor_loss": float(actor_loss.detach()),
            "mean_weight": float(weights.mean().detach()),
        }
