"""Conservative Q-learning with a stochastic actor.

The critic is trained on the Bellman error plus a penalty that pushes down the value of actions the
policy would choose and pushes up the value of the logged action, which is what makes an offline
bootstrap admissible without a behaviour constraint on the actor.

Ref: Table 1 (CQL).
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
class CQLAgent:
    """Twin critic with a conservative penalty and a squashed Gaussian actor."""

    theatre: TheatreConfig
    state_dim: int = 146
    gamma: float = 0.995
    tau: float = 0.005
    conservative: float = 1.0
    action_samples: int = 8
    batch: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.actor = GaussianActor(self.state_dim)
        self.critic = TwinQ(self.state_dim)
        self.critic_target = TwinQ(self.state_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.actor_optimiser = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_optimiser = torch.optim.Adam(self.critic.parameters(), lr=3e-4)
        self.log_alpha = torch.tensor(0.0, requires_grad=True)
        self.alpha_optimiser = torch.optim.Adam([self.log_alpha], lr=3e-4)

    def reset(self, batch: int) -> None:
        self.batch = batch

    def act(self, observation: Observation, explore: bool = False) -> Tensor:
        state = flatten_window(observation)
        with torch.no_grad():
            mean, _ = self.actor(state)
        return pump_action_from_mean(mean, self.theatre.propofol_max, self.theatre.remifentanil_max)

    def update(self, batch: TransitionBatch) -> dict[str, float]:
        action = normalise_action(
            batch.action, self.theatre.propofol_max, self.theatre.remifentanil_max
        )
        with torch.no_grad():
            next_mean, _ = self.actor(batch.next_state)
            next_action = torch.tanh(next_mean)
            target_q = self.critic_target.minimum(batch.next_state, next_action)
            target = batch.reward + self.gamma * (1.0 - batch.done) * target_q
        first, second = self.critic(batch.state, action)
        bellman = ((first - target) ** 2).mean() + ((second - target) ** 2).mean()

        batch_size = batch.state.shape[0]
        random_actions = torch.empty(
            batch_size, self.action_samples, 2, device=batch.state.device
        ).uniform_(-1.0, 1.0)
        expanded_state = batch.state.unsqueeze(1).expand(-1, self.action_samples, -1)
        random_q = self.critic.minimum(
            expanded_state.reshape(batch_size * self.action_samples, -1),
            random_actions.reshape(batch_size * self.action_samples, 2),
        ).reshape(batch_size, self.action_samples)
        with torch.no_grad():
            policy_mean, _ = self.actor(batch.state)
        policy_action = torch.tanh(policy_mean)
        policy_q = self.critic.minimum(batch.state, policy_action)
        conservative = (
            torch.logsumexp(random_q, dim=1).mean()
            - policy_q.mean()
            - self.critic.minimum(batch.state, action).mean()
        )
        critic_loss = bellman + self.conservative * conservative
        self.critic_optimiser.zero_grad()
        backward(critic_loss)
        self.critic_optimiser.step()

        mean, std = self.actor(batch.state)
        sampled = torch.tanh(mean + std * torch.randn_like(mean))
        actor_q = self.critic.minimum(batch.state, sampled)
        alpha = float(self.log_alpha.exp().detach())
        actor_loss = (alpha * -actor_q).mean()
        self.actor_optimiser.zero_grad()
        backward(actor_loss)
        self.actor_optimiser.step()

        with torch.no_grad():
            for target_parameter, parameter in zip(
                self.critic_target.parameters(), self.critic.parameters(), strict=True
            ):
                target_parameter.mul_(1.0 - self.tau).add_(self.tau * parameter)
        return {
            "critic_loss": float(critic_loss.detach()),
            "actor_loss": float(actor_loss.detach()),
            "conservative_term": float(conservative.detach()),
        }
