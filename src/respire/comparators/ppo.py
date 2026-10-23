"""Proximal policy optimisation on the depth-and-pressure reward.

A Gaussian policy and a state value, trained on the clipped objective with the advantage estimated
by generalised advantage over each case's own trajectory. No constraint handling: this is the arm
that buys depth at the cost of pressure.

Ref: Sec. 2.1 (reinforcement learning for propofol dosing), Table 1 (PPO).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor

from respire.comparators.nets import GaussianActor, ValueNet
from respire.comparators.policy_gradient import (
    collect_rollouts,
    generalised_advantage,
    ppo_core,
)
from respire.settings import ExperimentConfig
from respire.theatre.environment import Observation
from respire.theatre.patient import Patient
from respire.utils.seeding import set_seed


@dataclass
class PPOAgent:
    """Clipped-surrogate agent on the flat observation."""

    state_dim: int = 146
    batch: int = field(default=0, init=False)
    noise_scale: float = 0.5

    def __post_init__(self) -> None:
        self.actor = GaussianActor(self.state_dim)
        self.value = ValueNet(self.state_dim)
        self.optimiser = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.value.parameters()), lr=3e-4
        )
        self._generator = torch.Generator(device="cpu")

    def reset(self, batch: int) -> None:
        self.batch = batch

    def act(self, observation: Observation, explore: bool = False) -> Tensor:
        from respire.comparators.base import flatten_window

        state = flatten_window(observation)
        with torch.no_grad():
            mean, std = self.actor(state)
            if explore:
                self._generator.manual_seed(0)
                mean = mean + std * torch.randn(mean.shape, generator=self._generator) * (
                    self.noise_scale
                )
        return torch.stack([mean[:, 0].clamp(0.0, 12.0), mean[:, 1].clamp(0.0, 0.5)], dim=-1)

    def fit(
        self, patients: Patient, cfg: ExperimentConfig, iterations: int, seed: int = 0
    ) -> dict[str, float]:
        return fit_ppo(self, patients, cfg, iterations, seed)


def fit_ppo(
    agent: PPOAgent,
    patients: Patient,
    cfg: ExperimentConfig,
    iterations: int,
    seed: int = 0,
    epochs: int = 4,
    noise_scale: float = 0.5,
) -> dict[str, float]:
    """Alternate collection and clipped updates."""
    set_seed(seed)
    totals: dict[str, float] = {}
    steps = cfg.theatre.n_steps
    for iteration in range(iterations):
        rollout = collect_rollouts(
            agent.actor, agent.value, patients, cfg, seed + iteration, noise_scale
        )
        advantage, returns = generalised_advantage(rollout, steps, cfg.theatre.gamma)
        for _ in range(epochs):
            metrics = ppo_core(
                agent.actor, agent.value, agent.optimiser, rollout, advantage, returns
            )
            for key, value in metrics.items():
                totals[key] = totals.get(key, 0.0) + value
    count = max(iterations * epochs, 1)
    return {key: value / count for key, value in totals.items()}
