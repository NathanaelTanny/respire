"""Lagrangian PPO arms.

Two ways of putting the costs into a policy gradient. The single-multiplier arm aggregates the
three costs against their budgets and raises one price; the constrained arm keeps one multiplier per
constraint, forms a separate cost advantage for each, and projects the multipliers after every
update. The second is the arm that closes part of the safety gap without a certificate.

Ref: Sec. 2.2 (constrained reinforcement learning), Table 1 (PPO-Lagr., CPPO).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor

from respire.comparators.nets import GaussianActor, ValueNet
from respire.comparators.policy_gradient import (
    collect_rollouts,
    cost_advantage,
    generalised_advantage,
    ppo_core,
)
from respire.settings import ExperimentConfig
from respire.theatre.environment import Observation
from respire.theatre.patient import Patient
from respire.utils.seeding import set_seed


@dataclass
class LagrangianAgent:
    """PPO with one price on the budget-normalised aggregate cost."""

    budget: tuple[float, float, float] = (10.0, 1.0, 60.0)
    state_dim: int = 146
    multiplier: float = 0.1
    dual_lr: float = 0.05
    batch: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.actor = GaussianActor(self.state_dim)
        self.value = ValueNet(self.state_dim)
        self.optimiser = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.value.parameters()), lr=3e-4
        )

    def reset(self, batch: int) -> None:
        self.batch = batch

    def act(self, observation: Observation, explore: bool = False) -> Tensor:
        from respire.comparators.base import flatten_window

        state = flatten_window(observation)
        with torch.no_grad():
            mean, _ = self.actor(state)
        return torch.stack([mean[:, 0].clamp(0.0, 12.0), mean[:, 1].clamp(0.0, 0.5)], dim=-1)

    def aggregate(self, cost: Tensor) -> Tensor:
        weights = torch.tensor([1.0 / self.budget[0], 1.0 / self.budget[1], 1.0 / self.budget[2]])
        return cost @ weights

    def fit(
        self, patients: Patient, cfg: ExperimentConfig, iterations: int, seed: int = 0
    ) -> dict[str, float]:
        set_seed(seed)
        steps = cfg.theatre.n_steps
        totals: dict[str, float] = {}
        for iteration in range(iterations):
            rollout = collect_rollouts(self.actor, self.value, patients, cfg, seed + iteration, 0.5)
            advantage, returns = generalised_advantage(rollout, steps, cfg.theatre.gamma)
            aggregate = self.aggregate(rollout.cost)
            shaped = cost_advantage(aggregate, steps, cfg.theatre.gamma)
            penalty = torch.clamp(shaped - 1.0, min=0.0)
            metrics = ppo_core(
                self.actor,
                self.value,
                self.optimiser,
                rollout,
                advantage,
                returns,
                cost_advantages=penalty.unsqueeze(-1),
                cost_multipliers=torch.tensor([self.multiplier]),
            )
            violation = float(aggregate.mean()) - 1.0
            self.multiplier = max(self.multiplier + self.dual_lr * violation, 0.0)
            for key, value in metrics.items():
                totals[key] = totals.get(key, 0.0) + value
        count = max(iterations, 1)
        out = {key: value / count for key, value in totals.items()}
        out["multiplier"] = self.multiplier
        return out


@dataclass
class ConstrainedPPOAgent:
    """PPO with one multiplier per constraint and per-constraint cost advantages."""

    budget: tuple[float, float, float] = (10.0, 1.0, 60.0)
    state_dim: int = 146
    multipliers: Tensor = field(default_factory=lambda: torch.full((3,), 0.1))
    dual_lr: float = 0.05
    batch: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.actor = GaussianActor(self.state_dim)
        self.value = ValueNet(self.state_dim)
        self.optimiser = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.value.parameters()), lr=3e-4
        )

    def reset(self, batch: int) -> None:
        self.batch = batch

    def act(self, observation: Observation, explore: bool = False) -> Tensor:
        from respire.comparators.base import flatten_window

        state = flatten_window(observation)
        with torch.no_grad():
            mean, _ = self.actor(state)
        return torch.stack([mean[:, 0].clamp(0.0, 12.0), mean[:, 1].clamp(0.0, 0.5)], dim=-1)

    def project(self) -> None:
        self.multipliers = torch.clamp(self.multipliers, 0.0, 50.0)

    def fit(
        self, patients: Patient, cfg: ExperimentConfig, iterations: int, seed: int = 0
    ) -> dict[str, float]:
        set_seed(seed)
        steps = cfg.theatre.n_steps
        budget = torch.tensor(self.budget)
        totals: dict[str, float] = {}
        for iteration in range(iterations):
            rollout = collect_rollouts(self.actor, self.value, patients, cfg, seed + iteration, 0.5)
            advantage, returns = generalised_advantage(rollout, steps, cfg.theatre.gamma)
            cost_to_go = torch.stack(
                [
                    cost_advantage(rollout.cost[:, index], steps, cfg.theatre.gamma)
                    for index in range(3)
                ],
                dim=-1,
            )
            excess = (cost_to_go - budget.unsqueeze(0)).clamp(min=0.0)
            metrics = ppo_core(
                self.actor,
                self.value,
                self.optimiser,
                rollout,
                advantage,
                returns,
                cost_advantages=excess,
                cost_multipliers=self.multipliers,
            )
            measured = rollout.cost.mean(dim=0)
            self.multipliers = torch.clamp(
                self.multipliers + self.dual_lr * (measured - budget), min=0.0
            )
            self.project()
            for key, value in metrics.items():
                totals[key] = totals.get(key, 0.0) + value
        count = max(iterations, 1)
        out = {key: value / count for key, value in totals.items()}
        out.update(
            {f"multiplier_{index}": float(value) for index, value in enumerate(self.multipliers)}
        )
        return out


def lagrangian_arm() -> LagrangianAgent:
    return LagrangianAgent()


def constrained_arm() -> ConstrainedPPOAgent:
    return ConstrainedPPOAgent()
