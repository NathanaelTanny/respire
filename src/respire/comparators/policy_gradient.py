"""On-policy collection and the clipped objective shared by the three policy-gradient arms.

The arms differ only in how they treat the constraint costs, so the collector, the advantage
estimator and the clipped objective live here once. Each arm supplies its own multiplier rule.

Ref: Sec. 2.2 (constrained and robust reinforcement learning), Table 1 (PPO, PPO-Lagr., CPPO).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from respire.comparators.nets import GaussianActor, ValueNet
from respire.settings import ExperimentConfig
from respire.theatre.cohort import make_environment
from respire.theatre.patient import Patient
from respire.utils.autograd import backward


@dataclass
class EpisodeRollout:
    """Flattened on-policy data with the case boundaries kept."""

    state: Tensor
    action: Tensor
    log_prob: Tensor
    reward: Tensor
    cost: Tensor
    value: Tensor
    case: Tensor
    step: Tensor


def collect_rollouts(
    actor: GaussianActor,
    value: ValueNet,
    patients: Patient,
    cfg: ExperimentConfig,
    seed: int,
    noise_scale: float = 0.6,
) -> EpisodeRollout:
    """One batch of episodes with per-step log-probabilities and values under the current actor."""
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    env = make_environment(patients, cfg, seed)
    observation = env.reset()
    batch, steps = env.batch, env.steps
    states = torch.zeros(batch, steps, 144 + 2)
    actions = torch.zeros(batch, steps, 2)
    rewards = torch.zeros(batch, steps)
    costs = torch.zeros(batch, steps, 3)
    log_probs = torch.zeros(batch, steps)
    values = torch.zeros(batch, steps)
    for index in range(steps):
        flat = observation.window.reshape(batch, -1)
        flat = torch.cat([flat, observation.actions[:, -1, :]], dim=-1)
        with torch.no_grad():
            mean, std = actor(flat)
            value_estimate = value(flat)
            noise = torch.randn(mean.shape, generator=generator) * noise_scale
            raw = mean + std * noise
        action = torch.stack([raw[:, 0].clamp(0.0, 12.0), raw[:, 1].clamp(0.0, 0.5)], dim=-1)
        log_prob = (
            -0.5 * (((action - mean) / std) ** 2).sum(dim=-1)
            - std.log().sum(dim=-1)
            - torch.log(torch.tensor(2.0 * torch.pi))
        )
        transition = env.step(action)
        states[:, index] = flat
        actions[:, index] = transition.action
        rewards[:, index] = transition.reward
        costs[:, index, 0] = transition.cost_hypotension
        costs[:, index, 1] = transition.cost_suppression
        costs[:, index, 2] = transition.cost_variability
        log_probs[:, index] = log_prob
        values[:, index] = value_estimate
        observation = transition.observation
    case = torch.arange(batch).unsqueeze(1).expand(batch, steps).reshape(-1)
    step = torch.arange(steps).unsqueeze(0).expand(batch, steps).reshape(-1)
    return EpisodeRollout(
        state=states.reshape(batch * steps, -1),
        action=actions.reshape(batch * steps, 2),
        log_prob=log_probs.reshape(-1),
        reward=rewards.reshape(-1),
        cost=costs.reshape(batch * steps, 3),
        value=values.reshape(-1),
        case=case,
        step=step,
    )


def generalised_advantage(
    rollout: EpisodeRollout, steps: int, gamma: float, lam: float = 0.95
) -> tuple[Tensor, Tensor]:
    """Advantages and discounted returns for each case, computed along its own trajectory."""
    cases = int(rollout.case.max()) + 1
    reward = rollout.reward.reshape(cases, steps)
    value = rollout.value.reshape(cases, steps)
    next_value = torch.zeros(cases, steps)
    next_value[:, :-1] = value[:, 1:]
    delta = reward + gamma * next_value - value
    running = torch.zeros(cases)
    output = torch.zeros(cases, steps)
    for index in range(steps - 1, -1, -1):
        running = delta[:, index] + gamma * lam * running
        output[:, index] = running
    returns = output + value
    return output.reshape(-1), returns.reshape(-1)


def ppo_core(
    actor: GaussianActor,
    value: ValueNet,
    optimiser: torch.optim.Optimizer,
    rollout: EpisodeRollout,
    advantage: Tensor,
    returns: Tensor,
    clip: float = 0.2,
    entropy_weight: float = 0.01,
    cost_advantages: Tensor | None = None,
    cost_multipliers: Tensor | None = None,
) -> dict[str, float]:
    """One epoch of the clipped objective, optionally with multiplier-weighted cost advantages."""
    normalised = (advantage - advantage.mean()) / (advantage.std() + 1e-6)
    mean, std = actor(rollout.state)
    normalised_action = (rollout.action - mean) / std
    log_prob = (
        -0.5 * (normalised_action**2).sum(dim=-1)
        - std.log().sum(dim=-1)
        - torch.log(torch.tensor(2.0 * torch.pi))
    )
    ratio = torch.exp(log_prob - rollout.log_prob)
    objective = torch.minimum(
        ratio * normalised, torch.clamp(ratio, 1.0 - clip, 1.0 + clip) * normalised
    )
    if cost_advantages is not None and cost_multipliers is not None:
        per_constraint = cost_advantages.reshape(cost_advantages.shape[0], -1)
        weights = cost_multipliers.reshape(-1)
        cost_term = (per_constraint * weights.unsqueeze(0)).sum(dim=-1)
        objective = objective - cost_term.reshape(objective.shape)
    policy_loss = -objective.mean() - entropy_weight * std.mean()
    value_prediction = value(rollout.state)
    value_loss = ((value_prediction - returns) ** 2).mean()
    loss = policy_loss + 0.5 * value_loss
    optimiser.zero_grad()
    backward(loss)
    torch.nn.utils.clip_grad_norm_(list(actor.parameters()) + list(value.parameters()), 1.0)
    optimiser.step()
    return {
        "policy_loss": float(policy_loss.detach()),
        "value_loss": float(value_loss.detach()),
        "mean_ratio": float(ratio.mean().detach()),
    }


def cost_advantage(cost: Tensor, steps: int, gamma: float) -> Tensor:
    """Discounted cost-to-go per step, the quantity the constraint multiplies."""
    cases = cost.shape[0] // steps
    shaped = cost.reshape(cases, steps, -1)
    width = shaped.shape[-1]
    running = torch.zeros(cases, width)
    output = torch.zeros(cases, steps, width)
    for index in range(steps - 1, -1, -1):
        running = shaped[:, index] + gamma * running
        output[:, index] = running
    return output.reshape(cases * steps, width)
