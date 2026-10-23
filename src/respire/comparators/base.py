"""Shared plumbing for the comparison arms: state flattening, replay and the two drivers.

The drivers are deliberately thin. Everything that distinguishes one row of the comparison table
from another lives in that arm's own update rule, so a difference in the table cannot be a
difference in how many gradient steps were taken.

Ref: Table 1 (the comparison set), Sec. 5.1 (episode protocol).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch
from torch import Tensor

from respire.data.trace import EpisodeTrace
from respire.settings import ExperimentConfig
from respire.theatre.environment import Observation
from respire.theatre.patient import Patient
from respire.theatre.rollout import rollout

STATE_DIM = 146


def flatten_window(observation: Observation) -> Tensor:
    """The comparison arms see the flat window plus the last delivered action."""
    window = observation.window.reshape(observation.window.shape[0], -1)
    return torch.cat([window, observation.actions[:, -1, :]], dim=-1)


@dataclass(frozen=True)
class TransitionBatch:
    """A batch of transitions with the three constraint costs."""

    state: Tensor
    action: Tensor
    reward: Tensor
    next_state: Tensor
    done: Tensor
    costs: Tensor

    @property
    def size(self) -> int:
        return int(self.action.shape[0])

    def subset(self, index: Tensor) -> TransitionBatch:
        return TransitionBatch(
            state=self.state[index],
            action=self.action[index],
            reward=self.reward[index],
            next_state=self.next_state[index],
            done=self.done[index],
            costs=self.costs[index],
        )


def transitions_from_trace(trace: EpisodeTrace, window: int = 12) -> TransitionBatch:
    """Cut rolling windows out of a trace and pair each with the action delivered at its end.

    The comparison arms see the same sixty-second window the encoder sees, flattened, plus the last
    delivered action, so their state width matches ``STATE_DIM``.
    """
    if trace.steps <= window:
        raise ValueError("trace is shorter than one window")
    windows = trace.observation.unfold(1, window, 1)
    count = int(windows.shape[1])
    states = windows.permute(0, 1, 3, 2).reshape(-1, window * trace.channels)
    end = window - 1
    actions = trace.action[:, end : end + count].reshape(-1, 2)
    last = trace.action[:, end - 1 : end - 1 + count].reshape(-1, 2)
    states = torch.cat([states, last], dim=-1)
    following = trace.observation[:, 1:].unfold(1, window, 1)
    next_windows = following.permute(0, 1, 3, 2).reshape(-1, window * trace.channels)
    next_actions = trace.action[:, window : window + count].reshape(-1, 2)
    next_states = torch.cat([next_windows, next_actions], dim=-1)
    costs = torch.stack(
        [
            trace.hypotension_step[:, end : end + count].reshape(-1),
            trace.suppression_step[:, end : end + count].reshape(-1),
            trace.variability_step[:, end : end + count].reshape(-1),
        ],
        dim=-1,
    )
    return TransitionBatch(
        state=states,
        action=actions,
        reward=trace.reward[:, end : end + count].reshape(-1),
        next_state=next_states,
        done=torch.zeros(states.shape[0]),
        costs=costs,
    )


def stack_transitions(chunks: list[TransitionBatch]) -> TransitionBatch:
    return TransitionBatch(
        state=torch.cat([chunk.state for chunk in chunks], dim=0),
        action=torch.cat([chunk.action for chunk in chunks], dim=0),
        reward=torch.cat([chunk.reward for chunk in chunks], dim=0),
        next_state=torch.cat([chunk.next_state for chunk in chunks], dim=0),
        done=torch.cat([chunk.done for chunk in chunks], dim=0),
        costs=torch.cat([chunk.costs for chunk in chunks], dim=0),
    )


class ReplayBuffer:
    """A flat circular buffer of transitions."""

    def __init__(self, capacity: int, state_dim: int = STATE_DIM) -> None:
        self.capacity = capacity
        self.cursor = 0
        self.filled = 0
        self.state = torch.zeros(capacity, state_dim)
        self.action = torch.zeros(capacity, 2)
        self.reward = torch.zeros(capacity)
        self.next_state = torch.zeros(capacity, state_dim)
        self.done = torch.zeros(capacity)
        self.costs = torch.zeros(capacity, 3)

    def add(self, batch: TransitionBatch) -> None:
        count = batch.size
        index = (torch.arange(count) + self.cursor) % self.capacity
        self.state[index] = batch.state
        self.action[index] = batch.action
        self.reward[index] = batch.reward
        self.next_state[index] = batch.next_state
        self.done[index] = batch.done
        self.costs[index] = batch.costs
        self.cursor = int((self.cursor + count) % self.capacity)
        self.filled = min(self.filled + count, self.capacity)

    def sample(self, count: int, generator: torch.Generator) -> TransitionBatch:
        if self.filled == 0:
            raise RuntimeError("replay buffer is empty")
        index = torch.randint(0, self.filled, (count,), generator=generator)
        return TransitionBatch(
            state=self.state[index],
            action=self.action[index],
            reward=self.reward[index],
            next_state=self.next_state[index],
            done=self.done[index],
            costs=self.costs[index],
        )

    def __len__(self) -> int:
        return self.filled


class TrainableComparator(Protocol):
    """An arm that can play an action and take one gradient step on a batch."""

    def act(self, observation: Observation, explore: bool) -> Tensor: ...

    def update(self, batch: TransitionBatch) -> dict[str, float]: ...

    def reset(self, batch: int) -> None: ...


def train_online(
    agent: TrainableComparator,
    cfg: ExperimentConfig,
    patients: Patient,
    episodes: int,
    batch_size: int = 8,
    updates_per_episode: int = 4,
    capacity: int = 200000,
    seed: int = 0,
) -> dict[str, float]:
    """Collect episodes with exploration and take gradient steps on the replay buffer."""
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    buffer = ReplayBuffer(capacity)
    from respire.theatre.cohort import make_environment

    totals: dict[str, float] = {}
    count = 0
    for episode in range(episodes):
        index = torch.randint(
            0, len(patients), (min(batch_size, len(patients)),), generator=generator
        )
        env = make_environment(patients.select(index), cfg, seed + episode)

        class _Explorer:
            def reset(self, batch: int) -> None:
                agent.reset(batch)

            def act(self, observation: Observation) -> Tensor:
                return agent.act(observation, explore=True)

        trace = rollout(env, _Explorer())
        buffer.add(transitions_from_trace(trace))
        if len(buffer) >= 64:
            for _ in range(updates_per_episode):
                metrics = agent.update(buffer.sample(256, generator))
                for key, value in metrics.items():
                    totals[key] = totals.get(key, 0.0) + value
                count += 1
    if count:
        totals = {key: value / count for key, value in totals.items()}
    totals["buffer"] = float(len(buffer))
    return totals


def train_offline(
    agent: TrainableComparator,
    batch: TransitionBatch,
    epochs: int,
    batch_size: int,
    seed: int,
) -> dict[str, float]:
    """Take gradient steps on a fixed transition set."""
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    totals: dict[str, float] = {}
    count = 0
    for _ in range(epochs):
        permutation = torch.randperm(batch.size, generator=generator)
        for start in range(0, batch.size, batch_size):
            index = permutation[start : start + batch_size]
            metrics = agent.update(batch.subset(index))
            for key, value in metrics.items():
                totals[key] = totals.get(key, 0.0) + value
            count += 1
    if count:
        totals = {key: value / count for key, value in totals.items()}
    return totals


def pump_action(normalised: Tensor, propofol_max: float, remifentanil_max: float) -> Tensor:
    """Map a tanh-bounded actor output onto the pump box."""
    return denormalise_action(normalised, propofol_max, remifentanil_max)


def pump_action_from_mean(mean: Tensor, propofol_max: float, remifentanil_max: float) -> Tensor:
    """Map an unbounded actor mean onto the pump box through a tanh, the usual squashing."""
    return denormalise_action(torch.tanh(mean), propofol_max, remifentanil_max)


def normalise_action(action: Tensor, propofol_max: float, remifentanil_max: float) -> Tensor:
    """Map the pump box onto [-1, 1], the range a tanh head produces."""
    return torch.stack(
        [
            2.0 * action[:, 0] / propofol_max - 1.0,
            2.0 * action[:, 1] / remifentanil_max - 1.0,
        ],
        dim=-1,
    )


def denormalise_action(action: Tensor, propofol_max: float, remifentanil_max: float) -> Tensor:
    return torch.stack(
        [
            ((action[:, 0] + 1.0) * 0.5 * propofol_max).clamp(0.0, propofol_max),
            ((action[:, 1] + 1.0) * 0.5 * remifentanil_max).clamp(0.0, remifentanil_max),
        ],
        dim=-1,
    )
