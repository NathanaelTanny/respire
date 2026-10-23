"""Fitting the twin critics on admitted episodes.

The critic is what makes the constrained step differentiable: the maximisation term of the operator
is realised as the gradient of a return critic at the policy's own samples, and the three safety
costs are read from the same object. The return column is held in standardised units, the cost
columns in the units of Eq. (2), so a multiplier of one means one minute.

Ref: Eq. (2) (costs), Eq. (4) (constrained maximisation).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from respire.data.trace import EpisodeTrace
from respire.data.windows import build_windows
from respire.models.critic import TwinCritic
from respire.models.encoder import EncoderStandardiser, LatentEncoder
from respire.utils.autograd import backward
from respire.utils.logging_setup import get_logger
from respire.utils.numerics import discounted_returns_all

LOGGER = get_logger("training.critic")


@dataclass(frozen=True)
class StepDataset:
    """Per-decision features with their Monte-Carlo targets."""

    condition: Tensor
    action: Tensor
    return_target: Tensor
    cost_targets: Tensor

    @property
    def size(self) -> int:
        return int(self.action.shape[0])

    def subset(self, index: Tensor) -> StepDataset:
        return StepDataset(
            condition=self.condition[index],
            action=self.action[index],
            return_target=self.return_target[index],
            cost_targets=self.cost_targets[index],
        )


@dataclass(frozen=True)
class CriticFitReport:
    epochs: int
    samples: int
    train_rmse: float
    validation_rmse: float
    return_mean: float
    return_std: float

    def as_dict(self) -> dict[str, float | int]:
        return {
            "epochs": self.epochs,
            "samples": self.samples,
            "train_rmse": self.train_rmse,
            "validation_rmse": self.validation_rmse,
            "return_mean": self.return_mean,
            "return_std": self.return_std,
        }


def build_step_dataset(
    encoder: LatentEncoder,
    standardiser: EncoderStandardiser,
    traces: list[EpisodeTrace],
    window_steps: int,
    stride: int,
    gamma: float,
) -> StepDataset:
    """Assemble a step dataset from one or more traces, in the order the windows were cut."""
    conditions: list[Tensor] = []
    actions: list[Tensor] = []
    returns: list[Tensor] = []
    costs: list[Tensor] = []
    for trace in traces:
        windows = build_windows(trace, window_steps, stride)
        with torch.no_grad():
            chunked: list[Tensor] = []
            for start in range(0, windows.size, 512):
                stop = min(start + 512, windows.size)
                chunk = windows.subset(torch.arange(start, stop))
                output = encoder(standardiser.transform(chunk.window), chunk.actions)
                chunked.append(torch.cat([output.latent, output.hidden], dim=-1))
        starts = torch.arange(0, trace.steps - window_steps, stride)
        to_go = discounted_returns_all(trace.reward, gamma)[:, starts].reshape(-1)
        per_step = torch.stack(
            [
                _reverse_cumsum(trace.hypotension_step[:, starts]),
                _reverse_cumsum(trace.suppression_step[:, starts]),
                _reverse_cumsum(trace.variability_step[:, starts]),
            ],
            dim=-1,
        )
        conditions.append(torch.cat(chunked, dim=0))
        actions.append(windows.actions[:, -1, :])
        returns.append(to_go)
        costs.append(per_step.reshape(-1, 3))
    return StepDataset(
        condition=torch.cat(conditions, dim=0),
        action=torch.cat(actions, dim=0),
        return_target=torch.cat(returns, dim=0),
        cost_targets=torch.cat(costs, dim=0),
    )


def _reverse_cumsum(values: Tensor) -> Tensor:
    return torch.flip(torch.cumsum(torch.flip(values, dims=[1]), dim=1), dims=[1])


def fit_critics(
    dataset: StepDataset,
    input_dim: int,
    seed: int,
    epochs: int = 6,
    batch_size: int = 512,
    learning_rate: float = 1e-3,
    hidden: int = 128,
) -> tuple[TwinCritic, CriticFitReport]:
    """Regress the two critics on the return and the three costs."""
    torch.manual_seed(seed)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    critic = TwinCritic(input_dim, 4, hidden)
    raw_targets = torch.cat([dataset.return_target.unsqueeze(-1), dataset.cost_targets], dim=-1)
    critic.set_target_scaler(raw_targets)
    targets = critic.scale_targets(raw_targets)
    mean = float(critic.target_mean[0])
    std = float(critic.target_std[0])
    order = torch.randperm(dataset.size, generator=generator)
    cut = max(int(dataset.size * 0.8), 4)
    train_index, validation_index = order[:cut], order[cut:]
    optimiser = torch.optim.Adam(critic.parameters(), lr=learning_rate)
    for epoch in range(epochs):
        permutation = torch.randperm(train_index.numel(), generator=generator)
        total = 0.0
        batches = 0
        for start in range(0, train_index.numel(), batch_size):
            index = train_index[permutation[start : start + batch_size]]
            features = torch.cat([dataset.condition[index], dataset.action[index]], dim=-1)
            first, second = critic(features)
            loss = ((first - targets[index]) ** 2).mean() + ((second - targets[index]) ** 2).mean()
            optimiser.zero_grad()
            backward(loss)
            optimiser.step()
            total += float(loss.detach())
            batches += 1
        LOGGER.info("critic epoch %d/%d loss %.4f", epoch + 1, epochs, total / max(batches, 1))

    with torch.no_grad():
        train_features = torch.cat(
            [dataset.condition[train_index], dataset.action[train_index]], dim=-1
        )
        train_rmse = _column_rmse(
            critic.conservative(train_features)[:, 0], targets[train_index][:, 0]
        )
        if validation_index.numel() > 1:
            validation_features = torch.cat(
                [dataset.condition[validation_index], dataset.action[validation_index]], dim=-1
            )
            validation_rmse = _column_rmse(
                critic.conservative(validation_features)[:, 0], targets[validation_index][:, 0]
            )
        else:
            validation_rmse = float("nan")
    return critic, CriticFitReport(
        epochs=epochs,
        samples=dataset.size,
        train_rmse=train_rmse,
        validation_rmse=validation_rmse,
        return_mean=mean,
        return_std=std,
    )


def _column_rmse(prediction: Tensor, target: Tensor) -> float:
    return float(torch.sqrt(((prediction - target) ** 2).mean()))
