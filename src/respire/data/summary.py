"""Episode summaries the verifier scores.

A summary is a fixed-length read of what a case looked like from the monitor: four order statistics
per channel, the infusion statistics and the two aggregate quantities the return is made of. The
three safety costs are deliberately absent: the verifier exists to bound the return, and the costs
are handled by the dual variables, so letting them into the summary would make the certificate
report on the constraint it is supposed to be checked against.

Ref: Sec. 3.2 (verifier of a trajectory summary), Algorithm 1 line 3.
"""

from __future__ import annotations

import torch
from torch import Tensor

from respire.data.trace import EpisodeTrace

STAT_NAMES: tuple[str, ...] = ("mean", "std", "min", "max")
EXTRA_NAMES: tuple[str, ...] = (
    "propofol_mean",
    "propofol_std",
    "remifentanil_mean",
    "remifentanil_std",
    "duration_min",
    "mean_reward",
)


def summary_dim(channels: int) -> int:
    return channels * len(STAT_NAMES) + len(EXTRA_NAMES)


def summarise(trace: EpisodeTrace) -> Tensor:
    """Build the summary matrix, shape ``[batch, summary_dim]``."""
    observation = trace.observation
    statistics = torch.stack(
        [
            observation.mean(dim=1),
            observation.std(dim=1),
            observation.amin(dim=1),
            observation.amax(dim=1),
        ],
        dim=-1,
    ).reshape(observation.shape[0], -1)
    action = trace.action
    extras = torch.stack(
        [
            action[:, :, 0].mean(dim=1),
            action[:, :, 0].std(dim=1),
            action[:, :, 1].mean(dim=1),
            action[:, :, 1].std(dim=1),
            torch.full((observation.shape[0],), float(trace.steps)),
            trace.reward.mean(dim=1),
        ],
        dim=-1,
    )
    return torch.cat([statistics, extras], dim=-1)


def subset_summary(features: Tensor, index: Tensor) -> Tensor:
    return features[index]
