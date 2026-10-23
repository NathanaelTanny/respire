"""Rectified flow-matching objective with per-episode weights.

The velocity field is regressed on the straight path between noise and the logged action. When the
field is refitted inside the recursion, each episode carries the weight the admission rule gave it,
so the admitted set reweights the objective instead of replacing it.

Ref: Fig. 2 (admitted episodes reweight the flow-matching objective), Eq. (4) (maximisation term).
"""

from __future__ import annotations

import torch
from torch import Tensor

from respire.models.flow import FlowPolicy


def flow_matching_loss(
    policy: FlowPolicy,
    condition: Tensor,
    target: Tensor,
    weights: Tensor | None = None,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Mean squared velocity error along the rectified path."""
    batch = target.shape[0]
    if batch == 0:
        raise ValueError("empty batch")
    unit = policy.to_unit(target)
    time = torch.rand(batch, generator=generator)
    noise = torch.randn(unit.shape, generator=generator)
    interpolated = (1.0 - time).unsqueeze(-1) * noise + time.unsqueeze(-1) * unit
    velocity = policy.velocity(interpolated, time, condition)
    error = ((velocity - (unit - noise)) ** 2).mean(dim=-1)
    if weights is None:
        return error.mean()
    return (error * weights).sum() / torch.clamp(weights.sum(), min=1e-8)


def behaviour_cloning_loss(
    policy: FlowPolicy,
    condition: Tensor,
    action: Tensor,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Uniform-weight flow matching, used to initialise the policy on logged cases."""
    return flow_matching_loss(policy, condition, action, None, generator)
