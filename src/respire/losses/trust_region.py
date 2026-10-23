"""One-shot trust region around the previous policy.

The KL is estimated between two flows that share a base distribution: the updated policy's own
samples give the left-hand density by the change-of-variables identity, and the previous policy
scores the same actions by integrating its own field backwards. If the estimate exceeds the radius
the step is backtracked by interpolating the parameters towards the previous ones, which is the
one-shot trust region of the operator rather than a penalty.

Ref: Eq. (4) (KL penalty and KL <= B constraint), Proposition 2 (why the region must bind).
"""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor, nn

from respire.models.flow import FlowPolicy


def estimate_kl(
    updated: FlowPolicy,
    previous: FlowPolicy,
    condition: Tensor,
    samples: int,
    generator: torch.Generator | None = None,
) -> float:
    """Monte-Carlo estimate of KL(updated || previous) from updated-policy samples."""
    if samples <= 0:
        return 0.0
    index = torch.randperm(condition.shape[0], generator=generator)[:samples]
    subset = condition[index]
    with torch.no_grad():
        drawn = updated.sample(subset, generator)
        left = drawn.log_prob
        right = previous.log_prob(drawn.action, subset)
    return float((left - right).mean())


def interpolate_to(
    updated: FlowPolicy, previous_state: Mapping[str, Tensor], fraction: float
) -> None:
    """Move the parameters a fraction of the way back towards the previous iterate."""
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("fraction must lie in [0, 1]")
    with torch.no_grad():
        for name, parameter in updated.state_dict().items():
            reference = previous_state[name]
            parameter.copy_(reference + (parameter - reference) * (1.0 - fraction))


def backtrack(
    updated: FlowPolicy,
    previous: FlowPolicy,
    previous_state: Mapping[str, Tensor],
    condition: Tensor,
    budget: float,
    samples: int,
    steps: int,
    generator: torch.Generator | None = None,
) -> tuple[float, int]:
    """Halve the step until the estimated KL fits the radius; return (kl, steps used)."""
    estimate = estimate_kl(updated, previous, condition, samples, generator)
    used = 0
    while estimate > budget and used < steps:
        fraction = 0.5 ** (used + 1)
        interpolate_to(updated, previous_state, fraction)
        estimate = estimate_kl(updated, previous, condition, samples, generator)
        used += 1
    return estimate, used


def snapshot(module: nn.Module) -> dict[str, Tensor]:
    """Detached copy of a module's parameters, for the trust-region reference."""
    return {name: value.detach().clone() for name, value in module.state_dict().items()}
