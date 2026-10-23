"""Stratification of the benefit over the sensitivity axis.

The claim that the recursion repairs the tail rather than the average is a statement about where
the benefit sits on the axis, so the statistic is the share of the total reduction that accrues to
each quartile of the axis, computed per case and then pooled.

Ref: Fig. 1 (error and hypotension by sensitivity percentile), Fig. 5 right (share of total gain).
"""

from __future__ import annotations

import torch
from torch import Tensor

from respire.metrics.statistics import share_by_quartile


def quartile_index(percentile: Tensor, quartiles: int = 4) -> Tensor:
    """Quartile membership of a percentile vector."""
    width = 100.0 / quartiles
    return (percentile / width).floor().clamp(0, quartiles - 1).to(torch.int64)


def per_case_benefit(treated: Tensor, control: Tensor) -> Tensor:
    """Reduction in the metric, positive where the treated arm is better."""
    return control - treated


def benefit_shares(
    treated: Tensor, control: Tensor, percentile: Tensor, quartiles: int = 4
) -> list[float]:
    """Share of the pooled benefit in each quartile of the axis."""
    benefit = per_case_benefit(treated, control)
    membership = quartile_index(percentile, quartiles)
    return share_by_quartile(benefit, membership)


def benefit_by_quartile(
    treated: Tensor, control: Tensor, percentile: Tensor, quartiles: int = 4
) -> list[float]:
    """Mean benefit inside each quartile, which is the quantity the shares are built from."""
    benefit = per_case_benefit(treated, control)
    membership = quartile_index(percentile, quartiles)
    means: list[float] = []
    for index in range(quartiles):
        mask = membership == index
        means.append(float(benefit[mask].mean()) if bool(mask.any()) else float("nan"))
    return means


def tail_contrast(treated: Tensor, control: Tensor, percentile: Tensor) -> tuple[float, float]:
    """Benefit above the 75th percentile and below it."""
    benefit = per_case_benefit(treated, control)
    upper = percentile >= 75.0
    if not bool(upper.any()) or bool(upper.all()):
        return float("nan"), float("nan")
    return float(benefit[upper].mean()), float(benefit[~upper].mean())
