"""Resampling statistics for the paired case-level comparisons.

Both intervals are percentile bootstraps over cases, and the p-value is a paired sign-flip
permutation, which is the test that matches a within-case comparison of two controllers on the same
226 records.

Ref: Table 3 (bootstrap percentile intervals over cases, p-values).
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor


def _as_numpy(values: Tensor) -> np.ndarray:
    return values.detach().cpu().numpy().astype(np.float64).reshape(-1)


def bootstrap_interval(
    treated: Tensor, control: Tensor, samples: int = 2000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float]:
    """Percentile interval of the mean difference, resampling cases with replacement."""
    left = _as_numpy(treated)
    right = _as_numpy(control)
    if left.size != right.size:
        raise ValueError("paired comparison needs equal case counts")
    if left.size == 0:
        return float("nan"), float("nan")
    generator = np.random.default_rng(seed)
    index = generator.integers(0, left.size, size=(samples, left.size))
    differences = (left[index] - right[index]).mean(axis=1)
    return float(np.quantile(differences, alpha / 2.0)), float(
        np.quantile(differences, 1.0 - alpha / 2.0)
    )


def sign_flip_pvalue(treated: Tensor, control: Tensor, samples: int = 5000, seed: int = 0) -> float:
    """Two-sided paired permutation p-value on the mean difference."""
    difference = _as_numpy(treated) - _as_numpy(control)
    if difference.size == 0:
        return float("nan")
    observed = abs(difference.mean())
    generator = np.random.default_rng(seed)
    signs = generator.choice([-1.0, 1.0], size=(samples, difference.size))
    permuted = np.abs((signs * difference).mean(axis=1))
    return float((permuted >= observed).mean())


def paired_difference(treated: Tensor, control: Tensor) -> float:
    return float(treated.mean() - control.mean())


def welch_interval(
    values: Tensor, samples: int = 2000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float]:
    """Percentile interval of a single mean, for the one-sample columns."""
    data = _as_numpy(values)
    if data.size == 0:
        return float("nan"), float("nan")
    generator = np.random.default_rng(seed)
    index = generator.integers(0, data.size, size=(samples, data.size))
    means = data[index].mean(axis=1)
    return float(np.quantile(means, alpha / 2.0)), float(np.quantile(means, 1.0 - alpha / 2.0))


def pearson(left: Tensor, right: Tensor) -> float:
    """Correlation between two per-case vectors."""
    a = _as_numpy(left)
    b = _as_numpy(right)
    if a.size < 2:
        return float("nan")
    a_centred = a - a.mean()
    b_centred = b - b.mean()
    denominator = float(np.sqrt((a_centred**2).sum() * (b_centred**2).sum()))
    if denominator < 1e-12:
        return float("nan")
    return float((a_centred * b_centred).sum() / denominator)


def spearman(left: Tensor, right: Tensor) -> float:
    """Rank correlation, robust to the monotone warping of a latent axis."""
    a = _as_numpy(left)
    b = _as_numpy(right)
    if a.size < 2:
        return float("nan")
    ranks_a = np.argsort(np.argsort(a)).astype(np.float64)
    ranks_b = np.argsort(np.argsort(b)).astype(np.float64)
    return pearson(torch.from_numpy(ranks_a), torch.from_numpy(ranks_b))


def share_by_quartile(values: Tensor, quartile: Tensor) -> list[float]:
    """Share of the total of ``values`` that falls in each quartile of the axis."""
    total = float(values.clamp(min=0.0).sum())
    shares: list[float] = []
    for index in range(4):
        mask = quartile == index
        part = float(values[mask].clamp(min=0.0).sum()) if bool(mask.any()) else 0.0
        shares.append(100.0 * part / total if total > 0.0 else float("nan"))
    return shares
