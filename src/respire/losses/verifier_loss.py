"""Heteroscedastic regression loss for the verifier.

The variance head is trained on the same target as the mean, so the certificate's quantile is read
off a model that knows where it is uncertain rather than a constant residual scale.

Ref: Sec. 3.2, Eq. (3) (the quantile is taken over verifier error), Fig. 2 (heteroscedastic head).
"""

from __future__ import annotations

import torch
from torch import Tensor


def gaussian_nll(mean: Tensor, log_variance: Tensor, target: Tensor) -> Tensor:
    """Gaussian negative log-likelihood with a numerically safe variance."""
    variance = torch.exp(log_variance).clamp(min=1e-4)
    return (0.5 * ((target - mean) ** 2) / variance + 0.5 * log_variance).mean()


def absolute_error(mean: Tensor, target: Tensor) -> Tensor:
    return (mean - target).abs()


def calibration_error(mean: Tensor, target: Tensor) -> float:
    """Root mean squared error of the point estimate, the quantity the round reports."""
    return float(torch.sqrt(((mean - target) ** 2).mean()))
