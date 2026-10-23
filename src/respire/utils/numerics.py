"""Small numeric helpers used across the theatre, the losses and the metrics."""

from __future__ import annotations

import math

import numpy as np
import torch


def softplus_inverse(value: float) -> float:
    """Inverse of softplus, used to initialise the selective-SSM time scale."""
    return math.log(math.expm1(value)) if value > 1e-6 else math.log(value)


def safe_div(numerator: float, denominator: float, fallback: float = 0.0) -> float:
    if denominator == 0.0 or not math.isfinite(denominator):
        return fallback
    return numerator / denominator


def quantile(values: np.ndarray, level: float) -> float:
    """Linear-interpolated quantile on a flattened array."""
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    if flat.size == 0:
        raise ValueError("quantile of an empty array")
    return float(np.quantile(flat, level))


def conformal_index(n: int, alpha: float) -> int:
    """The ceil((n+1)(1-alpha)) rank of Proposition 1, clamped to the sample.

    The clamp matters: with small calibration sets the requested rank exceeds n, and the quantile
    then collapses onto the maximum residual rather than raising.
    """
    if n <= 0:
        raise ValueError("calibration set must be non-empty")
    rank = math.ceil((n + 1) * (1.0 - alpha))
    return min(max(rank, 1), n)


def conformal_quantile(residuals: np.ndarray, alpha: float) -> float:
    """The rank-based quantile used by the admission certificate (Eq. (3), Prop. 1)."""
    flat = np.sort(np.abs(np.asarray(residuals, dtype=np.float64)).reshape(-1))
    return float(flat[conformal_index(flat.size, alpha) - 1])


def normalised_rates(
    propofol: torch.Tensor, remifentanil: torch.Tensor, propofol_scale: float, remi_scale: float
) -> torch.Tensor:
    """Both rates on a comparable scale so a single L1 norm is meaningful."""
    return torch.stack([propofol / propofol_scale, remifentanil / remi_scale], dim=-1)


def discounted_return(rewards: torch.Tensor, gamma: float) -> torch.Tensor:
    """Discounted sum along the last axis, computed without a Python loop over time."""
    if rewards.ndim != 2:
        raise ValueError("rewards must be [batch, steps]")
    steps = rewards.shape[1]
    powers = torch.pow(
        torch.tensor(gamma, dtype=rewards.dtype, device=rewards.device),
        torch.arange(steps, dtype=rewards.dtype, device=rewards.device),
    )
    return (rewards * powers).sum(dim=1)


def discounted_returns_all(rewards: torch.Tensor, gamma: float) -> torch.Tensor:
    """Per-step discounted return-to-go, used as a verifier regression target."""
    out = torch.zeros_like(rewards)
    running = torch.zeros(rewards.shape[0], dtype=rewards.dtype, device=rewards.device)
    for index in range(rewards.shape[1] - 1, -1, -1):
        running = rewards[:, index] + gamma * running
        out[:, index] = running
    return out


def ema(values: list[float], decay: float) -> float:
    if not values:
        return 0.0
    acc = values[0]
    for value in values[1:]:
        acc = decay * acc + (1.0 - decay) * value
    return acc


class RunningScale:
    """Running mean and variance for observation standardisation."""

    def __init__(self, dim: int) -> None:
        self.count = 0
        self.mean = np.zeros(dim, dtype=np.float64)
        self.m2 = np.ones(dim, dtype=np.float64)

    def update(self, batch: np.ndarray) -> None:
        flat: np.ndarray = np.asarray(batch, dtype=np.float64).reshape(-1, self.mean.size)
        for row in flat:
            self.count += 1
            delta = row - self.mean
            self.mean += delta / self.count
            self.m2 += delta * (row - self.mean)

    @property
    def std(self) -> np.ndarray:
        var = self.m2 / max(self.count - 1, 1)
        return np.sqrt(np.maximum(var, 1e-6))

    def standardise(self, batch: np.ndarray) -> np.ndarray:
        scaled: np.ndarray = (np.asarray(batch, dtype=np.float64) - self.mean) / self.std
        return scaled

    def state(self) -> dict[str, list[float] | int]:
        return {
            "count": self.count,
            "mean": self.mean.tolist(),
            "m2": self.m2.tolist(),
        }

    @classmethod
    def from_state(cls, state: dict[str, object]) -> RunningScale:
        mean = np.asarray(state["mean"], dtype=np.float64)
        scale = cls(mean.size)
        scale.count = int(str(state["count"]))
        scale.mean = mean
        scale.m2 = np.asarray(state["m2"], dtype=np.float64)
        return scale
