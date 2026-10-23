"""The three closed-form claims of Sec. 3, evaluated as arithmetic rather than quoted.

Every function here is checkable by hand: the sample size of Corollary 1, the one-round slack of
Theorem 1 and the two-step recursion of Proposition 2. The release runs them on the configured
constants and reports the numbers next to the ones the article prints.

Ref: Eq. (6) (sample size), Eq. (7) (one-round bound), Proposition 2 (uncalibrated admission).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from torch import Tensor


def corollary_sample_size(R: float, delta: float, epsilon: float, target: float) -> float:
    """N_k >= 2 R^2 log(4/delta) / (g - 2 epsilon)^2, and infinity when g <= 2 epsilon."""
    if target <= 2.0 * epsilon:
        return float("inf")
    return 2.0 * R**2 * math.log(4.0 / delta) / (target - 2.0 * epsilon) ** 2


@dataclass(frozen=True)
class RoundBound:
    """Right-hand side of Eq. (7) and the two terms it is made of."""

    calibration_term: float
    sampling_term: float
    coverage_term: float
    total: float


def theorem_bound(epsilon: float, R: float, N: int, delta: float, delta_cov: float) -> RoundBound:
    """The three terms of the monotone-improvement bound."""
    if N <= 0:
        raise ValueError("N must be positive")
    calibration = 2.0 * epsilon
    sampling = 2.0 * R * math.sqrt(math.log(4.0 / delta) / (2.0 * N))
    coverage = 2.0 * R * delta_cov
    return RoundBound(
        calibration_term=calibration,
        sampling_term=sampling,
        coverage_term=coverage,
        total=calibration + sampling + coverage,
    )


def monotone_holds(
    previous_return: float,
    updated_return: float,
    bound: RoundBound,
) -> bool:
    """Whether the realised step obeys Eq. (7)."""
    return updated_return >= previous_return - bound.total


def cumulative_bound(bound: RoundBound, rounds: int) -> float:
    """Eq. (8): K times the per-round allowance."""
    if rounds < 0:
        raise ValueError("rounds must be non-negative")
    return rounds * bound.total


@dataclass(frozen=True)
class Proposition2Trajectory:
    """Iterates of the biased-verifier recursion with both readings of the reported value."""

    actions: list[float]
    reported_quadratic: list[float]
    reported_printed: list[float]
    true: list[float]
    bias: float


def proposition2_trajectory(bias: float, initial: float, rounds: int) -> Proposition2Trajectory:
    """Point-estimate admission under the multiplicative bias of Proposition 2.

    The verifier is ``V_hat(a) = V(a) + c a_k a`` with ``V(a) = -a^2``. Its maximiser satisfies
    ``a_{k+1} = (c/2) a_k`` and the value reported there is ``(c^2/4) a_k^2``, which is positive and
    grows exactly when ``c > 2``. The article prints ``(c-1) a_k^2`` for the same quantity and reads
    the growth condition off it as ``c > 1``; the two expressions coincide only at ``c = 2``.
    """
    if initial <= 0.0:
        raise ValueError("initial action must be positive")
    actions = [initial]
    action = initial
    for _ in range(rounds):
        action = bias * action / 2.0
        actions.append(action)
    reported_quadratic = [(bias**2 / 4.0) * value**2 for value in actions]
    reported_printed = [(bias - 1.0) * value**2 for value in actions]
    true = [-(value**2) for value in actions]
    return Proposition2Trajectory(
        actions=actions,
        reported_quadratic=reported_quadratic,
        reported_printed=reported_printed,
        true=true,
        bias=bias,
    )


def proposition2_growth_threshold(bias: float, initial: float = 1.0) -> bool:
    """Whether the exactly derived reported value grows from the first iterate."""
    trajectory = proposition2_trajectory(bias, initial, 2)
    return trajectory.reported_quadratic[1] > trajectory.reported_quadratic[0]


def propofol_ratio(treated: Tensor, control: Tensor) -> float:
    """Relative change of the mean hypnotic rate, in percent."""
    base = float(control.mean())
    if abs(base) < 1e-9:
        return float("nan")
    return 100.0 * (float(treated.mean()) - base) / base


def rate_variability_reduction(treated: Tensor, control: Tensor) -> float:
    """Percent reduction of a variability measure."""
    base = float(control.mean())
    if abs(base) < 1e-9:
        return float("nan")
    return 100.0 * (base - float(treated.mean())) / base
