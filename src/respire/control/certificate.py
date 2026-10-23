"""Conformal admission certificate.

The residual quantile is the rank of Proposition 1, clamped to the calibration set: with a small
reference split the requested rank exceeds the sample and the quantile collapses onto the largest
residual, which is the only available bound. Coverage is measured on the admitted set every round,
because the guarantee is marginal and only holds for episodes exchangeable with the calibration
draw.

Ref: Eq. (3) (lower confidence bound), Proposition 1 (Eq. (5)), Algorithm 1 lines 6-9.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class CertificateReport:
    """One round's certificate state."""

    quantile: float
    threshold: float
    coverage: float
    standard_error: float
    accepted: bool
    admitted_fraction: float
    reference_size: int

    def as_dict(self) -> dict[str, float | int | bool]:
        return {
            "quantile": self.quantile,
            "threshold": self.threshold,
            "coverage": self.coverage,
            "standard_error": self.standard_error,
            "accepted": self.accepted,
            "admitted_fraction": self.admitted_fraction,
            "reference_size": self.reference_size,
        }


@dataclass(frozen=True)
class ConformalCertificate:
    """Split-conformal lower bound on the true return of an episode."""

    alpha: float = 0.1
    delta: float = 0.05
    coverage_multiple: float = 2.0

    @property
    def target_coverage(self) -> float:
        return 1.0 - self.alpha

    def quantile(self, residuals: Tensor) -> float:
        """The ceil((n+1)(1-alpha))-th smallest absolute residual."""
        if residuals.numel() == 0:
            raise ValueError("calibration residuals are empty")
        ordered = torch.sort(residuals.abs().flatten()).values
        n = int(ordered.numel())
        rank = math.ceil((n + 1) * (1.0 - self.alpha))
        index = min(max(rank, 1), n) - 1
        return float(ordered[index])

    def lower_bound(self, prediction: Tensor, quantile: float) -> Tensor:
        return prediction - quantile

    def threshold(self, lower_bounds: Tensor, admitted_fraction: float) -> float:
        """Admission threshold that keeps the requested share of the pool."""
        if not 0.0 < admitted_fraction <= 1.0:
            raise ValueError("admitted_fraction must lie in (0, 1]")
        return float(torch.quantile(lower_bounds.flatten(), 1.0 - admitted_fraction))

    def coverage(
        self, lower_bounds: Tensor, realised: Tensor, admitted: Tensor
    ) -> tuple[float, float]:
        """Empirical coverage on the admitted set and its binomial standard error."""
        count = int(admitted.sum())
        if count == 0:
            return 0.0, 0.0
        covered = (realised >= lower_bounds)[admitted].to(torch.float32)
        mean = float(covered.mean())
        return mean, math.sqrt(max(mean * (1.0 - mean), 0.0) / count)

    def evaluate(
        self,
        lower_bounds: Tensor,
        realised: Tensor,
        admitted_fraction: float,
        quantile: float,
    ) -> tuple[CertificateReport, Tensor]:
        """Admit, measure coverage and decide whether the round may proceed."""
        threshold = self.threshold(lower_bounds, admitted_fraction)
        admitted = lower_bounds >= threshold
        coverage, se = self.coverage(lower_bounds, realised, admitted)
        accepted = coverage >= self.target_coverage - self.coverage_multiple * se
        report = CertificateReport(
            quantile=quantile,
            threshold=threshold,
            coverage=coverage,
            standard_error=se,
            accepted=accepted,
            admitted_fraction=float(admitted.to(torch.float32).mean()),
            reference_size=int(lower_bounds.numel()),
        )
        return report, admitted
