"""Return verifier with a heteroscedastic head.

The verifier is deliberately small, because the certificate depends on its error rather than on
its capacity, and it is refitted every round on the admitted episodes. Two outputs: the expected
discounted return of an episode summary and the log of its variance, which the calibration split
turns into a one-sided quantile.

Ref: Sec. 3.2 (verifier definition, Eq. (3)), Algorithm 1 lines 3 and 11, Fig. 2 (0.4M parameters).
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from respire.settings import VerifierConfig


class ReturnVerifier(nn.Module):
    """Episode summary to (mean, log-variance)."""

    def __init__(self, cfg: VerifierConfig, summary_dim: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.summary_dim = summary_dim
        self.register_buffer("feature_mean", torch.zeros(1, summary_dim))
        self.register_buffer("feature_std", torch.ones(1, summary_dim))
        self.register_buffer("target_mean", torch.zeros(1))
        self.register_buffer("target_std", torch.ones(1))
        first, second = cfg.hidden
        self.embed = nn.Linear(summary_dim, cfg.feature_dim)
        self.trunk = nn.Sequential(
            nn.Linear(cfg.feature_dim, first),
            nn.SiLU(),
            nn.Linear(first, second),
            nn.SiLU(),
        )
        self.mean_head = nn.Linear(second, 1)
        self.log_variance_head = nn.Linear(second, 1)

    def set_scaler(self, features: Tensor) -> None:
        """Freeze the summary statistics of the first pool the verifier sees."""
        mean = features.mean(dim=0, keepdim=True)
        std = features.std(dim=0, keepdim=True).clamp(min=1e-3)
        self.feature_mean = mean
        self.feature_std = std

    def standardise(self, features: Tensor) -> Tensor:
        return (features - self.feature_mean) / self.feature_std

    def set_target_scaler(self, targets: Tensor) -> None:
        """Returns are regressed in standardised units; the bound is reported in return units."""
        self.target_mean = targets.mean().reshape(1)
        self.target_std = targets.std().reshape(1).clamp(min=1e-3)

    def scale_target(self, targets: Tensor) -> Tensor:
        return (targets - self.target_mean.squeeze()) / self.target_std.squeeze()

    def unscale(self, values: Tensor) -> Tensor:
        return values * self.target_std.squeeze() + self.target_mean.squeeze()

    def forward(self, features: Tensor) -> tuple[Tensor, Tensor]:
        scaled = self.standardise(features)
        hidden = self.trunk(torch.nn.functional.silu(self.embed(scaled)))
        mean = self.mean_head(hidden).squeeze(-1)
        log_variance = self.log_variance_head(hidden).squeeze(-1).clamp(-8.0, 6.0)
        return mean, log_variance

    def predict(self, features: Tensor) -> Tensor:
        mean, _ = self.forward(features)
        return self.unscale(mean)

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
