"""Fitting the return verifier on a pool of episodes.

The verifier is refitted every round, so its fit is deliberately short and its capacity is fixed.
The calibration split never enters this loop: it is the reference that produces the quantile.

Ref: Eq. (3) (verifier and its error), Algorithm 1 lines 3 and 11.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from respire.losses.verifier_loss import calibration_error, gaussian_nll
from respire.models.verifier import ReturnVerifier
from respire.settings import VerifierConfig
from respire.utils.autograd import backward
from respire.utils.logging_setup import get_logger

LOGGER = get_logger("training.verifier")


@dataclass(frozen=True)
class VerifierFitReport:
    epochs: int
    samples: int
    train_rmse: float
    validation_rmse: float
    parameter_count: int

    def as_dict(self) -> dict[str, float | int]:
        return {
            "epochs": self.epochs,
            "samples": self.samples,
            "train_rmse": self.train_rmse,
            "validation_rmse": self.validation_rmse,
            "parameter_count": self.parameter_count,
        }


def fit_verifier(
    verifier: ReturnVerifier,
    features: Tensor,
    targets: Tensor,
    cfg: VerifierConfig,
    seed: int,
    validation_fraction: float = 0.2,
) -> VerifierFitReport:
    """Fit the verifier on summary features and discounted returns."""
    if features.shape[0] < 8:
        raise ValueError("verifier needs at least eight episodes to fit")
    torch.manual_seed(seed)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    order = torch.randperm(features.shape[0], generator=generator)
    cut = max(int(features.shape[0] * (1.0 - validation_fraction)), 4)
    train_index = order[:cut]
    validation_index = order[cut:]
    verifier.set_scaler(features[train_index])
    verifier.set_target_scaler(targets[train_index])
    scaled_targets = verifier.scale_target(targets)
    optimiser = torch.optim.Adam(verifier.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    train_features, train_targets = features[train_index], scaled_targets[train_index]
    verifier.train()
    for epoch in range(cfg.epochs):
        permutation = torch.randperm(train_features.shape[0], generator=generator)
        for start in range(0, train_features.shape[0], cfg.batch_size):
            index = permutation[start : start + cfg.batch_size]
            optimiser.zero_grad()
            mean, log_variance = verifier(train_features[index])
            loss = gaussian_nll(mean, log_variance, train_targets[index])
            backward(loss)
            optimiser.step()
        LOGGER.info("verifier epoch %d/%d", epoch + 1, cfg.epochs)
    verifier.eval()
    with torch.no_grad():
        train_mean = verifier.predict(train_features)
        report = VerifierFitReport(
            epochs=cfg.epochs,
            samples=int(features.shape[0]),
            train_rmse=calibration_error(train_mean, targets[train_index]),
            validation_rmse=(
                calibration_error(
                    verifier.predict(features[validation_index]), targets[validation_index]
                )
                if validation_index.numel() > 1
                else float("nan")
            ),
            parameter_count=verifier.parameter_count(),
        )
    return report
