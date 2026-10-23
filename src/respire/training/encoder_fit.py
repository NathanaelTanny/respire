"""Fitting the encoder on the development split.

The encoder is fitted once and then frozen; the recursion never revisits it. Each epoch draws a
fixed-size subsample of the windows, because the joint window set is larger than any round needs.

Ref: Fig. 2 (encoder trained once and frozen), Table 2 (interventional loss ablation).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from respire.data.windows import WindowBatch, sample_windows
from respire.losses.encoder_losses import encoder_objective
from respire.models.encoder import EncoderStandardiser, LatentEncoder
from respire.settings import EncoderConfig
from respire.theatre.probe import ProbeBatch
from respire.utils.autograd import backward
from respire.utils.logging_setup import get_logger

LOGGER = get_logger("training.encoder")


@dataclass(frozen=True)
class EncoderFitReport:
    """What the fit measured, including whether the axes came out aligned with the factors."""

    epochs: int
    windows: int
    probes: int
    terms: dict[str, float]
    axis_correlation: float
    axis_error: float

    def as_dict(self) -> dict[str, float | int]:
        out: dict[str, float | int] = {
            "epochs": self.epochs,
            "windows": self.windows,
            "probes": self.probes,
            "axis_correlation": self.axis_correlation,
            "axis_error": self.axis_error,
        }
        out.update({f"loss_{key}": value for key, value in self.terms.items()})
        return out


def default_weights(cfg: EncoderConfig) -> dict[str, float]:
    return {
        "reconstruction": 1.0,
        "one_step": 1.0,
        "interventional": cfg.interventional_weight,
        "axes": 0.25,
    }


def _correlation(left: Tensor, right: Tensor) -> float:
    if left.numel() < 2:
        return 0.0
    centred_left = left - left.mean()
    centred_right = right - right.mean()
    denominator = float(centred_left.norm() * centred_right.norm())
    if denominator < 1e-8:
        return 0.0
    return float((centred_left * centred_right).sum() / denominator)


def fit_encoder(
    encoder: LatentEncoder,
    standardiser: EncoderStandardiser,
    windows: WindowBatch,
    probes: ProbeBatch,
    cfg: EncoderConfig,
    seed: int,
    weights: dict[str, float] | None = None,
) -> EncoderFitReport:
    """Fit the encoder for the configured number of epochs."""
    torch.manual_seed(seed)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    standardiser.fit(torch.cat([windows.window, probes.window], dim=0))
    applied = weights if weights is not None else default_weights(cfg)
    optimiser = torch.optim.Adam(encoder.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    per_epoch = min(windows.size, cfg.batch_size * 32)
    terms: dict[str, float] = {}
    encoder.train()
    for epoch in range(cfg.epochs):
        epoch_windows = sample_windows(windows, per_epoch, generator)
        order = torch.randperm(epoch_windows.size, generator=generator)
        running: dict[str, float] = {}
        batches = 0
        for start in range(0, epoch_windows.size, cfg.batch_size):
            index = order[start : start + cfg.batch_size]
            chunk = epoch_windows.subset(index)
            optimiser.zero_grad()
            loss = encoder_objective(encoder, standardiser, chunk, probes, applied)
            backward(loss.total)
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
            optimiser.step()
            for key, value in loss.as_dict().items():
                running[key] = running.get(key, 0.0) + value
            batches += 1
        terms = {key: value / max(batches, 1) for key, value in running.items()}
        LOGGER.info("encoder epoch %d/%d loss %.4f", epoch + 1, cfg.epochs, terms["total"])

    encoder.eval()
    with torch.no_grad():
        scaled = standardiser.transform(probes.window)
        output = encoder(scaled, probes.action_history)
        correlation = _correlation(output.axes[:, 0], probes.axes[:, 0])
        error = float(((output.axes - probes.axes) ** 2).mean())
    return EncoderFitReport(
        epochs=cfg.epochs,
        windows=windows.size,
        probes=probes.size,
        terms=terms,
        axis_correlation=correlation,
        axis_error=error,
    )
