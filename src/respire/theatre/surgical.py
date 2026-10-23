"""Surgical schedule of a robot-assisted case.

Insufflation, reverse Trendelenburg positioning, stimulation events, blood loss and the relaxant
boluses are all exogenous to the controller; they enter as measured context, which is what makes
the pump an identified intervention rather than a hidden confounder.

Ref: Sec. 1 (pneumoperitoneum, positioning, case length), Sec. 3.1 (12 mmHg insufflation).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from respire.settings import SurgicalConfig


@dataclass(frozen=True)
class Schedule:
    """Per-step surgical context, shape ``[batch, steps]``."""

    insufflation_mmHg: Tensor
    trendelenburg: Tensor
    stimulation: Tensor
    blood_loss_ml: Tensor
    rocuronium_block: Tensor

    @property
    def steps(self) -> int:
        return int(self.insufflation_mmHg.shape[1])

    def index(self, step: int) -> dict[str, Tensor]:
        return {
            "insufflation_mmHg": self.insufflation_mmHg[:, step],
            "trendelenburg": self.trendelenburg[:, step],
            "stimulation": self.stimulation[:, step],
            "blood_loss_ml": self.blood_loss_ml[:, step],
            "rocuronium_block": self.rocuronium_block[:, step],
        }

    def select(self, index: Tensor) -> Schedule:
        return Schedule(
            insufflation_mmHg=self.insufflation_mmHg[index],
            trendelenburg=self.trendelenburg[index],
            stimulation=self.stimulation[index],
            blood_loss_ml=self.blood_loss_ml[index],
            rocuronium_block=self.rocuronium_block[index],
        )


def build_schedule(
    batch: int,
    steps: int,
    dt_s: float,
    cfg: SurgicalConfig,
    seed: int,
) -> Schedule:
    """Draw one schedule per episode.

    Onsets are jittered per case so the controller cannot key on a fixed clock.
    """
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    time_s = torch.arange(steps, dtype=torch.float32) * dt_s
    time_min = time_s / 60.0

    onset = cfg.insufflation_onset_min + 4.0 * torch.rand(batch, 1, generator=gen)
    ramp = torch.clamp((time_min.unsqueeze(0) - onset) / cfg.insufflation_ramp_min, 0.0, 1.0)
    insufflation = cfg.pneumo_pressure_mmHg * ramp

    start = cfg.trendelenburg_onset_min + 6.0 * torch.rand(batch, 1, generator=gen)
    angle = cfg.trendelenburg_fraction * (0.7 + 0.6 * torch.rand(batch, 1, generator=gen))
    positioning = torch.clamp((time_min.unsqueeze(0) - start) / 2.0, 0.0, 1.0) * angle

    stimulation = _stimulation_events(batch, steps, dt_s, cfg, gen)

    loss_start = cfg.blood_loss_onset_min + 8.0 * torch.rand(batch, 1, generator=gen)
    elapsed_min = torch.clamp(time_min.unsqueeze(0) - loss_start, min=0.0)
    loss_rate = cfg.blood_loss_ml_per_min * (0.6 + 0.8 * torch.rand(batch, 1, generator=gen))
    blood_loss = elapsed_min * loss_rate

    block = _relaxant_block(batch, steps, dt_s, cfg, gen)
    return Schedule(
        insufflation_mmHg=insufflation,
        trendelenburg=positioning,
        stimulation=stimulation,
        blood_loss_ml=blood_loss,
        rocuronium_block=block,
    )


def _stimulation_events(
    batch: int, steps: int, dt_s: float, cfg: SurgicalConfig, gen: torch.Generator
) -> Tensor:
    """Sum of decaying stimulation pulses, one train per episode."""
    duration_min = steps * dt_s / 60.0
    expected = max(cfg.stimulation_rate_per_h * duration_min / 60.0, 1.0)
    counts = torch.poisson(torch.full((batch,), expected), generator=gen).clamp(1.0, 24.0)
    time_s = torch.arange(steps, dtype=torch.float32) * dt_s
    onset = torch.rand(batch, 24, generator=gen) * duration_min * 60.0
    amplitude = 1.0 + cfg.stimulation_gain_sd * torch.randn(batch, 24, generator=gen)
    valid = torch.arange(24).unsqueeze(0) < counts.unsqueeze(1)
    delta = time_s.unsqueeze(0).unsqueeze(2) - onset.unsqueeze(1)
    pulse = torch.exp(-torch.clamp(delta, min=0.0) / cfg.stimulation_duration_s)
    pulse = pulse * (delta >= 0.0).to(torch.float32) * valid.unsqueeze(1)
    return (pulse * amplitude.unsqueeze(1)).sum(dim=2).clamp(0.0, 2.5)


def _relaxant_block(
    batch: int, steps: int, dt_s: float, cfg: SurgicalConfig, gen: torch.Generator
) -> Tensor:
    """Train-of-four block from the fixed relaxant boluses of the case."""
    time_min = torch.arange(steps, dtype=torch.float32) * dt_s / 60.0
    jitter = 3.0 * torch.randn(batch, len(cfg.rocuronium_bolus_min), generator=gen)
    onsets = torch.tensor(cfg.rocuronium_bolus_min).unsqueeze(0) + jitter
    delta = time_min.unsqueeze(0).unsqueeze(2) - onsets.unsqueeze(1)
    decay = torch.exp(-torch.clamp(delta, min=0.0) / 28.0) * (delta >= 0.0).to(torch.float32)
    strength = 0.85 + 0.15 * torch.rand(batch, len(cfg.rocuronium_bolus_min), generator=gen)
    return (decay * strength.unsqueeze(1)).amax(dim=2).clamp(0.0, 1.0)
