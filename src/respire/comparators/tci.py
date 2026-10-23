"""Model-based target-controlled infusion arms.

A target-controlled pump holds an effect-site target by inverting a population model. Two published
parameter sets are used, which is why the two rows of the comparison table differ: the pump is the
same, the population is not. The controller never sees the individual's transport constants, only
the population set, which is exactly the mismatch the paper opens with.

Ref: Sec. 2.1 (target-controlled infusion), Table 1 (Schnider TCU, Eleveld TCU).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor

from respire.settings import TheatreConfig
from respire.theatre.environment import Observation


@dataclass(frozen=True)
class PopulationModel:
    """Typical adult transport constants of one published model."""

    name: str
    v1: float
    v2: float
    v3: float
    clearance: float
    q2: float
    q3: float
    ke0: float


SCHNIDER = PopulationModel("schnider", 4.27, 18.9, 238.0, 1.89, 1.29, 0.836, 0.456)
ELEVELD = PopulationModel("eleveld", 6.28, 25.5, 273.0, 1.79, 1.75, 1.11, 0.146)


@dataclass
class TargetControlledInfusion:
    """Effect-site target tracking with a population model and a proportional correction."""

    model: PopulationModel
    theatre: TheatreConfig
    target_ce: float = 3.2
    depth_gain: float = 0.045
    correction_gain: float = 3.0
    remifentanil_rate: float = 0.15
    nominal_weight: float = 70.0
    batch: int = field(default=0, init=False)
    amount: Tensor = field(default_factory=lambda: torch.zeros(0, 3), init=False)
    effect: Tensor = field(default_factory=lambda: torch.zeros(0), init=False)

    def reset(self, batch: int) -> None:
        self.batch = batch
        self.amount = torch.zeros(batch, 3)
        self.effect = torch.zeros(batch)

    def _advance(self, rate: Tensor, dt_min: float) -> None:
        """Transport of the population model under the delivered rate."""
        k10 = self.model.clearance / self.model.v1
        k12 = self.model.q2 / self.model.v1
        k21 = self.model.q2 / self.model.v2
        k13 = self.model.q3 / self.model.v1
        k31 = self.model.q3 / self.model.v3
        central = self.amount[:, 0] + dt_min * (
            rate
            - (k10 + k12 + k13) * self.amount[:, 0]
            + k21 * self.amount[:, 1]
            + k31 * self.amount[:, 2]
        )
        rapid = self.amount[:, 1] + dt_min * (k12 * self.amount[:, 0] - k21 * self.amount[:, 1])
        slow = self.amount[:, 2] + dt_min * (k13 * self.amount[:, 0] - k31 * self.amount[:, 2])
        self.amount = torch.stack([central, rapid, slow], dim=-1)
        target = self.amount[:, 0] / self.model.v1
        self.effect = self.effect + dt_min * self.model.ke0 * (target - self.effect)

    def act(self, observation: Observation) -> Tensor:
        if self.batch != observation.window.shape[0]:
            self.reset(int(observation.window.shape[0]))
        depth = observation.current[:, 0]
        target = torch.clamp(
            self.target_ce + self.depth_gain * (depth - self.theatre.bis_target), 0.4, 8.0
        )
        mass_rate = self.model.clearance * target + self.correction_gain * (target - self.effect)
        mass_rate = mass_rate.clamp(min=0.0)
        rate = (mass_rate * 60.0 / self.nominal_weight).clamp(0.0, self.theatre.propofol_max)
        self._advance(mass_rate, self.theatre.dt_s / 60.0)
        remifentanil = torch.full_like(rate, self.remifentanil_rate)
        return torch.stack([rate, remifentanil], dim=-1)


def schnider_arm(theatre: TheatreConfig) -> TargetControlledInfusion:
    return TargetControlledInfusion(model=SCHNIDER, theatre=theatre)


def eleveld_arm(theatre: TheatreConfig) -> TargetControlledInfusion:
    return TargetControlledInfusion(model=ELEVELD, theatre=theatre)
