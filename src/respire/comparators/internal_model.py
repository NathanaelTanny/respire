"""Internal-model control arm.

An identified first-order model from infusion rate to depth index is inverted every step: the rate
that would bring the prediction to the target inside one horizon is applied, and the model state is
updated with the realised depth. This is the strongest of the model-based arms in the table.

Ref: Sec. 2.1 (internal-model compensation), Table 1 (Internal model).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor

from respire.settings import TheatreConfig
from respire.theatre.environment import Observation


@dataclass
class InternalModelController:
    """One-step inversion of a fitted gain and time constant."""

    theatre: TheatreConfig
    gain: float = 4.6
    time_constant_s: float = 55.0
    remifentanil_rate: float = 0.15
    disturbance_gain: float = 0.55
    rate: Tensor = field(default_factory=lambda: torch.zeros(0), init=False)
    predicted_depth: Tensor = field(default_factory=lambda: torch.zeros(0), init=False)
    batch: int = field(default=0, init=False)

    def reset(self, batch: int) -> None:
        self.batch = batch
        self.rate = torch.full((batch,), 5.5)
        self.predicted_depth = torch.full((batch,), 92.0)

    def act(self, observation: Observation) -> Tensor:
        if self.batch != observation.window.shape[0]:
            self.reset(int(observation.window.shape[0]))
        depth = observation.current[:, 0]
        pressure = observation.current[:, 2]
        dt = self.theatre.dt_s
        self.predicted_depth = self.predicted_depth + (dt / self.time_constant_s) * (
            92.0 - self.gain * self.rate - self.predicted_depth
        )
        model_error = depth - self.predicted_depth
        self.predicted_depth = self.predicted_depth + 0.35 * model_error
        target_rate = (self.predicted_depth - self.theatre.bis_target) / max(self.gain, 1e-3)
        pressure_relief = self.disturbance_gain * torch.clamp(
            self.theatre.map_floor + 6.0 - pressure, min=0.0
        )
        self.rate = (target_rate - pressure_relief).clamp(0.0, self.theatre.propofol_max)
        return torch.stack([self.rate, torch.full_like(self.rate, self.remifentanil_rate)], dim=-1)
