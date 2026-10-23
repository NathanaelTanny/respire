"""Discrete PID arm.

The historical depth controller: a proportional, integral and derivative law on the index error,
with the integral clamped so a long induction does not wind the pump up and a derivative term on
the measured slope. The opioid is a fixed background, as it was in the early depth-only systems.

Ref: Sec. 2.1 (proportional-integral-derivative depth control), Table 1 (PID).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor

from respire.settings import TheatreConfig
from respire.theatre.environment import Observation


@dataclass
class PIDDepthController:
    """PID on the depth error, saturated at the pump limits."""

    theatre: TheatreConfig
    kp: float = 0.11
    ki: float = 0.0007
    kd: float = 0.35
    base_rate: float = 5.5
    integral_clamp: float = 250.0
    remifentanil_rate: float = 0.15
    integral: Tensor = field(default_factory=lambda: torch.zeros(0), init=False)
    previous_error: Tensor = field(default_factory=lambda: torch.zeros(0), init=False)
    batch: int = field(default=0, init=False)

    def reset(self, batch: int) -> None:
        self.batch = batch
        self.integral = torch.zeros(batch)
        self.previous_error = torch.zeros(batch)

    def act(self, observation: Observation) -> Tensor:
        if self.batch != observation.window.shape[0]:
            self.reset(int(observation.window.shape[0]))
        depth = observation.current[:, 0]
        error = depth - self.theatre.bis_target
        self.integral = torch.clamp(
            self.integral + error * self.theatre.dt_s, -self.integral_clamp, self.integral_clamp
        )
        derivative = (error - self.previous_error) / self.theatre.dt_s
        self.previous_error = error
        rate = self.base_rate + self.kp * error + self.ki * self.integral + self.kd * derivative
        rate = rate.clamp(0.0, self.theatre.propofol_max)
        return torch.stack([rate, torch.full_like(rate, self.remifentanil_rate)], dim=-1)
