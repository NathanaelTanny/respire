"""Rule-based anaesthetist used as the behaviour source.

The logged action stream of a case is produced by an anaesthetist who titrates on the displayed
depth and reacts to pressure: a proportional-integral law on the depth index with an integral
clamp, a fixed opioid background, and a reduction when the pressure falls. This is the policy a
behaviour-cloning initialiser imitates, and it is the reference the safety comparison reads
against in the cohort.

Ref: Sec. 2.1 (manual titration as the behavioural reference), Sec. 5.1 (cohort).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor

from respire.settings import TheatreConfig
from respire.theatre.environment import Observation


@dataclass
class RuleBasedAnaesthetist:
    """Depth-driven titration with a pressure guard."""

    theatre: TheatreConfig
    propofol_base: float = 5.5
    propofol_gain: float = 0.09
    integral_gain: float = 0.0006
    integral_clamp: float = 200.0
    remifentanil_base: float = 0.15
    pressure_guard: float = 0.55
    guard_release: float = 12.0
    noise_sd: float = 0.0
    integral: Tensor = field(init=False)
    batch: int = field(init=False, default=0)

    def reset(self, batch: int) -> None:
        self.batch = batch
        self.integral = torch.zeros(batch)

    def act(self, observation: Observation) -> Tensor:
        """One two-column action from the current observation window.

        ``observation`` is the environment's :class:`Observation`; only the last row of its window
        is read, which is what a bedside titration does.
        """
        window = observation.window
        if not isinstance(window, Tensor):
            raise TypeError("observation must expose a tensor window")
        if self.batch != int(window.shape[0]):
            self.reset(int(window.shape[0]))
        current = window[:, -1, :]
        bis = current[:, 0]
        map_mmHg = current[:, 2]

        error = bis - self.theatre.bis_target
        self.integral = torch.clamp(
            self.integral + error * self.theatre.dt_s, -self.integral_clamp, self.integral_clamp
        )
        propofol = (
            self.propofol_base + self.propofol_gain * error + self.integral_gain * self.integral
        )
        penalty = torch.clamp((self.theatre.map_floor + self.guard_release - map_mmHg), min=0.0)
        propofol = propofol - self.pressure_guard * penalty / self.guard_release
        if self.noise_sd > 0.0:
            propofol = propofol + torch.randn_like(propofol) * self.noise_sd
        remifentanil = torch.full_like(propofol, self.remifentanil_base)
        return torch.stack(
            [
                propofol.clamp(0.0, self.theatre.propofol_max),
                remifentanil.clamp(0.0, self.theatre.remifentanil_max),
            ],
            dim=-1,
        )


@dataclass
class ConstantRate:
    """Fixed infusion, used to build a drift arm and to seed the replay pool."""

    theatre: TheatreConfig
    propofol: float = 6.0
    remifentanil: float = 0.15

    def act(self, observation: Observation) -> Tensor:
        window = observation.window
        if not isinstance(window, Tensor):
            raise TypeError("observation must expose a tensor window")
        batch = int(window.shape[0])
        return torch.tensor([[self.propofol, self.remifentanil]], dtype=torch.float32).repeat(
            batch, 1
        )
