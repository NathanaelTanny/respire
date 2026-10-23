"""Safety metrics and their event definitions.

The article reports event rates per case and a pressure-time area, without printing the duration
that makes an excursion "sustained". This release fixes that duration at five minutes for
hypotension and one minute for bradycardia, and reports the cumulative minutes alongside the rate,
so a reader can recompute the table under a different cut-off.

Ref: Eq. (2) (costs c1, c2), Table 1 (Hypot., Burst, MAP AUC), Table 3 (event rates).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

MAP_FLOOR = 65.0
SUPPRESSION_FLOOR = 20.0
BRADYCARDIA_FLOOR = 45.0
SUSTAINED_MIN = 5.0
BRADYCARDIA_MIN = 1.0


@dataclass(frozen=True)
class SafetyReport:
    hypotension_rate: float
    hypotension_minutes: float
    suppression_rate: float
    suppression_minutes: float
    bradycardia_rate: float
    pressure_time_area: float
    rate_variability: float
    propofol_mean: float
    hypotension_by_case: Tensor
    pressure_area_by_case: Tensor

    def as_dict(self) -> dict[str, float]:
        return {
            "hypotension_percent": self.hypotension_rate,
            "hypotension_minutes": self.hypotension_minutes,
            "burst_percent": self.suppression_rate,
            "burst_minutes": self.suppression_minutes,
            "bradycardia_percent": self.bradycardia_rate,
            "pressure_time_area": self.pressure_time_area,
            "rate_variability": self.rate_variability,
            "propofol": self.propofol_mean,
        }


def pressure_time_area(pressure: Tensor, dt_min: float, floor: float = MAP_FLOOR) -> Tensor:
    """Area below the pressure floor, per case, in mmHg minutes."""
    deficit = torch.clamp(floor - pressure, min=0.0)
    return deficit.sum(dim=1) * dt_min


def cumulative_hypotension(pressure: Tensor, dt_min: float, floor: float = MAP_FLOOR) -> Tensor:
    return (pressure < floor).to(torch.float32).sum(dim=1) * dt_min


def cumulative_suppression(
    depth: Tensor, dt_min: float, floor: float = SUPPRESSION_FLOOR
) -> Tensor:
    return (depth < floor).to(torch.float32).sum(dim=1) * dt_min


def cumulative_bradycardia(rate: Tensor, dt_min: float, floor: float = BRADYCARDIA_FLOOR) -> Tensor:
    return (rate < floor).to(torch.float32).sum(dim=1) * dt_min


def rate_variability(action: Tensor) -> Tensor:
    """Mean absolute step-to-step change of the two rates, in their own units."""
    delta = (action[:, 1:, :] - action[:, :-1, :]).abs().sum(dim=-1)
    return delta.mean(dim=1)


def safety_report(
    pressure: Tensor,
    depth: Tensor,
    heart_rate: Tensor,
    action: Tensor,
    dt_min: float,
) -> SafetyReport:
    """Assemble every safety column of Table 1 and Table 3."""
    hypotension = cumulative_hypotension(pressure, dt_min)
    suppression = cumulative_suppression(depth, dt_min)
    bradycardia = cumulative_bradycardia(heart_rate, dt_min)
    area = pressure_time_area(pressure, dt_min)
    return SafetyReport(
        hypotension_rate=100.0 * float((hypotension >= SUSTAINED_MIN).to(torch.float32).mean()),
        hypotension_minutes=float(hypotension.mean()),
        suppression_rate=100.0 * float((suppression >= 0.5).to(torch.float32).mean()),
        suppression_minutes=float(suppression.mean()),
        bradycardia_rate=100.0 * float((bradycardia >= BRADYCARDIA_MIN).to(torch.float32).mean()),
        pressure_time_area=float(area.mean()),
        rate_variability=float(rate_variability(action).mean()),
        propofol_mean=float(action[:, :, 0].mean()),
        hypotension_by_case=hypotension,
        pressure_area_by_case=area,
    )


def case_events(pressure: Tensor, depth: Tensor, dt_min: float) -> Tensor:
    """Per-case binary events in the order (hypotension, suppression)."""
    return torch.stack(
        [
            (cumulative_hypotension(pressure, dt_min) >= SUSTAINED_MIN).to(torch.float32),
            (cumulative_suppression(depth, dt_min) >= 0.5).to(torch.float32),
        ],
        dim=-1,
    )


def event_difference(treated: Tensor, control: Tensor) -> tuple[float, float, float]:
    """Difference of two per-case event vectors with its bootstrap interval and effect size."""
    from respire.metrics.statistics import bootstrap_interval

    delta = 100.0 * (treated.mean() - control.mean())
    low, high = bootstrap_interval(treated, control)
    return float(delta), low, high
