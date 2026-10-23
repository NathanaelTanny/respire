"""Depth metrics.

Two quantities, both per case and then averaged: the absolute error against the target index and
the share of the case spent inside the therapeutic band. The band is an engineering default, since
the article reports the metric without printing its bounds; forty to sixty index units is the
interval this release uses.

Ref: Eq. (1) (target index 50), Table 1 (BIS MAE, TIB).
"""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor

BIS_TARGET = 50.0
BAND_LOW = 40.0
BAND_HIGH = 60.0


@dataclass(frozen=True)
class DepthReport:
    mae: float
    mae_by_case: Tensor
    time_in_band: float
    band_by_case: Tensor

    def as_dict(self) -> dict[str, float]:
        return {"bis_mae": self.mae, "tib_percent": self.time_in_band}


def absolute_error(bis: Tensor, target: float = BIS_TARGET) -> Tensor:
    return (bis - target).abs()


def case_mae(bis: Tensor, target: float = BIS_TARGET) -> Tensor:
    """Mean absolute error per case; ``bis`` is ``[batch, steps]``."""
    return absolute_error(bis, target).mean(dim=1)


def in_band(bis: Tensor, low: float = BAND_LOW, high: float = BAND_HIGH) -> Tensor:
    return ((bis >= low) & (bis <= high)).to(bis.dtype)


def time_in_band(bis: Tensor, low: float = BAND_LOW, high: float = BAND_HIGH) -> Tensor:
    """Share of the case inside the band, per case."""
    return in_band(bis, low, high).mean(dim=1)


def depth_report(bis: Tensor) -> DepthReport:
    per_case = case_mae(bis)
    band = time_in_band(bis)
    return DepthReport(
        mae=float(per_case.mean()),
        mae_by_case=per_case,
        time_in_band=100.0 * float(band.mean()),
        band_by_case=100.0 * band,
    )


def tail_mae(bis: Tensor, percentile: Tensor, threshold: float) -> tuple[float, float]:
    """Depth error in the upper tail of the sensitivity axis against the rest."""
    selected = percentile >= threshold
    if not bool(selected.any()) or bool(selected.all()):
        return float("nan"), float("nan")
    per_case = case_mae(bis)
    return float(per_case[selected].mean()), float(per_case[~selected].mean())


def noise_floor_ratio(mae: float, floor: float) -> float:
    """How far the achieved error sits above the measurement noise of the index."""
    if floor <= 0.0:
        raise ValueError("noise floor must be positive")
    return mae / floor


def band_share(bis: Tensor, steps: int) -> Tensor:
    """Fraction of steps in band, kept for the per-quartile panels."""
    return in_band(bis).sum(dim=1) / max(steps, 1)
