"""Assembling the primary comparison and the safety table.

The primary table is one row per arm. The safety table compares the proposed controller with the
strongest unconstrained arm and with the observational reference, case by case, and reports the
paired difference with a bootstrap interval and a sign-flip p-value.

Ref: Table 1, Table 3, Sec. 6.1 (the cases-per-point conversion).
"""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor

from respire.evaluation.arms import ArmResult
from respire.metrics.safety import SUPPRESSION_FLOOR, SUSTAINED_MIN
from respire.metrics.statistics import bootstrap_interval, sign_flip_pvalue
from respire.settings import EvalConfig


@dataclass(frozen=True)
class EventRow:
    """One row of the safety table."""

    event: str
    treated: float
    control: float
    difference: float
    low: float
    high: float
    p_value: float

    def as_dict(self) -> dict[str, float | str]:
        return {
            "event": self.event,
            "treated": self.treated,
            "control": self.control,
            "difference": self.difference,
            "ci_low": self.low,
            "ci_high": self.high,
            "p_value": self.p_value,
        }


@dataclass(frozen=True)
class ComparisonReport:
    rows: list[dict[str, object]]
    safety: list[EventRow]
    derived: dict[str, float]
    reference: dict[str, object]

    def as_dict(self) -> dict[str, object]:
        return {
            "rows": self.rows,
            "safety": [row.as_dict() for row in self.safety],
            "derived": self.derived,
            "reference": self.reference,
        }


def _rate(values: Tensor, threshold: float) -> float:
    return 100.0 * float((values >= threshold).to(values.dtype).mean())


def _event_row(
    event: str,
    treated: Tensor,
    control: Tensor,
    threshold: float,
    cfg: EvalConfig,
) -> EventRow:
    treated_event = (treated >= threshold).to(treated.dtype)
    control_event = (control >= threshold).to(control.dtype)
    low, high = bootstrap_interval(
        treated_event, control_event, cfg.bootstrap_samples, cfg.bootstrap_alpha, seed=11
    )
    low, high = 100.0 * low, 100.0 * high
    treated_rate = _rate(treated, threshold)
    control_rate = _rate(control, threshold)
    return EventRow(
        event=event,
        treated=treated_rate,
        control=control_rate,
        difference=treated_rate - control_rate,
        low=low,
        high=high,
        p_value=sign_flip_pvalue(
            treated_event,
            control_event,
            cfg.permutation_samples,
            seed=13,
        ),
    )


def build_comparison(
    results: list[ArmResult],
    treated: str,
    control: str,
    reference: str,
    cfg: EvalConfig,
) -> ComparisonReport:
    """Fill the two tables plus the derived quantities the discussion quotes."""
    by_name = {result.name: result for result in results}
    treated_arm = by_name[treated]
    control_arm = by_name[control]
    events = [
        _event_row(
            "sustained hypotension",
            treated_arm.hypotension_by_case,
            control_arm.hypotension_by_case,
            SUSTAINED_MIN,
            cfg,
        ),
        _event_row(
            "burst suppression",
            treated_arm.suppression_by_case,
            control_arm.suppression_by_case,
            0.5,
            cfg,
        ),
        _event_row(
            "bradycardia",
            treated_arm.bradycardia_by_case,
            control_arm.bradycardia_by_case,
            1.0,
            cfg,
        ),
    ]
    events.append(
        EventRow(
            event="pressure-time area",
            treated=treated_arm.pressure_area,
            control=control_arm.pressure_area,
            difference=treated_arm.pressure_area - control_arm.pressure_area,
            low=bootstrap_interval(
                treated_arm.area_by_case,
                control_arm.area_by_case,
                cfg.bootstrap_samples,
                cfg.bootstrap_alpha,
                seed=17,
            )[0],
            high=bootstrap_interval(
                treated_arm.area_by_case,
                control_arm.area_by_case,
                cfg.bootstrap_samples,
                cfg.bootstrap_alpha,
                seed=17,
            )[1],
            p_value=sign_flip_pvalue(
                treated_arm.area_by_case,
                control_arm.area_by_case,
                cfg.permutation_samples,
                seed=19,
            ),
        )
    )
    events.append(
        EventRow(
            event="rate variability",
            treated=treated_arm.rate_variability,
            control=control_arm.rate_variability,
            difference=treated_arm.rate_variability - control_arm.rate_variability,
            low=bootstrap_interval(
                treated_arm.variability_by_case,
                control_arm.variability_by_case,
                cfg.bootstrap_samples,
                cfg.bootstrap_alpha,
                seed=23,
            )[0],
            high=bootstrap_interval(
                treated_arm.variability_by_case,
                control_arm.variability_by_case,
                cfg.bootstrap_samples,
                cfg.bootstrap_alpha,
                seed=23,
            )[1],
            p_value=sign_flip_pvalue(
                treated_arm.variability_by_case,
                control_arm.variability_by_case,
                cfg.permutation_samples,
                seed=29,
            ),
        )
    )

    cases = float(treated_arm.depth_by_case.numel())
    derived = {
        "hypotension_points_saved": control_arm.hypotension - treated_arm.hypotension,
        "cases_per_point": cases * 0.01,
        "cases_saved_hypotension": cases
        * 0.01
        * (control_arm.hypotension - treated_arm.hypotension),
        "cases_saved_suppression": cases * 0.01 * (control_arm.burst - treated_arm.burst),
        "rate_variability_reduction_percent": 100.0
        * (control_arm.rate_variability - treated_arm.rate_variability)
        / max(control_arm.rate_variability, 1e-9),
        "propofol_change_percent": 100.0
        * (treated_arm.propofol - control_arm.propofol)
        / max(control_arm.propofol, 1e-9),
        "depth_gap": control_arm.bis_mae - treated_arm.bis_mae,
        "suppression_floor": SUPPRESSION_FLOOR,
    }
    reference_row = by_name[reference]
    return ComparisonReport(
        rows=[item.as_dict() for item in results],
        safety=events,
        derived=derived,
        reference={"arm": reference, **reference_row.row()},
    )
