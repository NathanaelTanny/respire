"""Evaluating one controller on a held-out batch of cases.

Produces the columns of the primary comparison: the two utility metrics, the four constraint
metrics, the hypnotic consumption, and the per-case vectors the paired statistics need.

Ref: Table 1 (column set), Sec. 5.1 (held-out protocol).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from respire.data.trace import EpisodeTrace
from respire.metrics.depth import depth_report
from respire.metrics.safety import (
    cumulative_bradycardia,
    rate_variability,
    safety_report,
)
from respire.settings import ExperimentConfig
from respire.theatre.cohort import make_environment
from respire.theatre.patient import Patient
from respire.theatre.rollout import ActingPolicy, rollout

CHANNEL_HEART_RATE = 1


@dataclass(frozen=True)
class ArmResult:
    """One row of the comparison table plus its per-case vectors."""

    name: str
    bis_mae: float
    tib: float
    hypotension: float
    burst: float
    pressure_area: float
    propofol: float
    rate_variability: float
    depth_by_case: Tensor
    hypotension_by_case: Tensor
    suppression_by_case: Tensor
    area_by_case: Tensor
    bradycardia_by_case: Tensor
    variability_by_case: Tensor
    sensitivity_pct: Tensor
    trace: EpisodeTrace

    def row(self) -> dict[str, float]:
        return {
            "bis_mae": self.bis_mae,
            "tib": self.tib,
            "hypotension": self.hypotension,
            "burst": self.burst,
            "pressure_area": self.pressure_area,
            "propofol": self.propofol,
            "rate_variability": self.rate_variability,
        }

    def as_dict(self) -> dict[str, object]:
        return {"arm": self.name, **self.row()}


def evaluate_arm(
    name: str,
    policy: ActingPolicy,
    patients: Patient,
    cfg: ExperimentConfig,
    seed: int,
    chunk: int | None = None,
) -> ArmResult:
    """Roll the policy over the patient batch and read the metrics off the trace."""
    env = make_environment(patients, cfg, seed)
    trace = rollout(env, policy) if chunk is None else _chunked(env, policy, chunk)
    depth = depth_report(trace.bis)
    dt_min = cfg.theatre.dt_s / 60.0
    safety = safety_report(
        trace.map_mmHg,
        trace.bis,
        trace.observation[:, :, CHANNEL_HEART_RATE],
        trace.action,
        dt_min,
    )
    return ArmResult(
        name=name,
        bis_mae=depth.mae,
        tib=depth.time_in_band,
        hypotension=safety.hypotension_rate,
        burst=safety.suppression_rate,
        pressure_area=safety.pressure_time_area,
        propofol=safety.propofol_mean,
        rate_variability=safety.rate_variability,
        depth_by_case=depth.mae_by_case,
        hypotension_by_case=safety.hypotension_by_case,
        suppression_by_case=(trace.bis < 20.0).to(torch.float32).sum(dim=1) * dt_min,
        area_by_case=safety.pressure_area_by_case,
        bradycardia_by_case=cumulative_bradycardia(
            trace.observation[:, :, CHANNEL_HEART_RATE], dt_min
        ),
        variability_by_case=rate_variability(trace.action),
        sensitivity_pct=trace.sensitivity_pct,
        trace=trace,
    )


def _chunked(env: object, policy: ActingPolicy, chunk: int) -> EpisodeTrace:
    from respire.theatre.environment import AnaesthesiaEnv
    from respire.theatre.rollout import rollout_in_chunks

    if not isinstance(env, AnaesthesiaEnv):
        raise TypeError("expected an AnaesthesiaEnv")
    return rollout_in_chunks(env, policy, chunk)


def arm_table(results: list[ArmResult]) -> list[dict[str, object]]:
    return [result.as_dict() for result in results]


def best_unconstrained(results: list[ArmResult], avoid: tuple[str, ...] = ()) -> ArmResult:
    """The unconstrained arm with the lowest depth error, the reference of the safety margins."""
    candidates = [item for item in results if item.name not in avoid]
    if not candidates:
        raise ValueError("no candidate arms")
    return min(candidates, key=lambda item: item.bis_mae)
