"""Cohort construction and the three splits the protocol needs.

The development split trains the encoder, the pressure model and behaviour cloning; the
calibration split supplies the reference residuals of the certificate; the evaluation split is
read only at reporting time.

Ref: Sec. 5.1 (2{,}000 patients), Algorithm 1 lines 1-3 (development and reference splits).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from respire.settings import ExperimentConfig
from respire.theatre.environment import AnaesthesiaEnv
from respire.theatre.patient import Patient, sample_patients, sensitivity_percentile, split_indices
from respire.theatre.surgical import Schedule, build_schedule

DEVELOPMENT_FRACTION = 0.6
CALIBRATION_FRACTION = 0.2


@dataclass(frozen=True)
class CohortSplits:
    """Disjoint patient index sets."""

    development: Tensor
    calibration: Tensor
    evaluation: Tensor

    def named(self) -> dict[str, Tensor]:
        return {
            "development": self.development,
            "calibration": self.calibration,
            "evaluation": self.evaluation,
        }

    def get(self, name: str) -> Tensor:
        mapping = self.named()
        if name not in mapping:
            raise KeyError(f"unknown split {name!r}; expected one of {sorted(mapping)}")
        return mapping[name]


@dataclass(frozen=True)
class Cohort:
    """A sampled cohort with its splits."""

    patients: Patient
    splits: CohortSplits

    @property
    def size(self) -> int:
        return len(self.patients)

    def subset(self, split: str) -> Patient:
        return self.patients.select(self.splits.get(split))


def build_cohort(cfg: ExperimentConfig) -> Cohort:
    """Sample the cohort and carve the splits."""
    patients = sample_patients(
        cfg.cohort.n_patients,
        cfg.cohort,
        cfg.pharmacokinetics,
        cfg.pharmacodynamics,
        cfg.cohort.seed,
    )
    holdout, rest = split_indices(cfg.cohort.n_patients, CALIBRATION_FRACTION, cfg.cohort.seed + 1)
    share = DEVELOPMENT_FRACTION / (1.0 - CALIBRATION_FRACTION)
    development, evaluation = split_indices(int(rest.shape[0]), share, cfg.cohort.seed + 2)
    return Cohort(
        patients=patients,
        splits=CohortSplits(
            development=rest[development],
            calibration=holdout,
            evaluation=rest[evaluation],
        ),
    )


def episode_patients(cohort: Cohort, split: str, count: int, seed: int) -> Patient:
    """Draw ``count`` patients with replacement from a split."""
    pool = cohort.splits.get(split)
    if pool.numel() == 0:
        raise ValueError(f"split {split!r} is empty")
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    picks = pool[torch.randint(0, pool.numel(), (count,), generator=gen)]
    return cohort.patients.select(picks)


def make_environment(
    patients: Patient, cfg: ExperimentConfig, seed: int, schedule: Schedule | None = None
) -> AnaesthesiaEnv:
    """Pair a patient batch with a surgical schedule."""
    drawn = schedule
    if drawn is None:
        drawn = build_schedule(
            len(patients), cfg.theatre.n_steps, cfg.theatre.dt_s, cfg.surgical, seed
        )
    return AnaesthesiaEnv(patients, cfg.theatre, cfg.pharmacodynamics, drawn, seed)


def axes_of(patients: Patient) -> Tensor:
    return patients.axes.clone()


def percentiles_of(patients: Patient) -> Tensor:
    return sensitivity_percentile(patients)
