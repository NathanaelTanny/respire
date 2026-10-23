"""Patient cohort: covariates, transport constants and the three latent axes.

The cohort is parameterised by three standardised axes, because the encoder is asked to recover
them: how strongly this patient responds to the hypnotic, how strongly the pressure loop responds
to depth, and how much circulatory reserve is left. Sampling is done once per cohort with an
explicit generator so a cohort is reproducible from its seed alone.

Ref: Sec. 1 (three axes), Sec. 5.1 (2{,}000 patients, variability inflated by 1.25).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from respire.settings import CohortConfig, PharmacodynamicConfig, PharmacokineticConfig
from respire.theatre.pharmacodynamics import HaemodynamicParams, hill_occupancy
from respire.theatre.pharmacokinetics import (
    CompartmentParameters,
    PatientCovariates,
    propofol_parameters,
    remifentanil_parameters,
)

# Axis order is fixed: sensitivity to the hypnotic, pressure responsiveness, circulatory reserve.
AXIS_SENSITIVITY = 0
AXIS_RESPONSIVENESS = 1
AXIS_RESERVE = 2


@dataclass(frozen=True)
class Patient:
    """One row per patient; every field is a tensor over the batch."""

    covariates: PatientCovariates
    propofol: CompartmentParameters
    remifentanil: CompartmentParameters
    haemodynamic: HaemodynamicParams
    propofol_c50: Tensor
    remifentanil_c50: Tensor
    axes: Tensor

    def __len__(self) -> int:
        return int(self.axes.shape[0])

    def select(self, index: Tensor) -> Patient:
        """Take a subset of patients, preserving the field structure."""
        covariates = PatientCovariates(
            weight_kg=self.covariates.weight_kg[index],
            height_cm=self.covariates.height_cm[index],
            age_y=self.covariates.age_y[index],
            female=self.covariates.female[index],
            sensitivity_clearance=self.covariates.sensitivity_clearance[index],
        )
        propofol = CompartmentParameters(
            **{name: value[index] for name, value in self.propofol.as_dict().items()}
        )
        remifentanil = CompartmentParameters(
            **{name: value[index] for name, value in self.remifentanil.as_dict().items()}
        )
        haemodynamic = HaemodynamicParams(
            map_baseline=self.haemodynamic.map_baseline[index],
            hr_baseline=self.haemodynamic.hr_baseline[index],
            map_propofol_gain=self.haemodynamic.map_propofol_gain[index],
            map_remifentanil_gain=self.haemodynamic.map_remifentanil_gain[index],
            baroreflex=self.haemodynamic.baroreflex[index],
        )
        return Patient(
            covariates=covariates,
            propofol=propofol,
            remifentanil=remifentanil,
            haemodynamic=haemodynamic,
            propofol_c50=self.propofol_c50[index],
            remifentanil_c50=self.remifentanil_c50[index],
            axes=self.axes[index],
        )


def sample_patients(
    count: int,
    cohort: CohortConfig,
    pk: PharmacokineticConfig,
    pd: PharmacodynamicConfig,
    seed: int,
) -> Patient:
    """Draw a cohort of patients."""
    if count <= 0:
        raise ValueError("count must be positive")
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    spread = cohort.variability

    weight = torch.normal(
        cohort.weight_mean, cohort.weight_sd * spread, size=(count,), generator=gen
    ).clamp(45.0, 135.0)
    height = torch.normal(
        cohort.height_mean, cohort.height_sd * spread, size=(count,), generator=gen
    ).clamp(145.0, 200.0)
    age = torch.normal(cohort.age_mean, cohort.age_sd, size=(count,), generator=gen).clamp(
        18.0, 88.0
    )
    female = (torch.rand(count, generator=gen) < cohort.female_fraction).to(torch.float32)
    axes = torch.randn(count, 3, generator=gen) * (cohort.sensitivity_sd * spread)

    clearance_noise = torch.exp(0.18 * torch.randn(count, generator=gen) * spread)
    covariates = PatientCovariates(
        weight_kg=weight,
        height_cm=height,
        age_y=age,
        female=female,
        sensitivity_clearance=clearance_noise,
    )
    propofol = propofol_parameters(
        covariates,
        v1=pk.propofol_v1,
        v2=pk.propofol_v2,
        v3=pk.propofol_v3,
        cl=pk.propofol_cl,
        q2=pk.propofol_q2,
        q3=pk.propofol_q3,
        ke0=pk.propofol_ke0,
        weight_exponent=pk.weight_exponent,
        age_exponent=pk.age_exponent,
        effect_site_exponent=pk.effect_site_exponent,
    )
    remifentanil = remifentanil_parameters(
        covariates,
        v1=pk.remifentanil_v1,
        v2=pk.remifentanil_v2,
        v3=pk.remifentanil_v3,
        cl=pk.remifentanil_cl,
        q2=pk.remifentanil_q2,
        q3=pk.remifentanil_q3,
        ke0=pk.remifentanil_ke0,
        weight_exponent=pk.weight_exponent,
        age_exponent=pk.age_exponent,
    )

    sensitivity = axes[:, AXIS_SENSITIVITY]
    responsiveness = axes[:, AXIS_RESPONSIVENESS]
    reserve = axes[:, AXIS_RESERVE]

    propofol_c50 = pd.propofol_c50 * torch.exp(-0.30 * sensitivity)
    haemodynamic = HaemodynamicParams(
        map_baseline=pd.map_baseline + 0.12 * (age - 50.0) - 3.0 * reserve,
        hr_baseline=pd.hr_baseline - 0.08 * (age - 50.0),
        map_propofol_gain=pd.map_propofol_slope * torch.exp(0.25 * responsiveness),
        map_remifentanil_gain=pd.map_remifentanil_slope * torch.exp(0.15 * reserve),
        baroreflex=pd.hr_baroreflex * torch.exp(0.20 * reserve),
    )
    return Patient(
        covariates=covariates,
        propofol=propofol,
        remifentanil=remifentanil,
        haemodynamic=haemodynamic,
        propofol_c50=propofol_c50,
        remifentanil_c50=torch.full((count,), pd.remifentanil_c50),
        axes=axes,
    )


def occupancy(
    patient: Patient, propofol_ce: Tensor, remifentanil_ce: Tensor
) -> tuple[Tensor, Tensor]:
    """Effect-site occupancies of both drugs for a batch of patients."""
    return (
        hill_occupancy(propofol_ce, patient.propofol_c50, 1.5),
        hill_occupancy(remifentanil_ce, patient.remifentanil_c50, 2.0),
    )


def sensitivity_percentile(patient: Patient) -> Tensor:
    """Rank of each patient on the sensitivity axis, in percent; used by the stratified panels."""
    axis = patient.axes[:, AXIS_SENSITIVITY]
    order = torch.argsort(torch.argsort(axis))
    return 100.0 * order.to(torch.float32) / max(len(patient) - 1, 1)


def axis_quartile(patient: Patient) -> Tensor:
    """Quartile membership on the sensitivity axis, 0 being the least sensitive."""
    return (sensitivity_percentile(patient) / 25.0).floor().clamp(0.0, 3.0).to(torch.int64)


def split_indices(count: int, fraction: float, seed: int) -> tuple[Tensor, Tensor]:
    """Deterministic two-way split of a patient index range."""
    if not 0.0 < fraction < 1.0:
        raise ValueError("fraction must lie strictly between zero and one")
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    order = torch.randperm(count, generator=gen)
    cut = int(math.floor(count * fraction))
    return order[:cut], order[cut:]
