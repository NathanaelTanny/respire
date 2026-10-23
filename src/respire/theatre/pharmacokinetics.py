"""Mammillary transport for propofol and remifentanil, with an effect compartment.

Four states per drug: central, rapid peripheral, slow peripheral and effect site, all in mass
units. Concentrations are mass over volume, so the effect-site concentration is the state the
pharmacodynamic surface reads.

Ref: Sec. 5.1 (cohort built on the propofol and remifentanil models of refs. [6], [23]).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class PatientCovariates:
    """Per-patient covariates, all as float tensors of shape ``[batch]``."""

    weight_kg: Tensor
    height_cm: Tensor
    age_y: Tensor
    female: Tensor
    sensitivity_clearance: Tensor

    def __len__(self) -> int:
        return int(self.weight_kg.shape[0])


@dataclass(frozen=True)
class CompartmentParameters:
    """Solved rate constants for one drug and one batch of patients."""

    v1: Tensor
    v2: Tensor
    v3: Tensor
    k10: Tensor
    k12: Tensor
    k21: Tensor
    k13: Tensor
    k31: Tensor
    ke0: Tensor

    def as_dict(self) -> dict[str, Tensor]:
        return {
            "v1": self.v1,
            "v2": self.v2,
            "v3": self.v3,
            "k10": self.k10,
            "k12": self.k12,
            "k21": self.k21,
            "k13": self.k13,
            "k31": self.k31,
            "ke0": self.ke0,
        }


def _scaled(value: float, weight: Tensor, age: Tensor, cfg_w: float, cfg_a: float) -> Tensor:
    weight_ratio = weight / 70.0
    age_ratio = age / 35.0
    shape = torch.clamp(weight_ratio, 0.4, 2.5).pow(cfg_w)
    ageing = torch.clamp(age_ratio, 0.4, 2.5).pow(cfg_a)
    return value * shape * ageing


def propofol_parameters(
    covariates: PatientCovariates,
    *,
    v1: float,
    v2: float,
    v3: float,
    cl: float,
    q2: float,
    q3: float,
    ke0: float,
    weight_exponent: float,
    age_exponent: float,
    effect_site_exponent: float,
) -> CompartmentParameters:
    """Eleveld-type adult parameter set with allometric covariate scaling.

    Clearance and the effect-site rate are multiplied by the patient's unobserved sensitivity,
    which is the quantity the encoder is asked to recover (Sec. 1, Sec. 3.1).
    """
    weight = covariates.weight_kg
    age = covariates.age_y
    vol1 = _scaled(v1, weight, age, weight_exponent, 0.0)
    vol2 = _scaled(v2, weight, age, weight_exponent, 0.0)
    vol3 = _scaled(v3, weight, age, weight_exponent, age_exponent)
    clearance = _scaled(cl, weight, age, weight_exponent, age_exponent)
    clearance = clearance * covariates.sensitivity_clearance
    flow2 = _scaled(q2, weight, age, weight_exponent, 0.0)
    flow3 = _scaled(q3, weight, age, weight_exponent, 0.0)
    k10 = clearance / vol1
    return CompartmentParameters(
        v1=vol1,
        v2=vol2,
        v3=vol3,
        k10=k10,
        k12=flow2 / vol1,
        k21=flow2 / vol2,
        k13=flow3 / vol1,
        k31=flow3 / vol3,
        ke0=_scaled(ke0, weight, age, effect_site_exponent, 0.0),
    )


def remifentanil_parameters(
    covariates: PatientCovariates,
    *,
    v1: float,
    v2: float,
    v3: float,
    cl: float,
    q2: float,
    q3: float,
    ke0: float,
    weight_exponent: float,
    age_exponent: float,
) -> CompartmentParameters:
    """Minto-type adult parameter set with the same allometric scaling."""
    weight = covariates.weight_kg
    age = covariates.age_y
    vol1 = _scaled(v1, weight, age, weight_exponent, 0.0)
    vol2 = _scaled(v2, weight, age, weight_exponent, 0.0)
    vol3 = _scaled(v3, weight, age, weight_exponent, age_exponent)
    clearance = _scaled(cl, weight, age, weight_exponent, age_exponent)
    flow2 = _scaled(q2, weight, age, weight_exponent, 0.0)
    flow3 = _scaled(q3, weight, age, weight_exponent, 0.0)
    return CompartmentParameters(
        v1=vol1,
        v2=vol2,
        v3=vol3,
        k10=clearance / vol1,
        k12=flow2 / vol1,
        k21=flow2 / vol2,
        k13=flow3 / vol1,
        k31=flow3 / vol3,
        ke0=_scaled(ke0, weight, age, 0.0, 0.0),
    )


def initial_state(batch: int, dtype: torch.dtype = torch.float32) -> Tensor:
    """Zero mass in every compartment, shape ``[batch, 4]``."""
    return torch.zeros(batch, 4, dtype=dtype)


def derivative(state: Tensor, input_mass: Tensor, params: CompartmentParameters) -> Tensor:
    """Right-hand side of the four-state mammillary system.

    ``input_mass`` is the infused mass per minute entering the central compartment.
    """
    central = state[:, 0]
    rapid = state[:, 1]
    slow = state[:, 2]
    effect = state[:, 3]
    d_central = input_mass - (params.k10 + params.k12 + params.k13) * central
    d_central = d_central + params.k21 * rapid + params.k31 * slow
    d_rapid = params.k12 * central - params.k21 * rapid
    d_slow = params.k13 * central - params.k31 * slow
    d_effect = params.ke0 * (central - effect)
    return torch.stack([d_central, d_rapid, d_slow, d_effect], dim=-1)


def advance(
    state: Tensor,
    input_mass: Tensor,
    params: CompartmentParameters,
    dt_min: float,
    substeps: int,
) -> Tensor:
    """One decision interval of transport, integrated with fixed-step RK4."""
    if substeps < 1:
        raise ValueError("substeps must be at least one")
    step = dt_min / substeps
    current = state
    for _ in range(substeps):
        k1 = derivative(current, input_mass, params)
        k2 = derivative(current + 0.5 * step * k1, input_mass, params)
        k3 = derivative(current + 0.5 * step * k2, input_mass, params)
        k4 = derivative(current + step * k3, input_mass, params)
        current = current + (step / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return current


def concentrations(state: Tensor, params: CompartmentParameters) -> tuple[Tensor, Tensor]:
    """Plasma and effect-site concentrations for a drug with an effect volume equal to V1."""
    plasma = state[:, 0] / params.v1
    effect = state[:, 3] / params.v1
    return plasma, effect


def mass_rate(rate_per_kg: Tensor, weight_kg: Tensor, unit_per_min: float) -> Tensor:
    """Convert a weight-normalised per-hour rate into mass per minute."""
    return rate_per_kg * weight_kg * unit_per_min
