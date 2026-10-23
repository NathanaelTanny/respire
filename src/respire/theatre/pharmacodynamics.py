"""Depth-index and haemodynamic surfaces driven by effect-site concentrations.

The depth surface maps the two effect-site occupancies onto the processed EEG index; the
haemodynamic block integrates mean arterial pressure, heart rate, saturation, capnography,
temperature and the train-of-four count with first-order lags, so the shield has a one-step
pressure model to reason about.

Ref: Sec. 3.1 (observation set), Eq. (1)-(2) (depth target, pressure floor).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from respire.settings import PharmacodynamicConfig
from respire.theatre.pharmacokinetics import PatientCovariates


def hill_occupancy(concentration: Tensor, c50: Tensor, hill: float) -> Tensor:
    """Fractional receptor occupancy of a sigmoid Emax surface, per patient."""
    powered = torch.clamp(concentration, min=0.0).pow(hill)
    return powered / (c50.pow(hill) + powered + 1e-9)


def combined_effect(propofol: Tensor, remifentanil: Tensor, synergy: float) -> Tensor:
    """Additive depth effect of the hypnotic and the opioid, clipped to the unit interval."""
    return torch.clamp(propofol + synergy * remifentanil, 0.0, 1.0)


def depth_index(effect: Tensor, cfg: PharmacodynamicConfig) -> Tensor:
    """Processed EEG index, monotone decreasing in the combined effect."""
    span = cfg.bis_awake - cfg.bis_floor
    return cfg.bis_awake - span * effect


@dataclass(frozen=True)
class Effects:
    """Effect-site state read by both surfaces."""

    propofol_ce: Tensor
    propofol_occupancy: Tensor
    remifentanil_occupancy: Tensor
    combined: Tensor


@dataclass(frozen=True)
class HaemodynamicParams:
    """Per-patient gains of the pressure and rate loops."""

    map_baseline: Tensor
    hr_baseline: Tensor
    map_propofol_gain: Tensor
    map_remifentanil_gain: Tensor
    baroreflex: Tensor


@dataclass(frozen=True)
class Vitals:
    """Observed physiological block, one tensor per channel, shape ``[batch]``."""

    map_mmHg: Tensor
    heart_rate: Tensor
    systolic: Tensor
    diastolic: Tensor
    spo2: Tensor
    etco2: Tensor
    temperature: Tensor
    train_of_four: Tensor
    dopamine: Tensor
    propranolol: Tensor
    pneumo_pressure: Tensor


@dataclass(frozen=True)
class SurgicalDrive:
    """Per-step surgical context the physiology reacts to."""

    insufflation_mmHg: Tensor
    trendelenburg: Tensor
    stimulation: Tensor
    blood_loss_ml: Tensor
    rocuronium_block: Tensor


def initial_vitals(
    covariates: PatientCovariates,
    params: HaemodynamicParams,
    cfg: PharmacodynamicConfig,
) -> Vitals:
    """Awake baseline before induction."""
    batch = len(covariates)
    pulse_pressure = 30.0 + 0.1 * (params.map_baseline - cfg.map_baseline)
    return Vitals(
        map_mmHg=params.map_baseline,
        heart_rate=params.hr_baseline,
        systolic=params.map_baseline + 0.66 * pulse_pressure,
        diastolic=params.map_baseline - 0.34 * pulse_pressure,
        spo2=torch.full((batch,), cfg.spo2_baseline, dtype=torch.float32),
        etco2=torch.full((batch,), 38.0, dtype=torch.float32),
        temperature=torch.full((batch,), 36.6, dtype=torch.float32),
        train_of_four=torch.full((batch,), 4.0, dtype=torch.float32),
        dopamine=torch.zeros(batch, dtype=torch.float32),
        propranolol=torch.zeros(batch, dtype=torch.float32),
        pneumo_pressure=torch.zeros(batch, dtype=torch.float32),
    )


def step_vitals(
    vitals: Vitals,
    drive: SurgicalDrive,
    effects: Effects,
    params: HaemodynamicParams,
    cfg: PharmacodynamicConfig,
    dt_min: float,
) -> Vitals:
    """Advance every observed channel by one decision interval."""
    dt_s = dt_min * 60.0
    propofol_depression = params.map_propofol_gain * torch.clamp(effects.propofol_ce - 1.5, min=0.0)
    remifentanil_depression = params.map_remifentanil_gain * effects.remifentanil_occupancy
    insufflation = cfg.map_insufflation_slope * torch.clamp(drive.insufflation_mmHg - 8.0, min=0.0)
    positioning = cfg.map_trendelenburg_gain * drive.trendelenburg
    volume = 0.008 * drive.blood_loss_ml
    stimulation = cfg.map_stimulation_gain * drive.stimulation

    map_target = (
        params.map_baseline
        - propofol_depression
        - remifentanil_depression
        - insufflation
        - positioning
        - volume
        + stimulation
    )
    alpha = 1.0 - math.exp(-dt_s / cfg.map_tau_s)
    map_next = vitals.map_mmHg + alpha * (map_target - vitals.map_mmHg)

    hr_target = (
        params.hr_baseline
        + params.baroreflex * (params.map_baseline - map_next)
        - cfg.hr_remifentanil_slope * effects.remifentanil_occupancy
        + 0.25 * stimulation
    )
    hr_alpha = 1.0 - math.exp(-dt_s / cfg.hr_tau_s)
    hr_next = vitals.heart_rate + hr_alpha * (hr_target - vitals.heart_rate)

    apnoea = torch.clamp(effects.propofol_occupancy - 0.92, min=0.0) / 0.08
    spo2_target = cfg.spo2_baseline - 9.0 * apnoea
    spo2_next = vitals.spo2 + 0.5 * (spo2_target - vitals.spo2)

    etco2_target = cfg.etco2_baseline + 4.0 * torch.clamp(effects.propofol_occupancy - 0.6, min=0.0)
    etco2_target = etco2_target + 1.5 * stimulation
    etco2_next = vitals.etco2 + 0.35 * (etco2_target - vitals.etco2)

    temperature = vitals.temperature - 0.0006 * dt_min
    pulse_pressure = 30.0 + 0.1 * (map_next - params.map_baseline)
    tof = 4.0 * (1.0 - torch.clamp(drive.rocuronium_block, 0.0, 1.0))

    return Vitals(
        map_mmHg=map_next,
        heart_rate=hr_next,
        systolic=map_next + 0.66 * pulse_pressure,
        diastolic=map_next - 0.34 * pulse_pressure,
        spo2=spo2_next,
        etco2=etco2_next,
        temperature=temperature,
        train_of_four=tof,
        dopamine=torch.zeros_like(vitals.dopamine),
        propranolol=torch.zeros_like(vitals.propranolol),
        pneumo_pressure=drive.insufflation_mmHg,
    )
