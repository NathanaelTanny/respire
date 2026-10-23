"""Transport and depth surface.

The checks are conservation and monotonicity rather than numbers: a compartmental model that loses
mass, or a depth surface that rises with the effect-site concentration, would be wrong regardless of
the parameter set.
"""

from __future__ import annotations

import pytest
import torch

from respire.settings import PharmacodynamicConfig, PharmacokineticConfig
from respire.theatre.patient import sample_patients
from respire.theatre.pharmacodynamics import combined_effect, depth_index, hill_occupancy
from respire.theatre.pharmacokinetics import (
    CompartmentParameters,
    advance,
    concentrations,
    derivative,
    initial_state,
    mass_rate,
)


def _parameters(batch: int, v1: float = 6.28, cl: float = 1.79) -> CompartmentParameters:
    return CompartmentParameters(
        v1=torch.full((batch,), v1),
        v2=torch.full((batch,), 25.5),
        v3=torch.full((batch,), 273.0),
        k10=torch.full((batch,), cl / v1),
        k12=torch.full((batch,), 1.75 / v1),
        k21=torch.full((batch,), 1.75 / 25.5),
        k13=torch.full((batch,), 1.11 / v1),
        k31=torch.full((batch,), 1.11 / 273.0),
        ke0=torch.full((batch,), 0.146),
    )


def test_zero_input_decays_to_zero() -> None:
    parameters = _parameters(4)
    state = initial_state(4) + torch.tensor([10.0, 2.0, 5.0, 0.0])
    for _ in range(3000):
        state = advance(state, torch.zeros(4), parameters, 1.0, 1)
    assert torch.all(state.abs() < 1e-2)


def test_mass_is_conserved_when_nothing_is_cleared() -> None:
    parameters = CompartmentParameters(
        v1=torch.full((3,), 6.28),
        v2=torch.full((3,), 25.5),
        v3=torch.full((3,), 273.0),
        k10=torch.zeros(3),
        k12=torch.full((3,), 1.75 / 6.28),
        k21=torch.full((3,), 1.75 / 25.5),
        k13=torch.full((3,), 1.11 / 6.28),
        k31=torch.full((3,), 1.11 / 273.0),
        ke0=torch.zeros(3),
    )
    state = initial_state(3) + torch.tensor([5.0, 0.0, 0.0, 0.0])
    initial_mass = state.sum(dim=1)
    for _ in range(200):
        state = advance(state, torch.zeros(3), parameters, 0.5, 4)
    assert torch.allclose(state.sum(dim=1), initial_mass, atol=1e-3)


def test_effect_site_lags_the_central_compartment() -> None:
    parameters = _parameters(2)
    state = initial_state(2)
    rate = torch.full((2,), 7.0 / 60.0)
    for _ in range(6):
        state = advance(state, rate, parameters, 1.0, 4)
    plasma, effect = concentrations(state, parameters)
    assert bool((plasma > effect).all())
    assert bool((effect > 0).all())


def test_derivative_direction_is_negative_without_input() -> None:
    parameters = _parameters(2)
    state = initial_state(2) + torch.tensor([4.0, 0.0, 0.0, 0.0])
    delta = derivative(state, torch.zeros(2), parameters)
    assert bool((delta[:, 0] < 0).all())


def test_mass_rate_units() -> None:
    weight = torch.tensor([70.0])
    rate = torch.tensor([6.0])
    assert torch.allclose(mass_rate(rate, weight, 1.0 / 60.0), torch.tensor([7.0]))


def test_occupancy_is_monotone_and_bounded() -> None:
    c50 = torch.tensor([3.0])
    low = hill_occupancy(torch.tensor([1.0]), c50, 1.5)
    high = hill_occupancy(torch.tensor([6.0]), c50, 1.5)
    assert float(low) < float(high) < 1.0
    assert float(hill_occupancy(torch.tensor([3.0]), c50, 1.5)) == pytest.approx(0.5, abs=1e-6)


def test_depth_index_falls_with_effect() -> None:
    cfg = PharmacodynamicConfig()
    effect = torch.linspace(0.0, 1.0, 5)
    depth = depth_index(effect, cfg)
    assert bool((depth[1:] < depth[:-1]).all())
    assert float(depth[0]) == pytest.approx(cfg.bis_awake)
    assert float(depth[-1]) == pytest.approx(cfg.bis_floor)


def test_combined_effect_is_clipped() -> None:
    value = combined_effect(torch.tensor([1.0]), torch.tensor([1.0]), 0.35)
    assert float(value) == pytest.approx(1.0)


def test_cohort_sampling_is_reproducible() -> None:
    pk = PharmacokineticConfig()
    pd = PharmacodynamicConfig()
    from respire.settings import CohortConfig

    cohort = CohortConfig(n_patients=16)
    first = sample_patients(16, cohort, pk, pd, seed=4)
    second = sample_patients(16, cohort, pk, pd, seed=4)
    assert torch.allclose(first.axes, second.axes)
    assert torch.allclose(first.propofol.k10, second.propofol.k10)
    third = sample_patients(16, cohort, pk, pd, seed=5)
    assert not torch.allclose(first.axes, third.axes)


def test_sensitivity_axis_shifts_the_effect_site_target() -> None:
    pk = PharmacokineticConfig()
    pd = PharmacodynamicConfig()
    from respire.settings import CohortConfig

    cohort = CohortConfig(n_patients=64, sensitivity_sd=1.0)
    patients = sample_patients(64, cohort, pk, pd, seed=9)
    correlation = torch.corrcoef(torch.stack([patients.axes[:, 0], patients.propofol_c50]))[0, 1]
    assert float(correlation) < -0.7
