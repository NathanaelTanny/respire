"""The certificate, the three closed-form statements, and the safety metrics.

These checks are the release's own arithmetic: they recompute the quantities the article prints and
compare them with what the article prints, so a mismatch is recorded rather than absorbed.
"""

from __future__ import annotations

import math

import pytest
import torch

from respire.control.certificate import ConformalCertificate
from respire.evaluation.theory import (
    corollary_sample_size,
    cumulative_bound,
    monotone_holds,
    proposition2_growth_threshold,
    proposition2_trajectory,
    rate_variability_reduction,
    theorem_bound,
)
from respire.metrics.depth import BAND_HIGH, BAND_LOW, depth_report, in_band
from respire.metrics.safety import (
    SUSTAINED_MIN,
    cumulative_hypotension,
    pressure_time_area,
    rate_variability,
    safety_report,
)
from respire.metrics.statistics import bootstrap_interval, pearson, share_by_quartile
from respire.metrics.stratify import benefit_shares, quartile_index
from respire.utils.numerics import conformal_index, conformal_quantile


def test_quantile_uses_the_rank_of_proposition_one() -> None:
    residuals = torch.arange(1.0, 21.0)
    certificate = ConformalCertificate(alpha=0.1)
    rank = math.ceil((20 + 1) * 0.9)
    assert rank == 19
    assert certificate.quantile(residuals) == pytest.approx(19.0)


def test_rank_is_clamped_on_a_small_reference_set() -> None:
    assert conformal_index(5, 0.1) == 5
    assert conformal_quantile(torch.tensor([1.0, 2.0]), 0.1) == pytest.approx(2.0)


def test_coverage_reaches_the_target_on_exchangeable_data() -> None:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(11)
    calibration = 50.0 + 10.0 * torch.randn(2000, generator=generator)
    certificate = ConformalCertificate(alpha=0.1)
    quantile = certificate.quantile(calibration - calibration.mean())
    lower = torch.full((4000,), float(calibration.mean()) - quantile)
    fresh = 50.0 + 10.0 * torch.randn(4000, generator=generator)
    admitted = torch.ones(4000, dtype=torch.bool)
    coverage, error = certificate.coverage(lower, fresh, admitted)
    assert coverage >= 0.88
    assert error < 0.02


def test_coverage_breaks_when_the_pool_is_shifted() -> None:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(13)
    calibration = 50.0 + 10.0 * torch.randn(2000, generator=generator)
    certificate = ConformalCertificate(alpha=0.1)
    quantile = certificate.quantile(calibration - calibration.mean())
    lower = torch.full((4000,), float(calibration.mean()) - quantile)
    shifted = 10.0 + 10.0 * torch.randn(4000, generator=generator)
    admitted = torch.ones(4000, dtype=torch.bool)
    coverage, _ = certificate.coverage(lower, shifted, admitted)
    assert coverage < 0.05


def test_admission_keeps_the_requested_fraction() -> None:
    bounds = torch.linspace(-1.0, 1.0, 101)
    certificate = ConformalCertificate(alpha=0.1)
    report, admitted = certificate.evaluate(bounds, bounds, 0.5, quantile=0.0)
    assert report.admitted_fraction == pytest.approx(0.5, abs=0.02)
    assert int(admitted.sum()) in (50, 51)


def test_corollary_sample_size_matches_the_printed_example() -> None:
    size = corollary_sample_size(R=1.0, delta=0.05, epsilon=0.01, target=0.05)
    assert 9700.0 <= size <= 9750.0
    assert corollary_sample_size(R=1.0, delta=0.05, epsilon=0.03, target=0.05) == math.inf


def test_theorem_bound_sums_its_three_terms() -> None:
    bound = theorem_bound(epsilon=0.01, R=1.0, N=9700, delta=0.05, delta_cov=0.0)
    assert bound.total == pytest.approx(
        bound.calibration_term + bound.sampling_term + bound.coverage_term
    )
    assert bound.calibration_term == pytest.approx(0.02)
    assert monotone_holds(0.5, 0.5 - 0.5 * bound.total, bound)


def test_cumulative_bound_scales_with_rounds() -> None:
    bound = theorem_bound(epsilon=0.01, R=1.0, N=9700, delta=0.05, delta_cov=0.0)
    assert cumulative_bound(bound, 4) == pytest.approx(4.0 * bound.total)


def test_proposition_two_reported_value_matches_only_at_c_two() -> None:
    at_two = proposition2_trajectory(2.0, 1.0, 3)
    assert at_two.reported_quadratic[0] == pytest.approx(at_two.reported_printed[0])
    elsewhere = proposition2_trajectory(1.5, 1.0, 3)
    assert elsewhere.reported_quadratic[0] != pytest.approx(elsewhere.reported_printed[0])


def test_proposition_two_diverges_only_above_c_two() -> None:
    assert not proposition2_growth_threshold(1.5)
    assert proposition2_growth_threshold(2.5)
    trajectory = proposition2_trajectory(3.0, 1.0, 6)
    assert trajectory.true[-1] < trajectory.true[0]
    assert trajectory.actions[-1] > trajectory.actions[0]


def test_depth_report_is_zero_at_the_target() -> None:
    bis = torch.full((3, 10), 50.0)
    report = depth_report(bis)
    assert report.mae == pytest.approx(0.0)
    assert report.time_in_band == pytest.approx(100.0)


def test_band_edges_are_inclusive() -> None:
    bis = torch.tensor([BAND_LOW, BAND_HIGH, BAND_LOW - 1.0])
    assert float(in_band(bis).sum()) == 2.0


def test_hypotension_and_area_are_in_minutes() -> None:
    pressure = torch.tensor([[70.0, 60.0, 55.0, 70.0]])
    minutes = cumulative_hypotension(pressure, 0.5)
    assert float(minutes) == pytest.approx(1.0)
    area = pressure_time_area(pressure, 0.5)
    assert float(area) == pytest.approx((5.0 + 10.0) * 0.5)


def test_safety_report_counts_cases_not_minutes() -> None:
    steps = 120
    pressure = torch.full((4, steps), 80.0)
    pressure[0, :20] = 60.0
    depth = torch.full((4, steps), 45.0)
    heart = torch.full((4, steps), 70.0)
    action = torch.zeros(4, steps, 2)
    report = safety_report(pressure, depth, heart, action, dt_min=1.0)
    assert report.hypotension_rate == pytest.approx(25.0)
    assert report.hypotension_minutes == pytest.approx(20.0 / 4.0)
    assert report.suppression_rate == pytest.approx(0.0)
    assert report.pressure_time_area == pytest.approx((5.0 * 20.0) / 4.0)


def test_sustained_threshold_is_five_minutes() -> None:
    assert SUSTAINED_MIN == 5.0
    pressure = torch.full((1, 120), 80.0)
    pressure[0, :4] = 60.0
    assert float(cumulative_hypotension(pressure, 1.0)) == pytest.approx(4.0)


def test_rate_variability_is_the_step_to_step_change() -> None:
    action = torch.zeros(2, 5, 2)
    action[0, 1:, 0] = torch.tensor([1.0, 1.0, 1.0, 1.0])
    values = rate_variability(action)
    assert float(values[0]) == pytest.approx(0.25)
    assert float(values[1]) == pytest.approx(0.0)


def test_reduction_statistics() -> None:
    treated = torch.tensor([1.0, 2.0])
    control = torch.tensor([2.0, 4.0])
    assert rate_variability_reduction(treated, control) == pytest.approx(50.0)


def test_bootstrap_interval_brackets_the_difference() -> None:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(5)
    treated = torch.randn(200, generator=generator)
    control = treated + 0.5
    low, high = bootstrap_interval(treated, control, 500, 0.05, seed=1)
    assert low < -0.5 < high


def test_pearson_and_quartile_shares() -> None:
    left = torch.linspace(0.0, 1.0, 40)
    assert pearson(left, left) == pytest.approx(1.0)
    quartile = quartile_index(torch.linspace(0.0, 99.9, 40))
    shares = share_by_quartile(torch.ones(40), quartile)
    assert sum(shares) == pytest.approx(100.0)
    assert all(abs(value - 25.0) < 1e-6 for value in shares)


def test_benefit_shares_concentrate_on_the_tail() -> None:
    percentile = torch.linspace(0.0, 100.0, 100)
    benefit = (percentile > 75.0).to(torch.float32)
    shares = benefit_shares(-benefit, torch.zeros_like(benefit), percentile)
    assert shares[3] == pytest.approx(100.0)
