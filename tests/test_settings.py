"""The settings that the article fixes must survive the configuration layer unchanged."""

from __future__ import annotations

import pytest

from respire.settings import (
    ConfigError,
    ExperimentConfig,
    apply_overrides,
    load_experiment,
)


def test_primary_document_keeps_the_reported_values(primary_config: ExperimentConfig) -> None:
    assert primary_config.theatre.dt_s == 5.0
    assert primary_config.theatre.horizon_min == 120.0
    assert primary_config.theatre.bis_target == 50.0
    assert primary_config.theatre.bis_noise_sd == 2.1
    assert primary_config.theatre.gamma == 0.995
    assert primary_config.theatre.map_floor == 65.0
    assert primary_config.theatre.bis_suppression == 20.0
    assert primary_config.encoder.n_channels == 12
    assert primary_config.encoder.n_blocks == 6
    assert primary_config.encoder.latent_dim == 16
    assert primary_config.encoder.axis_dim == 3
    assert primary_config.policy.euler_steps == 4
    assert primary_config.cohort.n_patients == 2000
    assert primary_config.cohort.variability == 1.25
    assert primary_config.cohort.train_episodes == 20000
    assert primary_config.cohort.eval_episodes == 2000
    assert primary_config.rounds.cost_budget_hypotension_min == 10.0
    assert primary_config.rounds.cost_budget_suppression_min == 1.0
    assert primary_config.evaluation.heldout_cases == 226
    assert primary_config.evaluation.seeds == 5
    assert primary_config.surgical.pneumo_pressure_mmHg == 12.0


def test_window_matches_the_stated_seconds(primary_config: ExperimentConfig) -> None:
    steps = primary_config.encoder.window_steps
    assert steps * primary_config.theatre.dt_s == primary_config.encoder.window_s


def test_episode_length_matches_the_stated_minutes(primary_config: ExperimentConfig) -> None:
    assert primary_config.theatre.n_steps * primary_config.theatre.dt_s == 7200.0


def test_conditional_probability_budget_is_unique(primary_config: ExperimentConfig) -> None:
    assert 0.0 < primary_config.rounds.alpha < 1.0


def test_overrides_reach_nested_keys(primary_path: object) -> None:
    config = load_experiment(primary_path, ["rounds.rounds=3", "verifier.hidden=[128,64]"])
    assert config.rounds.rounds == 3
    assert config.verifier.hidden == (128, 64)


def test_unknown_section_type_is_rejected() -> None:
    with pytest.raises(ConfigError):
        ExperimentConfig.from_dict({"theatre": {"dt_s": "five"}})


def test_override_requires_assignment() -> None:
    with pytest.raises(ConfigError):
        apply_overrides({}, ["rounds.rounds"])
