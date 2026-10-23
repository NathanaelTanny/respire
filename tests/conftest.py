"""Fixtures shared by the suite.

Everything here is derived from the smoke document, so the suite runs at the same cadence and the
same window length as the real pipeline while staying inside a few seconds per test.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from respire.settings import CohortConfig, ExperimentConfig, load_experiment
from respire.theatre.cohort import Cohort, build_cohort

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def smoke_path() -> Path:
    return ROOT / "configs" / "round" / "_smoke.yaml"


@pytest.fixture(scope="session")
def primary_path() -> Path:
    return ROOT / "configs" / "round" / "primary.yaml"


@pytest.fixture(scope="session")
def smoke_config(smoke_path: Path) -> ExperimentConfig:
    return load_experiment(smoke_path)


@pytest.fixture(scope="session")
def primary_config(primary_path: Path) -> ExperimentConfig:
    return load_experiment(primary_path)


@pytest.fixture(scope="session")
def tiny_config(smoke_config: ExperimentConfig) -> ExperimentConfig:
    cohort = replace(
        smoke_config.cohort,
        n_patients=24,
        encoder_cases=8,
        train_episodes=8,
        eval_episodes=4,
        seed=3,
    )
    rounds = replace(smoke_config.rounds, rounds=1, pool_episodes=16)
    encoder = replace(smoke_config.encoder, epochs=1, batch_size=16)
    verifier = replace(smoke_config.verifier, epochs=1, batch_size=16)
    evaluation = replace(smoke_config.evaluation, heldout_cases=4, bootstrap_samples=50)
    return replace(
        smoke_config,
        cohort=cohort,
        rounds=rounds,
        encoder=encoder,
        verifier=verifier,
        evaluation=evaluation,
        seed=5,
    )


@pytest.fixture(scope="session")
def cohort(tiny_config: ExperimentConfig) -> Cohort:
    return build_cohort(tiny_config)


@pytest.fixture(scope="session")
def cohort_config() -> CohortConfig:
    return CohortConfig(n_patients=8, encoder_cases=4)
