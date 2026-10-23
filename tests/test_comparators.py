"""Every comparison arm must play and take a gradient step.

The arms are exercised on a handful of transitions, which is enough to catch a broken action
mapping, a missing parameter in an optimiser, or a head that returns the wrong width.
"""

from __future__ import annotations

import pytest
import torch

from respire.comparators.base import TransitionBatch, transitions_from_trace
from respire.comparators.constrained_pg import ConstrainedPPOAgent, LagrangianAgent
from respire.comparators.dreamer import DreamerAgent
from respire.comparators.internal_model import InternalModelController
from respire.comparators.pid import PIDDepthController
from respire.comparators.ppo import PPOAgent
from respire.comparators.registry import (
    analytic_names,
    build_analytic,
    build_learning,
    learned_names,
    spec,
)
from respire.comparators.tci import ELEVELD, SCHNIDER, schnider_arm
from respire.settings import ExperimentConfig
from respire.theatre.clinician import RuleBasedAnaesthetist
from respire.theatre.cohort import build_cohort, episode_patients, make_environment
from respire.theatre.environment import Observation
from respire.theatre.rollout import rollout


def _reading(batch: int, depth: float, pressure: float = 80.0) -> Observation:
    """An observation whose channels carry the reading the arm actually looks at."""
    current = torch.zeros(batch, 12)
    current[:, 0] = depth
    current[:, 1] = 70.0
    current[:, 2] = pressure
    return Observation(
        window=torch.zeros(batch, 12, 12),
        actions=torch.zeros(batch, 12, 2),
        current=current,
        bis=torch.full((batch,), depth),
        map_mmHg=torch.full((batch,), pressure),
    )


@pytest.fixture(scope="module")
def batch(tiny_config: ExperimentConfig) -> TransitionBatch:
    cohort = build_cohort(tiny_config)
    patients = episode_patients(cohort, "development", 2, tiny_config.cohort.seed)
    trace = rollout(
        make_environment(patients, tiny_config, tiny_config.cohort.seed),
        RuleBasedAnaesthetist(tiny_config.theatre, noise_sd=0.0),
    )
    return transitions_from_trace(trace)


@pytest.fixture(scope="module")
def observation(tiny_config: ExperimentConfig) -> Observation:
    cohort = build_cohort(tiny_config)
    patients = episode_patients(cohort, "development", 2, tiny_config.cohort.seed)
    env = make_environment(patients, tiny_config, tiny_config.cohort.seed)
    return env.reset()


def test_registry_covers_the_reported_arms() -> None:
    expected = {
        "schnider_tcu",
        "eleveld_tcu",
        "pid",
        "internal_model",
        "bc",
        "sac",
        "ppo",
        "td3bc",
        "cql",
        "iql",
        "ppo_lagr",
        "cppo",
        "dreamer",
    }
    assert set(analytic_names()) | set(learned_names()) == expected
    assert spec("cppo").constrained
    assert not spec("ppo").constrained


def test_analytic_arms_play_inside_the_box(
    tiny_config: ExperimentConfig, observation: Observation
) -> None:
    for name in analytic_names():
        arm = build_analytic(name, tiny_config.theatre)
        arm.reset(observation.batch)  # type: ignore[attr-defined]
        action = arm.act(observation)  # type: ignore[attr-defined]
        assert action.shape == (observation.batch, 2), name
        assert float(action[:, 0].min()) >= 0.0, name
        assert float(action[:, 0].max()) <= tiny_config.theatre.propofol_max + 1e-6, name
        assert float(action[:, 1].max()) <= tiny_config.theatre.remifentanil_max + 1e-6, name


def test_target_controlled_arms_differ_in_their_population() -> None:
    assert SCHNIDER.clearance != ELEVELD.clearance
    assert SCHNIDER.v1 != ELEVELD.v1


def test_tci_reaches_a_maintenance_rate(tiny_config: ExperimentConfig) -> None:
    arm = schnider_arm(tiny_config.theatre)
    arm.reset(4)
    observation = _reading(4, depth=90.0)
    rates = [float(arm.act(observation)[0, 0]) for _ in range(240)]
    assert min(rates) >= 0.0
    assert max(rates) <= tiny_config.theatre.propofol_max
    assert rates[-1] > 1.0


def test_pid_winds_up_but_stays_clipped(tiny_config: ExperimentConfig) -> None:
    arm = PIDDepthController(tiny_config.theatre)
    arm.reset(2)
    observation = _reading(2, depth=95.0)
    for _ in range(200):
        action = arm.act(observation)
    assert float(action[:, 0].max()) <= tiny_config.theatre.propofol_max
    assert float(arm.integral.abs().max()) <= arm.integral_clamp + 1e-3


def test_internal_model_tracks_a_falling_depth(tiny_config: ExperimentConfig) -> None:
    arm = InternalModelController(tiny_config.theatre)
    arm.reset(2)
    observation = _reading(2, depth=60.0)
    first = float(arm.act(observation)[0, 0])
    observation = _reading(2, depth=40.0)
    second = float(arm.act(observation)[0, 0])
    assert second < first


@pytest.mark.parametrize("name", ["sac", "td3bc", "cql", "iql"])
def test_off_policy_arms_take_a_step(
    name: str, tiny_config: ExperimentConfig, batch: TransitionBatch, observation: Observation
) -> None:
    agent = build_learning(name, tiny_config.theatre)
    agent.reset(observation.batch)  # type: ignore[attr-defined]
    action = agent.act(observation)  # type: ignore[attr-defined]
    assert action.shape == (observation.batch, 2)
    metrics = agent.update(batch.subset(torch.arange(64)))  # type: ignore[attr-defined]
    assert metrics
    assert all(torch.isfinite(torch.tensor(value)) for value in metrics.values())


def test_dreamer_takes_a_step(
    tiny_config: ExperimentConfig, batch: TransitionBatch, observation: Observation
) -> None:
    agent = DreamerAgent(tiny_config.theatre)
    agent.reset(observation.batch)
    action = agent.act(observation)
    assert action.shape == (observation.batch, 2)
    metrics = agent.update(batch.subset(torch.arange(48)))
    assert metrics["world_loss"] >= 0.0


def test_ppo_variants_take_a_step(tiny_config: ExperimentConfig) -> None:
    cohort = build_cohort(tiny_config)
    patients = episode_patients(cohort, "development", 2, tiny_config.cohort.seed)
    ppo = PPOAgent()
    assert ppo.fit(patients, tiny_config, iterations=1, seed=1)["value_loss"] >= 0.0
    lagrangian = LagrangianAgent(budget=(10.0, 1.0, 60.0))
    metrics = lagrangian.fit(patients, tiny_config, iterations=1, seed=1)
    assert metrics["multiplier"] >= 0.0
    constrained = ConstrainedPPOAgent(budget=(10.0, 1.0, 60.0))
    metrics = constrained.fit(patients, tiny_config, iterations=1, seed=1)
    assert all(value == value for value in metrics.values())
    assert all(abs(value) < 1e6 for value in metrics.values())


def test_action_mapping_round_trips() -> None:
    from respire.comparators.base import denormalise_action, normalise_action

    action = torch.tensor([[6.0, 0.25], [0.0, 0.0]])
    unit = normalise_action(action, 12.0, 0.5)
    restored = denormalise_action(unit, 12.0, 0.5)
    assert torch.allclose(restored, action, atol=1e-5)
