"""The theatre: shapes, reward, costs and the counterfactual look-ahead."""

from __future__ import annotations

import torch

from respire.settings import ExperimentConfig
from respire.theatre.clinician import RuleBasedAnaesthetist
from respire.theatre.cohort import episode_patients, make_environment
from respire.theatre.environment import CHANNELS, AnaesthesiaEnv, Observation
from respire.theatre.rollout import rollout


def _env(config: ExperimentConfig, episodes: int = 3) -> AnaesthesiaEnv:
    patients = episode_patients(_cohort(config), "development", episodes, config.cohort.seed)
    return make_environment(patients, config, config.cohort.seed)


def _cohort(config: ExperimentConfig) -> object:
    from respire.theatre.cohort import build_cohort

    return build_cohort(config)


def test_channel_schema_has_twelve_named_channels() -> None:
    assert len(CHANNELS) == 12
    assert CHANNELS[0] == "bis"
    assert CHANNELS[2] == "map"
    assert CHANNELS[-1] == "pneumo_pressure"


def test_reset_returns_a_full_window(tiny_config: ExperimentConfig) -> None:
    env = _env(tiny_config)
    observation = env.reset()
    assert observation.window.shape == (3, tiny_config.encoder.window_steps, 12)
    assert observation.current.shape == (3, 12)
    assert observation.actions.shape == (3, tiny_config.encoder.window_steps, 2)


def test_window_advances_one_row_per_step(tiny_config: ExperimentConfig) -> None:
    env = _env(tiny_config)
    env.reset()
    action = torch.full((3, 2), 6.0)
    action[:, 1] = 0.15
    first = env.step(action)
    assert torch.allclose(first.observation.window[:, -1, :], first.observation.current, atol=1e-5)
    assert torch.allclose(first.observation.actions[:, -1, :], action, atol=1e-5)


def test_reward_matches_equation_one(tiny_config: ExperimentConfig) -> None:
    env = _env(tiny_config)
    env.reset()
    action = torch.tensor([[6.0, 0.15], [6.0, 0.15], [6.0, 0.15]])
    before = env.step(action)
    depth_error = (before.observation.bis - tiny_config.theatre.bis_target).abs()
    expected = -depth_error - tiny_config.theatre.drug_cost_weight * (action[:, 0] + action[:, 1])
    assert torch.allclose(before.reward, expected, atol=1e-5)


def test_costs_accumulate_in_minutes(tiny_config: ExperimentConfig) -> None:
    env = _env(tiny_config)
    env.reset()
    action = torch.tensor([[9.0, 0.4]]).repeat(3, 1)
    hypotension = torch.zeros(3)
    suppression = torch.zeros(3)
    variability = torch.zeros(3)
    for _ in range(12):
        transition = env.step(action)
        hypotension += transition.cost_hypotension
        suppression += transition.cost_suppression
        variability += transition.cost_variability
    assert torch.allclose(env.state.cost_hypotension, hypotension, atol=1e-6)
    assert torch.allclose(env.state.cost_suppression, suppression, atol=1e-6)
    assert torch.allclose(env.state.cost_variability, variability, atol=1e-6)
    assert float(hypotension.max()) <= 12 * tiny_config.theatre.dt_s / 60.0 + 1e-6


def test_actions_are_clipped_to_the_pump(tiny_config: ExperimentConfig) -> None:
    env = _env(tiny_config)
    env.reset()
    transition = env.step(torch.tensor([[99.0, 9.0]]).repeat(3, 1))
    assert float(transition.action[:, 0].max()) == tiny_config.theatre.propofol_max
    assert float(transition.action[:, 1].max()) == tiny_config.theatre.remifentanil_max


def test_peek_leaves_the_state_untouched(tiny_config: ExperimentConfig) -> None:
    env = _env(tiny_config)
    env.reset()
    action = torch.full((3, 2), 5.0)
    for _ in range(3):
        env.step(action)
    before = env.state.propofol.clone()
    probe = env.peek(torch.full((3, 2), 9.0))
    assert torch.allclose(env.state.propofol, before)
    assert probe.true_bis.shape == (3,)
    assert bool((probe.true_bis <= 100.0).all())


def test_peek_separates_two_arms(tiny_config: ExperimentConfig) -> None:
    env = _env(tiny_config)
    env.reset()
    action = torch.full((3, 2), 5.0)
    action[:, 1] = 0.15
    for _ in range(6):
        env.step(action)
    base = env.true_depth()
    low = env.peek(torch.tensor([[2.0, 0.15]]).repeat(3, 1)).true_bis
    high = env.peek(torch.tensor([[12.0, 0.15]]).repeat(3, 1)).true_bis
    assert bool((high + base - base < low).all())


def test_episode_terminates_at_the_horizon(tiny_config: ExperimentConfig) -> None:
    env = _env(tiny_config)
    env.reset()
    action = torch.full((3, 2), 4.0)
    done = False
    for _ in range(env.steps):
        done = env.step(action).done
    assert done


def test_rollout_trace_is_consistent(tiny_config: ExperimentConfig) -> None:
    env = _env(tiny_config)
    trace = rollout(env, RuleBasedAnaesthetist(tiny_config.theatre, noise_sd=0.0))
    assert trace.observation.shape == (3, tiny_config.theatre.n_steps, 12)
    assert trace.action.shape == (3, tiny_config.theatre.n_steps, 2)
    assert torch.allclose(
        trace.cost_hypotension,
        trace.hypotension_step.sum(dim=1),
        atol=1e-5,
    )
    assert trace.axes.shape == (3, 3)
    assert bool((trace.sensitivity_pct >= 0.0).all())


def test_depth_falls_under_a_high_infusion(tiny_config: ExperimentConfig) -> None:
    env = _env(tiny_config)
    trace = rollout(env, RuleBasedAnaesthetist(tiny_config.theatre, propofol_base=9.0))
    assert float(trace.bis[:, -1].mean()) < float(trace.bis[:, 0].mean())


def test_observation_type_is_exposed(tiny_config: ExperimentConfig) -> None:
    env = _env(tiny_config)
    observation = env.reset()
    assert isinstance(observation, Observation)
    assert observation.window.shape[0] == observation.batch
