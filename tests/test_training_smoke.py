"""Closed-loop checks: a loss that must fall, a checkpoint that must round-trip, and the pipeline.

The overfit test is deliberately small: one batch, twelve passes, and the requirement that the
training loss falls. It catches a detached graph, a frozen parameter and a mis-scaled target, which
are the three ways an objective can look healthy while learning nothing.
"""

from __future__ import annotations

from dataclasses import replace

import torch

from respire.control.shield import AnalyticShield
from respire.data.windows import build_windows
from respire.losses.encoder_losses import encoder_objective
from respire.losses.flow_matching import behaviour_cloning_loss
from respire.models.encoder import EncoderStandardiser, LatentEncoder
from respire.models.flow import FlowPolicy
from respire.models.pressure import OneStepPressureModel
from respire.settings import ExperimentConfig
from respire.theatre.clinician import RuleBasedAnaesthetist
from respire.theatre.cohort import build_cohort, episode_patients, make_environment
from respire.theatre.probe import counterfactual_probes
from respire.theatre.rollout import rollout
from respire.training.checkpoint import ReleaseBundle
from respire.training.encoder_fit import fit_encoder
from respire.training.verifier_fit import fit_verifier


def _windows(config: ExperimentConfig, episodes: int = 6) -> tuple[object, object, object]:
    cohort = build_cohort(config)
    patients = episode_patients(cohort, "development", episodes, config.seed)
    trace = rollout(
        make_environment(patients, config, config.seed),
        RuleBasedAnaesthetist(config.theatre, noise_sd=0.05),
    )
    windows = build_windows(trace, config.encoder.window_steps, stride=4)
    probes = counterfactual_probes(
        patients,
        config,
        RuleBasedAnaesthetist(config.theatre, noise_sd=0.05),
        warmup_steps=(12, 36),
        seed=config.seed,
    )
    return cohort, windows, probes


def test_encoder_objective_falls_on_one_batch(tiny_config: ExperimentConfig) -> None:
    _, windows, probes = _windows(tiny_config)
    standardiser = EncoderStandardiser(12)
    standardiser.fit(windows.window)
    encoder = LatentEncoder(replace(tiny_config.encoder, epochs=1))
    weights = {"reconstruction": 1.0, "one_step": 1.0, "interventional": 0.5, "axes": 0.25}
    optimiser = torch.optim.Adam(encoder.parameters(), lr=3e-3)
    chunk = windows.subset(torch.arange(min(64, windows.size)))
    losses: list[float] = []
    for _ in range(12):
        optimiser.zero_grad()
        terms = encoder_objective(encoder, standardiser, chunk, probes, weights)
        terms.total.backward()
        optimiser.step()
        losses.append(float(terms.total.detach()))
    assert losses[-1] < losses[0]
    assert losses[-1] < 0.6 * losses[0]


def test_flow_matching_falls_on_one_batch(tiny_config: ExperimentConfig) -> None:
    _, windows, _ = _windows(tiny_config)
    standardiser = EncoderStandardiser(12)
    standardiser.fit(windows.window)
    encoder = LatentEncoder(tiny_config.encoder)
    policy = FlowPolicy(replace(tiny_config.policy, hidden=32, depth=2), 48)
    policy.configure_action_scale(
        tiny_config.theatre.propofol_max, tiny_config.theatre.remifentanil_max
    )
    optimiser = torch.optim.Adam(policy.parameters(), lr=6e-3)
    chunk = windows.subset(torch.arange(min(64, windows.size)))
    with torch.no_grad():
        output = encoder(standardiser.transform(chunk.window), chunk.actions)
        condition = torch.cat([output.latent, output.hidden], dim=-1)
    target = chunk.actions[:, -1, :]
    generator = torch.Generator(device="cpu")
    generator.manual_seed(0)
    losses: list[float] = []
    for _ in range(80):
        optimiser.zero_grad()
        loss = behaviour_cloning_loss(policy, condition, target, generator)
        loss.backward()
        optimiser.step()
        losses.append(float(loss.detach()))
    assert losses[-1] < 0.6 * losses[0]


def test_encoder_fit_report_tracks_the_axis(tiny_config: ExperimentConfig) -> None:
    _, windows, probes = _windows(tiny_config)
    standardiser = EncoderStandardiser(12)
    encoder = LatentEncoder(replace(tiny_config.encoder, epochs=2, lr=3e-3))
    report = fit_encoder(
        encoder,
        standardiser,
        windows.subset(torch.arange(min(96, windows.size))),
        probes,
        replace(tiny_config.encoder, epochs=2, lr=3e-3),
        seed=1,
    )
    assert report.epochs == 2
    assert -1.0 <= report.axis_correlation <= 1.0
    assert report.terms["total"] >= 0.0


def test_verifier_fit_reduces_the_error(tiny_config: ExperimentConfig) -> None:
    from respire.data.summary import summarise, summary_dim
    from respire.models.verifier import ReturnVerifier

    cohort = build_cohort(tiny_config)
    patients = episode_patients(cohort, "development", 16, tiny_config.seed)
    trace = rollout(
        make_environment(patients, tiny_config, tiny_config.seed),
        RuleBasedAnaesthetist(tiny_config.theatre, noise_sd=0.05),
    )
    features = summarise(trace)
    verifier = ReturnVerifier(
        replace(tiny_config.verifier, hidden=(64, 32), feature_dim=32), summary_dim(12)
    )
    tuning = replace(tiny_config.verifier, epochs=30, lr=1e-2)
    report = fit_verifier(verifier, features, trace.episode_return, tuning, seed=2)
    spread = float(trace.episode_return.std())
    assert report.train_rmse < spread


def test_shield_projects_into_the_half_space(tiny_config: ExperimentConfig) -> None:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(4)
    state = torch.randn(64, 5, generator=generator) * 2.0
    action = torch.rand(64, 2, generator=generator)
    target = 85.0 + 2.0 * state[:, 0] - 3.0 * action[:, 0] - 2.0 * action[:, 1]
    model = OneStepPressureModel(ridge=1e-6)
    model.fit(state, action, target)
    shield = AnalyticShield(model, tiny_config.theatre, tiny_config.shield)
    requested = torch.tensor([[12.0, 0.5]]).repeat(8, 1)
    outcome = shield(state[:8], requested)
    assert bool((outcome.action[:, 0] <= requested[:, 0] + 1e-6).all())
    feasible = outcome.feasible
    if bool(feasible.any()):
        assert bool((outcome.bound[feasible] >= shield.bound - 1e-3).all())
    if bool((~feasible).any()):
        assert float(outcome.action[~feasible].abs().max()) == 0.0
    assert bool((outcome.violation >= 0.0).all())


def test_checkpoint_round_trip(tmp_path: object, tiny_config: ExperimentConfig) -> None:
    from pathlib import Path

    from respire.data.summary import summary_dim
    from respire.models.deterministic_head import DeterministicHead
    from respire.models.verifier import ReturnVerifier

    directory = Path(str(tmp_path))
    encoder = LatentEncoder(tiny_config.encoder)
    standardiser = EncoderStandardiser(12)
    standardiser.fit(torch.randn(32, 12, 12))
    verifier = ReturnVerifier(tiny_config.verifier, summary_dim(12))
    flow = FlowPolicy(tiny_config.policy, 48)
    pressure = OneStepPressureModel()
    pressure.fit(torch.randn(32, 5), torch.rand(32, 2), 80.0 + torch.randn(32))
    bundle = ReleaseBundle(
        encoder=encoder,
        standardiser=standardiser,
        verifier=verifier,
        flow=flow,
        pressure=pressure,
        seed=17,
        round_index=2,
        metrics={"bis_mae": 1.0},
        certificates=[{"coverage": 0.9}],
    )
    path = bundle.save(directory / "bundle.pt")
    restored = ReleaseBundle.load(path)
    window = torch.randn(3, 12, 12)
    actions = torch.zeros(3, 12, 2)
    with torch.no_grad():
        first = encoder(window, actions)
        second = restored.encoder(window, actions)
    assert torch.allclose(first.latent, second.latent, atol=1e-6)
    assert restored.seed == 17
    assert restored.round_index == 2
    assert restored.metrics["bis_mae"] == 1.0
    assert torch.allclose(
        restored.pressure.predict(torch.randn(4, 5).zero_(), torch.zeros(4, 2)),
        pressure.predict(torch.zeros(4, 5), torch.zeros(4, 2)),
        atol=1e-5,
    )
    assert DeterministicHead(tiny_config.policy, 48).parameter_count() > 0


def test_pipeline_runs_end_to_end(tmp_path: object, tiny_config: ExperimentConfig) -> None:
    from pathlib import Path

    from respire.flags import FULL
    from respire.pipeline import run_pipeline

    directory = Path(str(tmp_path))
    result = run_pipeline(tiny_config, FULL, seed=tiny_config.seed, output_dir=directory)
    assert result.run.records
    assert result.evaluation.bis_mae > 0.0
    assert result.evaluation.tib >= 0.0
    assert result.behaviour.bis_mae > 0.0
    assert all(torch.isfinite(torch.tensor(value)) for value in result.evaluation.row().values())
    saved = result.bundle.save(directory / "bundle.pt")
    assert saved.exists()
    record = result.run.records[0]
    assert 0.0 <= float(record.certificate["coverage"]) <= 1.0
    assert record.certificate["quantile"] >= 0.0
