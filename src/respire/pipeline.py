"""The end-to-end driver: cohort, encoder, certificate, rounds, evaluation.

This is the only place that knows the order of the stages. The CLI commands are thin wrappers
around it, and the component analysis is the same function with one switch turned off.

Ref: Fig. 2 (the three components and the frozen encoder), Algorithm 1 (the round order).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from respire.control.adaptation import RidgeAdapter
from respire.control.certificate import ConformalCertificate
from respire.control.deployment import DeployedController, OverrideRule
from respire.control.policy import LatentFlowController
from respire.control.shield import AnalyticShield
from respire.data.summary import summarise, summary_dim
from respire.data.trace import EpisodeTrace
from respire.data.windows import WindowBatch, build_windows, sample_windows
from respire.evaluation.arms import ArmResult, evaluate_arm
from respire.flags import FULL, VariantFlags, configure
from respire.models.deterministic_head import DeterministicHead
from respire.models.encoder import EncoderStandardiser, LatentEncoder
from respire.models.flow import FlowPolicy
from respire.models.pressure import OneStepPressureModel, state_features
from respire.models.verifier import ReturnVerifier
from respire.settings import ExperimentConfig
from respire.theatre.clinician import RuleBasedAnaesthetist
from respire.theatre.cohort import Cohort, build_cohort, episode_patients, make_environment
from respire.theatre.probe import ProbeBatch, counterfactual_probes
from respire.theatre.rollout import rollout
from respire.training.checkpoint import ReleaseBundle
from respire.training.encoder_fit import EncoderFitReport, fit_encoder
from respire.training.round_loop import RoundRun, run_recursion
from respire.training.verifier_fit import VerifierFitReport, fit_verifier
from respire.utils.logging_setup import get_logger
from respire.utils.seeding import derive_seed, set_seed

LOGGER = get_logger("pipeline")

WARMUP_MIN = 5.0


@dataclass
class PreparedData:
    """Everything the fit stages consume."""

    cohort: Cohort
    behaviour: RuleBasedAnaesthetist
    development: EpisodeTrace
    calibration: EpisodeTrace
    windows: WindowBatch
    probes: ProbeBatch

    def as_summary(self) -> dict[str, float]:
        return {
            "development_cases": float(self.development.batch),
            "calibration_cases": float(self.calibration.batch),
            "windows": float(self.windows.size),
            "probes": float(self.probes.size),
        }


@dataclass
class PipelineResult:
    """The artefacts of one run."""

    config: ExperimentConfig
    flags: VariantFlags
    bundle: ReleaseBundle
    run: RoundRun
    encoder_report: EncoderFitReport
    verifier_report: VerifierFitReport
    evaluation: ArmResult
    behaviour: ArmResult
    data: dict[str, float]

    def summary(self) -> dict[str, object]:
        return {
            "variant": self.flags.name,
            "rounds": len(self.run.records),
            "encoder": self.encoder_report.as_dict(),
            "verifier": self.verifier_report.as_dict(),
            "evaluation": self.evaluation.row(),
            "behaviour": self.behaviour.row(),
            "data": self.data,
        }


def prepare_data(cfg: ExperimentConfig, seed: int) -> PreparedData:
    """Sample the cohort, generate the development and calibration pools, cut the windows."""
    cohort = build_cohort(cfg)
    behaviour = RuleBasedAnaesthetist(cfg.theatre, noise_sd=0.05)
    cases = min(cfg.cohort.encoder_cases, int(cohort.splits.development.numel()))
    development_patients = episode_patients(cohort, "development", cases, derive_seed(seed, "dev"))
    development = rollout(
        make_environment(development_patients, cfg, derive_seed(seed, "dev")), behaviour
    )
    calibration_patients = episode_patients(
        cohort,
        "calibration",
        min(cfg.cohort.encoder_cases, int(cohort.splits.calibration.numel())),
        derive_seed(seed, "cal"),
    )
    calibration = rollout(
        make_environment(calibration_patients, cfg, derive_seed(seed, "cal")), behaviour
    )
    windows = build_windows(development, cfg.encoder.window_steps, stride=3)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(derive_seed(seed, "windows"))
    windows = sample_windows(windows, min(windows.size, 12000), generator)
    probes = counterfactual_probes(
        development_patients,
        cfg,
        RuleBasedAnaesthetist(cfg.theatre, noise_sd=0.05),
        warmup_steps=(24, 72, 144, 240),
        perturb=2.0,
        seed=derive_seed(seed, "probe"),
    )
    return PreparedData(
        cohort=cohort,
        behaviour=behaviour,
        development=development,
        calibration=calibration,
        windows=windows,
        probes=probes,
    )


def _fit_pressure(windows: WindowBatch) -> OneStepPressureModel:
    """Ridge fit of the next-step pressure on the five state summaries and the two rates."""
    model = OneStepPressureModel()
    state = state_features(windows.window)
    target = windows.current[:, 2] + windows.pressure_delta
    model.fit(state, windows.actions[:, -1, :], target)
    return model


def build_flow(
    cfg: ExperimentConfig, flags: VariantFlags, cond_dim: int
) -> FlowPolicy | DeterministicHead:
    if flags.flow_head:
        policy = FlowPolicy(cfg.policy, cond_dim)
        policy.configure_action_scale(cfg.theatre.propofol_max, cfg.theatre.remifentanil_max)
        return policy
    return DeterministicHead(cfg.policy, cond_dim)


def run_pipeline(
    cfg: ExperimentConfig,
    flags: VariantFlags = FULL,
    seed: int | None = None,
    output_dir: str | Path | None = None,
    prepared: PreparedData | None = None,
) -> PipelineResult:
    """Fit the encoder, initialise by cloning, run the rounds and evaluate the result."""
    active_cfg = configure(cfg, flags)
    chosen_seed = cfg.seed if seed is None else seed
    set_seed(chosen_seed)
    data = prepared if prepared is not None else prepare_data(active_cfg, chosen_seed)
    standardiser = EncoderStandardiser(active_cfg.encoder.n_channels)
    encoder = LatentEncoder(active_cfg.encoder)
    encoder_report = fit_encoder(
        encoder,
        standardiser,
        data.windows,
        data.probes,
        active_cfg.encoder,
        derive_seed(chosen_seed, "encoder"),
    )
    cond_dim = active_cfg.encoder.latent_dim + active_cfg.encoder.hidden_dim
    flow = build_flow(active_cfg, flags, cond_dim)
    from respire.comparators.bc import fit_behaviour_cloning

    fit_behaviour_cloning(
        encoder,
        standardiser,
        flow,
        data.windows,
        derive_seed(chosen_seed, "cloning"),
        epochs=active_cfg.policy.bc_epochs,
        batch_size=256,
    )
    width = summary_dim(active_cfg.encoder.n_channels)
    verifier = ReturnVerifier(active_cfg.verifier, width)
    development_features = summarise(data.development)
    verifier.set_scaler(development_features)
    verifier_report = fit_verifier(
        verifier,
        development_features,
        data.development.episode_return,
        active_cfg.verifier,
        derive_seed(chosen_seed, "verifier"),
    )
    pressure = _fit_pressure(data.windows)
    run, controller, _duals = run_recursion(
        active_cfg,
        data.cohort,
        encoder,
        standardiser,
        verifier,
        flow,
        data.behaviour,
        data.development,
        chosen_seed,
        flags=flags,
        output_dir=output_dir,
    )
    _ = controller
    evaluation = evaluate_controller(
        active_cfg, flags, encoder, standardiser, flow, pressure, data.cohort, chosen_seed
    )
    behaviour_policy = RuleBasedAnaesthetist(active_cfg.theatre, noise_sd=0.05)
    evaluation_patients = episode_patients(
        data.cohort,
        "evaluation",
        min(active_cfg.evaluation.heldout_cases, int(data.cohort.splits.evaluation.numel())),
        derive_seed(chosen_seed, "eval"),
    )
    behaviour = evaluate_arm(
        "clinician", behaviour_policy, evaluation_patients, active_cfg, chosen_seed
    )
    bundle = ReleaseBundle(
        encoder=encoder,
        standardiser=standardiser,
        verifier=verifier,
        flow=flow if isinstance(flow, FlowPolicy) else FlowPolicy(active_cfg.policy, cond_dim),
        pressure=pressure,
        seed=chosen_seed,
        round_index=len(run.records),
        metrics={key: float(value) for key, value in evaluation.row().items()},
        certificates=[record.certificate for record in run.records],
    )
    return PipelineResult(
        config=active_cfg,
        flags=flags,
        bundle=bundle,
        run=run,
        encoder_report=encoder_report,
        verifier_report=verifier_report,
        evaluation=evaluation,
        behaviour=behaviour,
        data=data.as_summary(),
    )


def evaluate_controller(
    cfg: ExperimentConfig,
    flags: VariantFlags,
    encoder: LatentEncoder,
    standardiser: EncoderStandardiser,
    flow: FlowPolicy | DeterministicHead,
    pressure: OneStepPressureModel,
    cohort: Cohort,
    seed: int,
) -> ArmResult:
    """Roll the controller through the deployment wrapper on the evaluation split."""
    patients = episode_patients(
        cohort,
        "evaluation",
        min(cfg.evaluation.heldout_cases, 64, int(cohort.splits.evaluation.numel())),
        derive_seed(seed, "eval"),
    )
    behaviour = RuleBasedAnaesthetist(cfg.theatre, noise_sd=0.0)
    shield = AnalyticShield(pressure, cfg.theatre, cfg.shield) if flags.shield else None
    active = LatentFlowController(
        encoder,
        standardiser,
        flow,
        cfg.theatre,
        shield=shield,
        seed=derive_seed(seed, "eval-controller"),
        use_latent=flags.latent_axes,
    )
    if flags.adaptation:
        warmup = int(WARMUP_MIN * 60.0 / cfg.theatre.dt_s)
        deployed = DeployedController(
            active,
            behaviour,
            RidgeAdapter(),
            warmup_steps=warmup,
            override=OverrideRule(
                map_floor=cfg.theatre.map_floor,
                dt_s=cfg.theatre.dt_s,
            ),
            dt_s=cfg.theatre.dt_s,
        )
        return evaluate_arm("respire", deployed, patients, cfg, seed)
    return evaluate_arm("respire", active, patients, cfg, seed)


def certificate_report(
    cfg: ExperimentConfig, calibration: EpisodeTrace, pool: EpisodeTrace
) -> dict[str, float]:
    """Standalone coverage read used by the verification suite."""
    certificate = ConformalCertificate(alpha=cfg.rounds.alpha, delta=cfg.rounds.delta)
    residuals = (calibration.episode_return - calibration.episode_return.mean()).abs()
    quantile = certificate.quantile(residuals)
    lower = pool.episode_return - quantile
    threshold = certificate.threshold(lower, cfg.rounds.admitted_fraction)
    admitted = lower >= threshold
    coverage, se = certificate.coverage(lower, pool.episode_return, admitted)
    return {
        "quantile": quantile,
        "coverage": coverage,
        "standard_error": se,
        "admitted_fraction": float(admitted.to(torch.float32).mean()),
        "target": certificate.target_coverage,
    }
