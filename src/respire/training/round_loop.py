"""The recursion of Algorithm 1.

One pass: draw a pool from the current policy plus replay plus a perturbed arm, certify the pool on
the reference residuals, admit the upper part of the pool, refit the verifier on the admitted
episodes, move the dual variables on the measured costs, take one constrained policy step and
measure the certified return again. A round whose coverage falls below the target is rejected and
the policy is left untouched.

Ref: Algorithm 1 (lines 4-13), Eq. (3)-(4), Theorem 1 (the certified quantity), Fig. 4 (the curve).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch import Tensor

from respire.control.certificate import ConformalCertificate
from respire.control.policy import LatentFlowController
from respire.data.summary import summarise, summary_dim
from respire.data.trace import EpisodeTrace
from respire.data.windows import build_windows
from respire.flags import VariantFlags
from respire.losses.lagrangian import DualAscent, violations
from respire.models.deterministic_head import DeterministicHead
from respire.models.encoder import EncoderStandardiser, LatentEncoder
from respire.models.flow import FlowPolicy
from respire.models.verifier import ReturnVerifier
from respire.settings import ExperimentConfig
from respire.theatre.clinician import RuleBasedAnaesthetist
from respire.theatre.cohort import Cohort, episode_patients, make_environment
from respire.theatre.environment import Observation
from respire.theatre.rollout import rollout
from respire.training.checkpoint import certificate_row
from respire.training.critic_fit import build_step_dataset, fit_critics
from respire.training.policy_step import constrained_step, reference_copy
from respire.training.verifier_fit import fit_verifier
from respire.utils.logging_setup import get_logger
from respire.utils.seeding import derive_seed

LOGGER = get_logger("training.rounds")


@dataclass
class RoundRecord:
    """One row of the recursion log."""

    round_index: int
    accepted: bool
    certificate: dict[str, float | int | bool]
    costs: dict[str, float]
    policy: dict[str, float | int]
    verifier: dict[str, float | int]
    certified_return: float
    certified_return_scaled: float
    pool_return: float
    admitted_return: float

    def as_dict(self) -> dict[str, object]:
        return {
            "round": self.round_index,
            "accepted": self.accepted,
            "certificate": self.certificate,
            "costs": self.costs,
            "policy": self.policy,
            "verifier": self.verifier,
            "certified_return": self.certified_return,
            "certified_return_scaled": self.certified_return_scaled,
            "pool_return": self.pool_return,
            "admitted_return": self.admitted_return,
        }


@dataclass
class RoundRun:
    """Everything the recursion produced, in round order."""

    records: list[RoundRecord] = field(default_factory=list)
    calibration_errors: list[float] = field(default_factory=list)

    @property
    def certified_curve(self) -> list[float]:
        return [record.certified_return_scaled for record in self.records]

    @property
    def calibration_curve(self) -> list[float]:
        return list(self.calibration_errors)

    def round_summary(self) -> dict[str, list[float]]:
        return {
            "certified_return": [record.certified_return for record in self.records],
            "certified_return_scaled": self.certified_curve,
            "pool_return": [record.pool_return for record in self.records],
            "admitted_return": [record.admitted_return for record in self.records],
            "coverage": [float(record.certificate["coverage"]) for record in self.records],
            "quantile": [float(record.certificate["quantile"]) for record in self.records],
            "hypotension": [record.costs["hypotension_min"] for record in self.records],
            "suppression": [record.costs["suppression_min"] for record in self.records],
            "variability": [record.costs["variability"] for record in self.records],
        }


def _controller(
    encoder: LatentEncoder,
    standardiser: EncoderStandardiser,
    flow: FlowPolicy | DeterministicHead,
    cfg: ExperimentConfig,
    seed: int,
) -> LatentFlowController:
    return LatentFlowController(encoder, standardiser, flow, cfg.theatre, seed=seed)


def _perturbed_behaviour(behaviour: RuleBasedAnaesthetist, offset: float) -> OffsetPolicy:
    return OffsetPolicy(behaviour, offset)


class OffsetPolicy:
    """Behaviour policy with a fixed propofol offset, the shifted arm of the pool."""

    def __init__(self, behaviour: RuleBasedAnaesthetist, offset: float) -> None:
        self.behaviour = behaviour
        self.offset = offset

    def reset(self, batch: int) -> None:
        self.behaviour.reset(batch)

    def act(self, observation: Observation) -> Tensor:
        action = self.behaviour.act(observation)
        shifted = torch.stack(
            [
                (action[:, 0] + self.offset).clamp(0.0, 20.0),
                action[:, 1],
            ],
            dim=-1,
        )
        return shifted


@dataclass
class Pool:
    """Episodes, summaries and realised returns of one round's draw."""

    traces: list[EpisodeTrace]
    features: Tensor
    returns: Tensor
    costs: Tensor

    @classmethod
    def build(cls, traces: list[EpisodeTrace]) -> Pool:
        features = torch.cat([summarise(trace) for trace in traces], dim=0)
        returns = torch.cat([trace.episode_return for trace in traces], dim=0)
        costs = torch.stack(
            [
                torch.cat([trace.cost_hypotension for trace in traces], dim=0),
                torch.cat([trace.cost_suppression for trace in traces], dim=0),
                torch.cat([trace.cost_variability for trace in traces], dim=0),
            ],
            dim=-1,
        )
        return cls(traces=traces, features=features, returns=returns, costs=costs)

    def take(self, index: Tensor) -> Pool:
        traces: list[EpisodeTrace] = []
        offset = 0
        for trace in self.traces:
            count = trace.batch
            local = index[(index >= offset) & (index < offset + count)] - offset
            if local.numel() > 0:
                traces.append(trace.select(local))
            offset += count
        if not traces:
            return Pool(
                traces=[],
                features=self.features[:0],
                returns=self.returns[:0],
                costs=self.costs[:0],
            )
        return Pool(
            traces=traces,
            features=self.features[index],
            returns=self.returns[index],
            costs=self.costs[index],
        )


def _draw_pool(
    cfg: ExperimentConfig,
    cohort: Cohort,
    encoder: LatentEncoder,
    standardiser: EncoderStandardiser,
    flow: FlowPolicy | DeterministicHead,
    behaviour: RuleBasedAnaesthetist,
    replay: EpisodeTrace,
    round_index: int,
) -> Pool:
    """Policy episodes, replayed behaviour episodes and a perturbed arm."""
    base_seed = derive_seed(cfg.cohort.seed, "pool", round_index)
    episodes = max(cfg.rounds.pool_episodes, 8)
    policy_count = max(
        int(episodes * (1.0 - cfg.rounds.replay_fraction - cfg.rounds.perturb_fraction)), 4
    )
    replay_count = max(int(episodes * cfg.rounds.replay_fraction), 2)
    perturb_count = max(episodes - policy_count - replay_count, 2)

    traces: list[EpisodeTrace] = []
    controller = _controller(encoder, standardiser, flow, cfg, base_seed)
    patients = episode_patients(cohort, "development", policy_count, base_seed)
    traces.append(rollout(make_environment(patients, cfg, base_seed), controller))

    replay_index = torch.randperm(replay.batch)[: min(replay_count, replay.batch)]
    traces.append(replay.select(replay_index))

    perturbed = _perturbed_behaviour(behaviour, 1.5)
    patients = episode_patients(cohort, "development", perturb_count, base_seed + 1)
    traces.append(rollout(make_environment(patients, cfg, base_seed + 1), perturbed))
    return Pool.build(traces)


def _condition_batch(
    encoder: LatentEncoder,
    standardiser: EncoderStandardiser,
    trace: EpisodeTrace,
    window_steps: int,
) -> tuple[Tensor, Tensor]:
    windows = build_windows(trace, window_steps, stride=max(trace.steps // 24, 1))
    with torch.no_grad():
        output = encoder(standardiser.transform(windows.window), windows.actions)
    return torch.cat([output.latent, output.hidden], dim=-1), windows.actions[:, -1, :]


def run_recursion(
    cfg: ExperimentConfig,
    cohort: Cohort,
    encoder: LatentEncoder,
    standardiser: EncoderStandardiser,
    verifier: ReturnVerifier,
    flow: FlowPolicy | DeterministicHead,
    behaviour: RuleBasedAnaesthetist,
    replay: EpisodeTrace,
    seed: int,
    flags: VariantFlags | None = None,
    output_dir: str | Path | None = None,
) -> tuple[RoundRun, LatentFlowController, DualAscent]:
    """Run the rounds and return the log, the final controller and the multipliers."""
    certificate = ConformalCertificate(
        alpha=cfg.rounds.alpha,
        delta=cfg.rounds.delta,
        coverage_multiple=cfg.rounds.coverage_se_multiple,
    )
    active = flags if flags is not None else VariantFlags("full")
    duals = DualAscent(
        learning_rate=cfg.rounds.dual_lr if active.lagrangian else 0.0,
        initial=cfg.rounds.dual_init if active.lagrangian else 0.0,
    )
    budgets = torch.tensor(
        [
            cfg.rounds.cost_budget_hypotension_min,
            cfg.rounds.cost_budget_suppression_min,
            float(
                torch.quantile(replay.cost_variability, cfg.rounds.cost_budget_variability_quantile)
            ),
        ]
    )
    run = RoundRun()
    reference = reference_copy(flow)
    scale_reference = 1.0

    initial = _draw_pool(cfg, cohort, encoder, standardiser, flow, behaviour, replay, 900)
    warmup_features = torch.cat([initial.features, summarise(replay)], dim=0)
    warmup_returns = torch.cat([initial.returns, replay.episode_return], dim=0)
    fit_verifier(
        verifier,
        warmup_features,
        warmup_returns,
        cfg.verifier,
        derive_seed(seed, "verifier", "warm"),
    )

    for round_index in range(cfg.rounds.rounds):
        pool = _draw_pool(cfg, cohort, encoder, standardiser, flow, behaviour, replay, round_index)
        split_generator = torch.Generator(device="cpu")
        split_generator.manual_seed(derive_seed(seed, "split", round_index))
        order = torch.randperm(int(pool.features.shape[0]), generator=split_generator)
        reference_count = max(int(pool.features.shape[0] * cfg.rounds.calibration_fraction), 4)
        reference_pool = pool.take(order[:reference_count])
        candidate_pool = pool.take(order[reference_count:])

        with torch.no_grad():
            reference_prediction = verifier.predict(reference_pool.features)
            candidate_prediction = verifier.predict(candidate_pool.features)
        residuals = (reference_prediction - reference_pool.returns).abs()
        quantile = certificate.quantile(residuals) if active.certificate else 0.0
        lower = certificate.lower_bound(candidate_prediction, quantile)
        report, admitted = certificate.evaluate(
            lower, candidate_pool.returns, cfg.rounds.admitted_fraction, quantile
        )
        admitted_pool = candidate_pool.take(torch.nonzero(admitted, as_tuple=False).flatten())
        count = int(admitted_pool.features.shape[0])
        cost_means = (
            admitted_pool.costs.mean(dim=0) if count > 0 else candidate_pool.costs.mean(dim=0)
        )
        certified = float(lower[admitted].mean()) if bool(admitted.any()) else float("nan")
        if round_index == 0:
            scale_reference = abs(float(candidate_pool.returns.mean())) or 1.0

        record = RoundRecord(
            round_index=round_index,
            accepted=report.accepted,
            certificate=certificate_row(
                report, round_index, {"pool_return": float(candidate_pool.returns.mean())}
            ),
            costs={
                "hypotension_min": float(cost_means[0]),
                "suppression_min": float(cost_means[1]),
                "variability": float(cost_means[2]),
                "budget_variability": float(budgets[2]),
            },
            policy={},
            verifier={},
            certified_return=certified,
            certified_return_scaled=certified / scale_reference,
            pool_return=float(candidate_pool.returns.mean()),
            admitted_return=float(admitted_pool.returns.mean()) if count > 0 else float("nan"),
        )

        if not report.accepted or count < 8:
            LOGGER.info("round %d rejected; policy unchanged", round_index)
            run.records.append(record)
            run.calibration_errors.append(float(torch.sqrt((residuals**2).mean())))
            continue

        duals.update(violations(cost_means.detach(), budgets, 0.0, cfg.rounds.trust_region))
        retrain_features = torch.cat([admitted_pool.features, reference_pool.features], dim=0)
        retrain_returns = torch.cat([admitted_pool.returns, reference_pool.returns], dim=0)
        verifier_report = fit_verifier(
            verifier,
            retrain_features,
            retrain_returns,
            cfg.verifier,
            derive_seed(seed, "verifier", round_index),
        )
        record.verifier = verifier_report.as_dict()

        dataset = build_step_dataset(
            encoder,
            standardiser,
            admitted_pool.traces,
            cfg.encoder.window_steps,
            stride=max(cfg.theatre.n_steps // 24, 1),
            gamma=cfg.theatre.gamma,
        )
        critic, critic_report = fit_critics(
            dataset,
            input_dim=int(dataset.condition.shape[1]) + 2,
            seed=derive_seed(seed, "critic", round_index),
            epochs=4,
        )
        condition_parts: list[Tensor] = []
        action_parts: list[Tensor] = []
        weight_parts: list[Tensor] = []
        for trace in admitted_pool.traces:
            condition, action = _condition_batch(
                encoder, standardiser, trace, cfg.encoder.window_steps
            )
            condition_parts.append(condition)
            action_parts.append(action)
            value = float(trace.episode_return.mean())
            weight_parts.append(torch.full((condition.shape[0],), max(value, -1e6)))
        condition = torch.cat(condition_parts, dim=0)
        target_action = torch.cat(action_parts, dim=0)
        weights = torch.cat(weight_parts, dim=0)
        weights = weights - weights.min() + 1e-6

        step = constrained_step(
            flow,
            reference,
            critic,
            condition,
            target_action,
            weights,
            duals.values,
            budgets,
            cfg.rounds,
            cfg.theatre,
            derive_seed(seed, "policy", round_index),
        )
        reference = reference_copy(flow)
        record.policy = dict(step.as_dict())
        record.policy["critic_rmse"] = critic_report.train_rmse
        run.records.append(record)
        run.calibration_errors.append(verifier_report.validation_rmse)

    final = LatentFlowController(
        encoder,
        standardiser,
        flow,
        cfg.theatre,
        seed=derive_seed(seed, "final"),
        use_latent=active.latent_axes,
    )
    if output_dir is not None:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    return run, final, duals


def describe_pool(pool: Pool) -> dict[str, float]:
    return {
        "episodes": float(pool.features.shape[0]),
        "return_mean": float(pool.returns.mean()),
        "summary_width": float(pool.features.shape[1]),
    }


def summary_width(cfg: ExperimentConfig) -> int:
    return summary_dim(cfg.encoder.n_channels)
