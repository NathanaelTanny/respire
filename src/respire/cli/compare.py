"""Build the primary comparison table.

Analytic arms need no training and are evaluated from their parameters; the learned arms can be
fitted for a stated number of iterations, which is what the comparison protocol requires. The
proposed controller is read from a saved bundle so the table and the released weights are the same
object.

Ref: Table 1 (rows and columns), Table 3 (the paired safety comparison).
"""

from __future__ import annotations

from pathlib import Path

import torch

from respire.cli.common import base_parser, emit, output_directory, resolve, write_json
from respire.comparators.base import train_offline, train_online, transitions_from_trace
from respire.comparators.registry import analytic_names, build_analytic, build_learning, spec
from respire.control.adaptation import RidgeAdapter
from respire.control.deployment import DeployedController, OverrideRule
from respire.control.policy import LatentFlowController
from respire.control.shield import AnalyticShield
from respire.evaluation.arms import ArmResult, best_unconstrained, evaluate_arm
from respire.evaluation.comparison import build_comparison
from respire.pipeline import prepare_data
from respire.settings import ExperimentConfig
from respire.theatre.clinician import RuleBasedAnaesthetist
from respire.theatre.cohort import build_cohort, episode_patients
from respire.theatre.patient import Patient
from respire.training.checkpoint import ReleaseBundle
from respire.utils.logging_setup import configure
from respire.utils.seeding import derive_seed, set_seed

DEFAULT_ARMS = (*analytic_names(), "respire", "clinician")


def _evaluation_patients(config: ExperimentConfig, seed: int) -> Patient:
    cohort = build_cohort(config)
    return episode_patients(
        cohort,
        "evaluation",
        min(config.evaluation.heldout_cases, 128, int(cohort.splits.evaluation.numel())),
        derive_seed(seed, "eval"),
    )


def _respire_policy(config: ExperimentConfig, bundle: ReleaseBundle) -> DeployedController:
    shield = AnalyticShield(bundle.pressure, config.theatre, config.shield)
    controller = LatentFlowController(
        bundle.encoder,
        bundle.standardiser,
        bundle.flow,
        config.theatre,
        shield=shield,
        seed=derive_seed(bundle.seed, "compare"),
    )
    warmup = int(5.0 * 60.0 / config.theatre.dt_s)
    return DeployedController(
        controller,
        RuleBasedAnaesthetist(config.theatre, noise_sd=0.0),
        RidgeAdapter(),
        warmup_steps=warmup,
        override=OverrideRule(map_floor=config.theatre.map_floor, dt_s=config.theatre.dt_s),
        dt_s=config.theatre.dt_s,
    )


def _fit_learning_arm(
    name: str,
    config: ExperimentConfig,
    patients: Patient,
    data: object,
    iterations: int,
    seed: int,
) -> object:
    agent = build_learning(name, config.theatre)
    arm_spec = spec(name)
    if arm_spec.family == "online":
        if name in {"ppo", "ppo_lagr", "cppo"}:
            agent.fit(patients, config, iterations, seed)  # type: ignore[attr-defined]
        else:
            train_online(agent, config, patients, iterations, seed=seed)  # type: ignore[arg-type]
    else:
        from respire.pipeline import PreparedData

        if not isinstance(data, PreparedData):
            raise TypeError("offline arms need the prepared development pool")
        trace = data.development
        dataset = transitions_from_trace(trace)
        train_offline(agent, dataset, max(iterations, 1), 256, seed)  # type: ignore[arg-type]
    return agent


def main(argv: list[str] | None = None) -> int:
    parser = base_parser("evaluate the comparison arms on the held-out cohort")
    parser.add_argument("--bundle", default="runs/rounds/bundle.pt")
    parser.add_argument("--arms", default=",".join(DEFAULT_ARMS))
    parser.add_argument(
        "--iterations", type=int, default=0, help="training iterations for learned arms"
    )
    parser.add_argument("--treated", default="respire")
    parser.add_argument("--control", default="dreamer")
    args = parser.parse_args(argv)
    config = resolve(args)
    configure(args.log_level)
    set_seed(config.seed)
    patients = _evaluation_patients(config, config.seed)
    requested = [item.strip() for item in args.arms.split(",") if item.strip()]
    results: list[ArmResult] = []
    data = None
    learned_requested = any(
        name not in {"respire", "clinician"} and spec(name).learns for name in requested
    )
    if learned_requested:
        data = prepare_data(config, config.seed)
    for name in requested:
        if name == "clinician":
            results.append(
                evaluate_arm(
                    "clinician",
                    RuleBasedAnaesthetist(config.theatre, noise_sd=0.05),
                    patients,
                    config,
                    config.seed,
                )
            )
            continue
        if name == "respire":
            bundle = ReleaseBundle.load(args.bundle)
            results.append(
                evaluate_arm(
                    "respire", _respire_policy(config, bundle), patients, config, config.seed
                )
            )
            continue
        if spec(name).learns:
            if args.iterations <= 0:
                continue
            agent = _fit_learning_arm(name, config, patients, data, args.iterations, config.seed)
            results.append(evaluate_arm(name, agent, patients, config, config.seed))  # type: ignore[arg-type]
            continue
        arm = build_analytic(name, config.theatre)
        results.append(evaluate_arm(name, arm, patients, config, config.seed))  # type: ignore[arg-type]
    if not results:
        print("no arm was evaluated")
        return 1
    control = args.control
    available = {item.name for item in results}
    if control not in available:
        control = best_unconstrained(results, avoid=("respire", "clinician")).name
    treated = args.treated if args.treated in available else results[0].name
    reference = "clinician" if "clinician" in available else results[0].name
    comparison = build_comparison(results, treated, control, reference, config.evaluation)
    rows = [item.as_dict() for item in results]
    print(
        emit(rows, ["arm", "bis_mae", "tib", "hypotension", "burst", "pressure_area", "propofol"])
    )
    print(
        emit(
            [row.as_dict() for row in comparison.safety],
            ["event", "treated", "control", "difference", "ci_low", "ci_high", "p_value"],
        )
    )
    directory = output_directory(config, "comparison")
    write_json(directory / "comparison.json", comparison.as_dict())
    torch.save(
        {item.name: {"hypotension": item.hypotension_by_case} for item in results},
        directory / "per_case.pt",
    )
    print(f"artefacts in {directory} ({Path(args.bundle).name} for the proposed controller)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
