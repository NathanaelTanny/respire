"""Evaluate a saved bundle on the held-out split.

Reports the utility, safety and consumption columns of the primary comparison for the loaded
controller, next to the observational reference, and writes the per-case vectors so the paired
statistics can be recomputed downstream.

Ref: Table 1, Table 3 (metric definitions), Sec. 5.1 (held-out protocol).
"""

from __future__ import annotations

import torch

from respire.cli.common import base_parser, emit, output_directory, resolve, write_json
from respire.control.adaptation import RidgeAdapter
from respire.control.deployment import DeployedController, OverrideRule
from respire.control.policy import LatentFlowController
from respire.control.shield import AnalyticShield
from respire.evaluation.arms import evaluate_arm
from respire.metrics.statistics import bootstrap_interval
from respire.theatre.clinician import RuleBasedAnaesthetist
from respire.theatre.cohort import build_cohort, episode_patients
from respire.training.checkpoint import ReleaseBundle
from respire.utils.logging_setup import configure
from respire.utils.seeding import derive_seed, set_seed


def main(argv: list[str] | None = None) -> int:
    parser = base_parser("evaluate a saved bundle on the held-out cohort")
    parser.add_argument(
        "--bundle", default="runs/rounds/bundle.pt", help="path to the saved bundle"
    )
    parser.add_argument(
        "--no-shield", action="store_true", help="evaluate the policy without the guard"
    )
    args = parser.parse_args(argv)
    config = resolve(args)
    configure(args.log_level)
    bundle = ReleaseBundle.load(args.bundle)
    set_seed(bundle.seed)
    cohort = build_cohort(config)
    patients = episode_patients(
        cohort,
        "evaluation",
        min(config.evaluation.heldout_cases, int(cohort.splits.evaluation.numel())),
        derive_seed(bundle.seed, "eval"),
    )
    shield = (
        None if args.no_shield else AnalyticShield(bundle.pressure, config.theatre, config.shield)
    )
    controller = LatentFlowController(
        bundle.encoder,
        bundle.standardiser,
        bundle.flow,
        config.theatre,
        shield=shield,
        seed=derive_seed(bundle.seed, "eval-controller"),
    )
    warmup = int(5.0 * 60.0 / config.theatre.dt_s)
    deployed = DeployedController(
        controller,
        RuleBasedAnaesthetist(config.theatre, noise_sd=0.0),
        RidgeAdapter(),
        warmup_steps=warmup,
        override=OverrideRule(map_floor=config.theatre.map_floor, dt_s=config.theatre.dt_s),
        dt_s=config.theatre.dt_s,
    )
    treated = evaluate_arm("respire", deployed, patients, config, bundle.seed)
    reference = evaluate_arm(
        "clinician",
        RuleBasedAnaesthetist(config.theatre, noise_sd=0.05),
        patients,
        config,
        bundle.seed,
    )
    rows = [treated.as_dict(), reference.as_dict()]
    print(
        emit(rows, ["arm", "bis_mae", "tib", "hypotension", "burst", "pressure_area", "propofol"])
    )
    low, high = bootstrap_interval(
        treated.hypotension_by_case,
        reference.hypotension_by_case,
        config.evaluation.bootstrap_samples,
        config.evaluation.bootstrap_alpha,
        seed=bundle.seed,
    )
    directory = output_directory(config, "evaluation")
    write_json(
        directory / "evaluation.json",
        {
            "bundle": str(args.bundle),
            "seed": bundle.seed,
            "cases": int(treated.depth_by_case.numel()),
            "respire": treated.row(),
            "clinician": reference.row(),
            "hypotension_difference": {
                "value": treated.hypotension - reference.hypotension,
                "ci_low": low,
                "ci_high": high,
            },
            "shield": not args.no_shield,
            "diagnostics": controller.diagnostics.summary(),
            "overrides": deployed.overrides,
        },
    )
    torch.save(
        {
            "depth_by_case": treated.depth_by_case,
            "hypotension_by_case": treated.hypotension_by_case,
            "area_by_case": treated.area_by_case,
        },
        directory / "per_case.pt",
    )
    print(f"artefacts in {directory}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
