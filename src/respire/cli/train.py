"""Fit and run the recursion, then write the bundle and the round log.

One command per document under ``configs/round``. The bundle carries the seed it was trained with;
loading it restores that seed before anything else runs.

Ref: Algorithm 1, Fig. 4 (the round log).
"""

from __future__ import annotations

import json

from respire.cli.common import base_parser, emit, output_directory, resolve, write_json
from respire.flags import by_name
from respire.pipeline import run_pipeline
from respire.utils.logging_setup import configure


def main(argv: list[str] | None = None) -> int:
    parser = base_parser("fit the encoder and run the improvement rounds")
    parser.add_argument("--variant", default="full", help="component variant to run")
    args = parser.parse_args(argv)
    config = resolve(args)
    configure(args.log_level)
    flags = by_name(args.variant)
    directory = output_directory(config, "rounds")
    result = run_pipeline(config, flags, seed=config.seed, output_dir=directory)
    result.bundle.save(directory / "bundle.pt")
    curve = result.run.round_summary()
    write_json(
        directory / "rounds.json",
        {
            "variant": flags.name,
            "seed": config.seed,
            "curve": curve,
            "records": [record.as_dict() for record in result.run.records],
            "encoder": result.encoder_report.as_dict(),
            "verifier": result.verifier_report.as_dict(),
            "evaluation": result.evaluation.row(),
            "behaviour": result.behaviour.row(),
        },
    )
    rows: list[dict[str, object]] = [
        {
            "round": record.round_index,
            "accepted": record.accepted,
            "coverage": record.certificate["coverage"],
            "certified": record.certified_return_scaled,
            "pool": record.pool_return,
        }
        for record in result.run.records
    ]
    print(emit(rows, ["round", "accepted", "coverage", "certified", "pool"]))
    print(json.dumps(result.evaluation.row(), indent=2))
    print(f"artefacts in {directory}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
