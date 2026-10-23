"""Component analysis: one element removed at a time.

Each variant removes a single element and is otherwise identical, so the differences between rows
of the table are attributable to that element. The encoder is refitted for the variant that removes
the interventional term, because that term is part of the encoder's own objective.

Ref: Table 2 (component analysis on the held-out robot-assisted cases).
"""

from __future__ import annotations

import torch

from respire.cli.common import base_parser, emit, output_directory, resolve, write_json
from respire.data.summary import summary_dim
from respire.flags import COMPONENTS, VariantFlags, by_name
from respire.pipeline import run_pipeline
from respire.utils.logging_setup import configure

COLUMNS = ["variant", "bis_mae", "tib", "hypotension", "burst"]


def _run(config: object, flags: VariantFlags, seed: int) -> dict[str, object]:
    from respire.settings import ExperimentConfig

    if not isinstance(config, ExperimentConfig):
        raise TypeError("expected an ExperimentConfig")
    result = run_pipeline(config, flags, seed=seed)
    row = result.evaluation.row()
    return {
        "variant": flags.name,
        "bis_mae": row["bis_mae"],
        "tib": row["tib"],
        "hypotension": row["hypotension"],
        "burst": row["burst"],
        "propofol": row["propofol"],
        "rate_variability": row["rate_variability"],
        "flags": flags.as_dict(),
        "encoder_axes": result.encoder_report.axis_correlation,
        "certified": result.run.certified_curve,
        "summary_width": summary_dim(config.encoder.n_channels),
    }


def main(argv: list[str] | None = None) -> int:
    parser = base_parser("run the component analysis")
    parser.add_argument("--variants", default=",".join(variant.name for variant in COMPONENTS))
    args = parser.parse_args(argv)
    config = resolve(args)
    configure(args.log_level)
    rows: list[dict[str, object]] = []
    for name in [item.strip() for item in args.variants.split(",") if item.strip()]:
        flags = by_name(name)
        rows.append(_run(config, flags, config.seed))
    print(emit(rows, COLUMNS))
    directory = output_directory(config, "components")
    write_json(directory / "components.json", {"seed": config.seed, "rows": rows})
    torch.save({"rows": rows}, directory / "components.pt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
