"""Generate the cohort and export its behaviour pool.

The cohort is reproducible from the document's seed alone, so an exported trace can be regenerated
byte for byte and compared against the one a reader has.

Ref: Sec. 5.1 (2{,}000 patients, 120-minute episodes, five-second decisions).
"""

from __future__ import annotations

import torch

from respire.cli.common import base_parser, output_directory, resolve, write_json
from respire.pipeline import prepare_data
from respire.utils.logging_setup import configure


def main(argv: list[str] | None = None) -> int:
    parser = base_parser("build the cohort and its behaviour pool")
    parser.add_argument("--cases", type=int, default=None, help="override the encoder case count")
    args = parser.parse_args(argv)
    config = resolve(args)
    configure(args.log_level)
    if args.cases is not None:
        from dataclasses import replace

        config = replace(config, cohort=replace(config.cohort, encoder_cases=args.cases))
    data = prepare_data(config, config.seed)
    directory = output_directory(config, "cohort")
    torch.save(
        {
            "development_return": data.development.episode_return,
            "calibration_return": data.calibration.episode_return,
            "development_observation": data.development.observation,
            "axes": data.cohort.patients.axes,
        },
        directory / "cohort.pt",
    )
    write_json(
        directory / "cohort.json",
        {
            "seed": config.seed,
            "patients": data.cohort.size,
            "splits": {
                name: int(index.numel()) for name, index in data.cohort.splits.named().items()
            },
            "pool": data.as_summary(),
            "development_return_mean": float(data.development.episode_return.mean()),
            "calibration_return_mean": float(data.calibration.episode_return.mean()),
        },
    )
    print(f"cohort written to {directory}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
