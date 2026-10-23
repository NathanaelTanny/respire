"""Shared argument handling for the command line."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from respire.settings import ExperimentConfig, load_experiment
from respire.utils.logging_setup import configure, level_from_env


def base_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--config", default="configs/round/primary.yaml", help="experiment document"
    )
    parser.add_argument("--base", default=None, help="document to inherit before applying --config")
    parser.add_argument("--override", action="append", default=[], help="dotted key=value")
    parser.add_argument("--seed", type=int, default=None, help="override the document seed")
    parser.add_argument("--output", default=None, help="directory for artefacts")
    parser.add_argument("--log-level", default=level_from_env())
    return parser


def resolve(parser_args: Any) -> ExperimentConfig:
    configure(parser_args.log_level)
    config = load_experiment(parser_args.config, parser_args.override, parser_args.base)
    if parser_args.seed is not None:
        from dataclasses import replace

        config = replace(config, seed=int(parser_args.seed))
    if parser_args.output is not None:
        from dataclasses import replace

        config = replace(config, output_dir=str(parser_args.output))
    return config


def output_directory(config: ExperimentConfig, name: str) -> Path:
    directory = Path(config.output_dir) / name
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    return destination


def read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        loaded = json.load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"{path} does not contain an object")
    return loaded


def emit(rows: Sequence[Mapping[str, object]], keys: Sequence[str]) -> str:
    """Fixed-width table for the console."""
    widths = {key: max(len(key), 12) for key in keys}
    header = "  ".join(key.ljust(widths[key]) for key in keys)
    lines = [header, "-" * len(header)]
    for row in rows:
        cells = []
        for key in keys:
            value = row.get(key, "")
            if isinstance(value, float):
                cells.append(f"{value:12.3f}")
            else:
                cells.append(str(value).ljust(widths[key]))
        lines.append("  ".join(cells))
    return "\n".join(lines)
