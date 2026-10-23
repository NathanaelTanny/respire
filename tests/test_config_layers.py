"""The section documents and the primary document must not drift apart.

`configs/{theatre,encoder,controller}` hold the same values as the corresponding sections of
`configs/round/primary.yaml`, because they are meant to be read next to the code they configure.
This test is what keeps the two copies equal: it walks every leaf of every section document and
compares it with the primary document at the same path.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from respire.settings import load_experiment

ROOT = Path(__file__).resolve().parents[1]
SECTIONS = ("configs/theatre", "configs/encoder", "configs/controller")


def _leaves(node: Any, prefix: str = "") -> list[tuple[str, Any]]:
    if isinstance(node, dict):
        out: list[tuple[str, Any]] = []
        for key, value in node.items():
            out.extend(_leaves(value, f"{prefix}.{key}" if prefix else str(key)))
        return out
    return [(prefix, node)]


def _lookup(document: dict[str, Any], path: str) -> Any:
    cursor: Any = document
    for part in path.split("."):
        if not isinstance(cursor, dict) or part not in cursor:
            raise KeyError(path)
        cursor = cursor[part]
    return cursor


@pytest.mark.parametrize("directory", SECTIONS)
def test_section_matches_the_primary_document(directory: str) -> None:
    primary_path = ROOT / "configs" / "round" / "primary.yaml"
    primary = yaml.safe_load(primary_path.read_text(encoding="utf-8"))
    for document in sorted((ROOT / directory).glob("*.yaml")):
        loaded = yaml.safe_load(document.read_text(encoding="utf-8"))
        for path, value in _leaves(loaded):
            assert _lookup(primary, path) == value, f"{document.name} drifts at {path}"


def test_section_documents_load_as_base_documents() -> None:
    base = ROOT / "configs" / "theatre" / "eleveld_adult.yaml"
    config = load_experiment(ROOT / "configs" / "round" / "primary.yaml", base_path=base)
    assert config.theatre.dt_s == 5.0
    assert config.encoder.n_blocks == 6
