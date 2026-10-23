"""Readers for the official exports of the credentialed corpora.

The corpora are not redistributed, so the release ships the readers and the layout they expect. A
reader turns one case export into the same twelve-channel block the theatre produces, which is
what lets the encoder and the metrics run on a record without a second code path.

None of these readers infers a column that is absent: a missing channel is a refusal, because a
silently zero-filled pressure trace would make every downstream metric meaningless.

Ref: Sec. 5.1 (four thousand eight hundred and six intraoperative records; 1,876 sedation episodes).
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from respire.theatre.environment import CHANNELS

VITALDB_COLUMNS: dict[str, str] = {
    "time_s": "time_s",
    "bis": "bis",
    "heart_rate": "hr",
    "map": "map",
    "systolic": "sbp",
    "diastolic": "dbp",
    "spo2": "spo2",
    "etco2": "etco2",
    "temperature": "temp",
    "train_of_four": "tof",
    "dopamine": "dopamine_rate",
    "propranolol": "propranolol_rate",
    "pneumo_pressure": "pneumo_pressure",
}

MIMIC_COLUMNS: dict[str, str] = {
    "time_s": "charttime_s",
    "bis": "bis",
    "heart_rate": "heart_rate",
    "map": "mbp",
    "systolic": "sbp",
    "diastolic": "dbp",
    "spo2": "spo2",
    "etco2": "etco2",
    "temperature": "temperature",
    "train_of_four": "tof_count",
    "dopamine": "dopamine_rate",
    "propranolol": "propranolol_rate",
    "pneumo_pressure": "pneumo_pressure",
}


class ExportError(ValueError):
    """Raised when an export does not carry the columns the reader needs."""


@dataclass(frozen=True)
class RecordedCase:
    """One case as a twelve-channel block on a fixed decision grid, plus the delivered rates."""

    identifier: str
    time_s: np.ndarray
    channels: np.ndarray
    propofol: np.ndarray
    remifentanil: np.ndarray
    source: str

    @property
    def steps(self) -> int:
        return int(self.channels.shape[0])

    @property
    def channels_named(self) -> tuple[str, ...]:
        return CHANNELS

    def channel(self, name: str) -> np.ndarray:
        try:
            index = CHANNELS.index(name)
        except ValueError as error:
            raise KeyError(f"unknown channel {name!r}") from error
        return self.channels[:, index]


def _read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        header = list(reader.fieldnames or [])
    if not header:
        raise ExportError(f"{path.name} has no header row")
    if not rows:
        raise ExportError(f"{path.name} has no data rows")
    return header, rows


def read_case(
    path: str | Path,
    schema: dict[str, str],
    identifier: str | None = None,
    propofol_column: str = "propofol_rate",
    remifentanil_column: str = "remifentanil_rate",
) -> RecordedCase:
    """Read one case export with an explicit column schema."""
    source = Path(path)
    header, rows = _read_rows(source)
    missing = [name for name in schema.values() if name not in header]
    if missing:
        raise ExportError(f"{source.name} is missing columns: {sorted(missing)}")
    for column in (propofol_column, remifentanil_column):
        if column not in header:
            raise ExportError(f"{source.name} is missing the infusion column {column}")

    time_s = np.array([float(row[schema["time_s"]]) for row in rows], dtype=np.float64)
    channels = np.array(
        [[float(row[schema[name]]) for name in CHANNELS] for row in rows], dtype=np.float64
    )
    propofol = np.array([float(row[propofol_column]) for row in rows], dtype=np.float64)
    remifentanil = np.array([float(row[remifentanil_column]) for row in rows], dtype=np.float64)
    if np.isnan(channels).any():
        raise ExportError(f"{source.name} contains a missing channel value")
    if not np.all(np.diff(time_s) > 0):
        raise ExportError(f"{source.name} is not ordered by time")
    return RecordedCase(
        identifier=identifier or source.stem,
        time_s=time_s,
        channels=channels,
        propofol=propofol,
        remifentanil=remifentanil,
        source=source.name,
    )


def read_vitaldb_case(path: str | Path, identifier: str | None = None) -> RecordedCase:
    """Read a case exported from the VitalDB case file."""
    return read_case(path, VITALDB_COLUMNS, identifier)


def read_mimic_case(path: str | Path, identifier: str | None = None) -> RecordedCase:
    """Read a derived sedation episode exported from the MIMIC-IV tables."""
    return read_case(path, MIMIC_COLUMNS, identifier)


def resample(case: RecordedCase, dt_s: float) -> RecordedCase:
    """Put a case on the decision grid by linear interpolation, without inventing channels."""
    if case.steps < 2:
        raise ExportError("a case needs at least two samples to be resampled")
    grid = np.arange(case.time_s[0], case.time_s[-1] + dt_s * 0.5, dt_s)
    channels = np.stack(
        [
            np.interp(grid, case.time_s, case.channels[:, index])
            for index in range(case.channels.shape[1])
        ],
        axis=-1,
    )
    propofol = np.interp(grid, case.time_s, case.propofol)
    remifentanil = np.interp(grid, case.time_s, case.remifentanil)
    return RecordedCase(
        identifier=case.identifier,
        time_s=grid,
        channels=channels,
        propofol=propofol,
        remifentanil=remifentanil,
        source=case.source,
    )


def windows(case: RecordedCase, length: int, stride: int = 1) -> np.ndarray:
    """Cut rolling windows from a recorded case, shape ``[n, length, channels]``."""
    if length > case.steps:
        raise ExportError("the case is shorter than one window")
    starts = range(0, case.steps - length + 1, stride)
    return np.stack([case.channels[start : start + length] for start in starts], axis=0)


def detect_layout(root: Path) -> str | None:
    """Recognise the documented directory layouts, or return ``None``."""
    if not root.exists() or not root.is_dir():
        return None
    names = {entry.name for entry in root.iterdir()}
    if {"cases", "waves"} & names:
        return "vitaldb"
    if {"hosp", "icu"} <= names:
        return "mimic-iv"
    if {"patient.csv", "infusionDrug.csv"} <= names:
        return "eicu-crd"
    return None


def describe_export(root: str | Path) -> dict[str, object]:
    """Report what an export contains before any of it is read.

    This is the step a credentialed user runs first: it names the layout, states that nothing may be
    redistributed, and lists the channel schema the reader will insist on.
    """
    directory = Path(root)
    layout = detect_layout(directory)
    report: dict[str, object] = {
        "root": directory.name,
        "layout": layout or "unrecognised",
        "redistributable": False,
        "channels_required": list(CHANNELS),
        "note": "the export stays under its own agreement; nothing is copied into the release",
    }
    if layout is None:
        report["expected"] = [
            "cases/ or waves/ (VitalDB)",
            "hosp/ and icu/ (MIMIC-IV)",
            "patient.csv and infusionDrug.csv (eICU-CRD)",
        ]
        return report
    tables = sorted(entry.name for entry in directory.glob("*.csv"))
    report["tables"] = tables[:32]
    if tables:
        with (directory / tables[0]).open(
            "r", encoding="utf-8", errors="replace", newline=""
        ) as handle:
            header = next(csv.reader(handle), [])
        report["first_table_columns"] = [column.strip() for column in header][:32]
    return report
