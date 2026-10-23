"""Atomic checkpoints for every object the recursion carries between rounds.

A checkpoint is written to a temporary file in the destination directory and moved into place, so
an interrupted write cannot leave a half-file that a later round would load. The seed and the round
index travel with the weights, and a resumed run restores the seed before anything else.

Ref: Sec. 4 (frozen encoder, refitted verifier and policy), Algorithm 1 (round bookkeeping).
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import nn

from respire.control.certificate import CertificateReport
from respire.models.encoder import EncoderStandardiser, LatentEncoder
from respire.models.flow import FlowPolicy
from respire.models.pressure import OneStepPressureModel
from respire.models.verifier import ReturnVerifier
from respire.settings import EncoderConfig, PolicyConfig, VerifierConfig


def atomic_torch_save(payload: Mapping[str, Any], path: str | Path) -> Path:
    """Write a checkpoint atomically."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=destination.parent, suffix=".tmp")
    os.close(handle)
    staged = Path(temporary)
    try:
        torch.save(dict(payload), staged)
        staged.replace(destination)
    finally:
        if staged.exists():
            staged.unlink()
    return destination


@dataclass
class ReleaseBundle:
    """All modules one round needs, plus the round's own bookkeeping."""

    encoder: LatentEncoder
    standardiser: EncoderStandardiser
    verifier: ReturnVerifier
    flow: FlowPolicy
    pressure: OneStepPressureModel
    seed: int = 0
    round_index: int = 0
    metrics: dict[str, float] = field(default_factory=dict)
    certificates: list[dict[str, float | int | bool]] = field(default_factory=list)

    def save(self, path: str | Path) -> Path:
        payload: dict[str, Any] = {
            "encoder": self.encoder.state_dict(),
            "verifier": self.verifier.state_dict(),
            "flow": self.flow.state_dict(),
            "standardiser": self.standardiser.state(),
            "pressure": self.pressure.state_dict(),
            "encoder_config": self.encoder.cfg.__dict__,
            "verifier_config": self.verifier.cfg.__dict__,
            "policy_config": self.flow.cfg.__dict__,
            "summary_dim": self.verifier.summary_dim,
            "cond_dim": self.flow.cond_dim,
            "seed": self.seed,
            "round_index": self.round_index,
            "metrics": self.metrics,
            "certificates": self.certificates,
        }
        return atomic_torch_save(payload, path)

    @classmethod
    def load(cls, path: str | Path) -> ReleaseBundle:
        payload = torch.load(Path(path), map_location="cpu", weights_only=False)
        encoder_cfg = EncoderConfig(**payload["encoder_config"])
        verifier_cfg = VerifierConfig(**payload["verifier_config"])
        policy_cfg = PolicyConfig(**payload["policy_config"])
        encoder = LatentEncoder(encoder_cfg)
        encoder.load_state_dict(payload["encoder"])
        verifier = ReturnVerifier(verifier_cfg, int(payload["summary_dim"]))
        verifier.load_state_dict(payload["verifier"])
        flow = FlowPolicy(policy_cfg, int(payload["cond_dim"]))
        flow.load_state_dict(payload["flow"])
        standardiser = EncoderStandardiser.from_state(payload["standardiser"])
        pressure = OneStepPressureModel.from_state(payload["pressure"])
        return cls(
            encoder=encoder,
            standardiser=standardiser,
            verifier=verifier,
            flow=flow,
            pressure=pressure,
            seed=int(payload["seed"]),
            round_index=int(payload["round_index"]),
            metrics={str(k): float(v) for k, v in payload["metrics"].items()},
            certificates=list(payload["certificates"]),
        )


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def certificate_row(
    report: CertificateReport, round_index: int, extra: Mapping[str, float]
) -> dict[str, float | int | bool]:
    """Flatten a certificate into the row the round log stores."""
    row: dict[str, float | int | bool] = {"round": round_index}
    row.update(report.as_dict())
    row.update({key: float(value) for key, value in extra.items()})
    return row
