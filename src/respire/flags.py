"""Switches that remove exactly one element of the system.

Every row of the component analysis is one of these settings. The switches are read in the driver
and in the round loop, nowhere else, so a variant cannot quietly change two things at once.

Ref: Table 2 (the nine rows of the component analysis).
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from respire.settings import ExperimentConfig


@dataclass(frozen=True)
class VariantFlags:
    """One row of the component analysis."""

    name: str
    latent_axes: bool = True
    interventional: bool = True
    recursion: bool = True
    certificate: bool = True
    shield: bool = True
    lagrangian: bool = True
    adaptation: bool = True
    flow_head: bool = True

    @property
    def is_full(self) -> bool:
        return all(
            [
                self.latent_axes,
                self.interventional,
                self.recursion,
                self.certificate,
                self.shield,
                self.lagrangian,
                self.adaptation,
                self.flow_head,
            ]
        )

    def as_dict(self) -> dict[str, bool | str]:
        return {
            "variant": self.name,
            "latent_axes": self.latent_axes,
            "interventional": self.interventional,
            "recursion": self.recursion,
            "certificate": self.certificate,
            "shield": self.shield,
            "lagrangian": self.lagrangian,
            "adaptation": self.adaptation,
            "flow_head": self.flow_head,
        }


FULL = VariantFlags("full")


def without(element: str) -> VariantFlags:
    """The variant with one element removed, named after that element."""
    mapping = {
        "latent_axes": ("no latent axes", "latent_axes"),
        "interventional": ("no interventional loss", "interventional"),
        "recursion": ("no recursive operator", "recursion"),
        "certificate": ("no conformal certificate", "certificate"),
        "shield": ("no shield", "shield"),
        "lagrangian": ("no Lagrangian costs", "lagrangian"),
        "adaptation": ("no test-time adaptation", "adaptation"),
        "flow_head": ("no flow head", "flow_head"),
    }
    if element not in mapping:
        raise KeyError(f"unknown element {element!r}")
    label, field = mapping[element]
    return replace(FULL, name=label, **{field: False})


COMPONENTS: tuple[VariantFlags, ...] = (
    FULL,
    without("latent_axes"),
    without("interventional"),
    without("recursion"),
    without("certificate"),
    without("shield"),
    without("lagrangian"),
    without("adaptation"),
    without("flow_head"),
)


def configure(cfg: ExperimentConfig, flags: VariantFlags) -> ExperimentConfig:
    """Apply the switches that live in the configuration rather than in the loop."""
    rounds = cfg.rounds if flags.recursion else replace(cfg.rounds, rounds=0)
    encoder = cfg.encoder
    if not flags.interventional:
        encoder = replace(encoder, interventional_weight=0.0)
    return replace(cfg, rounds=rounds, encoder=encoder)


def by_name(name: str) -> VariantFlags:
    for variant in COMPONENTS:
        if variant.name == name:
            return variant
    raise KeyError(f"unknown variant {name!r}")
