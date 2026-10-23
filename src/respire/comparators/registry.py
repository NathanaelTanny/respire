"""Named construction of every comparison arm.

Each entry states whether the arm learns and whether it enforces a constraint, so the comparison
table can be read against the release without opening the individual modules.

Ref: Table 1 (the thirteen arms and the observational clinician row).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from respire.comparators.constrained_pg import ConstrainedPPOAgent, LagrangianAgent
from respire.comparators.cql import CQLAgent
from respire.comparators.dreamer import DreamerAgent
from respire.comparators.internal_model import InternalModelController
from respire.comparators.iql import IQLAgent
from respire.comparators.pid import PIDDepthController
from respire.comparators.ppo import PPOAgent
from respire.comparators.sac import SACAgent
from respire.comparators.tci import eleveld_arm, schnider_arm
from respire.comparators.td3bc import TD3BCAgent
from respire.settings import TheatreConfig


@dataclass(frozen=True)
class ArmSpec:
    """Metadata of one arm."""

    name: str
    learns: bool
    constrained: bool
    family: str


SPECS: tuple[ArmSpec, ...] = (
    ArmSpec("schnider_tcu", False, False, "model-based"),
    ArmSpec("eleveld_tcu", False, False, "model-based"),
    ArmSpec("pid", False, False, "model-based"),
    ArmSpec("internal_model", False, False, "model-based"),
    ArmSpec("bc", True, False, "imitation"),
    ArmSpec("sac", True, False, "online"),
    ArmSpec("ppo", True, False, "online"),
    ArmSpec("td3bc", True, False, "offline"),
    ArmSpec("cql", True, False, "offline"),
    ArmSpec("iql", True, False, "offline"),
    ArmSpec("ppo_lagr", True, True, "online"),
    ArmSpec("cppo", True, True, "online"),
    ArmSpec("dreamer", True, False, "world-model"),
)


def spec(name: str) -> ArmSpec:
    for item in SPECS:
        if item.name == name:
            return item
    raise KeyError(f"unknown arm {name!r}; known arms are {[item.name for item in SPECS]}")


def build_analytic(name: str, theatre: TheatreConfig) -> object:
    """Construct one of the arms that needs no training."""
    if name == "schnider_tcu":
        return schnider_arm(theatre)
    if name == "eleveld_tcu":
        return eleveld_arm(theatre)
    if name == "pid":
        return PIDDepthController(theatre)
    if name == "internal_model":
        return InternalModelController(theatre)
    raise KeyError(f"{name!r} is not an analytic arm")


def build_learning(name: str, theatre: TheatreConfig) -> object:
    """Construct one of the arms that is trained against the theatre."""
    constructors: dict[str, Callable[[], object]] = {
        "sac": lambda: SACAgent(theatre),
        "ppo": lambda: PPOAgent(),
        "td3bc": lambda: TD3BCAgent(theatre),
        "cql": lambda: CQLAgent(theatre),
        "iql": lambda: IQLAgent(theatre),
        "ppo_lagr": lambda: LagrangianAgent(),
        "cppo": lambda: ConstrainedPPOAgent(),
        "dreamer": lambda: DreamerAgent(theatre),
    }
    if name not in constructors:
        raise KeyError(f"{name!r} is not a learning arm")
    return constructors[name]()


def names() -> list[str]:
    return [item.name for item in SPECS]


def learned_names() -> list[str]:
    return [item.name for item in SPECS if item.learns]


def analytic_names() -> list[str]:
    return [item.name for item in SPECS if not item.learns]
