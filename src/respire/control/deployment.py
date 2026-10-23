"""Shielded deployment with test-time latent adaptation.

The first five minutes are the anaesthetist's; the controller takes over once the latent prior has
been adapted on that stretch. A supervisor override hands control back for the remainder of the
case and is recorded, so the trace keeps the override as an event rather than as a silent gap.

Ref: Algorithm 2 (lines 1-11), Sec. 4 (shield before the pump).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor

from respire.control.adaptation import RidgeAdapter
from respire.control.policy import LatentFlowController
from respire.theatre.environment import Observation
from respire.theatre.rollout import ActingPolicy


@dataclass
class OverrideRule:
    """Hands control back when the case leaves the envelope the controller was certified on."""

    map_floor: float = 65.0
    suppression_floor: float = 20.0
    sustained_min: float = 1.0
    dt_s: float = 5.0
    triggered: Tensor = field(default_factory=lambda: torch.zeros(0, dtype=torch.bool))
    episodes: int = 0

    def reset(self, batch: int) -> None:
        self.triggered = torch.zeros(batch, dtype=torch.bool)
        self.episodes = 0

    def check(self, observation: Observation, elapsed_min: float) -> Tensor:
        """Boolean mask of episodes that must revert to clinician rates."""
        pressure = observation.current[:, 2]
        depth = observation.current[:, 0]
        below = (pressure < self.map_floor) | (depth < self.suppression_floor)
        minutes = elapsed_min if elapsed_min > 0 else 0.0
        if below.any() and minutes >= self.sustained_min:
            newly = below & ~self.triggered
            self.episodes += int(newly.sum())
            self.triggered = self.triggered | newly
        return self.triggered.clone()


class DeployedController:
    """Warm start, adaptation, then policy plus shield, with the override in the loop."""

    def __init__(
        self,
        controller: LatentFlowController,
        behaviour: ActingPolicy,
        adapter: RidgeAdapter,
        warmup_steps: int,
        override: OverrideRule | None = None,
        dt_s: float = 5.0,
    ) -> None:
        self.controller = controller
        self.behaviour = behaviour
        self.adapter = adapter
        self.warmup_steps = warmup_steps
        self.override = override if override is not None else OverrideRule(dt_s=dt_s)
        self.dt_s = dt_s
        self._step = 0
        self._window_log: list[Tensor] = []
        self.adapted = False

    def reset(self, batch: int) -> None:
        self._step = 0
        self._window_log = []
        self.adapted = False
        self.override.reset(batch)
        self.behaviour.reset(batch)
        self.controller.reset(batch)

    def act(self, observation: Observation) -> Tensor:
        if self._step < self.warmup_steps:
            self._window_log.append(observation.window.clone())
            self._step += 1
            return self.behaviour.act(observation)

        if not self.adapted:
            self._adapt()
            self.adapted = True

        elapsed_min = self._step * self.dt_s / 60.0
        reverted = self.override.check(observation, elapsed_min)
        clinical = self.behaviour.act(observation)
        proposed = self.controller.act(observation)
        action = torch.where(reverted.unsqueeze(-1), clinical, proposed)
        self._step += 1
        return action

    def _adapt(self) -> None:
        """Fit the ridge map on the warm-start stretch, one sample per case and step."""
        if not self._window_log:
            return
        windows = torch.cat(self._window_log, dim=0)
        with torch.no_grad():
            outputs = self.controller.encoder(
                self.controller.standardiser.transform(windows),
                torch.zeros(windows.shape[0], windows.shape[1], 2),
            )
        self.adapter.fit(windows, outputs.latent)

    @property
    def overrides(self) -> int:
        return self.override.episodes
