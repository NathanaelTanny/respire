"""Analytic shield between the policy and the pump.

The guard is a half-space in the two infusion rates. The one-step pressure model is linear in the
two rates, so the worst-case next pressure is a linear function minus the model's propagated
error; the largest admissible hypnotic rate follows in closed form, with no search and no learned
projection. Because both action coefficients are negative, moving along the constraint normal can
only lower the rates, and the box therefore preserves feasibility.

Ref: Sec. 4 (shield on the delivered action, not the intended one), Algorithm 2 line 5, Fig. 6.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from respire.models.pressure import OneStepPressureModel
from respire.settings import ShieldConfig, TheatreConfig


@dataclass(frozen=True)
class ShieldOutcome:
    """Delivered action together with the guard's own verdict."""

    action: Tensor
    requested: Tensor
    bound: Tensor
    projected: Tensor
    violation: Tensor
    feasible: Tensor


class AnalyticShield:
    """Projection onto the set of rates whose worst-case next pressure clears the floor."""

    def __init__(
        self,
        model: OneStepPressureModel,
        theatre: TheatreConfig,
        cfg: ShieldConfig,
    ) -> None:
        self.model = model
        self.theatre = theatre
        self.cfg = cfg
        self._coefficients = model.action_coefficients.clone()
        self._margin = cfg.lipschitz * cfg.noise_eta

    @property
    def bound(self) -> float:
        return self.cfg.map_floor + self.cfg.safety_margin

    def worst_case(self, state: Tensor, action: Tensor) -> Tensor:
        """Lower bound on the next pressure under the model's error budget."""
        return self.model.predict(state, action) - self._margin

    def __call__(self, state: Tensor, action: Tensor) -> ShieldOutcome:
        if action.shape[-1] != 2:
            raise ValueError("shield expects a two-column action")
        coefficients = self._coefficients
        offset = self.model.offset(state)
        budget = self.bound + self._margin - offset
        projection = action @ coefficients
        deficit = torch.clamp(budget - projection, min=0.0)
        norm = torch.clamp((coefficients**2).sum(), min=1e-8)
        corrected = action + (deficit / norm).unsqueeze(-1) * coefficients.unsqueeze(0)
        clipped = torch.stack(
            [
                corrected[:, 0].clamp(0.0, self.theatre.propofol_max),
                corrected[:, 1].clamp(0.0, self.theatre.remifentanil_max),
            ],
            dim=-1,
        )
        realised = self.worst_case(state, clipped)
        feasible = realised >= self.bound - 1e-3
        delivered = torch.where(
            feasible.unsqueeze(-1),
            clipped,
            torch.zeros_like(clipped),
        )
        violation = (self.worst_case(state, action) < self.bound).to(torch.float32)
        return ShieldOutcome(
            action=delivered,
            requested=action,
            bound=realised,
            projected=clipped,
            violation=violation,
            feasible=feasible,
        )
