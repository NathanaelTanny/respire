"""The deployed controller: frozen encoder, flow policy, shield.

The controller is the only object the deployment loop and the comparators share, so the safety
instruments are attached here rather than inside the policy: switching the shield off is a
constructor argument, not a branch in the network.

Ref: Fig. 2 (encoder frozen; recursion refits verifier and policy), Algorithm 2 lines 3-6.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor

from respire.control.adaptation import RidgeAdapter
from respire.control.shield import AnalyticShield, ShieldOutcome
from respire.models.deterministic_head import DeterministicHead
from respire.models.encoder import EncoderStandardiser, LatentEncoder
from respire.models.flow import FlowPolicy
from respire.models.pressure import state_features
from respire.settings import TheatreConfig
from respire.theatre.environment import Observation


@dataclass
class ControllerDiagnostics:
    """Running tallies the evaluator reads after a rollout."""

    steps: int = 0
    violations: float = 0.0
    infeasible: float = 0.0
    latent_norm: float = 0.0
    axis_sum: Tensor = field(default_factory=lambda: torch.zeros(3))
    sparsity: float = 0.0

    def add(
        self, outcome: ShieldOutcome | None, latent: Tensor, axes: Tensor, sparsity: Tensor
    ) -> None:
        self.steps += 1
        if outcome is not None:
            self.violations += float(outcome.violation.mean())
            self.infeasible += float((~outcome.feasible).to(torch.float32).mean())
        self.latent_norm += float(latent.detach().norm(dim=-1).mean())
        self.axis_sum = self.axis_sum + axes.mean(dim=0).detach()
        self.sparsity += float(sparsity.detach().mean())

    def summary(self) -> dict[str, float]:
        steps = max(self.steps, 1)
        return {
            "shield_violation_rate": self.violations / steps,
            "shield_infeasible_rate": self.infeasible / steps,
            "latent_norm": self.latent_norm / steps,
            "axis_mean_0": float(self.axis_sum[0]) / steps,
            "axis_mean_1": float(self.axis_sum[1]) / steps,
            "axis_mean_2": float(self.axis_sum[2]) / steps,
            "sparsity": self.sparsity / steps,
        }


class LatentFlowController:
    """Policy conditioned on the latent and the recurrent summary, with optional instruments."""

    def __init__(
        self,
        encoder: LatentEncoder,
        standardiser: EncoderStandardiser,
        flow: FlowPolicy | DeterministicHead,
        theatre: TheatreConfig,
        shield: AnalyticShield | None = None,
        adapter: RidgeAdapter | None = None,
        seed: int = 0,
        use_latent: bool = True,
    ) -> None:
        self.encoder = encoder.eval()
        self.standardiser = standardiser
        self.flow = flow
        self.theatre = theatre
        self.shield = shield
        self.adapter = adapter
        self.use_latent = use_latent
        self.diagnostics = ControllerDiagnostics()
        self._generator = torch.Generator(device="cpu")
        self._generator.manual_seed(seed)

    def reset(self, batch: int) -> None:
        self.diagnostics = ControllerDiagnostics()

    def condition(self, window: Tensor, actions: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Standardised window to (latent, axes, hidden, sparsity)."""
        scaled = self.standardiser.transform(window)
        output = self.encoder(scaled, actions)
        latent = output.latent
        if not self.use_latent:
            latent = torch.zeros_like(latent)
        if self.adapter is not None:
            latent = self.adapter.transform(window, latent)
        return latent, output.axes, output.hidden, output.sparsity

    def act(self, observation: Observation) -> Tensor:
        latent, axes, hidden, sparsity = self.condition(observation.window, observation.actions)
        condition = torch.cat([latent, hidden], dim=-1)
        sample = self.flow.sample(condition, self._generator)
        action = torch.stack(
            [
                sample.action[:, 0].clamp(0.0, self.theatre.propofol_max),
                sample.action[:, 1].clamp(0.0, self.theatre.remifentanil_max),
            ],
            dim=-1,
        )
        outcome: ShieldOutcome | None = None
        if self.shield is not None:
            outcome = self.shield(state_features(observation.window), action)
            action = outcome.action
        self.diagnostics.add(outcome, latent, axes, sparsity)
        return action

    def act_deterministic(self, observation: Observation) -> Tensor:
        """Same path with the sampling noise pinned, for reproducibility checks."""
        generator = torch.Generator(device="cpu")
        generator.manual_seed(12345)
        latent, axes, hidden, sparsity = self.condition(observation.window, observation.actions)
        sample = self.flow.sample(torch.cat([latent, hidden], dim=-1), generator)
        action = torch.stack(
            [
                sample.action[:, 0].clamp(0.0, self.theatre.propofol_max),
                sample.action[:, 1].clamp(0.0, self.theatre.remifentanil_max),
            ],
            dim=-1,
        )
        if self.shield is not None:
            action = self.shield(state_features(observation.window), action).action
        self.diagnostics.add(None, latent, axes, sparsity)
        return action
