"""Conditional flow-matching policy over the two infusion rates.

A rectified flow from a standard normal to the action space, conditioned on the recurrent summary
and the latent. Sampling costs four Euler steps, which is the budget the deployment loop allows.
The change-of-variables identity along the same ODE gives an exact log-density in two dimensions,
which is what the trust region needs: the KL between the updated and the previous policy is
estimated from the new policy's own samples.

Ref: Sec. 4.1-4.2, Fig. 2 (flow policy, four Euler steps), Eq. (4) (KL term and trust region).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import cast

import torch
from torch import Tensor, nn

from respire.settings import PolicyConfig


class SinusoidalTime(nn.Module):
    """Fourier features of the flow time, so the field is smooth in t without a deep net."""

    def __init__(self, features: int) -> None:
        super().__init__()
        self.features = features
        frequencies = torch.arange(features // 2, dtype=torch.float32)
        self.register_buffer("frequencies", frequencies)
        self.frequencies: Tensor = frequencies

    def forward(self, time: Tensor) -> Tensor:
        scaled = time.unsqueeze(-1) * torch.exp(
            -math.log(1000.0) * self.frequencies / max(self.features // 2 - 1, 1)
        )
        return torch.cat([torch.sin(scaled), torch.cos(scaled)], dim=-1)


class VelocityField(nn.Module):
    """Residual MLP predicting the flow velocity at (state, time, condition)."""

    def __init__(
        self, action_dim: int, cond_dim: int, hidden: int, depth: int, time_features: int
    ) -> None:
        super().__init__()
        self.time = SinusoidalTime(time_features)
        width = action_dim + cond_dim + time_features
        layers: list[nn.Module] = [nn.Linear(width, hidden), nn.SiLU()]
        for _ in range(depth - 1):
            layers.extend([nn.Linear(hidden, hidden), nn.SiLU()])
        layers.append(nn.Linear(hidden, action_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, state: Tensor, time: Tensor, condition: Tensor) -> Tensor:
        joined = torch.cat([state, condition, self.time(time)], dim=-1)
        return cast(Tensor, self.net(joined))


@dataclass(frozen=True)
class FlowSample:
    """Sampled actions with their exact log-density under the current field."""

    action: Tensor
    log_prob: Tensor
    divergence_integral: Tensor


class FlowPolicy(nn.Module):
    """Rectified-flow policy with exact two-dimensional log-density."""

    def __init__(self, cfg: PolicyConfig, cond_dim: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.cond_dim = cond_dim
        self.kind = "flow"
        self.field = VelocityField(
            cfg.action_dim, cond_dim, cfg.hidden, cfg.depth, cfg.time_features
        )
        self.register_buffer("action_scale", torch.ones(cfg.action_dim))
        self.register_buffer("action_offset", torch.zeros(cfg.action_dim))

    def configure_action_scale(self, propofol_max: float, remifentanil_max: float) -> None:
        """The flow is transported on the pump's own box, so the base noise is close to the data."""
        self.action_scale = torch.tensor([propofol_max, remifentanil_max])
        self.action_offset = torch.zeros(2)

    def to_unit(self, action: Tensor) -> Tensor:
        return (action - self.action_offset) / self.action_scale

    def to_raw(self, unit: Tensor) -> Tensor:
        return unit * self.action_scale + self.action_offset

    def velocity(self, state: Tensor, time: Tensor, condition: Tensor) -> Tensor:
        return cast(Tensor, self.field(state, time, condition))

    def sample(self, condition: Tensor, generator: torch.Generator | None = None) -> FlowSample:
        """Integrate the flow from noise to actions, accumulating the divergence.

        The divergence needs a graph even when the caller is inside ``no_grad``, so the integration
        enables grad locally and returns detached values; the density is a constant downstream.
        """
        batch = condition.shape[0]
        with torch.enable_grad():
            state = torch.randn(batch, self.cfg.action_dim, generator=generator)
            steps = self.cfg.euler_steps
            dt = 1.0 / steps
            divergence = torch.zeros(batch)
            for index in range(steps):
                time = torch.full((batch,), index * dt)
                state = state.detach().requires_grad_(True)
                velocity = self.velocity(state, time, condition)
                divergence = divergence + self._divergence(velocity, state) * dt
                state = (state + dt * velocity).detach()
            log_base = -0.5 * (state**2).sum(dim=-1)
            log_base = log_base - 0.5 * self.cfg.action_dim * math.log(2 * math.pi)
        return FlowSample(
            action=self.to_raw(state).detach(),
            log_prob=(log_base - divergence).detach(),
            divergence_integral=divergence.detach(),
        )

    def sample_pathwise(self, condition: Tensor) -> Tensor:
        """Differentiable sampling, used by the actor step; no divergence is accumulated."""
        batch = condition.shape[0]
        state = torch.randn(batch, self.cfg.action_dim)
        steps = self.cfg.euler_steps
        dt = 1.0 / steps
        for index in range(steps):
            time = torch.full((batch,), index * dt)
            state = state + dt * self.velocity(state, time, condition)
        return self.to_raw(state)

    def log_prob(self, action: Tensor, condition: Tensor) -> Tensor:
        """Exact log-density at an arbitrary action, by integrating the flow backwards."""
        batch = action.shape[0]
        with torch.enable_grad():
            state = self.to_unit(action).detach().clone()
            steps = self.cfg.euler_steps
            dt = 1.0 / steps
            divergence = torch.zeros(batch)
            for index in range(steps):
                time = torch.full((batch,), 1.0 - index * dt)
                state = state.detach().requires_grad_(True)
                velocity = self.velocity(state, time, condition)
                divergence = divergence + self._divergence(velocity, state) * dt
                state = (state - dt * velocity).detach()
            log_base = -0.5 * (state**2).sum(dim=-1)
            log_base = log_base - 0.5 * self.cfg.action_dim * math.log(2 * math.pi)
        return (log_base - divergence).detach()

    @staticmethod
    def _divergence(velocity: Tensor, state: Tensor) -> Tensor:
        """Trace of the Jacobian, exact because the action has two coordinates."""
        total = torch.zeros(state.shape[0], dtype=velocity.dtype)
        for coordinate in range(velocity.shape[-1]):
            gradient = torch.autograd.grad(
                velocity[:, coordinate].sum(), state, create_graph=True, retain_graph=True
            )[0]
            total = total + gradient[:, coordinate]
        return total

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
