"""One-step pressure model behind the shield.

The article's shield is analytic, so the model it is built on has to be analytic too: a linear
one-step predictor of mean arterial pressure whose action coefficients give both the gradient the
projection follows and the Lipschitz constant the worst-case guard uses.

Ref: Sec. 4 (analytic shield), Algorithm 1 line 1 (fit the pressure model, estimate eta and Lg),
Fig. 6 (a blocked propofol increase under insufflation).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

STATE_FEATURES: tuple[str, ...] = ("map", "bis", "heart_rate", "map_slope", "bis_slope")


def state_features(window: Tensor) -> Tensor:
    """Five observable summaries of the current window, shape ``[batch, 5]``."""
    current = window[:, -1, :]
    previous = window[:, -2, :] if window.shape[1] > 1 else window[:, -1, :]
    pressure = current[:, 2]
    depth = current[:, 0]
    rate = current[:, 1]
    pressure_slope = pressure - previous[:, 2]
    depth_slope = depth - previous[:, 0]
    return torch.stack([pressure, depth, rate, pressure_slope, depth_slope], dim=-1)


@dataclass(frozen=True)
class PressureFit:
    """Fitted coefficients: state part, action part and the intercept."""

    state_mean: Tensor
    state_scale: Tensor
    state_coef: Tensor
    action_coef: Tensor
    intercept: Tensor
    residual_sd: float
    ridge: float


class OneStepPressureModel:
    """Ridge regression of the next-step pressure on five state summaries and the two rates."""

    def __init__(self, ridge: float = 1e-2) -> None:
        self.fit_result: PressureFit | None = None
        self.ridge = ridge

    @property
    def fitted(self) -> bool:
        return self.fit_result is not None

    def fit(self, state: Tensor, action: Tensor, target: Tensor) -> PressureFit:
        if state.ndim != 2 or action.ndim != 2 or target.ndim != 1:
            raise ValueError("state, action and target must be [batch, d], [batch, 2], [batch]")
        mean = state.mean(dim=0)
        scale = state.std(dim=0).clamp(min=1e-3)
        scaled = (state - mean) / scale
        design = torch.cat([scaled, action, torch.ones(state.shape[0], 1)], dim=-1)
        gram = design.T @ design
        penalty = self.ridge * torch.eye(gram.shape[0])
        penalty[-1, -1] = 0.0
        coef = torch.linalg.solve(gram + penalty, design.T @ target)
        state_coef = coef[: state.shape[1]]
        action_coef = coef[state.shape[1] : state.shape[1] + action.shape[1]]
        intercept = coef[-1]
        residual = target - (scaled @ state_coef + action @ action_coef + intercept)
        self.fit_result = PressureFit(
            state_mean=mean,
            state_scale=scale,
            state_coef=state_coef,
            action_coef=action_coef,
            intercept=intercept,
            residual_sd=float(residual.std(unbiased=True)),
            ridge=self.ridge,
        )
        return self.fit_result

    def _require(self) -> PressureFit:
        if self.fit_result is None:
            raise RuntimeError("pressure model must be fitted before it is used")
        return self.fit_result

    def offset(self, state: Tensor) -> Tensor:
        """Predicted next pressure at zero infusion: the state-dependent part of the constraint."""
        fit = self._require()
        scaled = (state - fit.state_mean) / fit.state_scale
        return scaled @ fit.state_coef + fit.intercept

    def predict(self, state: Tensor, action: Tensor) -> Tensor:
        return self.offset(state) + action @ self._require().action_coef

    @property
    def action_coefficients(self) -> Tensor:
        """Gradient of the next pressure with respect to the two rates."""
        return self._require().action_coef

    @property
    def lipschitz(self) -> float:
        return float(torch.linalg.norm(self._require().action_coef))

    @property
    def residual_sd(self) -> float:
        return self._require().residual_sd

    def state_dict(self) -> dict[str, object]:
        fit = self._require()
        return {
            "state_mean": fit.state_mean.tolist(),
            "state_scale": fit.state_scale.tolist(),
            "state_coef": fit.state_coef.tolist(),
            "action_coef": fit.action_coef.tolist(),
            "intercept": float(fit.intercept),
            "residual_sd": fit.residual_sd,
            "ridge": fit.ridge,
        }

    @classmethod
    def from_state(cls, state: dict[str, object]) -> OneStepPressureModel:
        model = cls(ridge=float(str(state["ridge"])))
        state_mean = torch.tensor(state["state_mean"], dtype=torch.float32)
        model.fit_result = PressureFit(
            state_mean=state_mean,
            state_scale=torch.tensor(state["state_scale"], dtype=torch.float32),
            state_coef=torch.tensor(state["state_coef"], dtype=torch.float32),
            action_coef=torch.tensor(state["action_coef"], dtype=torch.float32),
            intercept=torch.tensor(float(str(state["intercept"]))),
            residual_sd=float(str(state["residual_sd"])),
            ridge=float(str(state["ridge"])),
        )
        return model
