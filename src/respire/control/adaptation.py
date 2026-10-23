"""Test-time adaptation of the latent prior.

During the first five minutes the anaesthetist holds the rates, which yields a clean stretch of
observations under a known action. A ridge map from the observable window summaries to the
encoder's latent is fitted on that stretch in eight fixed steps, and the latent used by the policy
is then blended towards the fitted prediction. With no adaptation the blend weight is zero and the
policy runs on the frozen encoder alone.

Ref: Algorithm 2 lines 1-3 (hold recorded rates, adapt the latent prior with eight ridge steps).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor

from respire.models.pressure import state_features
from respire.utils.autograd import backward


@dataclass
class RidgeAdapter:
    """Eight-step ridge fit of latent targets on observation summaries."""

    steps: int = 8
    learning_rate: float = 0.05
    ridge: float = 1e-2
    blend: float = 0.5
    fitted: bool = False
    weights: Tensor | None = None

    def fit(self, windows: Tensor, latents: Tensor) -> None:
        """``windows`` is ``[batch, steps, channels]``, ``latents`` is ``[batch, latent]``."""
        if windows.shape[0] == 0:
            raise ValueError("adaptation needs at least one window")
        features = state_features(windows)
        mean = features.mean(dim=0)
        scale = features.std(dim=0).clamp(min=1e-3)
        design = (features - mean) / scale
        design = torch.cat([design, torch.ones(design.shape[0], 1)], dim=-1)
        weights = torch.zeros(design.shape[1], latents.shape[1], requires_grad=True)
        optimiser = torch.optim.SGD([weights], lr=self.learning_rate)
        for _ in range(self.steps):
            optimiser.zero_grad()
            prediction = design @ weights
            loss: Tensor = ((prediction - latents) ** 2).mean() + self.ridge * (weights**2).mean()
            backward(loss)
            optimiser.step()
        self.weights = weights.detach()
        self._mean = mean
        self._scale = scale
        self.fitted = True

    def predict(self, windows: Tensor) -> Tensor:
        if self.weights is None:
            raise RuntimeError("adapter is not fitted")
        features = state_features(windows)
        design = (features - self._mean) / self._scale
        design = torch.cat([design, torch.ones(design.shape[0], 1)], dim=-1)
        return design @ self.weights

    def transform(self, windows: Tensor, latent: Tensor) -> Tensor:
        """Blend the encoder's latent towards the adapted prediction."""
        if not self.fitted or self.blend <= 0.0:
            return latent
        return (1.0 - self.blend) * latent + self.blend * self.predict(windows)

    _mean: Tensor = field(default_factory=lambda: torch.zeros(5))
    _scale: Tensor = field(default_factory=lambda: torch.ones(5))
