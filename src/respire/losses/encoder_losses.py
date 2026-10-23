"""The four terms that train the encoder.

Reconstruction keeps the latent faithful to the window; the one-step head keeps it predictive; the
interventional head is the term that makes the axes identifiable, because it is asked for the depth
change under one infusion and then under another at the same state, so a latent that merely records
the observation cannot satisfy it; the axis anchor is available only where the generating factors
are known.

Ref: Sec. 1 (predicting the reaction to the perturbation rather than reconstructing it),
Sec. 4.2, Table 2 (interventional loss ablation).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from respire.data.windows import WindowBatch
from respire.models.encoder import EncoderStandardiser, LatentEncoder
from respire.theatre.probe import ProbeBatch


@dataclass(frozen=True)
class EncoderLossTerms:
    total: Tensor
    reconstruction: Tensor
    one_step: Tensor
    interventional: Tensor
    axes: Tensor

    def as_dict(self) -> dict[str, float]:
        return {
            "total": float(self.total.detach()),
            "reconstruction": float(self.reconstruction.detach()),
            "one_step": float(self.one_step.detach()),
            "interventional": float(self.interventional.detach()),
            "axes": float(self.axes.detach()),
        }


def reconstruction_loss(reconstruction: Tensor, target: Tensor) -> Tensor:
    """Mean squared error on the standardised currents."""
    return ((reconstruction - target) ** 2).mean()


def one_step_loss(prediction: Tensor, target: Tensor) -> Tensor:
    """Mean squared error on the pair (pressure change, depth change)."""
    return ((prediction - target) ** 2).mean()


def interventional_loss(
    encoder: LatentEncoder,
    latent: Tensor,
    action_a: Tensor,
    action_b: Tensor,
    delta_a: Tensor,
    delta_b: Tensor,
    scale: float = 10.0,
) -> Tensor:
    """Match both counterfactual depth changes and their difference."""
    predicted_a = encoder.predict_delta_depth(latent, action_a)
    predicted_b = encoder.predict_delta_depth(latent, action_b)
    absolute = ((predicted_a - delta_a) ** 2).mean() + ((predicted_b - delta_b) ** 2).mean()
    relative = (((predicted_a - predicted_b) - (delta_a - delta_b)) ** 2).mean()
    return absolute + scale * relative


def axis_loss(axes: Tensor, target: Tensor) -> Tensor:
    """Anchors the axes on the generating factors, where the theatre supplies them."""
    return ((axes - target) ** 2).mean()


def probe_objective(
    encoder: LatentEncoder,
    standardiser: EncoderStandardiser,
    probes: ProbeBatch,
    scale: float = 10.0,
) -> Tensor:
    """Interventional term on the probe batch, with the axis anchor of the same states."""
    scaled = standardiser.transform(probes.window)
    output = encoder(scaled, probes.action_history)
    term = interventional_loss(
        encoder,
        output.latent,
        probes.action_a,
        probes.action_b,
        probes.delta_a,
        probes.delta_b,
        scale,
    )
    return term + axis_loss(output.axes, probes.axes)


def encoder_objective(
    encoder: LatentEncoder,
    standardiser: EncoderStandardiser,
    windows: WindowBatch,
    probes: ProbeBatch,
    weights: dict[str, float],
) -> EncoderLossTerms:
    """Assemble the four terms."""
    scaled_window = standardiser.transform(windows.window)
    output = encoder(scaled_window, windows.actions)
    reconstruction = reconstruction_loss(
        output.reconstruction, standardiser.transform(windows.current)
    )
    one_step = one_step_loss(
        encoder.predict_one_step(output.latent, output.hidden, windows.actions[:, -1, :]),
        torch.stack([windows.pressure_delta, windows.depth_delta], dim=-1) / 10.0,
    )
    scaled_probes = standardiser.transform(probes.window)
    probe_output = encoder(scaled_probes, probes.action_history)
    interventional = interventional_loss(
        encoder,
        probe_output.latent,
        probes.action_a,
        probes.action_b,
        probes.delta_a,
        probes.delta_b,
    )
    anchored = axis_loss(output.axes, windows.axes) + axis_loss(probe_output.axes, probes.axes)
    total = (
        weights["reconstruction"] * reconstruction
        + weights["one_step"] * one_step
        + weights["interventional"] * interventional
        + weights["axes"] * anchored
    )
    return EncoderLossTerms(
        total=total,
        reconstruction=reconstruction,
        one_step=one_step,
        interventional=interventional,
        axes=anchored,
    )
