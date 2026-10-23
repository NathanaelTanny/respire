"""Behaviour cloning on the logged action stream.

The initialiser of Algorithm 1 line 2, used here as a comparison arm in its own right: the same
encoder, the same flow head, and no recursion at all. Whatever the table shows about this row is
therefore attributable to the recursion and not to the policy class.

Ref: Algorithm 1 line 2 (initialise by behaviour cloning with the flow-matching loss), Table 1 (BC).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from respire.data.windows import WindowBatch
from respire.losses.flow_matching import behaviour_cloning_loss
from respire.models.deterministic_head import DeterministicHead, deterministic_regulariser
from respire.models.encoder import EncoderStandardiser, LatentEncoder
from respire.models.flow import FlowPolicy
from respire.settings import TheatreConfig
from respire.theatre.environment import Observation
from respire.utils.autograd import backward


@dataclass
class BehaviourCloningAgent:
    """Frozen encoder plus a flow head fitted by cloning."""

    encoder: LatentEncoder
    standardiser: EncoderStandardiser
    flow: FlowPolicy
    theatre: TheatreConfig
    seed: int = 0
    _generator: torch.Generator | None = None

    def __post_init__(self) -> None:
        self._generator = torch.Generator(device="cpu")
        self._generator.manual_seed(self.seed)

    def reset(self, batch: int) -> None:
        return None

    def act(self, observation: Observation, explore: bool = False) -> Tensor:
        scaled = self.standardiser.transform(observation.window)
        with torch.no_grad():
            output = self.encoder(scaled, observation.actions)
            sample = self.flow.sample(
                torch.cat([output.latent, output.hidden], dim=-1), self._generator
            )
        return torch.stack(
            [
                sample.action[:, 0].clamp(0.0, self.theatre.propofol_max),
                sample.action[:, 1].clamp(0.0, self.theatre.remifentanil_max),
            ],
            dim=-1,
        )


def fit_behaviour_cloning(
    encoder: LatentEncoder,
    standardiser: EncoderStandardiser,
    flow: FlowPolicy | DeterministicHead,
    windows: WindowBatch,
    seed: int,
    epochs: int = 6,
    batch_size: int = 256,
    learning_rate: float = 3e-4,
) -> dict[str, float]:
    """Regress the flow on the observed action at the end of each window."""
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    optimiser = torch.optim.Adam(flow.parameters(), lr=learning_rate)
    total = 0.0
    batches = 0
    for _ in range(epochs):
        permutation = torch.randperm(windows.size, generator=generator)
        for start in range(0, windows.size, batch_size):
            index = permutation[start : start + batch_size]
            chunk = windows.subset(index)
            with torch.no_grad():
                output = encoder(standardiser.transform(chunk.window), chunk.actions)
                condition = torch.cat([output.latent, output.hidden], dim=-1)
            target = chunk.actions[:, -1, :]
            target = torch.stack(
                [
                    target[:, 0].clamp(0.0, 12.0),
                    target[:, 1].clamp(0.0, 0.5),
                ],
                dim=-1,
            )
            if isinstance(flow, FlowPolicy):
                loss = behaviour_cloning_loss(flow, condition, target, generator)
            else:
                loss = deterministic_regulariser(flow.mean_action(condition), target)
            optimiser.zero_grad()
            backward(loss)
            torch.nn.utils.clip_grad_norm_(flow.parameters(), 1.0)
            optimiser.step()
            total += float(loss.detach())
            batches += 1
    return {"bc_loss": total / max(batches, 1), "samples": float(windows.size)}
