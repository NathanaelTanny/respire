"""An episode trace: everything one pass over the theatre produces, per case and per step."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor


@dataclass(frozen=True)
class EpisodeTrace:
    """Step-wise and per-case quantities of a batch of episodes."""

    observation: Tensor
    action: Tensor
    reward: Tensor
    bis: Tensor
    map_mmHg: Tensor
    true_bis: Tensor
    propofol_ce: Tensor
    cost_hypotension: Tensor
    cost_suppression: Tensor
    cost_variability: Tensor
    hypotension_step: Tensor
    suppression_step: Tensor
    variability_step: Tensor
    episode_return: Tensor
    axes: Tensor
    sensitivity_pct: Tensor

    @property
    def batch(self) -> int:
        return int(self.observation.shape[0])

    @property
    def steps(self) -> int:
        return int(self.observation.shape[1])

    @property
    def channels(self) -> int:
        return int(self.observation.shape[2])

    def select(self, index: Tensor) -> EpisodeTrace:
        return EpisodeTrace(
            observation=self.observation[index],
            action=self.action[index],
            reward=self.reward[index],
            bis=self.bis[index],
            map_mmHg=self.map_mmHg[index],
            true_bis=self.true_bis[index],
            propofol_ce=self.propofol_ce[index],
            cost_hypotension=self.cost_hypotension[index],
            cost_suppression=self.cost_suppression[index],
            cost_variability=self.cost_variability[index],
            hypotension_step=self.hypotension_step[index],
            suppression_step=self.suppression_step[index],
            variability_step=self.variability_step[index],
            episode_return=self.episode_return[index],
            axes=self.axes[index],
            sensitivity_pct=self.sensitivity_pct[index],
        )

    def concat(self, other: EpisodeTrace) -> EpisodeTrace:
        return EpisodeTrace(
            observation=torch.cat([self.observation, other.observation], dim=0),
            action=torch.cat([self.action, other.action], dim=0),
            reward=torch.cat([self.reward, other.reward], dim=0),
            bis=torch.cat([self.bis, other.bis], dim=0),
            map_mmHg=torch.cat([self.map_mmHg, other.map_mmHg], dim=0),
            true_bis=torch.cat([self.true_bis, other.true_bis], dim=0),
            propofol_ce=torch.cat([self.propofol_ce, other.propofol_ce], dim=0),
            cost_hypotension=torch.cat([self.cost_hypotension, other.cost_hypotension], dim=0),
            cost_suppression=torch.cat([self.cost_suppression, other.cost_suppression], dim=0),
            cost_variability=torch.cat([self.cost_variability, other.cost_variability], dim=0),
            hypotension_step=torch.cat([self.hypotension_step, other.hypotension_step], dim=0),
            suppression_step=torch.cat([self.suppression_step, other.suppression_step], dim=0),
            variability_step=torch.cat([self.variability_step, other.variability_step], dim=0),
            episode_return=torch.cat([self.episode_return, other.episode_return], dim=0),
            axes=torch.cat([self.axes, other.axes], dim=0),
            sensitivity_pct=torch.cat([self.sensitivity_pct, other.sensitivity_pct], dim=0),
        )

    def prefix(self, steps: int) -> EpisodeTrace:
        """Truncate on the time axis, used by the horizon ablations."""
        return EpisodeTrace(
            observation=self.observation[:, :steps],
            action=self.action[:, :steps],
            reward=self.reward[:, :steps],
            bis=self.bis[:, :steps],
            map_mmHg=self.map_mmHg[:, :steps],
            true_bis=self.true_bis[:, :steps],
            propofol_ce=self.propofol_ce[:, :steps],
            cost_hypotension=self.cost_hypotension,
            cost_suppression=self.cost_suppression,
            cost_variability=self.cost_variability,
            hypotension_step=self.hypotension_step[:, :steps],
            suppression_step=self.suppression_step[:, :steps],
            variability_step=self.variability_step[:, :steps],
            episode_return=self.episode_return,
            axes=self.axes,
            sensitivity_pct=self.sensitivity_pct,
        )

    def numpy_observation(self) -> np.ndarray:
        return self.observation.detach().cpu().numpy().astype(np.float64)

    def depth_error(self) -> Tensor:
        return (self.bis - 50.0).abs()

    def summary(self) -> dict[str, float]:
        return {
            "bis_mae": float(self.depth_error().mean()),
            "return_mean": float(self.episode_return.mean()),
            "hypotension_min": float(self.cost_hypotension.mean()),
            "suppression_min": float(self.cost_suppression.mean()),
            "variability": float(self.cost_variability.mean()),
        }
