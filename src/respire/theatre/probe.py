"""Counterfactual probes.

A probe is a state reached under the behaviour policy, at which two infusions are applied and the
resulting depth changes are recorded from the theatre's own state rather than from the noisy
monitor. That difference is the label the interventional head is trained on, and it is the only
target in the release that could not be obtained from a logged record.

Ref: Sec. 1 (exogenous perturbation with logged timing), Table 2 (interventional loss ablation).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from respire.settings import ExperimentConfig
from respire.theatre.cohort import make_environment
from respire.theatre.patient import Patient
from respire.theatre.rollout import ActingPolicy


@dataclass(frozen=True)
class ProbeBatch:
    """Probe windows with the two action arms and their noiseless depth deltas."""

    window: Tensor
    action_history: Tensor
    action_a: Tensor
    action_b: Tensor
    delta_a: Tensor
    delta_b: Tensor
    axes: Tensor

    @property
    def size(self) -> int:
        return int(self.window.shape[0])


def counterfactual_probes(
    patients: Patient,
    cfg: ExperimentConfig,
    behaviour: ActingPolicy,
    warmup_steps: tuple[int, ...],
    perturb: float = 2.0,
    seed: int = 0,
) -> ProbeBatch:
    """Walk each case to a set of decision points and look one step ahead twice."""
    env = make_environment(patients, cfg, seed)
    observation = env.reset()
    behaviour.reset(env.batch)
    windows: list[Tensor] = []
    histories: list[Tensor] = []
    base_actions: list[Tensor] = []
    alternates: list[Tensor] = []
    deltas_a: list[Tensor] = []
    deltas_b: list[Tensor] = []
    marks = sorted(set(warmup_steps))
    target = max(marks) if marks else 0
    mark_index = 0

    for step in range(target + 1):
        action = behaviour.act(observation)
        if mark_index < len(marks) and step == marks[mark_index]:
            alternate = torch.stack(
                [
                    (action[:, 0] + perturb).clamp(0.0, cfg.theatre.propofol_max),
                    action[:, 1],
                ],
                dim=-1,
            )
            baseline = env.true_depth()
            first = env.peek(action)
            second = env.peek(alternate)
            windows.append(observation.window.clone())
            histories.append(observation.actions.clone())
            base_actions.append(action.clone())
            alternates.append(alternate.clone())
            deltas_a.append(first.true_bis - baseline)
            deltas_b.append(second.true_bis - baseline)
            mark_index += 1
        observation = env.step(action).observation

    return ProbeBatch(
        window=torch.cat(windows, dim=0),
        action_history=torch.cat(histories, dim=0),
        action_a=torch.cat(base_actions, dim=0),
        action_b=torch.cat(alternates, dim=0),
        delta_a=torch.cat(deltas_a, dim=0),
        delta_b=torch.cat(deltas_b, dim=0),
        axes=patients.axes.repeat(len(marks), 1),
    )
