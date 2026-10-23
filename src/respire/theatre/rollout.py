"""Rolling a policy through the theatre and recording the trace.

Traces are the only currency between components: the encoder learns from them, the verifier
regresses their returns, the metrics read them and the admission rule scores them.

Ref: Sec. 3.1 (episode return), Algorithm 2 line 6 (the case record).
"""

from __future__ import annotations

from typing import Protocol

import torch
from torch import Tensor

from respire.data.trace import EpisodeTrace
from respire.theatre.environment import AnaesthesiaEnv, Observation
from respire.theatre.patient import sensitivity_percentile


class ActingPolicy(Protocol):
    """Anything that maps an observation window to a two-column infusion rate."""

    def reset(self, batch: int) -> None: ...

    def act(self, observation: Observation) -> Tensor: ...


def rollout(env: AnaesthesiaEnv, policy: ActingPolicy) -> EpisodeTrace:
    """Run one full episode per row of the environment batch."""
    observation = env.reset()
    policy.reset(env.batch)
    batch = env.batch
    steps = env.steps
    channels = int(observation.window.shape[2])

    observations = torch.zeros(batch, steps, channels)
    actions = torch.zeros(batch, steps, 2)
    rewards = torch.zeros(batch, steps)
    bis = torch.zeros(batch, steps)
    pressure = torch.zeros(batch, steps)
    true_bis = torch.zeros(batch, steps)
    effect_site = torch.zeros(batch, steps)
    hypotension = torch.zeros(batch, steps)
    suppression = torch.zeros(batch, steps)
    variability = torch.zeros(batch, steps)

    for index in range(steps):
        action = policy.act(observation)
        transition = env.step(action)
        observations[:, index] = transition.observation.current
        actions[:, index] = transition.action
        rewards[:, index] = transition.reward
        bis[:, index] = transition.observation.bis
        pressure[:, index] = transition.observation.map_mmHg
        true_bis[:, index] = transition.true_bis
        effect_site[:, index] = transition.propofol_ce
        hypotension[:, index] = transition.cost_hypotension
        suppression[:, index] = transition.cost_suppression
        variability[:, index] = transition.cost_variability
        observation = transition.observation

    state = env.state
    return EpisodeTrace(
        observation=observations,
        action=actions,
        reward=rewards,
        bis=bis,
        map_mmHg=pressure,
        true_bis=true_bis,
        propofol_ce=effect_site,
        cost_hypotension=state.cost_hypotension,
        cost_suppression=state.cost_suppression,
        cost_variability=state.cost_variability,
        hypotension_step=hypotension,
        suppression_step=suppression,
        variability_step=variability,
        episode_return=state.episode_return,
        axes=env.patient.axes.clone(),
        sensitivity_pct=sensitivity_percentile(env.patient),
    )


def rollout_in_chunks(env: AnaesthesiaEnv, policy: ActingPolicy, chunk: int) -> EpisodeTrace:
    """Roll a large batch in pieces so a 2{,}000-case read stays inside memory."""
    if chunk <= 0:
        raise ValueError("chunk must be positive")
    pieces: list[EpisodeTrace] = []
    total = env.batch
    for start in range(0, total, chunk):
        stop = min(start + chunk, total)
        index = torch.arange(start, stop)
        piece_env = AnaesthesiaEnv(
            env.patient.select(index),
            env.theatre,
            env.pd_cfg,
            env.schedule.select(index),
            seed=start,
        )
        pieces.append(rollout(piece_env, policy))
    merged = pieces[0]
    for piece in pieces[1:]:
        merged = merged.concat(piece)
    return merged
