"""Sliding sixty-second windows and their supervised targets.

The encoder learns from windows, not from episodes: each window carries the current twelve
channels, the action history that led to them, the one-step change in pressure and depth, and the
ground-truth axes when the window comes from the cohort.

Ref: Sec. 4.2 (60-s window), Algorithm 2 line 3 (recurrent update from the observation).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from respire.data.trace import EpisodeTrace


@dataclass(frozen=True)
class WindowBatch:
    """A batch of windows with every target the encoder heads consume."""

    window: Tensor
    actions: Tensor
    current: Tensor
    next_depth: Tensor
    depth_delta: Tensor
    pressure_delta: Tensor
    axes: Tensor

    @property
    def size(self) -> int:
        return int(self.window.shape[0])

    def subset(self, index: Tensor) -> WindowBatch:
        return WindowBatch(
            window=self.window[index],
            actions=self.actions[index],
            current=self.current[index],
            next_depth=self.next_depth[index],
            depth_delta=self.depth_delta[index],
            pressure_delta=self.pressure_delta[index],
            axes=self.axes[index],
        )

    def concat(self, other: WindowBatch) -> WindowBatch:
        return WindowBatch(
            window=torch.cat([self.window, other.window], dim=0),
            actions=torch.cat([self.actions, other.actions], dim=0),
            current=torch.cat([self.current, other.current], dim=0),
            next_depth=torch.cat([self.next_depth, other.next_depth], dim=0),
            depth_delta=torch.cat([self.depth_delta, other.depth_delta], dim=0),
            pressure_delta=torch.cat([self.pressure_delta, other.pressure_delta], dim=0),
            axes=torch.cat([self.axes, other.axes], dim=0),
        )


def build_windows(trace: EpisodeTrace, window_steps: int, stride: int = 1) -> WindowBatch:
    """Slide a window over a trace and attach the one-step targets that follow each window."""
    if window_steps < 2:
        raise ValueError("window_steps must be at least two")
    if stride < 1:
        raise ValueError("stride must be positive")
    steps = trace.steps
    starts = list(range(0, steps - window_steps, stride))
    if not starts:
        raise ValueError("trace shorter than one window")
    index = torch.tensor(starts)
    offsets = torch.arange(window_steps)
    gather = index.unsqueeze(1) + offsets.unsqueeze(0)

    observation = trace.observation
    action = trace.action
    windows = observation[:, gather]
    action_windows = action[:, gather]
    current = observation[:, gather[:, -1]]
    next_depth = trace.bis[:, index + window_steps]
    depth_delta = next_depth - trace.bis[:, index + window_steps - 1]
    pressure_delta = (
        trace.map_mmHg[:, index + window_steps] - trace.map_mmHg[:, index + window_steps - 1]
    )

    batch, count = windows.shape[0], windows.shape[1]
    flattened = WindowBatch(
        window=windows.reshape(batch * count, window_steps, windows.shape[-1]),
        actions=action_windows.reshape(batch * count, window_steps, 2),
        current=current.reshape(batch * count, current.shape[-1]),
        next_depth=next_depth.reshape(batch * count),
        depth_delta=depth_delta.reshape(batch * count),
        pressure_delta=pressure_delta.reshape(batch * count),
        axes=trace.axes.unsqueeze(1).expand(batch, count, 3).reshape(batch * count, 3),
    )
    return flattened


def sample_windows(batch: WindowBatch, count: int, generator: torch.Generator) -> WindowBatch:
    """Uniformly subsample a window batch without replacement when possible."""
    total = batch.size
    take = min(count, total)
    index = torch.randperm(total, generator=generator)[:take]
    return batch.subset(index)


def split_windows(
    batch: WindowBatch, fraction: float, generator: torch.Generator
) -> tuple[WindowBatch, WindowBatch]:
    """Deterministic two-way split of a window batch."""
    if not 0.0 < fraction < 1.0:
        raise ValueError("fraction must lie strictly between zero and one")
    order = torch.randperm(batch.size, generator=generator)
    cut = int(batch.size * fraction)
    return batch.subset(order[:cut]), batch.subset(order[cut:])
