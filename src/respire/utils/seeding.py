"""Process-wide seeding.

One entry point so that a round, a comparator run and a replay evaluation all consume the same
stream regardless of which CLI invoked them.
"""

from __future__ import annotations

import os
import random
from collections.abc import Iterator
from contextlib import contextmanager

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed every generator the release touches and make cuBLAS deterministic."""
    if seed < 0:
        raise ValueError("seed must be non-negative")
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(False)


def derive_seed(base: int, *tags: str | int) -> int:
    """Deterministic child seed, so a stage's stream does not shift when another stage moves."""
    values = [base]
    for tag in tags:
        if isinstance(tag, int):
            values.append(tag)
        else:
            values.extend(ord(ch) for ch in tag)
    acc = 1469598103934665603
    for value in values:
        acc = (acc ^ (value & 0xFFFFFFFF)) * 1099511628211
        acc &= 0xFFFFFFFFFFFFFFFF
    return acc % (2**31 - 1)


@contextmanager
def fork_rng(base: int, *tags: str | int) -> Iterator[int]:
    """Run a block under a derived seed and restore the ambient generator state afterwards."""
    seed = derive_seed(base, *tags)
    py_state = random.getstate()
    np_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    set_seed(seed)
    try:
        yield seed
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
        torch.set_rng_state(torch_state)


def generator(seed: int) -> torch.Generator:
    """A CPU generator for reproducible sampling inside a module."""
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    return gen
