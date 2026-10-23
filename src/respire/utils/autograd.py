"""One typed entry point for the backward pass.

The stubs shipped with torch leave ``Tensor.backward`` unannotated, which a strict run reports at
every call site. Routing the call through this module keeps the strictness setting honest: the
single cast below is the only place where the signature is asserted.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

from torch import Tensor


def backward(loss: Tensor) -> None:
    """Accumulate gradients of ``loss``."""
    cast(Callable[[], None], loss.backward)()
