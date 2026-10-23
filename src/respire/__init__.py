"""Recursive self-improving closed-loop anaesthesia with latent physiological patterns.

The package is layered by distance from the patient: ``theatre`` builds the room, ``data`` cuts it
into windows, ``models`` and ``losses`` fit it, ``control`` enforces the safety instruments,
``training`` runs the recursion, ``comparators`` holds the arms of the comparison and ``evaluation``
reads the result.
"""

from __future__ import annotations

__version__ = "1.0.0"

__all__ = ["__version__"]
