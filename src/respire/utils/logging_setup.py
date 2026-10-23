"""Logging setup shared by the CLI and the round driver."""

from __future__ import annotations

import logging
import os
import sys

_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"


def configure(level: str = "INFO") -> None:
    """Attach a single stderr handler to the package logger."""
    root = logging.getLogger("respire")
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    if not root.handlers:
        handler = logging.StreamHandler(stream=sys.stderr)
        handler.setFormatter(logging.Formatter(_FORMAT, datefmt="%H:%M:%S"))
        root.addHandler(handler)
    root.propagate = False


def get_logger(name: str) -> logging.Logger:
    """A namespaced logger; the package root is configured once by :func:`configure`."""
    return logging.getLogger(f"respire.{name}")


def level_from_env(default: str = "INFO") -> str:
    return os.environ.get("RESPIRE_LOG_LEVEL", default)
