"""Helpers for PyInstaller windowed builds (no console streams)."""

from __future__ import annotations

import os
import sys


def ensure_valid_stdio() -> None:
    """Windowed frozen apps often have stdout/stderr set to None; loguru/boxmot need a sink."""
    if sys.stdout is None:
        sys.stdout = open(os.devnull, "w")
    if sys.stderr is None:
        sys.stderr = open(os.devnull, "w")
