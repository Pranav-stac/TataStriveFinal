"""Resolve bundled/runtime model weight paths for dev and frozen builds."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List, Optional


def resolve_weights_dir() -> tuple[Optional[str], Optional[str]]:
    """
    Return (weights_dir, base_path) for the current runtime.

    weights_dir is the preferred Models/ directory when present.
    base_path is the project or executable root used for fallbacks.
    """
    base_path: Optional[str] = None
    weights_dir: Optional[str] = None
    try:
        if getattr(sys, "frozen", False):
            exe_dir = os.path.dirname(os.path.abspath(sys.executable))
            exe_models = os.path.join(exe_dir, "Models")
            bundle_models = os.path.join(sys._MEIPASS, "Models")
            base_path = exe_dir
            if os.path.isdir(exe_models):
                weights_dir = exe_models
            elif os.path.isdir(bundle_models):
                weights_dir = bundle_models
            else:
                weights_dir = sys._MEIPASS
        else:
            base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            weights_dir = os.path.join(base_path, "Models")
            if not os.path.exists(weights_dir):
                weights_dir = base_path
    except Exception:
        base_path = os.getcwd()
        weights_dir = os.path.join(base_path, "Models")
        if not os.path.exists(weights_dir):
            weights_dir = base_path
    return weights_dir, base_path


def model_weight_candidates(
    weight_file: str,
    weights_dir: Optional[str] = None,
    base_path: Optional[str] = None,
) -> List[str]:
    """All locations build_exe / runtime may place a .pt (order = load preference)."""
    if getattr(sys, "frozen", False):
        exe_dir = os.path.dirname(os.path.abspath(sys.executable))
        mp = sys._MEIPASS
        wd = weights_dir or mp
        return [
            os.path.join(exe_dir, "Models", weight_file),
            os.path.join(exe_dir, weight_file),
            os.path.join(wd, weight_file),
            os.path.join(mp, "Models", weight_file),
            os.path.join(mp, weight_file),
        ]
    bp = base_path or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    wd = weights_dir or os.path.join(bp, "Models")
    return [
        os.path.join(wd, weight_file),
        os.path.join(bp, "Models", weight_file),
        os.path.join(bp, weight_file),
    ]


def resolve_model_weight(
    weight_file: str,
    weights_dir: Optional[str] = None,
    base_path: Optional[str] = None,
) -> Optional[str]:
    """Return the first existing path for a weight file, or None."""
    for candidate in model_weight_candidates(weight_file, weights_dir, base_path):
        if os.path.isfile(candidate):
            return candidate
    return None


def resolve_ultralytics_botsort_yaml() -> str:
    """Absolute path to the bundled Ultralytics BoT-SORT tracker config."""
    import ultralytics

    return str(Path(ultralytics.__file__).resolve().parent / "cfg" / "trackers" / "botsort.yaml")
