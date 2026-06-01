"""Helpers for PyInstaller windowed builds (no console streams, DLL paths)."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Iterable, List, Optional

_onnxruntime_ready = False


def ensure_valid_stdio() -> None:
    """Windowed frozen apps often have stdout/stderr set to None; loguru/boxmot need a sink."""
    if sys.stdout is None:
        sys.stdout = open(os.devnull, "w")
    if sys.stderr is None:
        sys.stderr = open(os.devnull, "w")


def _unique_existing_dirs(paths: Iterable[Path]) -> List[Path]:
    seen: set[str] = set()
    out: List[Path] = []
    for path in paths:
        try:
            resolved = path.resolve()
        except OSError:
            continue
        key = str(resolved).lower()
        if key in seen or not resolved.is_dir():
            continue
        seen.add(key)
        out.append(resolved)
    return out


def frozen_dll_directories() -> List[Path]:
    """Directories that must be on PATH / add_dll_directory for onnxruntime when frozen."""
    dirs: List[Path] = []
    if getattr(sys, "frozen", False):
        if hasattr(sys, "_MEIPASS"):
            base = Path(sys._MEIPASS)
            dirs.extend(
                [
                    base,
                    base / "onnxruntime" / "capi",
                    base / "onnxruntime.libs",
                ]
            )
        try:
            exe_dir = Path(sys.executable).resolve().parent
            dirs.extend(
                [
                    exe_dir,
                    exe_dir / "_internal",
                    exe_dir / "_internal" / "onnxruntime" / "capi",
                    exe_dir / "onnxruntime" / "capi",
                ]
            )
        except Exception:
            pass
    return _unique_existing_dirs(dirs)


def configure_frozen_dll_paths() -> None:
    """Register DLL search paths for the frozen exe (safe to call repeatedly)."""
    if os.name != "nt":
        return
    dirs = frozen_dll_directories()
    if not dirs:
        return
    existing = os.environ.get("PATH", "")
    prepend = ";".join(str(d) for d in dirs)
    if prepend and not existing.startswith(prepend):
        os.environ["PATH"] = prepend + ";" + existing
    add_dll = getattr(os, "add_dll_directory", None)
    if add_dll is None:
        return
    for dll_dir in dirs:
        try:
            add_dll(str(dll_dir))
        except OSError:
            pass


def _preload_onnxruntime_dlls() -> None:
    if os.name != "nt":
        return
    import ctypes

    for dll_dir in frozen_dll_directories():
        for name in (
            "onnxruntime_providers_shared.dll",
            "onnxruntime.dll",
        ):
            dll_path = dll_dir / name
            if dll_path.is_file():
                try:
                    ctypes.CDLL(str(dll_path))
                except OSError:
                    pass


def ensure_onnxruntime_loaded() -> tuple[bool, Optional[str]]:
    """
    Make onnxruntime importable in frozen builds. Returns (ok, error_message).
    Call before ``from insightface.app import FaceAnalysis``.
    """
    global _onnxruntime_ready
    if _onnxruntime_ready:
        return True, None

    if getattr(sys, "frozen", False):
        configure_frozen_dll_paths()
        _preload_onnxruntime_dlls()

    try:
        import onnxruntime as ort  # noqa: F401

        _ = ort.__version__
        _onnxruntime_ready = True
        return True, None
    except Exception as exc:
        hint = ""
        if getattr(sys, "frozen", False):
            hint = (
                " Copy the entire TataStriveAnalytics folder (not only the .exe), "
                "run Run_TataStrive.bat, and install vc_redist.x64.exe if prompted."
            )
        return False, f"{exc}.{hint}"


def insightface_providers() -> List[str]:
    """Execution providers for FaceAnalysis; frozen builds stay CPU-only for reliability."""
    if getattr(sys, "frozen", False):
        return ["CPUExecutionProvider"]
    try:
        import torch

        if torch.cuda.is_available():
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    except Exception:
        pass
    return ["CPUExecutionProvider"]
