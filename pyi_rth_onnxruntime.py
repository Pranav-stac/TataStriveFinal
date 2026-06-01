"""
PyInstaller runtime hook: onnxruntime DLL paths before any imports.
Runs before main.py (app package may not be importable yet).
"""
import os
import sys
from pathlib import Path

if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
    _dirs: list[str] = []
    _base = Path(sys._MEIPASS)
    for _candidate in (
        _base,
        _base / "onnxruntime" / "capi",
        _base / "onnxruntime.libs",
    ):
        if _candidate.is_dir():
            _dirs.append(str(_candidate))
    try:
        _exe = Path(sys.executable).resolve().parent
        for _candidate in (
            _exe,
            _exe / "_internal",
            _exe / "_internal" / "onnxruntime" / "capi",
        ):
            if _candidate.is_dir():
                _dirs.append(str(_candidate))
    except Exception:
        pass

    _seen: set[str] = set()
    _unique = []
    for _d in _dirs:
        _key = _d.lower()
        if _key not in _seen:
            _seen.add(_key)
            _unique.append(_d)

    if _unique:
        os.environ["PATH"] = ";".join(_unique) + ";" + os.environ.get("PATH", "")
        _add = getattr(os, "add_dll_directory", None)
        if _add:
            for _d in _unique:
                try:
                    _add(_d)
                except OSError:
                    pass

        if os.name == "nt":
            import ctypes

            for _d in _unique:
                for _name in ("onnxruntime_providers_shared.dll", "onnxruntime.dll"):
                    _dll = Path(_d) / _name
                    if _dll.is_file():
                        try:
                            ctypes.CDLL(str(_dll))
                        except OSError:
                            pass
