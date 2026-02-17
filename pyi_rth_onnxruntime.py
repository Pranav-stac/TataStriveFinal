"""
PyInstaller runtime hook: add onnxruntime DLL path before any imports.
Runs before main.py - ensures DLLs are findable when onnxruntime loads.
"""
import os
import sys
from pathlib import Path

if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
    base = Path(sys._MEIPASS)
    # Possible locations for onnxruntime DLLs
    dll_dirs = [
        str(base),
        str(base / "onnxruntime" / "capi"),
        str(base / "onnxruntime.libs"),  # Some versions use this
    ]
    
    # Add to PATH (older method, works on more systems)
    existing_path = os.environ.get("PATH", "")
    os.environ["PATH"] = ";".join([d for d in dll_dirs if Path(d).exists()]) + ";" + existing_path
    
    # Also use add_dll_directory (modern method, Windows 10+)
    for dll_dir in dll_dirs:
        if Path(dll_dir).exists():
            try:
                os.add_dll_directory(dll_dir)
            except (OSError, AttributeError):
                pass
    
    # Try to preload onnxruntime DLL using ctypes (helps with initialization issues)
    import ctypes
    for dll_dir in dll_dirs:
        ort_dll = Path(dll_dir) / "onnxruntime.dll"
        if ort_dll.exists():
            try:
                ctypes.CDLL(str(ort_dll))
            except OSError:
                pass
            break
