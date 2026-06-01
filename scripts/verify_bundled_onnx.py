"""
Verify onnxruntime files exist in a PyInstaller onedir build.
Run after build_exe.py:

  python scripts/verify_bundled_onnx.py
  python scripts/verify_bundled_onnx.py "C:\\path\\to\\TataStriveAnalytics"
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DIST = ROOT / "dist" / "TataStriveAnalytics"


def check(dist: Path) -> int:
    internal = dist / "_internal"
    if not internal.is_dir():
        print(f"FAIL: missing {internal}")
        return 1

    required = [
        internal / "onnxruntime" / "capi" / "onnxruntime.dll",
        internal / "onnxruntime" / "capi" / "onnxruntime_pybind11_state.pyd",
        internal / "onnxruntime" / "capi" / "onnxruntime_providers_shared.dll",
    ]
    ok = True
    for path in required:
        if path.is_file():
            print(f"OK  {path.relative_to(dist)}")
        else:
            print(f"MISSING  {path}")
            ok = False

    pyd_candidates = list(internal.rglob("onnxruntime_pybind11_state.pyd"))
    if not pyd_candidates:
        print("FAIL: onnxruntime_pybind11_state.pyd not found anywhere under _internal")
        ok = False

    buffalo = dist / "models" / "buffalo_l"
    if buffalo.is_dir() and any(buffalo.glob("*.onnx")):
        print(f"OK  models/buffalo_l ({len(list(buffalo.glob('*.onnx')))} onnx files)")
    else:
        print("WARN models/buffalo_l missing — rebuild after running buffalo_l once in dev")

    vc = dist / "vc_redist.x64.exe"
    print(f"{'OK' if vc.is_file() else 'WARN'}  vc_redist.x64.exe")

    return 0 if ok else 1


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DIST
    print(f"Checking: {target}\n")
    raise SystemExit(check(target))
