"""
Build Script for TataStrive Analytics

Creates a standalone .exe using PyInstaller.
Auto-downloads required models (face, osnet) before building.
Uses the project's env venv (has all dependencies).

Usage:
    python build_exe.py
    # Or: env\Scripts\python.exe build_exe.py
"""

import os
import sys
from pathlib import Path

# Use project env venv (has torch, ultralytics, etc.)
_PROJECT_ROOT = Path(__file__).resolve().parent
_ENV_PYTHON = _PROJECT_ROOT / "env" / "Scripts" / "python.exe"
if _ENV_PYTHON.exists() and Path(sys.executable).resolve() != _ENV_PYTHON.resolve():
    print(f"Re-running with env Python: {_ENV_PYTHON}")
    script = str(Path(__file__).resolve())
    os.execv(str(_ENV_PYTHON), [str(_ENV_PYTHON), script] + sys.argv[1:])
import shutil
import subprocess
import urllib.request

# Ensure build dependencies (PyInstaller + pip packages)
for mod, pkg in [("pefile", "pefile"), ("typing_extensions", "typing_extensions"), ("dotenv", "python-dotenv")]:
    try:
        __import__(mod)
    except ImportError:
        print(f"Installing {pkg}...")
        subprocess.run([sys.executable, "-m", "pip", "install", pkg, "-q"], check=True)

# Check for PyInstaller
try:
    import PyInstaller.__main__
except ImportError:
    print("PyInstaller not found. Installing...")
    os.system(f"{sys.executable} -m pip install pyinstaller")
    import PyInstaller.__main__


def ensure_models(project_root: Path) -> bool:
    """Download required models if missing. Returns True if Models folder is ready."""
    models_dir = project_root / "Models"
    models_dir.mkdir(exist_ok=True)
    
    # 1. Face model (yolov8n-face.pt) - not in Ultralytics
    face_path = models_dir / "yolov8n-face.pt"
    if not face_path.exists():
        print("Downloading yolov8n-face.pt...")
        try:
            url = "https://huggingface.co/deepghs/yolo-face/resolve/main/yolov8n-face/model.pt"
            urllib.request.urlretrieve(url, face_path)
            print(f"  Saved to {face_path}")
        except Exception as e:
            print(f"  WARNING: Face model download failed: {e}")
            print("  Run 'python download_face_model.py' manually before building.")
    
    # 2. OSNet - BoxMOT downloads on first run; we ensure Models/ exists for it
    # (yolov8m.pt, yolov8n-pose.pt are auto-downloaded by Ultralytics)
    return models_dir.exists()


def get_onnxruntime_binaries():
    """Get onnxruntime DLLs/pyd for bundling. PyInstaller often misses these."""
    try:
        import onnxruntime as ort
        ort_dir = Path(ort.__file__).parent / "capi"
        if not ort_dir.exists():
            return []
        binaries = []
        for f in ort_dir.iterdir():
            if f.suffix in (".dll", ".pyd"):
                # Dest: onnxruntime/capi/ so package finds them when frozen
                binaries.append((str(f), "onnxruntime/capi"))
        return binaries
    except ImportError:
        return []


def get_vcruntime_binaries():
    """Get VC++ runtime DLLs from Python installation for bundling."""
    binaries = []
    python_dir = Path(sys.executable).parent
    for dll_name in ["vcruntime140.dll", "vcruntime140_1.dll", "msvcp140.dll", "msvcp140_1.dll"]:
        dll_path = python_dir / dll_name
        if dll_path.exists():
            binaries.append((str(dll_path), "."))
    return binaries


def build():
    """Build the executable."""
    # Paths
    project_root = Path(__file__).parent
    app_dir = project_root / "app"
    resources_dir = app_dir / "resources"
    dist_dir = project_root / "dist"
    build_dir = project_root / "build"
    
    # Pre-build: ensure required models exist
    print("Ensuring required models...")
    ensure_models(project_root)
    models_dir = project_root / "Models"
    
    # Pre-build: ensure onnxruntime (CPU) for reliable bundling - onnxruntime-gpu has different DLLs
    try:
        import onnxruntime as ort
        if "CUDAExecutionProvider" in ort.get_available_providers():
            print("WARNING: onnxruntime-gpu detected. For reliable builds, use CPU-only:")
            print("  pip uninstall onnxruntime-gpu -y")
            print("  pip install onnxruntime")
    except Exception:
        pass
    
    # Clean previous builds (skip if files are locked - e.g. exe is running)
    if dist_dir.exists():
        print("Cleaning previous dist folder...")
        try:
            shutil.rmtree(dist_dir)
        except PermissionError:
            print("WARNING: Could not delete dist folder (files may be in use).")
            print("         Close TataStriveAnalytics.exe and any file explorers, then retry.")
            print("         Building anyway - PyInstaller will overwrite where possible...")
    if build_dir.exists():
        print("Cleaning previous build folder...")
        try:
            shutil.rmtree(build_dir)
        except PermissionError:
            print("WARNING: Could not delete build folder. Building anyway...")
    
    # Add site-packages to path so PyInstaller finds typing_extensions, onnx, etc.
    import site
    site_packages = site.getsitepackages()
    if not site_packages:
        site_packages = [str(Path(sys.executable).parent / "Lib" / "site-packages")]
    
    # PyInstaller arguments
    args = [
        str(app_dir / "main.py"),
        "--name=TataStriveAnalytics",
        "--windowed",  # No console window
        "--onedir",  # Create a directory with all files (more reliable than onefile)
        f"--distpath={dist_dir}",
        f"--workpath={build_dir}",
        "--clean",
        "--exclude-module=onnx.reference",  # Avoids subprocess crash (0xC0000005) during binary dep analysis
        f"--paths={site_packages[0]}",  # Help PyInstaller find typing_extensions etc.
        f"--runtime-hook={project_root / 'pyi_rth_onnxruntime.py'}",  # Add DLL path before main
        
        # Add data files
        f"--add-data={resources_dir};resources",
        
        # Add the app package
        f"--add-data={app_dir};app",
        
        # Add existing analysis code (stitch_logic, vlm_metadata)
        f"--add-data={project_root / 'classroom_analysis'};classroom_analysis",
    ]
    
    # Add Models folder (bundled so frozen app finds yolov8n-face.pt etc.)
    if models_dir.exists():
        args.append(f"--add-data={models_dir};Models")
    
    # Add onnxruntime DLLs (PyInstaller often misses these - causes "DLL load failed" when frozen)
    ort_binaries = get_onnxruntime_binaries()
    for src, dest in ort_binaries:
        args.append(f"--add-binary={src};{dest}")
    if ort_binaries:
        print("Bundling onnxruntime DLLs for face matching...")
    
    # Add VC++ runtime DLLs (onnxruntime depends on these)
    vcrt_binaries = get_vcruntime_binaries()
    for src, dest in vcrt_binaries:
        args.append(f"--add-binary={src};{dest}")
    if vcrt_binaries:
        print(f"Bundling VC++ runtime DLLs: {[Path(b[0]).name for b in vcrt_binaries]}")
    
    # Collect all onnxruntime binaries using PyInstaller's collector
    args.append("--collect-binaries=onnxruntime")
    
    args += [
        # Hidden imports (modules that PyInstaller might miss)
        "--hidden-import=typing_extensions",  # Must be before ultralytics
        "--hidden-import=ultralytics",
        "--hidden-import=ultralytics.models",
        "--hidden-import=ultralytics.nn",
        "--hidden-import=ultralytics.utils",
        "--hidden-import=onnx",
        "--hidden-import=omegaconf",
        "--hidden-import=boxmot",
        "--hidden-import=insightface",
        "--hidden-import=onnxruntime",
        "--hidden-import=scipy.spatial.distance",
        "--hidden-import=cv2",
        "--hidden-import=torch",
        "--hidden-import=torchvision",
        "--hidden-import=numpy",
        "--hidden-import=PIL",
        "--hidden-import=groq",
        "--hidden-import=dotenv",
        "--hidden-import=PyQt6",
        "--hidden-import=PyQt6.QtWidgets",
        "--hidden-import=PyQt6.QtCore",
        "--hidden-import=PyQt6.QtGui",
        
        # Skip --collect-all (fails when subprocess can't import typing_extensions/onnx)
        # Hidden imports + runtime discovery should suffice
        
        # Don't confirm overwrite
        "--noconfirm",
    ]
    
    # Add icon if exists
    icon_path = resources_dir / "icons" / "app.ico"
    if icon_path.exists():
        args.append(f"--icon={icon_path}")
    
    print("Building TataStrive Analytics...")
    print("This may take several minutes...")
    print()
    
    # Run PyInstaller
    PyInstaller.__main__.run(args)
    
    # Post-build: Copy Models folder if exists
    models_src = project_root / "Models"
    models_dst = dist_dir / "TataStriveAnalytics" / "Models"
    if models_src.exists():
        print("Copying Models folder...")
        shutil.copytree(models_src, models_dst)
    
    # Create a batch file to run the app (cd ensures DLLs load from exe's folder)
    batch_content = '''@echo off
cd /d "%~dp0"
set "PATH=%CD%\\_internal;%CD%;%PATH%"
start "" "TataStriveAnalytics.exe"
'''
    batch_path = dist_dir / "TataStriveAnalytics" / "Run_TataStrive.bat"
    with open(batch_path, 'w') as f:
        f.write(batch_content)
    
    print()
    print("=" * 50)
    print("BUILD COMPLETE!")
    print("=" * 50)
    print()
    print(f"Output location: {dist_dir / 'TataStriveAnalytics'}")
    print()
    print("To run the application:")
    print(f"  1. Navigate to: {dist_dir / 'TataStriveAnalytics'}")
    print("  2. Run: TataStriveAnalytics.exe (or Run_TataStrive.bat)")
    print()
    print("Models (yolov8n-face.pt, etc.) are bundled. Detection/pose models")
    print("download automatically on first run.")


if __name__ == "__main__":
    build()
