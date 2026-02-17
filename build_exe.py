"""
Build Script for TataStrive Analytics

Creates a standalone .exe using PyInstaller.
Auto-downloads required models (face, osnet) before building.

Usage:
    python build_exe.py
"""

import os
import sys
import shutil
import subprocess
import urllib.request
from pathlib import Path

# Ensure build dependencies (PyInstaller + collect-all need these)
for mod in ("pefile", "typing_extensions"):
    try:
        __import__(mod)
    except ImportError:
        print(f"Installing {mod}...")
        subprocess.run([sys.executable, "-m", "pip", "install", mod, "-q"], check=True)

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
        f"--paths={site_packages[0]}",  # Help PyInstaller find typing_extensions etc.
        
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
        "--hidden-import=python-dotenv",
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
    
    # Create a batch file to run the app
    batch_content = '''@echo off
cd /d "%~dp0"
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
    print("  2. Run: TataStriveAnalytics.exe")
    print()
    print("Models (yolov8n-face.pt, etc.) are bundled. Detection/pose models")
    print("download automatically on first run.")


if __name__ == "__main__":
    build()
