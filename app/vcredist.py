"""Detect and install MSVC 2015-2022 redistributable on Windows."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Optional

VC_REDIST_DOWNLOAD_URL = "https://aka.ms/vs/17/release/vc_redist.x64.exe"
VC_REDIST_FILENAME = "vc_redist.x64.exe"


def _runtime_registry_paths() -> Iterable[str]:
    yield r"SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\X64"
    yield r"SOFTWARE\WOW6432Node\Microsoft\VisualStudio\14.0\VC\Runtimes\X64"


def is_vc_redist_installed() -> bool:
    if os.name != "nt":
        return True
    try:
        import winreg
    except ImportError:
        return True

    for path in _runtime_registry_paths():
        try:
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, path) as key:
                installed, _ = winreg.QueryValueEx(key, "Installed")
                if installed:
                    return True
        except OSError:
            continue
    return False


def find_bundled_installer() -> Optional[Path]:
    candidates: list[Path] = []
    if getattr(sys, "frozen", False):
        candidates.append(Path(sys.executable).resolve().parent / VC_REDIST_FILENAME)
    if hasattr(sys, "_MEIPASS"):
        candidates.append(Path(sys._MEIPASS) / VC_REDIST_FILENAME)
    for path in candidates:
        if path.is_file():
            return path
    return None


def install_bundled_vc_redist(installer: Path) -> bool:
    if is_vc_redist_installed():
        return True
    if os.name != "nt":
        return False

    installer = installer.resolve()
    command = (
        f"Start-Process -FilePath '{installer}' "
        "-ArgumentList '/install','/passive','/norestart' "
        "-Verb RunAs -Wait"
    )
    proc = subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-Command",
            command,
        ],
        check=False,
    )
    return proc.returncode == 0 and is_vc_redist_installed()


def ensure_vc_redist(parent=None) -> bool:
    """Return True when the MSVC runtime is present or was installed successfully."""
    if is_vc_redist_installed():
        return True
    if os.name != "nt":
        return True

    installer = find_bundled_installer()
    if installer is None:
        return False

    try:
        from PyQt6.QtWidgets import QMessageBox
    except ImportError:
        return install_bundled_vc_redist(installer)

    reply = QMessageBox.question(
        parent,
        "Install Microsoft Visual C++ Runtime",
        "This app needs the Microsoft Visual C++ 2015-2022 Redistributable (x64).\n\n"
        "Install it now? Windows may ask for administrator permission.",
        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        QMessageBox.StandardButton.Yes,
    )
    if reply != QMessageBox.StandardButton.Yes:
        return False

    if install_bundled_vc_redist(installer):
        QMessageBox.information(
            parent,
            "Runtime Installed",
            "Microsoft Visual C++ 2015-2022 Redistributable (x64) is installed.",
        )
        return True

    QMessageBox.warning(
        parent,
        "Runtime Install Failed",
        "The Visual C++ runtime could not be installed automatically.\n\n"
        f"Run {VC_REDIST_FILENAME} from the app folder, or download it from:\n"
        f"{VC_REDIST_DOWNLOAD_URL}",
    )
    return False
