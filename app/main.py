"""
TataStrive Analytics - Main Entry Point
A professional desktop application for classroom analysis and attendance tracking.
"""

import os
import shutil
import sys

# Force CPU mode to avoid CUDA DLL errors on Windows (must be before any torch import)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from pathlib import Path


def _merge_partial_app_overlay_from_bundle() -> None:
    """
    Delta patches only ship changed files. If app/ exists next to the exe but is
    incomplete, Python would load that partial package first and crash (e.g. no
    app.config). Copy any missing modules from the bundled _internal/app tree.
    """
    if not getattr(sys, "frozen", False) or not hasattr(sys, "_MEIPASS"):
        return
    exe_dir = Path(sys.executable).resolve().parent
    overlay = exe_dir / "app"
    bundle_app = Path(sys._MEIPASS) / "app"
    if not overlay.is_dir() or not bundle_app.is_dir():
        return
    for src in bundle_app.rglob("*"):
        if not src.is_file():
            continue
        rel = src.relative_to(bundle_app)
        dest = overlay / rel
        if not dest.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)


# Dev: repo root on path. Frozen: bundle first; prepend exe dir only if overlay is complete
# (after merge) so "import app" sees patched files without a broken partial package.
project_root = Path(__file__).parent.parent
if getattr(sys, "frozen", False):
    _merge_partial_app_overlay_from_bundle()
    exe_dir = Path(sys.executable).resolve().parent
    sys.path.insert(0, str(project_root))
    if (exe_dir / "app" / "config.py").is_file():
        sys.path.insert(0, str(exe_dir))
    from app.frozen_runtime import ensure_valid_stdio

    ensure_valid_stdio()
else:
    sys.path.insert(0, str(project_root))


def _load_env() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return

    load_dotenv()
    load_dotenv(project_root / ".env", override=False)
    if getattr(sys, "frozen", False):
        load_dotenv(Path(sys.executable).resolve().parent / ".env", override=False)


_load_env()

# Version string for About / updater (reads app/__init__.py from overlay when patched)
from app import __version__

# Frozen exe: register DLL directories early (import happens after VC++ in main()).
if getattr(sys, "frozen", False):
    from app.frozen_runtime import configure_frozen_dll_paths

    configure_frozen_dll_paths()

from PyQt6.QtWidgets import QApplication, QMessageBox
from PyQt6.QtCore import Qt, QFile, QTextStream
from PyQt6.QtGui import QIcon


def load_stylesheet(app: QApplication) -> None:
    """Load the application stylesheet."""
    # Try multiple paths for the stylesheet
    possible_paths = [
        Path(__file__).parent / "resources" / "styles.qss",
        project_root / "app" / "resources" / "styles.qss",
        Path(sys._MEIPASS) / "resources" / "styles.qss" if getattr(sys, 'frozen', False) else None
    ]
    
    for style_path in possible_paths:
        if style_path and style_path.exists():
            with open(style_path, 'r', encoding='utf-8') as f:
                app.setStyleSheet(f.read())
            print(f"Loaded stylesheet from: {style_path}")
            return
    
    print("Warning: Could not find stylesheet")


def main():
    """Main entry point for the application."""
    # High DPI support
    if hasattr(Qt, 'AA_EnableHighDpiScaling'):
        QApplication.setAttribute(Qt.ApplicationAttribute.AA_EnableHighDpiScaling, True)
    if hasattr(Qt, 'AA_UseHighDpiPixmaps'):
        QApplication.setAttribute(Qt.ApplicationAttribute.AA_UseHighDpiPixmaps, True)

    # Create application (needed for message box)
    app = QApplication(sys.argv)

    if getattr(sys, "frozen", False) and os.name == "nt":
        from app.vcredist import ensure_vc_redist

        ensure_vc_redist()

    # After VC++ runtime: load onnxruntime before PyTorch (DLL order on Windows).
    if getattr(sys, "frozen", False):
        from app.frozen_runtime import ensure_onnxruntime_loaded

        ort_ok, ort_err = ensure_onnxruntime_loaded()
        if not ort_ok:
            QMessageBox.warning(
                None,
                "Face matching unavailable",
                "ONNX Runtime could not load. Attendance will run without InsightFace "
                "(no student ID photos / weaker identity matching).\n\n"
                f"{ort_err}\n\n"
                "Fix: run vc_redist.x64.exe from this folder, use Run_TataStrive.bat, "
                "and copy the full TataStriveAnalytics folder (not only the .exe).",
            )

    # Pre-load PyTorch in main thread (avoids DLL issues when loading in worker thread)
    torch_available = False
    try:
        import torch
        _ = torch.__version__
        torch_available = True
    except (OSError, ImportError):
        from app.vcredist import VC_REDIST_DOWNLOAD_URL, ensure_vc_redist, find_bundled_installer

        if ensure_vc_redist():
            try:
                import torch
                _ = torch.__version__
                torch_available = True
            except (OSError, ImportError):
                pass

        if not torch_available:
            if find_bundled_installer() is None:
                vc_hint = f"Download: {VC_REDIST_DOWNLOAD_URL}\n\n"
            else:
                vc_hint = (
                    "Run vc_redist.x64.exe from the app folder, or choose Install "
                    "when prompted on the next launch.\n\n"
                )
            reply = QMessageBox.warning(
                None,
                "PyTorch Not Available",
                "PyTorch failed to load (DLL error).\n\n"
                "Most common cause on Windows: Microsoft Visual C++ 2015-2022 "
                "Redistributable (x64) is missing.\n"
                f"{vc_hint}"
                "Analysis features are disabled. You can still use Report Viewer.\n\n"
                "To fix: Run in Command Prompt:\n"
                "  pip uninstall torch torchvision -y\n"
                "  pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu\n\n"
                "Continue in limited mode?",
                QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel
            )
            if reply == QMessageBox.StandardButton.Cancel:
                sys.exit(1)

    app.setApplicationName("TataStrive Analytics")
    app.setApplicationVersion(__version__)
    app.setOrganizationName("TataStrive")

    # Set application icon
    icon_paths = [
        Path(__file__).parent / "resources" / "icons" / "app.ico",
        project_root / "app" / "resources" / "icons" / "app.ico"
    ]
    for icon_path in icon_paths:
        if icon_path.exists():
            app.setWindowIcon(QIcon(str(icon_path)))
            break

    # Load stylesheet
    load_stylesheet(app)

    # ── Center-ID setup ───────────────────────────────────────────────────
    from app.config import get_config
    from app.ui.center_dialog import CenterDialog

    cfg = get_config()
    center_id = cfg.get("center_id", "").strip()
    roster_center = cfg.get("student_roster_center_name", "").strip()

    if not roster_center:
        dlg = CenterDialog(
            parent=None,
            existing_name=center_id,
            existing_roster_center=roster_center,
        )
        dlg.exec()
        roster_center = dlg.roster_center_name()
        center_id = dlg.center_name() or roster_center or "DefaultCenter"
        cfg.set("student_roster_center_name", roster_center)
        cfg.set("center_id", center_id)
    elif not center_id:
        center_id = roster_center
        cfg.set("center_id", center_id)

    # ── BigQuery service init ─────────────────────────────────────────────
    from app.bigquery_sync import get_sync_service, _creds_path
    bq_service = get_sync_service(
        center_id=center_id,
        credentials_path=_creds_path()
    )

    # ── Import and create main window ─────────────────────────────────────
    from app.ui.main_window import MainWindow

    window = MainWindow(torch_available=torch_available, bq_service=bq_service)
    window.show()

    # ── Kick off daily BigQuery sync (background thread) ──────────────────
    auto_sync = cfg.get("bigquery.auto_sync", True)
    if auto_sync:
        output_dirs = [
            d for d in [
                cfg.get("last_classroom_output_dir", ""),
                cfg.get("last_crossday_output_dir", ""),
            ] if d
        ]
        bq_service.trigger_daily_sync(
            output_dirs=output_dirs,
            log_callback=lambda msg: print(f"[BQ] {msg}"),
            done_callback=lambda summary: window.on_bq_sync_done(summary)
        )

    # ── Auto-update: silent background polling (no user interaction) ──────
    _start_update_polling(window, app)

    # Run application
    sys.exit(app.exec())


def _start_update_polling(window, app) -> None:
    """
    Poll GitHub Releases on a timer (see updater.DEFAULT_POLL_INTERVAL_MINUTES).
    When a newer release exists, the patch is downloaded and applied in the
    background and the process restarts — no dialogs or clicks.
    """
    from app.updater import UpdateChecker

    checker = UpdateChecker(current_version=__version__, silent_auto_apply=True)

    def _on_error(msg: str) -> None:
        print(f"[Updater] Check failed: {msg}")

    checker.on_error = _on_error

    app.aboutToQuit.connect(checker.stop_polling)
    checker.start_polling()

    window._update_checker = checker


if __name__ == "__main__":
    main()

