"""
TataStrive Analytics - Main Entry Point
A professional desktop application for classroom analysis and attendance tracking.
"""

import os
import sys

# Force CPU mode to avoid CUDA DLL errors on Windows (must be before any torch import)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from pathlib import Path

# Add the project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Load onnxruntime as early as possible (before PyTorch/PyQt) - DLL order matters on Windows
try:
    import onnxruntime as _ort
    _ = _ort.__version__
except (OSError, ImportError):
    pass

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
    
    # Pre-load PyTorch in main thread (avoids DLL issues when loading in worker thread)
    torch_available = False
    try:
        import torch
        _ = torch.__version__
        torch_available = True
    except (OSError, ImportError):
        reply = QMessageBox.warning(
            None,
            "PyTorch Not Available",
            "PyTorch failed to load (DLL error).\n\n"
            "Analysis features are disabled. You can still use Report Viewer.\n\n"
            "To fix: Run in Command Prompt:\n"
            "  pip uninstall torch torchvision -y\n"
            "  pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu\n\n"
            "Or install Visual C++ Redistributable:\n"
            "https://aka.ms/vs/17/release/vc_redist.x64.exe\n\n"
            "Continue in limited mode?",
            QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel
        )
        if reply == QMessageBox.StandardButton.Cancel:
            sys.exit(1)
    
    app.setApplicationName("TataStrive Analytics")
    app.setApplicationVersion("1.0.0")
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
    
    # Import and create main window
    from app.ui.main_window import MainWindow
    
    window = MainWindow(torch_available=torch_available)
    window.show()
    
    # Run application
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
