"""
Update Dialog for TataStrive Analytics.

Shows an available-update notification with changelog, file count, and a
progress bar while the patch is being downloaded and applied.

Thread-safety: the dialog is created on the Qt main thread.  The worker
callbacks (_Signals) use Qt signals to marshal results back to the main
thread automatically.
"""

from __future__ import annotations

from PyQt6.QtCore import Qt, QObject, pyqtSignal
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QDialog, QFrame, QHBoxLayout, QLabel, QMessageBox,
    QProgressBar, QPushButton, QTextEdit, QVBoxLayout,
)

from app.updater import UpdateChecker, UpdateInfo


# ── Signal bridge (keeps Qt signal in a QObject subclass) ────────────────────

class _Bridge(QObject):
    """Carries update-worker results back to the Qt main thread."""
    progress = pyqtSignal(int, int)   # bytes_done, total_bytes
    done     = pyqtSignal(bool, str)  # success, message


# ─────────────────────────────────────────────────────────────────────────────

class UpdateDialog(QDialog):
    """
    Modal dialog that presents an available update and handles the
    download-and-apply workflow with a live progress bar.

    Usage
    -----
    ::

        dlg = UpdateDialog(info, parent=main_window)
        dlg.exec()   # blocks; restarts the app on success
    """

    def __init__(self, info: UpdateInfo, parent=None) -> None:
        super().__init__(parent)
        self._info    = info
        self._bridge  = _Bridge()
        self._checker = UpdateChecker(current_version="")  # only used for apply

        self._bridge.progress.connect(self._on_progress)
        self._bridge.done.connect(self._on_done)

        self._build_ui()

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        self.setWindowTitle("Update Available — TataStrive Analytics")
        self.setFixedWidth(500)
        self.setModal(True)
        self.setWindowFlags(
            self.windowFlags() & ~Qt.WindowType.WindowContextHelpButtonHint
        )

        root = QVBoxLayout(self)
        root.setSpacing(14)
        root.setContentsMargins(24, 20, 24, 20)

        # ── Version headline ──────────────────────────────────────────────────
        headline = QLabel(
            f"TataStrive Analytics  v{self._info.version}  is available"
        )
        f = QFont("Segoe UI", 13, QFont.Weight.Bold)
        headline.setFont(f)
        headline.setAlignment(Qt.AlignmentFlag.AlignCenter)
        root.addWidget(headline)

        # ── Separator ─────────────────────────────────────────────────────────
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setFrameShadow(QFrame.Shadow.Sunken)
        root.addWidget(sep)

        # ── Changelog ─────────────────────────────────────────────────────────
        root.addWidget(QLabel("What's new:"))
        notes = QTextEdit()
        notes.setReadOnly(True)
        notes.setFixedHeight(110)
        notes.setPlainText(self._info.changelog)
        notes.setStyleSheet("border: 1px solid #555; border-radius: 4px; padding: 4px;")
        root.addWidget(notes)

        # ── Delta info ────────────────────────────────────────────────────────
        n = len(self._info.changed_files)
        info_lbl = QLabel(
            f"{n} file(s) will be patched  —  only changes are downloaded, not the full app"
        )
        info_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        info_lbl.setStyleSheet("color: #9a9a9a; font-size: 11px;")
        root.addWidget(info_lbl)

        # ── Progress bar (hidden until download starts) ───────────────────────
        self._progress = QProgressBar()
        self._progress.setVisible(False)
        self._progress.setTextVisible(True)
        self._progress.setFixedHeight(20)
        root.addWidget(self._progress)

        self._status_lbl = QLabel("")
        self._status_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._status_lbl.setStyleSheet("font-size: 11px; color: #aaa;")
        self._status_lbl.setVisible(False)
        root.addWidget(self._status_lbl)

        # ── Buttons ───────────────────────────────────────────────────────────
        btn_row = QHBoxLayout()
        btn_row.setSpacing(10)

        self._skip_btn   = QPushButton("Skip")
        self._later_btn  = QPushButton("Remind me later")
        self._update_btn = QPushButton("  Update Now  ")
        self._update_btn.setDefault(True)
        self._update_btn.setStyleSheet(
            "QPushButton { background: #0d6efd; color: white; "
            "border-radius: 4px; padding: 6px 16px; font-weight: bold; }"
            "QPushButton:hover { background: #0b5ed7; }"
            "QPushButton:disabled { background: #444; color: #888; }"
        )

        btn_row.addWidget(self._skip_btn)
        btn_row.addStretch()
        btn_row.addWidget(self._later_btn)
        btn_row.addWidget(self._update_btn)
        root.addLayout(btn_row)

        self._skip_btn.clicked.connect(self.reject)
        self._later_btn.clicked.connect(self.reject)
        self._update_btn.clicked.connect(self._start_download)

    # ── Button handler ────────────────────────────────────────────────────────

    def _start_download(self) -> None:
        self._update_btn.setEnabled(False)
        self._skip_btn.setEnabled(False)
        self._later_btn.setEnabled(False)

        self._progress.setRange(0, 0)   # indeterminate spinner while connecting
        self._progress.setVisible(True)
        self._status_lbl.setText("Connecting…")
        self._status_lbl.setVisible(True)

        self._checker.download_and_apply(
            self._info,
            progress_cb=lambda done, total: self._bridge.progress.emit(done, total),
            done_cb=lambda ok, msg:         self._bridge.done.emit(ok, msg),
        )

    # ── Signal slots (called on Qt main thread) ───────────────────────────────

    def _on_progress(self, done: int, total: int) -> None:
        self._status_lbl.setText("Downloading update…")
        if total > 0:
            self._progress.setRange(0, total)
            self._progress.setValue(done)
            pct      = int(done / total * 100)
            mb_done  = done  / 1_048_576
            mb_total = total / 1_048_576
            self._progress.setFormat(f"{pct}%  ({mb_done:.1f} / {mb_total:.1f} MB)")
        else:
            self._progress.setRange(0, 0)   # keep indeterminate if no Content-Length

    def _on_done(self, success: bool, message: str) -> None:
        self._progress.setRange(0, 100)
        self._progress.setValue(100 if success else 0)
        self._status_lbl.setText(message)

        if success:
            QMessageBox.information(
                self,
                "Update Ready",
                f"Update applied successfully!\n\nThe application will now restart.",
            )
            self.accept()
            UpdateChecker.restart_app()
        else:
            QMessageBox.warning(
                self,
                "Update Failed",
                f"{message}\n\nYou can try again later or update manually.",
            )
            self._update_btn.setEnabled(True)
            self._skip_btn.setEnabled(True)
            self._later_btn.setEnabled(True)
            self._progress.setVisible(False)
