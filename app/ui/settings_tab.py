"""Settings page for application preferences."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from PyQt6.QtCore import QThread, Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
    QPushButton, QCheckBox, QMessageBox,
    QLabel, QLineEdit, QFileDialog, QProgressBar, QTextEdit,
    QScrollArea, QFrame,
)

from app.config import get_config
from app.student_embeddings_sync import (
    SyncResult,
    count_enrolled_students,
    resolve_enrollments_db_path,
)
from app.workers.student_roster_worker import StudentRosterSyncWorker


class SettingsTab(QWidget):
    """Full-page settings view for application preferences and student roster preparation."""

    settings_saved = pyqtSignal()
    back_requested = pyqtSignal()

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.config = get_config()
        self._sync_thread: QThread | None = None
        self._sync_worker: StudentRosterSyncWorker | None = None
        self._setup_ui()
        self._load_settings()
        self._refresh_student_db_status()

    def _setup_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(16)

        header_layout = QHBoxLayout()
        back_btn = QPushButton("Back to analysis")
        back_btn.clicked.connect(self.back_requested.emit)
        header_layout.addWidget(back_btn)
        header_layout.addStretch()
        layout.addLayout(header_layout)

        general_group = QGroupBox("General")
        general_form = QVBoxLayout(general_group)
        self.preview_checkbox = QCheckBox("Enable video preview by default")
        general_form.addWidget(self.preview_checkbox)
        layout.addWidget(general_group)

        roster_group = QGroupBox("Student enrollment database")
        roster_form = QVBoxLayout(roster_group)

        self._roster_center_label = QLabel()
        self._roster_center_label.setWordWrap(True)
        roster_form.addWidget(self._roster_center_label)

        path_layout = QHBoxLayout()
        path_layout.addWidget(QLabel("SQLite path:"))
        self._db_path_edit = QLineEdit()
        self._db_path_edit.setPlaceholderText("Defaults to student_enrollments.db in the project folder")
        path_layout.addWidget(self._db_path_edit)
        browse_btn = QPushButton("Browse…")
        browse_btn.clicked.connect(self._browse_student_db_path)
        path_layout.addWidget(browse_btn)
        roster_form.addLayout(path_layout)

        self._db_status_label = QLabel()
        self._db_status_label.setWordWrap(True)
        roster_form.addWidget(self._db_status_label)

        self._sync_progress = QProgressBar()
        self._sync_progress.setRange(0, 100)
        self._sync_progress.setValue(0)
        self._sync_progress.setFormat("Idle")
        roster_form.addWidget(self._sync_progress)

        self._sync_log = QTextEdit()
        self._sync_log.setReadOnly(True)
        self._sync_log.setMinimumHeight(180)
        self._sync_log.setPlaceholderText("Roster sync progress appears here.")
        roster_form.addWidget(self._sync_log)

        sync_btn_layout = QHBoxLayout()
        self._sync_button = QPushButton("Prepare / sync roster")
        self._sync_button.clicked.connect(self._start_student_sync)
        sync_btn_layout.addWidget(self._sync_button)

        self._force_sync_button = QPushButton("Force re-embed all")
        self._force_sync_button.clicked.connect(self._start_force_student_sync)
        sync_btn_layout.addWidget(self._force_sync_button)
        sync_btn_layout.addStretch()
        roster_form.addLayout(sync_btn_layout)

        roster_hint = QLabel(
            "Builds the local face gallery from BigQuery roster + S3 enrollment photos. "
            "Attendance runs check for new or changed students in the background and only "
            "embed missing roster entries before matching."
        )
        roster_hint.setWordWrap(True)
        roster_form.addWidget(roster_hint)
        layout.addWidget(roster_group)

        processing_group = QGroupBox("Processing")
        processing_form = QVBoxLayout(processing_group)
        self.sync_roster_on_run_checkbox = QCheckBox(
            "Check roster for new students during each attendance run"
        )
        self.sync_roster_on_run_checkbox.setToolTip(
            "Runs an incremental BigQuery + S3 sync in the background while attendance "
            "video processing continues."
        )
        processing_form.addWidget(self.sync_roster_on_run_checkbox)
        self.delete_classroom_checkbox = QCheckBox(
            "Delete source video after engagement analysis completes"
        )
        self.delete_classroom_checkbox.setToolTip(
            "When enabled, the source video file is removed after a successful engagement run."
        )
        self.delete_crossday_checkbox = QCheckBox(
            "Delete source video after attendance analysis completes"
        )
        self.delete_crossday_checkbox.setToolTip(
            "When enabled, the source video file is removed after a successful attendance run."
        )
        self.save_classroom_video_checkbox = QCheckBox(
            "Save annotated output video (engagement analysis)"
        )
        self.save_classroom_video_checkbox.setToolTip(
            "When enabled, engagement runs write an annotated video in the output folder."
        )
        self.save_crossday_video_checkbox = QCheckBox(
            "Save annotated output video (attendance analysis)"
        )
        self.save_crossday_video_checkbox.setToolTip(
            "When enabled, attendance runs write an annotated video in the output folder."
        )
        processing_form.addWidget(self.delete_classroom_checkbox)
        processing_form.addWidget(self.delete_crossday_checkbox)
        processing_form.addWidget(self.save_classroom_video_checkbox)
        processing_form.addWidget(self.save_crossday_video_checkbox)
        layout.addWidget(processing_group)

        button_layout = QHBoxLayout()
        reset_btn = QPushButton("Reset to Defaults")
        reset_btn.clicked.connect(self._reset_defaults)
        button_layout.addWidget(reset_btn)
        button_layout.addStretch()

        self._save_button = QPushButton("Save Settings")
        self._save_button.setObjectName("primaryButton")
        self._save_button.clicked.connect(self._save_settings)
        button_layout.addWidget(self._save_button)
        layout.addLayout(button_layout)
        layout.addStretch()

        scroll.setWidget(content)
        outer.addWidget(scroll)

    def activate(self) -> None:
        """Refresh read-only status when the tab is opened."""
        self._refresh_student_db_status()

    def sync_running(self) -> bool:
        return self._sync_thread is not None and self._sync_thread.isRunning()

    def _load_settings(self) -> None:
        self.preview_checkbox.setChecked(self.config.get("preview_enabled", False))
        self.delete_classroom_checkbox.setChecked(
            self.config.get("classroom.delete_video_after_processing", False)
        )
        self.delete_crossday_checkbox.setChecked(
            self.config.get("crossday.delete_video_after_processing", False)
        )
        self.save_classroom_video_checkbox.setChecked(
            self.config.get("classroom.save_output_video", False)
        )
        self.save_crossday_video_checkbox.setChecked(
            self.config.get("crossday.save_output_video", True)
        )
        self.sync_roster_on_run_checkbox.setChecked(
            self.config.get("crossday.sync_student_roster_on_run", True)
        )
        self._db_path_edit.setText(self.config.get("crossday.student_db_path", ""))

    def _selected_db_path(self) -> Path:
        return resolve_enrollments_db_path(self._db_path_edit.text().strip())

    def _refresh_student_db_status(self) -> None:
        roster_center = (self.config.get("student_roster_center_name") or "").strip()
        if roster_center:
            self._roster_center_label.setText(f"BigQuery roster center: {roster_center}")
        else:
            self._roster_center_label.setText(
                "BigQuery roster center is not set. Choose it from BigQuery → Change Center…"
            )

        db_path = self._selected_db_path()
        embedded = count_enrolled_students(db_path)
        exists = db_path.is_file()
        status = self.config.get_section("student_roster")
        last_sync = (status.get("last_sync_at") or "").strip()
        if exists:
            file_line = f"Database: {db_path} ({embedded} embedded students)"
        else:
            file_line = f"Database: {db_path} (not created yet)"
        if last_sync:
            summary = (
                f"Last sync {last_sync}: roster {status.get('roster_total', 0)}, "
                f"updated {status.get('processed', 0)}, skipped {status.get('skipped', 0)}, "
                f"no photo {status.get('no_photo', 0)}, no face {status.get('no_face', 0)}, "
                f"errors {status.get('failed', 0)}."
            )
            self._db_status_label.setText(f"{file_line}\n{summary}")
        else:
            self._db_status_label.setText(file_line)

    def _browse_student_db_path(self) -> None:
        start_dir = str(self._selected_db_path().parent)
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Student enrollment SQLite database",
            start_dir,
            "SQLite database (*.db);;All files (*.*)",
        )
        if file_path:
            if not file_path.lower().endswith(".db"):
                file_path = f"{file_path}.db"
            self._db_path_edit.setText(file_path)
            self._refresh_student_db_status()

    def _append_sync_log(self, message: str, level: str = "info") -> None:
        stamp = datetime.now().strftime("%H:%M:%S")
        self._sync_log.append(f"[{stamp}] {message}")
        scrollbar = self._sync_log.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def _set_sync_controls_enabled(self, enabled: bool) -> None:
        self._sync_button.setEnabled(enabled)
        self._force_sync_button.setEnabled(enabled)

    def _start_student_sync(self) -> None:
        self._run_student_sync(force_all=False)

    def _start_force_student_sync(self) -> None:
        reply = QMessageBox.question(
            self,
            "Force re-embed all",
            "Re-download and re-embed every student photo, even if unchanged?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self._run_student_sync(force_all=True)

    def _run_student_sync(self, *, force_all: bool) -> None:
        if self.sync_running():
            return

        roster_center = (self.config.get("student_roster_center_name") or "").strip()
        if not roster_center:
            QMessageBox.warning(
                self,
                "Roster center required",
                "Set the BigQuery roster center first (BigQuery → Change Center…).",
            )
            return

        out_path = self._selected_db_path()
        self._sync_log.clear()
        self._append_sync_log(
            f"Starting roster sync for {roster_center} → {out_path}",
            "info",
        )
        self._sync_progress.setRange(0, 100)
        self._sync_progress.setValue(0)
        self._sync_progress.setFormat("Starting…")
        self._set_sync_controls_enabled(False)

        self._sync_thread = QThread(self)
        self._sync_worker = StudentRosterSyncWorker(
            out_path,
            center_name=roster_center,
            force_all=force_all,
        )
        self._sync_worker.moveToThread(self._sync_thread)
        self._sync_thread.started.connect(self._sync_worker.run)
        self._sync_worker.log_message.connect(self._append_sync_log)
        self._sync_worker.progress.connect(self._on_student_sync_progress)
        self._sync_worker.finished.connect(self._on_student_sync_finished)
        self._sync_worker.finished.connect(self._sync_thread.quit)
        self._sync_worker.finished.connect(self._sync_worker.deleteLater)
        self._sync_thread.finished.connect(self._sync_thread.deleteLater)
        self._sync_thread.start()

    def _on_student_sync_progress(self, done: int, total: int) -> None:
        if total <= 0:
            self._sync_progress.setRange(0, 0)
            self._sync_progress.setFormat("Working…")
            return
        self._sync_progress.setRange(0, total)
        self._sync_progress.setValue(done)
        self._sync_progress.setFormat(f"{done} / {total}")

    def _on_student_sync_finished(self, result: SyncResult) -> None:
        self._set_sync_controls_enabled(True)
        if result.output_path is not None:
            self._db_path_edit.setText(str(result.output_path))
        if result.ok and result.message == "complete":
            self.config.set("student_roster.last_sync_at", datetime.now().strftime("%Y-%m-%d %H:%M:%S"), save=False)
            self.config.set("student_roster.roster_total", result.roster_total, save=False)
            self.config.set("student_roster.processed", result.processed, save=False)
            self.config.set("student_roster.skipped", result.skipped, save=False)
            self.config.set("student_roster.no_photo", result.no_photo, save=False)
            self.config.set("student_roster.no_face", result.no_face, save=False)
            self.config.set("student_roster.failed", result.failed, save=False)
            self.config.set("student_roster.db_path", str(result.output_path or self._selected_db_path()), save=True)
            self._sync_progress.setValue(self._sync_progress.maximum())
            self._sync_progress.setFormat("Complete")
        elif result.ok:
            self._sync_progress.setFormat(result.message or "Finished")
        else:
            self._sync_progress.setFormat("Failed")
        self._refresh_student_db_status()

    def _save_settings(self) -> None:
        if self.sync_running():
            QMessageBox.warning(
                self,
                "Roster sync running",
                "Wait for the student enrollment sync to finish before saving settings.",
            )
            return

        self.config.set("preview_enabled", self.preview_checkbox.isChecked(), save=False)
        self.config.set(
            "classroom.delete_video_after_processing",
            self.delete_classroom_checkbox.isChecked(),
            save=False,
        )
        self.config.set(
            "crossday.delete_video_after_processing",
            self.delete_crossday_checkbox.isChecked(),
            save=False,
        )
        self.config.set(
            "classroom.save_output_video",
            self.save_classroom_video_checkbox.isChecked(),
            save=False,
        )
        self.config.set(
            "crossday.save_output_video",
            self.save_crossday_video_checkbox.isChecked(),
            save=False,
        )
        self.config.set(
            "crossday.sync_student_roster_on_run",
            self.sync_roster_on_run_checkbox.isChecked(),
            save=False,
        )
        self.config.set("crossday.student_db_path", self._db_path_edit.text().strip(), save=True)
        self.settings_saved.emit()

    def _reset_defaults(self) -> None:
        reply = QMessageBox.question(
            self,
            "Reset Settings",
            "Reset all stored settings to their default values?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self.config.reset()
            self._load_settings()
            self._refresh_student_db_status()
            self.settings_saved.emit()
