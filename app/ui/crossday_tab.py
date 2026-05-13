"""
Attendance Tab.
UI for running attendance analysis.
Run mode is auto-detected from the DB file. Session info is derived from the current date.
Open Settings (Ctrl+,) for preview preferences; analysis defaults are fixed in the app.
"""

import os
from datetime import datetime
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QFrame, QSplitter,
    QMessageBox, QScrollArea
)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer

from app.config import get_config
from app.ui.widgets.file_picker import FilePicker, FolderPicker
from app.ui.widgets.progress_panel import ProgressPanel
from app.ui.widgets.video_preview import VideoPreview


class CrossDayTab(QWidget):
    """Tab for attendance analysis execution."""

    analysis_complete = pyqtSignal(str)
    VIDEO_EXTENSIONS = (".mp4", ".avi", ".mkv", ".mov", ".m4v", ".wmv")

    def __init__(self, parent: QWidget = None):
        super().__init__(parent)
        self.config = get_config()
        self._worker = None
        self._watch_timer = QTimer(self)
        self._watch_timer.setInterval(5000)
        self._watch_timer.timeout.connect(self._poll_video_folder)
        self._pending_videos = []
        self._known_videos = set()
        self._completed_videos = set()  # persisted; survives restart / app update
        self._size_probe = {}
        self._current_video = ""
        self._current_run_output_dir = ""
        self._runtime_db_path = ""
        self._is_monitoring = False
        self._stop_requested = False
        self._setup_ui()
        self._load_config()
        QTimer.singleShot(300, self._try_auto_start)

    def _setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(16)

        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        left_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        left_scroll.setFrameShape(QFrame.Shape.NoFrame)
        left_scroll.setMinimumWidth(340)

        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 8, 0)
        left_layout.setSpacing(12)

        self.video_picker = FolderPicker(
            label="Video Input Folder",
            placeholder="Select a folder to watch for videos..."
        )
        self.video_picker.path_changed.connect(self._on_video_changed)
        left_layout.addWidget(self.video_picker)

        self.db_picker = FilePicker(
            label="Database File",
            placeholder="Select master_database.db or .pkl  (leave empty for Day 1)...",
            file_filter="Database Files (*.db *.pkl);;SQLite DB (*.db);;Pickle Files (*.pkl);;All Files (*.*)"
        )
        self.db_picker.setToolTip(
            "Leave empty on Day 1 (BUILD_DB mode is auto-selected).\n"
            "Provide an existing database file for Day 2+ (EVAL_DAY mode is auto-selected)."
        )
        left_layout.addWidget(self.db_picker)

        self.output_picker = FolderPicker(
            label="Output Directory",
            placeholder="Select output folder..."
        )
        left_layout.addWidget(self.output_picker)

        self.video_preview = VideoPreview()
        self.video_preview.preview_toggled.connect(self._on_preview_toggled)
        left_layout.addWidget(self.video_preview)

        left_layout.addStretch()

        button_layout = QHBoxLayout()
        button_layout.setSpacing(12)

        self.start_btn = QPushButton("Start Monitoring")
        self.start_btn.setObjectName("primaryButton")
        self.start_btn.setMinimumHeight(40)
        self.start_btn.clicked.connect(self._start_analysis)
        button_layout.addWidget(self.start_btn)

        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setObjectName("dangerButton")
        self.stop_btn.setMinimumHeight(40)
        self.stop_btn.setVisible(False)
        self.stop_btn.clicked.connect(self._stop_analysis)
        button_layout.addWidget(self.stop_btn)

        left_layout.addLayout(button_layout)
        left_scroll.setWidget(left_panel)

        self.progress_panel = ProgressPanel()

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(left_scroll)
        splitter.addWidget(self.progress_panel)
        splitter.setSizes([420, 580])

        layout.addWidget(splitter)

    # ── Auto-detection helpers ────────────────────────────────────────

    def _get_run_mode(self) -> str:
        """Auto-detect run mode: EVAL_DAY if a DB file is available, else BUILD_DB."""
        db_path = self._runtime_db_path or self.db_picker.get_path().strip()
        if db_path and os.path.isfile(db_path):
            return "EVAL_DAY"
        return "BUILD_DB"

    def _get_day_label(self) -> str:
        """Auto-generate day label from today's date (e.g. Day0313)."""
        return f"Day{datetime.now().strftime('%m%d')}"

    # ── Config ───────────────────────────────────────────────────────

    def _load_config(self):
        last_folder = self.config.get("last_crossday_video_folder", "")
        if last_folder:
            self.video_picker.set_path(last_folder)

        last_db = self.config.get("last_db_path", "")
        if last_db:
            self.db_picker.set_path(last_db)

        last_output = self.config.get("last_crossday_output_dir", "")
        if last_output:
            self.output_picker.set_path(last_output)

        self.video_preview.set_enabled(self.config.get("preview_enabled", False))
        self._reload_completed_videos()

    @staticmethod
    def _norm_video_path(p: str) -> str:
        if not p:
            return ""
        return os.path.normcase(os.path.normpath(os.path.abspath(p)))

    def _reload_completed_videos(self):
        """Load completed paths for the current input folder from config (resume after restart)."""
        raw = self.config.get("crossday_completed_videos") or {}
        folder = self.video_picker.get_path().strip()
        if not folder:
            self._completed_videos = set()
            return
        fn = self._norm_video_path(folder)
        if self._norm_video_path(raw.get("folder") or "") != fn:
            self._completed_videos = set()
            return
        self._completed_videos = {self._norm_video_path(p) for p in (raw.get("paths") or []) if p}

    def _persist_completed_videos(self):
        folder = self.video_picker.get_path().strip()
        if not folder:
            return
        self.config.set(
            "crossday_completed_videos",
            {
                "folder": self._norm_video_path(folder),
                "paths": sorted(self._completed_videos),
            },
            save=True,
        )

    def _save_config(self):
        self.config.set("last_crossday_video_folder", self.video_picker.get_path(), save=False)
        self.config.set("last_db_path", self.db_picker.get_path(), save=False)
        self.config.set("last_crossday_output_dir", self.output_picker.get_path(), save=False)
        self.config.set("preview_enabled", self.video_preview.is_enabled(), save=True)

    def reload_config(self):
        self._load_config()

    # ── Slot handlers ────────────────────────────────────────────────

    def _on_video_changed(self, path: str):
        self._reload_completed_videos()
        if path and os.path.isdir(path):
            if not self.output_picker.get_path():
                self.output_picker.set_path(os.path.join(path, "Outputs"))

    def _on_preview_toggled(self, enabled: bool):
        self.config.set("preview_enabled", enabled)

    def _try_auto_start(self):
        """Auto-start monitoring if paths are already configured."""
        folder = self.video_picker.get_path()
        output = self.output_picker.get_path()
        if folder and os.path.isdir(folder) and output:
            self._start_analysis()

    # ── External setters ─────────────────────────────────────────────

    def set_video_path(self, path: str):
        folder = os.path.dirname(path) if os.path.isfile(path) else path
        if folder:
            self.video_picker.set_path(folder)

    def set_video_folder(self, folder_path: str):
        self.video_picker.set_path(folder_path)

    # ── Validation ───────────────────────────────────────────────────

    def _validate_inputs(self) -> bool:
        folder_path = self.video_picker.get_path()
        output_dir = self.output_picker.get_path()

        if not folder_path:
            QMessageBox.warning(self, "Validation Error", "Please select a video input folder.")
            return False

        if not os.path.isdir(folder_path):
            QMessageBox.warning(self, "Validation Error", "The selected video folder does not exist.")
            return False

        if not output_dir:
            QMessageBox.warning(self, "Validation Error", "Please select an output directory.")
            return False

        return True

    # ── Analysis control ─────────────────────────────────────────────

    def _start_analysis(self):
        if not self._validate_inputs():
            return

        self._save_config()
        os.makedirs(self.output_picker.get_path(), exist_ok=True)

        self._set_inputs_enabled(False)
        self.start_btn.setVisible(False)
        self.stop_btn.setVisible(True)

        self.progress_panel.reset()
        mode = self._get_run_mode()
        self.progress_panel.log_info(
            f"Folder listener started  [{mode}]  —  Waiting for new videos..."
        )
        self._runtime_db_path = self.db_picker.get_path().strip()
        self._is_monitoring = True
        self._stop_requested = False
        self._pending_videos.clear()
        self._known_videos.clear()
        self._size_probe.clear()
        self._current_video = ""
        self._current_run_output_dir = ""
        self._reload_completed_videos()
        if self._completed_videos:
            self.progress_panel.log_info(
                f"Resume: {len(self._completed_videos)} video(s) already completed in this folder — will skip."
            )
        self._poll_video_folder()
        self._watch_timer.start()

    def _stop_analysis(self):
        self._stop_requested = True
        self._watch_timer.stop()
        self._pending_videos.clear()
        self._size_probe.clear()
        if self._worker and self._worker.isRunning():
            self.progress_panel.log_warning("Stopping active analysis...")
            self._worker.stop()
        else:
            self._is_monitoring = False
            self._set_inputs_enabled(True)
            self.start_btn.setVisible(True)
            self.stop_btn.setVisible(False)
            self.progress_panel.log_info("Folder listener stopped.")

    def stop_analysis(self):
        self._stop_analysis()

    # ── Worker signals ───────────────────────────────────────────────

    def _on_progress(self, percent: int, message: str):
        self.progress_panel.update_progress(percent, message)

    def _on_log(self, message: str, level: str):
        self.progress_panel.log(message, level)

    def _on_finished(self, report_path: str):
        self._worker = None
        self.video_preview.clear()

        if report_path:
            self.progress_panel.update_progress(100, "Complete")
            self.progress_panel.log_success(f"Analysis complete! Report saved to: {report_path}")
            if self._current_video:
                self._completed_videos.add(self._norm_video_path(self._current_video))
                self._persist_completed_videos()
            # Delete source video if setting is enabled
            if self._current_video and self.config.get("crossday.delete_video_after_processing", False):
                try:
                    if os.path.isfile(self._current_video):
                        os.remove(self._current_video)
                        self.progress_panel.log_info(f"Deleted source video: {os.path.basename(self._current_video)}")
                except OSError as e:
                    self.progress_panel.log_warning(f"Could not delete video: {e}")
            self._update_runtime_db_after_run()
            self.analysis_complete.emit(report_path)
        else:
            self.progress_panel.log_warning("Analysis stopped by user.")

        self._current_video = ""
        self._current_run_output_dir = ""
        if self._is_monitoring and not self._stop_requested:
            self._try_start_next_video()
        else:
            self._is_monitoring = False
            self._set_inputs_enabled(True)
            self.start_btn.setVisible(True)
            self.stop_btn.setVisible(False)

    def _on_error(self, error_message: str):
        self._worker = None
        self.video_preview.clear()
        self.progress_panel.log_error(f"Error: {error_message}")
        self._current_video = ""
        self._current_run_output_dir = ""

        if self._is_monitoring and not self._stop_requested:
            self.progress_panel.log_warning("Listener continues. Waiting for next video...")
            self._try_start_next_video()
        else:
            QMessageBox.critical(self, "Analysis Error", f"An error occurred:\n\n{error_message}")
            self._is_monitoring = False
            self._set_inputs_enabled(True)
            self.start_btn.setVisible(True)
            self.stop_btn.setVisible(False)

    # ── Folder polling ───────────────────────────────────────────────

    def _poll_video_folder(self):
        folder = self.video_picker.get_path().strip()
        if not folder or not os.path.isdir(folder):
            return
        try:
            entries = sorted(os.listdir(folder))
        except OSError as e:
            self.progress_panel.log_warning(f"Could not read input folder: {e}")
            return

        newly_queued: list[str] = []
        for name in entries:
            file_path = os.path.join(folder, name)
            if not os.path.isfile(file_path):
                continue
            if not name.lower().endswith(self.VIDEO_EXTENSIONS):
                continue
            if self._norm_video_path(file_path) in self._completed_videos:
                continue
            if file_path in self._known_videos or file_path in self._pending_videos or file_path == self._current_video:
                continue
            try:
                size_now = os.path.getsize(file_path)
            except OSError:
                continue
            previous_size = self._size_probe.get(file_path)
            if previous_size is None or previous_size != size_now:
                self._size_probe[file_path] = size_now
                continue
            self._size_probe.pop(file_path, None)
            self._pending_videos.append(file_path)
            newly_queued.append(file_path)

        if newly_queued:
            self.progress_panel.log_batched_queued_videos(newly_queued)
            self.progress_panel.log_video_queue_summary(
                self._pending_videos, self._current_video
            )
        self._try_start_next_video()

    def _try_start_next_video(self):
        if not self._is_monitoring or self._stop_requested:
            return
        if self._worker and self._worker.isRunning():
            return
        if not self._pending_videos:
            self.progress_panel.update_progress(0, "Waiting for new videos...")
            return

        from app.workers.crossday_worker import CrossDayWorker

        video_path = self._pending_videos.pop(0)
        self._known_videos.add(video_path)
        self._current_video = video_path
        base_output_dir = self.output_picker.get_path()
        safe_name = os.path.splitext(os.path.basename(video_path))[0]
        run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_output_dir = os.path.join(base_output_dir, f"{safe_name}_{run_stamp}")
        os.makedirs(run_output_dir, exist_ok=True)
        self._current_run_output_dir = run_output_dir

        mode = self._get_run_mode()
        day_label = self._get_day_label()
        crossday_config = self.config.get_section("crossday") or {}

        worker_config = {
            "run_mode": mode,
            "current_date": datetime.now().strftime("%Y-%m-%d"),
            "day_label": day_label,
            "t_strict_merge": crossday_config.get("t_strict_merge", 0.55),
            "t_new_id": crossday_config.get("t_new_id", 0.35),
            "t_ratio_margin": crossday_config.get("t_ratio_margin", 0.10),
            "min_samples": crossday_config.get("min_samples", 8),
            "visitor_upgrade_days": crossday_config.get("visitor_upgrade_days", 3),
        }
        inference_cfg = self.config.get_section("inference") or {}
        worker_config["inference"] = inference_cfg
        worker_config["crossday"] = crossday_config
        # Shared ClassRoom Name (set in Settings) — written into every attendance report.
        worker_config["classroom_name"] = self.config.get("general.classroom_name", "")

        db_path = self._runtime_db_path if self._runtime_db_path else self.db_picker.get_path().strip()

        self.progress_panel.reset()
        self.progress_panel.log_info(
            f"Processing: {os.path.basename(video_path)}  [{mode}  |  {day_label}]"
        )
        self.progress_panel.log_video_queue_summary(
            self._pending_videos, self._current_video
        )
        self._worker = CrossDayWorker(
            video_path=video_path,
            output_dir=run_output_dir,
            db_path=db_path,
            config=worker_config,
            preview_enabled=self.video_preview.is_enabled()
        )
        qc = Qt.ConnectionType.QueuedConnection
        self._worker.progress.connect(self._on_progress, qc)
        self._worker.log_message.connect(self._on_log, qc)
        self._worker.frame_ready.connect(self.video_preview.update_frame, qc)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _update_runtime_db_after_run(self):
        """Chain the output database to the next run automatically."""
        if not self._current_run_output_dir:
            return
        mode = self._get_run_mode()
        if mode == "BUILD_DB":
            if self._runtime_db_path and os.path.isfile(self._runtime_db_path):
                return
            candidate = os.path.join(self._current_run_output_dir, "master_database.db")
            if not os.path.isfile(candidate):
                candidate = os.path.join(self._current_run_output_dir, "master_database.pkl")
        else:
            candidate = os.path.join(self._current_run_output_dir, "updated_master_database.db")
            if not os.path.isfile(candidate):
                candidate = os.path.join(self._current_run_output_dir, "updated_master_database.pkl")
        if os.path.isfile(candidate):
            self._runtime_db_path = candidate
            self.db_picker.set_path(candidate)
            self.config.set("last_db_path", candidate)

    def _set_inputs_enabled(self, enabled: bool):
        self.video_picker.setEnabled(enabled)
        self.db_picker.setEnabled(enabled)
        self.output_picker.setEnabled(enabled)

    def is_running(self) -> bool:
        return self._is_monitoring or (self._worker is not None and self._worker.isRunning())

    def is_monitoring(self) -> bool:
        return self._is_monitoring

    def pending_video_queue_count(self) -> int:
        """Videos still waiting in the folder-listener queue (after current one finishes)."""
        return len(self._pending_videos)
