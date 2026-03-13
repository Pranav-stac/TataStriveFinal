"""
Classroom Analysis Tab.
UI for configuring and running classroom engagement analysis.
"""

import os
from datetime import datetime
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QSpinBox, QDoubleSpinBox, QPushButton, QFrame, QSplitter,
    QSizePolicy, QMessageBox, QScrollArea
)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer

from app.config import get_config
from app.ui.widgets.file_picker import FolderPicker
from app.ui.widgets.progress_panel import ProgressPanel
from app.ui.widgets.video_preview import VideoPreview


class ClassroomTab(QWidget):
    """Tab for classroom analysis configuration and execution."""
    
    analysis_complete = pyqtSignal(str)  # Emits report path
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
        self._size_probe = {}
        self._current_video = ""
        self._is_monitoring = False
        self._stop_requested = False
        self._setup_ui()
        self._load_config()
        
    def _setup_ui(self):
        """Setup the tab UI."""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(16)
        
        # Left panel - Configuration (scrollable)
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        left_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        left_scroll.setFrameShape(QFrame.Shape.NoFrame)
        left_scroll.setMinimumWidth(380)
        
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 8, 0)
        left_layout.setSpacing(12)
        
        # File inputs
        self.video_picker = FolderPicker(
            label="Video Input Folder",
            placeholder="Select a folder to watch for videos..."
        )
        self.video_picker.path_changed.connect(self._on_video_changed)
        left_layout.addWidget(self.video_picker)
        
        self.output_picker = FolderPicker(
            label="Output Directory",
            placeholder="Select output folder..."
        )
        left_layout.addWidget(self.output_picker)
        
        # Settings button - opens global settings
        from app.ui.settings_dialog import SettingsDialog
        settings_btn = QPushButton("Settings...")
        settings_btn.setToolTip("Configure analysis settings (Ctrl+,)")
        settings_btn.clicked.connect(lambda: self._open_settings())
        left_layout.addWidget(settings_btn)
        
        # Video preview
        self.video_preview = VideoPreview()
        self.video_preview.preview_toggled.connect(self._on_preview_toggled)
        left_layout.addWidget(self.video_preview)
        
        # Buttons
        button_layout = QHBoxLayout()
        button_layout.setSpacing(12)
        
        self.start_btn = QPushButton("Start Analysis")
        self.start_btn.setObjectName("primaryButton")
        self.start_btn.setMinimumHeight(40)
        self.start_btn.clicked.connect(self._start_analysis)
        button_layout.addWidget(self.start_btn)
        
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setObjectName("dangerButton")
        self.stop_btn.setMinimumHeight(40)
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._stop_analysis)
        button_layout.addWidget(self.stop_btn)
        
        left_layout.addLayout(button_layout)
        
        left_scroll.setWidget(left_panel)
        
        # Right panel - Progress
        self.progress_panel = ProgressPanel()
        
        # Add to main layout with splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(left_scroll)
        splitter.addWidget(self.progress_panel)
        splitter.setSizes([500, 500])
        
        layout.addWidget(splitter)
        
    def _load_config(self):
        """Load configuration values."""
        # Load last paths
        last_folder = self.config.get("last_video_folder", "")
        if last_folder:
            self.video_picker.set_path(last_folder)
            
        last_output = self.config.get("last_output_dir", "")
        if last_output:
            self.output_picker.set_path(last_output)
            
        # Preview state
        self.video_preview.set_enabled(self.config.get("preview_enabled", False))
        
    def _save_config(self):
        """Save configuration values."""
        self.config.set("last_video_folder", self.video_picker.get_path(), save=False)
        self.config.set("last_output_dir", self.output_picker.get_path(), save=False)
        self.config.set("preview_enabled", self.video_preview.is_enabled(), save=True)
        
    def reload_config(self):
        """Reload configuration from file."""
        self._load_config()
        
    def _on_video_changed(self, path: str):
        """Handle video path change."""
        if path and os.path.isdir(path):
            # Auto-set output directory if not set
            if not self.output_picker.get_path():
                output_dir = os.path.join(path, "analysis_output")
                self.output_picker.set_path(output_dir)
                
    def _on_preview_toggled(self, enabled: bool):
        """Handle preview toggle."""
        self.config.set("preview_enabled", enabled)
        
    def _open_settings(self):
        """Open the settings dialog."""
        from app.ui.settings_dialog import SettingsDialog
        dialog = SettingsDialog(self)
        if dialog.exec():
            self.reload_config()
        
    def set_video_path(self, path: str):
        """Set the video path from external source."""
        folder = os.path.dirname(path) if os.path.isfile(path) else path
        if folder:
            self.video_picker.set_path(folder)

    def set_video_folder(self, folder_path: str):
        """Set the video folder from external source."""
        self.video_picker.set_path(folder_path)
        
    def _validate_inputs(self) -> bool:
        """Validate inputs before starting analysis."""
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
        
    def _start_analysis(self):
        """Start the classroom analysis."""
        if not self._validate_inputs():
            return
            
        # Save config
        self._save_config()
        
        # Create output directory
        output_dir = self.output_picker.get_path()
        os.makedirs(output_dir, exist_ok=True)
        
        # Disable inputs
        self._set_inputs_enabled(False)
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        
        # Reset progress
        self.progress_panel.reset()
        self.progress_panel.log_info("Folder listener started. Waiting for new videos...")
        self._is_monitoring = True
        self._stop_requested = False
        self._pending_videos.clear()
        self._known_videos.clear()
        self._size_probe.clear()
        self._current_video = ""
        self._poll_video_folder()
        self._watch_timer.start()
        
    def _stop_analysis(self):
        """Stop the running analysis."""
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
            self.start_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            self.progress_panel.log_info("Folder listener stopped.")
            
    def stop_analysis(self):
        """Public method to stop analysis."""
        self._stop_analysis()
        
    def _on_progress(self, percent: int, message: str):
        """Handle progress update."""
        self.progress_panel.update_progress(percent, message)
        
    def _on_log(self, message: str, level: str):
        """Handle log message."""
        self.progress_panel.log(message, level)
        
    def _on_finished(self, report_path: str):
        """Handle analysis completion (or stop)."""
        self._worker = None
        self.video_preview.clear()
        
        if report_path:
            self.progress_panel.update_progress(100, "Complete")
            self.progress_panel.log_success(f"Analysis complete! Report saved to: {report_path}")
            self.analysis_complete.emit(report_path)
        else:
            self.progress_panel.log_warning("Analysis stopped by user.")
        self._current_video = ""
        if self._is_monitoring and not self._stop_requested:
            self._try_start_next_video()
        else:
            self._is_monitoring = False
            self._set_inputs_enabled(True)
            self.start_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
        
    def _on_error(self, error_message: str):
        """Handle analysis error."""
        self._worker = None
        self.video_preview.clear()
        
        self.progress_panel.log_error(f"Error: {error_message}")
        self._current_video = ""
        if self._is_monitoring and not self._stop_requested:
            self.progress_panel.log_warning("Listener continues. Waiting for next video...")
            self._try_start_next_video()
        else:
            QMessageBox.critical(self, "Analysis Error", f"An error occurred:\n\n{error_message}")
            self._is_monitoring = False
            self._set_inputs_enabled(True)
            self.start_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)

    def _poll_video_folder(self):
        """Scan input folder and enqueue new stable video files."""
        folder = self.video_picker.get_path().strip()
        if not folder or not os.path.isdir(folder):
            return
        try:
            entries = sorted(os.listdir(folder))
        except OSError as e:
            self.progress_panel.log_warning(f"Could not read input folder: {e}")
            return

        for name in entries:
            file_path = os.path.join(folder, name)
            if not os.path.isfile(file_path):
                continue
            if not name.lower().endswith(self.VIDEO_EXTENSIONS):
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
            self.progress_panel.log_info(f"Queued: {os.path.basename(file_path)}")
        self._try_start_next_video()

    def _try_start_next_video(self):
        """Start processing next video in queue if idle."""
        if not self._is_monitoring or self._stop_requested:
            return
        if self._worker and self._worker.isRunning():
            return
        if not self._pending_videos:
            self.progress_panel.update_progress(0, "Waiting for new videos...")
            return

        from app.workers.classroom_worker import ClassroomWorker

        video_path = self._pending_videos.pop(0)
        self._known_videos.add(video_path)
        self._current_video = video_path
        base_output_dir = self.output_picker.get_path()
        safe_name = os.path.splitext(os.path.basename(video_path))[0]
        run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_output_dir = os.path.join(base_output_dir, f"{safe_name}_{run_stamp}")
        os.makedirs(run_output_dir, exist_ok=True)

        self.progress_panel.reset()
        self.progress_panel.log_info(f"Processing: {os.path.basename(video_path)}")

        classroom_config = self.config.get_section("classroom")
        self._worker = ClassroomWorker(
            video_path=video_path,
            output_dir=run_output_dir,
            config=classroom_config,
            preview_enabled=self.video_preview.is_enabled()
        )
        self._worker.progress.connect(self._on_progress)
        self._worker.log_message.connect(self._on_log)
        self._worker.frame_ready.connect(self.video_preview.update_frame)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.start()
        
    def _set_inputs_enabled(self, enabled: bool):
        """Enable or disable input widgets."""
        self.video_picker.setEnabled(enabled)
        self.output_picker.setEnabled(enabled)
        
    def is_running(self) -> bool:
        """Check if analysis is running."""
        return self._is_monitoring or (self._worker is not None and self._worker.isRunning())

    def is_monitoring(self) -> bool:
        """Check if folder listener is active."""
        return self._is_monitoring
