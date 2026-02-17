"""
Attendance Tab.
UI for configuring and running attendance analysis.
"""

import os
from datetime import datetime
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QSpinBox, QDoubleSpinBox, QPushButton, QFrame, QSplitter,
    QRadioButton, QButtonGroup, QDateEdit, QLineEdit,
    QSizePolicy, QMessageBox, QScrollArea
)
from PyQt6.QtCore import Qt, pyqtSignal, QDate

from app.config import get_config
from app.ui.widgets.file_picker import FilePicker, FolderPicker
from app.ui.widgets.progress_panel import ProgressPanel
from app.ui.widgets.video_preview import VideoPreview


class CrossDayTab(QWidget):
    """Tab for attendance analysis configuration and execution."""
    
    analysis_complete = pyqtSignal(str)  # Emits report path
    
    def __init__(self, parent: QWidget = None):
        super().__init__(parent)
        self.config = get_config()
        self._worker = None
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
        
        # Mode selection
        mode_group = QGroupBox("Run Mode")
        mode_layout = QVBoxLayout(mode_group)
        
        self.mode_group = QButtonGroup(self)
        
        self.build_db_radio = QRadioButton("BUILD_DB - Create baseline database (Day 1)")
        self.build_db_radio.setToolTip("Use this mode on the first day to build the initial face database")
        self.mode_group.addButton(self.build_db_radio, 0)
        mode_layout.addWidget(self.build_db_radio)
        
        self.eval_day_radio = QRadioButton("EVAL_DAY - Evaluate against existing database (Day 2+)")
        self.eval_day_radio.setToolTip("Use this mode on subsequent days to match against the existing database")
        self.mode_group.addButton(self.eval_day_radio, 1)
        mode_layout.addWidget(self.eval_day_radio)
        
        self.build_db_radio.setChecked(True)
        self.mode_group.buttonClicked.connect(self._on_mode_changed)
        
        left_layout.addWidget(mode_group)
        
        # Date and Label
        date_group = QGroupBox("Session Info")
        date_layout = QVBoxLayout(date_group)
        
        date_row = QHBoxLayout()
        date_row.addWidget(QLabel("Current Date:"))
        self.date_edit = QDateEdit()
        self.date_edit.setCalendarPopup(True)
        self.date_edit.setDate(QDate.currentDate())
        self.date_edit.dateChanged.connect(self._on_date_changed)
        date_row.addWidget(self.date_edit)
        date_row.addStretch()
        date_layout.addLayout(date_row)
        
        label_row = QHBoxLayout()
        label_row.addWidget(QLabel("Day Label:"))
        self.day_label_edit = QLineEdit()
        self.day_label_edit.setPlaceholderText("e.g., Day2")
        self.day_label_edit.setToolTip("Label for visitors created on this day (e.g., Day2_V_001)")
        label_row.addWidget(self.day_label_edit)
        label_row.addStretch()
        date_layout.addLayout(label_row)
        
        left_layout.addWidget(date_group)
        
        # File inputs
        self.video_picker = FilePicker(
            label="Video Input",
            placeholder="Select a video file...",
            file_filter="Video Files (*.mp4 *.avi *.mkv *.mov);;All Files (*.*)"
        )
        self.video_picker.path_changed.connect(self._on_video_changed)
        left_layout.addWidget(self.video_picker)
        
        self.db_picker = FilePicker(
            label="Database File",
            placeholder="Select master_database.pkl...",
            file_filter="Pickle Files (*.pkl);;All Files (*.*)"
        )
        self.db_picker.setToolTip("Path to the master database file (required for EVAL_DAY)")
        left_layout.addWidget(self.db_picker)
        
        self.output_picker = FolderPicker(
            label="Output Directory",
            placeholder="Select output folder..."
        )
        left_layout.addWidget(self.output_picker)
        
        # Settings button
        from app.ui.settings_dialog import SettingsDialog
        settings_btn = QPushButton("Settings...")
        settings_btn.setToolTip("Configure default values (Ctrl+,)")
        settings_btn.clicked.connect(lambda: self._open_settings())
        left_layout.addWidget(settings_btn)
        
        # Configuration group
        config_group = QGroupBox("Threshold Configuration")
        config_layout = QVBoxLayout(config_group)
        config_layout.setSpacing(10)
        
        # T_STRICT_MERGE
        strict_layout = QHBoxLayout()
        strict_layout.addWidget(QLabel("Strict Merge Threshold:"))
        self.t_strict_merge_spin = QDoubleSpinBox()
        self.t_strict_merge_spin.setRange(0.3, 0.9)
        self.t_strict_merge_spin.setSingleStep(0.05)
        self.t_strict_merge_spin.setDecimals(2)
        self.t_strict_merge_spin.setToolTip("Minimum similarity to match to existing ID (default: 0.55)")
        strict_layout.addWidget(self.t_strict_merge_spin)
        strict_layout.addStretch()
        config_layout.addLayout(strict_layout)
        
        # T_NEW_ID
        new_id_layout = QHBoxLayout()
        new_id_layout.addWidget(QLabel("New ID Threshold:"))
        self.t_new_id_spin = QDoubleSpinBox()
        self.t_new_id_spin.setRange(0.1, 0.6)
        self.t_new_id_spin.setSingleStep(0.05)
        self.t_new_id_spin.setDecimals(2)
        self.t_new_id_spin.setToolTip("Maximum similarity to create new ID (default: 0.35)")
        new_id_layout.addWidget(self.t_new_id_spin)
        new_id_layout.addStretch()
        config_layout.addLayout(new_id_layout)
        
        # T_RATIO_MARGIN
        margin_layout = QHBoxLayout()
        margin_layout.addWidget(QLabel("Ratio Margin:"))
        self.t_ratio_margin_spin = QDoubleSpinBox()
        self.t_ratio_margin_spin.setRange(0.05, 0.3)
        self.t_ratio_margin_spin.setSingleStep(0.05)
        self.t_ratio_margin_spin.setDecimals(2)
        self.t_ratio_margin_spin.setToolTip("Minimum gap between best and second-best match (default: 0.10)")
        margin_layout.addWidget(self.t_ratio_margin_spin)
        margin_layout.addStretch()
        config_layout.addLayout(margin_layout)
        
        # MIN_SAMPLES
        samples_layout = QHBoxLayout()
        samples_layout.addWidget(QLabel("Min Samples:"))
        self.min_samples_spin = QSpinBox()
        self.min_samples_spin.setRange(3, 20)
        self.min_samples_spin.setToolTip("Minimum face samples before identity assignment (default: 8)")
        samples_layout.addWidget(self.min_samples_spin)
        samples_layout.addStretch()
        config_layout.addLayout(samples_layout)
        
        # VISITOR_UPGRADE_DAYS
        upgrade_layout = QHBoxLayout()
        upgrade_layout.addWidget(QLabel("Visitor Upgrade Days:"))
        self.visitor_upgrade_spin = QSpinBox()
        self.visitor_upgrade_spin.setRange(1, 10)
        self.visitor_upgrade_spin.setToolTip("Days present to upgrade visitor to permanent ID (default: 3)")
        upgrade_layout.addWidget(self.visitor_upgrade_spin)
        upgrade_layout.addStretch()
        config_layout.addLayout(upgrade_layout)
        
        left_layout.addWidget(config_group)
        
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
        
        # Initial mode update
        self._on_mode_changed()
        
    def _load_config(self):
        """Load configuration values."""
        crossday_config = self.config.get_section("crossday")
        
        self.t_strict_merge_spin.setValue(crossday_config.get("t_strict_merge", 0.55))
        self.t_new_id_spin.setValue(crossday_config.get("t_new_id", 0.35))
        self.t_ratio_margin_spin.setValue(crossday_config.get("t_ratio_margin", 0.10))
        self.min_samples_spin.setValue(crossday_config.get("min_samples", 8))
        self.visitor_upgrade_spin.setValue(crossday_config.get("visitor_upgrade_days", 3))
        
        # Load last paths
        last_video = self.config.get("last_video_path", "")
        if last_video:
            self.video_picker.set_path(last_video)
            
        last_db = self.config.get("last_db_path", "")
        if last_db:
            self.db_picker.set_path(last_db)
            
        last_output = self.config.get("last_output_dir", "")
        if last_output:
            self.output_picker.set_path(last_output)
            
        # Preview state
        self.video_preview.set_enabled(self.config.get("preview_enabled", False))
        
        # Auto-generate day label
        self._update_day_label()
        
    def _save_config(self):
        """Save configuration values."""
        self.config.set("crossday.t_strict_merge", self.t_strict_merge_spin.value(), save=False)
        self.config.set("crossday.t_new_id", self.t_new_id_spin.value(), save=False)
        self.config.set("crossday.t_ratio_margin", self.t_ratio_margin_spin.value(), save=False)
        self.config.set("crossday.min_samples", self.min_samples_spin.value(), save=False)
        self.config.set("crossday.visitor_upgrade_days", self.visitor_upgrade_spin.value(), save=False)
        self.config.set("last_video_path", self.video_picker.get_path(), save=False)
        self.config.set("last_db_path", self.db_picker.get_path(), save=False)
        self.config.set("last_output_dir", self.output_picker.get_path(), save=False)
        self.config.set("preview_enabled", self.video_preview.is_enabled(), save=True)
        
    def reload_config(self):
        """Reload configuration from file."""
        self._load_config()
        
    def _on_mode_changed(self):
        """Handle mode selection change."""
        is_eval_day = self.eval_day_radio.isChecked()
        self.db_picker.setEnabled(True)  # Always allow DB selection
        self.day_label_edit.setEnabled(is_eval_day)
        
        if is_eval_day:
            self.db_picker.label.setText("Database File (Required)")
        else:
            self.db_picker.label.setText("Database File (Output)")
            
    def _on_date_changed(self):
        """Handle date change."""
        self._update_day_label()
        
    def _update_day_label(self):
        """Auto-generate day label from date."""
        date = self.date_edit.date()
        day_label = f"Day{date.toString('MMdd')}"
        self.day_label_edit.setText(day_label)
        
    def _on_video_changed(self, path: str):
        """Handle video path change."""
        if path and os.path.isfile(path):
            # Auto-set output directory if not set
            if not self.output_picker.get_path():
                video_dir = os.path.dirname(path)
                output_dir = os.path.join(video_dir, "Outputs")
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
        self.video_picker.set_path(path)
        
    def _validate_inputs(self) -> bool:
        """Validate inputs before starting analysis."""
        video_path = self.video_picker.get_path()
        output_dir = self.output_picker.get_path()
        db_path = self.db_picker.get_path()
        is_eval_day = self.eval_day_radio.isChecked()
        
        if not video_path:
            QMessageBox.warning(self, "Validation Error", "Please select a video file.")
            return False
            
        if not os.path.isfile(video_path):
            QMessageBox.warning(self, "Validation Error", "The selected video file does not exist.")
            return False
            
        if not output_dir:
            QMessageBox.warning(self, "Validation Error", "Please select an output directory.")
            return False
            
        if is_eval_day:
            if not db_path:
                QMessageBox.warning(self, "Validation Error", "Please select a database file for EVAL_DAY mode.")
                return False
            if not os.path.isfile(db_path):
                QMessageBox.warning(self, "Validation Error", "The selected database file does not exist.")
                return False
            if not self.day_label_edit.text().strip():
                QMessageBox.warning(self, "Validation Error", "Please enter a day label.")
                return False
                
        return True
        
    def _start_analysis(self):
        """Start the attendance analysis."""
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
        
        mode = "BUILD_DB" if self.build_db_radio.isChecked() else "EVAL_DAY"
        self.progress_panel.log_info(f"Starting attendance analysis in {mode} mode...")
        
        # Create and start worker
        from app.workers.crossday_worker import CrossDayWorker
        
        self._worker = CrossDayWorker(
            video_path=self.video_picker.get_path(),
            output_dir=output_dir,
            db_path=self.db_picker.get_path(),
            config={
                "run_mode": mode,
                "current_date": self.date_edit.date().toString("yyyy-MM-dd"),
                "day_label": self.day_label_edit.text().strip(),
                "t_strict_merge": self.t_strict_merge_spin.value(),
                "t_new_id": self.t_new_id_spin.value(),
                "t_ratio_margin": self.t_ratio_margin_spin.value(),
                "min_samples": self.min_samples_spin.value(),
                "visitor_upgrade_days": self.visitor_upgrade_spin.value()
            },
            preview_enabled=self.video_preview.is_enabled()
        )
        
        # Connect signals
        self._worker.progress.connect(self._on_progress)
        self._worker.log_message.connect(self._on_log)
        self._worker.frame_ready.connect(self.video_preview.update_frame)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        
        self._worker.start()
        
    def _stop_analysis(self):
        """Stop the running analysis."""
        if self._worker and self._worker.isRunning():
            self.progress_panel.log_warning("Stopping analysis...")
            self._worker.stop()
            
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
        self._set_inputs_enabled(True)
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.video_preview.clear()
        
        if report_path:
            self.progress_panel.update_progress(100, "Complete")
            self.progress_panel.log_success(f"Analysis complete! Report saved to: {report_path}")
            self.analysis_complete.emit(report_path)
        else:
            self.progress_panel.log_warning("Analysis stopped by user.")
        
    def _on_error(self, error_message: str):
        """Handle analysis error."""
        self._set_inputs_enabled(True)
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.video_preview.clear()
        
        self.progress_panel.log_error(f"Error: {error_message}")
        QMessageBox.critical(self, "Analysis Error", f"An error occurred:\n\n{error_message}")
        
    def _set_inputs_enabled(self, enabled: bool):
        """Enable or disable input widgets."""
        self.build_db_radio.setEnabled(enabled)
        self.eval_day_radio.setEnabled(enabled)
        self.date_edit.setEnabled(enabled)
        self.day_label_edit.setEnabled(enabled and self.eval_day_radio.isChecked())
        self.video_picker.setEnabled(enabled)
        self.db_picker.setEnabled(enabled)
        self.output_picker.setEnabled(enabled)
        self.t_strict_merge_spin.setEnabled(enabled)
        self.t_new_id_spin.setEnabled(enabled)
        self.t_ratio_margin_spin.setEnabled(enabled)
        self.min_samples_spin.setEnabled(enabled)
        self.visitor_upgrade_spin.setEnabled(enabled)
        
    def is_running(self) -> bool:
        """Check if analysis is running."""
        return self._worker is not None and self._worker.isRunning()
