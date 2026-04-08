"""
Settings Dialog.
Application settings configuration dialog.
"""

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QSpinBox, QDoubleSpinBox, QPushButton, QTabWidget,
    QWidget, QCheckBox, QDialogButtonBox, QMessageBox, QComboBox
)
from PyQt6.QtCore import Qt

from app.config import get_config


class SettingsDialog(QDialog):
    """Dialog for configuring application settings."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.config = get_config()
        self.setWindowTitle("Settings")
        self.setMinimumWidth(500)
        self.setModal(True)
        self._setup_ui()
        self._load_settings()
        
    def _setup_ui(self):
        """Setup the dialog UI."""
        layout = QVBoxLayout(self)
        layout.setSpacing(16)
        
        # Tab widget for categories
        tabs = QTabWidget()
        
        # Classroom settings tab
        classroom_tab = QWidget()
        classroom_layout = QVBoxLayout(classroom_tab)
        
        classroom_group = QGroupBox("Classroom Analysis Defaults")
        classroom_form = QVBoxLayout(classroom_group)
        
        # Probe Duration
        probe_dur_layout = QHBoxLayout()
        probe_dur_layout.addWidget(QLabel("Default Probe Duration:"))
        self.probe_duration_spin = QSpinBox()
        self.probe_duration_spin.setRange(60, 1800)
        self.probe_duration_spin.setSuffix(" sec")
        probe_dur_layout.addWidget(self.probe_duration_spin)
        probe_dur_layout.addStretch()
        classroom_form.addLayout(probe_dur_layout)
        
        # Probe Interval
        probe_int_layout = QHBoxLayout()
        probe_int_layout.addWidget(QLabel("Default Probe Interval:"))
        self.probe_interval_spin = QSpinBox()
        self.probe_interval_spin.setRange(300, 7200)
        self.probe_interval_spin.setSuffix(" sec")
        probe_int_layout.addWidget(self.probe_interval_spin)
        probe_int_layout.addStretch()
        classroom_form.addLayout(probe_int_layout)
        
        # Frame Skip
        frame_skip_layout = QHBoxLayout()
        frame_skip_layout.addWidget(QLabel("Default Frame Skip:"))
        self.frame_skip_spin = QSpinBox()
        self.frame_skip_spin.setRange(1, 10)
        frame_skip_layout.addWidget(self.frame_skip_spin)
        frame_skip_layout.addStretch()
        classroom_form.addLayout(frame_skip_layout)
        
        # Similarity Threshold
        sim_layout = QHBoxLayout()
        sim_layout.addWidget(QLabel("Default Similarity Threshold:"))
        self.similarity_spin = QDoubleSpinBox()
        self.similarity_spin.setRange(0.5, 1.0)
        self.similarity_spin.setSingleStep(0.05)
        self.similarity_spin.setDecimals(2)
        sim_layout.addWidget(self.similarity_spin)
        sim_layout.addStretch()
        classroom_form.addLayout(sim_layout)
        
        # Max Time Gap
        time_gap_layout = QHBoxLayout()
        time_gap_layout.addWidget(QLabel("Max Time Gap (stitching):"))
        self.max_time_gap_spin = QSpinBox()
        self.max_time_gap_spin.setRange(60, 1800)
        self.max_time_gap_spin.setSuffix(" sec")
        time_gap_layout.addWidget(self.max_time_gap_spin)
        time_gap_layout.addStretch()
        classroom_form.addLayout(time_gap_layout)
        
        # Max Pixel Distance
        pixel_dist_layout = QHBoxLayout()
        pixel_dist_layout.addWidget(QLabel("Max Pixel Distance (stitching):"))
        self.max_pixel_dist_spin = QSpinBox()
        self.max_pixel_dist_spin.setRange(50, 500)
        self.max_pixel_dist_spin.setSuffix(" px")
        pixel_dist_layout.addWidget(self.max_pixel_dist_spin)
        pixel_dist_layout.addStretch()
        classroom_form.addLayout(pixel_dist_layout)
        
        # Delete video after processing complete
        self.delete_video_classroom_checkbox = QCheckBox("Delete source video after processing completes")
        self.delete_video_classroom_checkbox.setToolTip(
            "When enabled, the source video file will be deleted automatically after classroom analysis "
            "(attendance + engagement) completes successfully. Use to save disk space."
        )
        classroom_form.addWidget(self.delete_video_classroom_checkbox)
        
        classroom_layout.addWidget(classroom_group)
        classroom_layout.addStretch()
        
        tabs.addTab(classroom_tab, "Classroom")
        
        # Cross-day settings tab
        crossday_tab = QWidget()
        crossday_layout = QVBoxLayout(crossday_tab)
        
        crossday_group = QGroupBox("Attendance Defaults")
        crossday_form = QVBoxLayout(crossday_group)
        
        # T_STRICT_MERGE (lower = more lenient / easier to match same person)
        strict_layout = QHBoxLayout()
        strict_layout.addWidget(QLabel("Match similarity (lower = lenient):"))
        self.t_strict_merge_spin = QDoubleSpinBox()
        self.t_strict_merge_spin.setRange(0.15, 0.9)
        self.t_strict_merge_spin.setSingleStep(0.05)
        self.t_strict_merge_spin.setDecimals(2)
        self.t_strict_merge_spin.setToolTip(
            "Minimum similarity to reuse an existing face id (G_*). Lower = easier matches; higher = stricter."
        )
        strict_layout.addWidget(self.t_strict_merge_spin)
        strict_layout.addStretch()
        crossday_form.addLayout(strict_layout)
        
        # T_NEW_ID
        new_id_layout = QHBoxLayout()
        new_id_layout.addWidget(QLabel("New ID Threshold:"))
        self.t_new_id_spin = QDoubleSpinBox()
        self.t_new_id_spin.setRange(0.1, 0.6)
        self.t_new_id_spin.setSingleStep(0.05)
        self.t_new_id_spin.setDecimals(2)
        self.t_new_id_spin.setToolTip(
            "When best match is weaker than this, a new id may be created (depends on mode)."
        )
        new_id_layout.addWidget(self.t_new_id_spin)
        new_id_layout.addStretch()
        crossday_form.addLayout(new_id_layout)
        
        # T_RATIO_MARGIN
        margin_layout = QHBoxLayout()
        margin_layout.addWidget(QLabel("Ratio Margin:"))
        self.t_ratio_margin_spin = QDoubleSpinBox()
        self.t_ratio_margin_spin.setRange(0.02, 0.3)
        self.t_ratio_margin_spin.setSingleStep(0.02)
        self.t_ratio_margin_spin.setDecimals(2)
        self.t_ratio_margin_spin.setToolTip(
            "Min. lead of best vs 2nd gallery match. Smaller = more lenient (accept closer races)."
        )
        margin_layout.addWidget(self.t_ratio_margin_spin)
        margin_layout.addStretch()
        crossday_form.addLayout(margin_layout)
        
        # MIN_SAMPLES
        samples_layout = QHBoxLayout()
        samples_layout.addWidget(QLabel("Min Samples:"))
        self.min_samples_spin = QSpinBox()
        self.min_samples_spin.setRange(1, 20)
        self.min_samples_spin.setToolTip(
            "Max face embeddings to collect per track; matching can start earlier (min_embeds_for_match in config)"
        )
        samples_layout.addWidget(self.min_samples_spin)
        samples_layout.addStretch()
        crossday_form.addLayout(samples_layout)
        
        # VISITOR_UPGRADE_DAYS
        upgrade_layout = QHBoxLayout()
        upgrade_layout.addWidget(QLabel("Visitor Upgrade Days:"))
        self.visitor_upgrade_spin = QSpinBox()
        self.visitor_upgrade_spin.setRange(1, 10)
        upgrade_layout.addWidget(self.visitor_upgrade_spin)
        upgrade_layout.addStretch()
        crossday_form.addLayout(upgrade_layout)

        # Save output video
        self.save_video_checkbox = QCheckBox("Save annotated output video")
        self.save_video_checkbox.setToolTip(
            "Off by default — skipping video saves disk space and speeds up processing.\n"
            "Enable only when you need a labelled video file as output."
        )
        crossday_form.addWidget(self.save_video_checkbox)

        # Delete video after processing complete
        self.delete_video_crossday_checkbox = QCheckBox("Delete source video after processing completes")
        self.delete_video_crossday_checkbox.setToolTip(
            "When enabled, the source video file will be deleted automatically after attendance analysis "
            "completes successfully. Use to save disk space."
        )
        crossday_form.addWidget(self.delete_video_crossday_checkbox)

        # Motion detection (trim video to motion segments before attendance)
        self.enable_motion_checkbox = QCheckBox("Enable motion detection (trim to motion segments)")
        self.enable_motion_checkbox.setToolTip(
            "When enabled: first scan the video for segments with motion, then run attendance only on those parts.\n"
            "Skips static/empty scenes to speed up processing and focus on relevant footage.\n\n"
            "IMPORTANT: Disable this if your video is already motion-filtered (e.g. output_motion_only.mp4).\n"
            "Running motion detection on pre-filtered video can skip many frames and undercount people."
        )
        crossday_form.addWidget(self.enable_motion_checkbox)

        crossday_layout.addWidget(crossday_group)
        crossday_layout.addStretch()
        
        tabs.addTab(crossday_tab, "Attendance")
        
        # Inference / Performance tab
        inference_tab = QWidget()
        inference_layout = QVBoxLayout(inference_tab)
        
        inference_group = QGroupBox("CPU Acceleration (Intel OpenVINO)")
        inference_form = QVBoxLayout(inference_group)
        
        self.use_openvino_checkbox = QCheckBox("Use OpenVINO for faster CPU inference (2-3x speedup on Intel)")
        self.use_openvino_checkbox.setToolTip("Requires: pip install openvino. Falls back to PyTorch/ONNX if unavailable.")
        inference_form.addWidget(self.use_openvino_checkbox)
        
        self.force_cpu_checkbox = QCheckBox("Force CPU mode (disable GPU/CUDA even if available)")
        self.force_cpu_checkbox.setToolTip("Useful for compatibility or debugging. When enabled, all analysis runs on CPU.")
        inference_form.addWidget(self.force_cpu_checkbox)
        
        yolo_imgsz_layout = QHBoxLayout()
        yolo_imgsz_layout.addWidget(QLabel("YOLO inference size:"))
        self.yolo_imgsz_spin = QSpinBox()
        self.yolo_imgsz_spin.setRange(320, 640)
        self.yolo_imgsz_spin.setSingleStep(32)
        self.yolo_imgsz_spin.setSuffix(" px")
        self.yolo_imgsz_spin.setToolTip("320=fastest, 640=best quality. 416 is a good balance.")
        yolo_imgsz_layout.addWidget(self.yolo_imgsz_spin)
        yolo_imgsz_layout.addStretch()
        inference_form.addLayout(yolo_imgsz_layout)
        
        face_det_layout = QHBoxLayout()
        face_det_layout.addWidget(QLabel("Face detection size:"))
        self.face_det_size_spin = QSpinBox()
        self.face_det_size_spin.setRange(320, 640)
        self.face_det_size_spin.setSingleStep(32)
        self.face_det_size_spin.setSuffix(" px")
        self.face_det_size_spin.setToolTip("320=fastest, 640=best quality. 416 is a good balance.")
        face_det_layout.addWidget(self.face_det_size_spin)
        face_det_layout.addStretch()
        inference_form.addLayout(face_det_layout)
        
        frame_skip_layout = QHBoxLayout()
        frame_skip_layout.addWidget(QLabel("Face frame skip (attendance):"))
        self.frame_skip_attendance_spin = QSpinBox()
        self.frame_skip_attendance_spin.setRange(1, 5)
        self.frame_skip_attendance_spin.setToolTip("Run face detection every Nth frame. 1=every frame (best accuracy), 2-5=faster, slight accuracy trade-off.")
        frame_skip_layout.addWidget(self.frame_skip_attendance_spin)
        frame_skip_layout.addStretch()
        inference_form.addLayout(frame_skip_layout)
        
        preview_mode_layout = QHBoxLayout()
        preview_mode_layout.addWidget(QLabel("Preview mode:"))
        self.preview_mode_combo = QComboBox()
        self.preview_mode_combo.addItem("cv2 (fast, separate window)", "cv2")
        self.preview_mode_combo.addItem("PyQt (integrated in app)", "pyqt")
        self.preview_mode_combo.setToolTip("cv2.imshow is faster with no cross-thread overhead. PyQt shows in app but can slow processing.")
        preview_mode_layout.addWidget(self.preview_mode_combo)
        preview_mode_layout.addStretch()
        inference_form.addLayout(preview_mode_layout)
        
        inference_layout.addWidget(inference_group)
        inference_layout.addStretch()
        
        tabs.addTab(inference_tab, "Performance")
        
        # General settings tab
        general_tab = QWidget()
        general_layout = QVBoxLayout(general_tab)
        
        general_group = QGroupBox("General Settings")
        general_form = QVBoxLayout(general_group)
        
        self.preview_checkbox = QCheckBox("Enable video preview by default")
        general_form.addWidget(self.preview_checkbox)
        
        general_layout.addWidget(general_group)
        general_layout.addStretch()
        
        tabs.addTab(general_tab, "General")
        
        layout.addWidget(tabs)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        reset_btn = QPushButton("Reset to Defaults")
        reset_btn.clicked.connect(self._reset_defaults)
        button_layout.addWidget(reset_btn)
        
        button_layout.addStretch()
        
        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(self._save_and_accept)
        button_box.rejected.connect(self.reject)
        button_layout.addWidget(button_box)
        
        layout.addLayout(button_layout)
        
    def _load_settings(self):
        """Load current settings."""
        # Classroom
        classroom = self.config.get_section("classroom")
        self.probe_duration_spin.setValue(classroom.get("probe_duration", 300))
        self.probe_interval_spin.setValue(classroom.get("probe_interval", 3600))
        self.frame_skip_spin.setValue(classroom.get("frame_skip", 3))
        self.similarity_spin.setValue(classroom.get("similarity_threshold", 0.75))
        self.max_time_gap_spin.setValue(classroom.get("max_time_gap", 600))
        self.max_pixel_dist_spin.setValue(classroom.get("max_pixel_dist", 200))
        self.delete_video_classroom_checkbox.setChecked(classroom.get("delete_video_after_processing", False))
        
        # Cross-day
        crossday = self.config.get_section("crossday")
        self.t_strict_merge_spin.setValue(crossday.get("t_strict_merge", 0.36))
        self.t_new_id_spin.setValue(crossday.get("t_new_id", 0.22))
        self.t_ratio_margin_spin.setValue(crossday.get("t_ratio_margin", 0.05))
        self.min_samples_spin.setValue(crossday.get("min_samples", 2))
        self.visitor_upgrade_spin.setValue(crossday.get("visitor_upgrade_days", 3))
        self.save_video_checkbox.setChecked(crossday.get("save_output_video", False))
        self.delete_video_crossday_checkbox.setChecked(crossday.get("delete_video_after_processing", False))
        self.enable_motion_checkbox.setChecked(crossday.get("enable_motion_detection", False))
        
        # Inference
        inference = self.config.get_section("inference") or {}
        self.use_openvino_checkbox.setChecked(inference.get("use_openvino", False))
        self.force_cpu_checkbox.setChecked(inference.get("force_cpu", False))
        self.yolo_imgsz_spin.setValue(inference.get("yolo_imgsz", 640))
        self.face_det_size_spin.setValue(inference.get("face_det_size", 640))
        self.frame_skip_attendance_spin.setValue(inference.get("frame_skip", 1))
        idx = self.preview_mode_combo.findData(inference.get("preview_mode", "cv2"))
        self.preview_mode_combo.setCurrentIndex(idx if idx >= 0 else 0)
        
        # General
        self.preview_checkbox.setChecked(self.config.get("preview_enabled", False))
        
    def _save_and_accept(self):
        """Save settings and close dialog."""
        # Classroom
        self.config.set("classroom.probe_duration", self.probe_duration_spin.value(), save=False)
        self.config.set("classroom.probe_interval", self.probe_interval_spin.value(), save=False)
        self.config.set("classroom.frame_skip", self.frame_skip_spin.value(), save=False)
        self.config.set("classroom.similarity_threshold", self.similarity_spin.value(), save=False)
        self.config.set("classroom.max_time_gap", self.max_time_gap_spin.value(), save=False)
        self.config.set("classroom.max_pixel_dist", self.max_pixel_dist_spin.value(), save=False)
        self.config.set("classroom.delete_video_after_processing", self.delete_video_classroom_checkbox.isChecked(), save=False)
        
        # Cross-day
        self.config.set("crossday.t_strict_merge", self.t_strict_merge_spin.value(), save=False)
        self.config.set("crossday.t_new_id", self.t_new_id_spin.value(), save=False)
        self.config.set("crossday.t_ratio_margin", self.t_ratio_margin_spin.value(), save=False)
        self.config.set("crossday.min_samples", self.min_samples_spin.value(), save=False)
        self.config.set("crossday.visitor_upgrade_days", self.visitor_upgrade_spin.value(), save=False)
        self.config.set("crossday.save_output_video", self.save_video_checkbox.isChecked(), save=False)
        self.config.set("crossday.delete_video_after_processing", self.delete_video_crossday_checkbox.isChecked(), save=False)
        self.config.set("crossday.enable_motion_detection", self.enable_motion_checkbox.isChecked(), save=False)
        
        # Inference
        self.config.set("inference.use_openvino", self.use_openvino_checkbox.isChecked(), save=False)
        self.config.set("inference.force_cpu", self.force_cpu_checkbox.isChecked(), save=False)
        self.config.set("inference.yolo_imgsz", self.yolo_imgsz_spin.value(), save=False)
        self.config.set("inference.face_det_size", self.face_det_size_spin.value(), save=False)
        self.config.set("inference.frame_skip", self.frame_skip_attendance_spin.value(), save=False)
        self.config.set("inference.preview_mode", self.preview_mode_combo.currentData(), save=False)
        
        # General
        self.config.set("preview_enabled", self.preview_checkbox.isChecked(), save=True)
        
        self.accept()
        
    def _reset_defaults(self):
        """Reset all settings to defaults."""
        reply = QMessageBox.question(
            self,
            "Reset Settings",
            "Are you sure you want to reset all settings to their default values?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            self.config.reset()
            self._load_settings()
