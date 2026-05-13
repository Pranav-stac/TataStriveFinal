"""
Settings Dialog.
User-facing preferences only; pipeline defaults live in app/config.py.
"""

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGroupBox,
    QPushButton, QCheckBox, QDialogButtonBox, QMessageBox,
)

from app.config import get_config


class SettingsDialog(QDialog):
    """Dialog for application preferences."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.config = get_config()
        self.setWindowTitle("Settings")
        self.setMinimumWidth(420)
        self.setModal(True)
        self._setup_ui()
        self._load_settings()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(16)

        general_group = QGroupBox("General")
        general_form = QVBoxLayout(general_group)
        self.preview_checkbox = QCheckBox("Enable video preview by default")
        general_form.addWidget(self.preview_checkbox)
        layout.addWidget(general_group)

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
        self.preview_checkbox.setChecked(self.config.get("preview_enabled", False))

    def _save_and_accept(self):
        self.config.set("preview_enabled", self.preview_checkbox.isChecked(), save=True)
        self.accept()

    def _reset_defaults(self):
        reply = QMessageBox.question(
            self,
            "Reset Settings",
            "Reset all stored settings to their default values?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self.config.reset()
            self._load_settings()
