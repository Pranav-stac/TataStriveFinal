"""
Center Setup Dialog.

Captures the BigQuery center_name used for student roster + S3 embeddings,
and the center_id tag applied to analytics uploaded from this device.
"""

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QFrame, QMessageBox, QComboBox, QCompleter,
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont

from app.center_catalog import TATA_STRIVE_CENTER_NAMES, resolve_center_name


class CenterDialog(QDialog):
    """Modal dialog for device center_id and BigQuery roster center_name."""

    def __init__(
        self,
        parent=None,
        existing_name: str = "",
        existing_roster_center: str = "",
    ):
        super().__init__(parent)
        self.setWindowTitle("TataStrive Analytics – Center Setup")
        self.setWindowFlags(
            Qt.WindowType.Dialog
            | Qt.WindowType.WindowTitleHint
            | Qt.WindowType.CustomizeWindowHint
        )
        self.setMinimumWidth(520)
        self.setModal(True)
        self._setup_ui(existing_name, existing_roster_center)

    def _setup_ui(self, existing_name: str, existing_roster_center: str) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(32, 28, 32, 24)
        layout.setSpacing(18)

        header_label = QLabel("Welcome to TataStrive Analytics")
        hfont = QFont()
        hfont.setPointSize(14)
        hfont.setBold(True)
        header_label.setFont(hfont)
        header_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(header_label)

        divider = QFrame()
        divider.setFrameShape(QFrame.Shape.HLine)
        divider.setFrameShadow(QFrame.Shadow.Sunken)
        layout.addWidget(divider)

        info_label = QLabel(
            "Choose the Tata STRIVE center for the student roster in BigQuery.\n\n"
            "That center_name drives engagement_id lookup, S3 enrollment photos,\n"
            "and face matching during attendance.\n\n"
            "Reports uploaded from this device are tagged with the device center ID."
        )
        info_label.setWordWrap(True)
        info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(info_label)

        roster_layout = QHBoxLayout()
        roster_label = QLabel("BigQuery center:")
        roster_label.setMinimumWidth(130)
        roster_layout.addWidget(roster_label)

        self._roster_combo = QComboBox()
        self._roster_combo.addItems(TATA_STRIVE_CENTER_NAMES)
        self._roster_combo.setEditable(True)
        self._roster_combo.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        self._roster_combo.setMinimumHeight(34)
        completer = self._roster_combo.completer()
        if completer is not None:
            completer.setCompletionMode(QCompleter.CompletionMode.PopupCompletion)
            completer.setFilterMode(Qt.MatchFlag.MatchContains)
            completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)

        roster_seed = resolve_center_name(existing_roster_center) or resolve_center_name(existing_name)
        if roster_seed:
            idx = self._roster_combo.findText(roster_seed, Qt.MatchFlag.MatchExactly)
            if idx >= 0:
                self._roster_combo.setCurrentIndex(idx)
            else:
                self._roster_combo.setEditText(roster_seed)
        roster_layout.addWidget(self._roster_combo)
        layout.addLayout(roster_layout)

        device_layout = QHBoxLayout()
        device_label = QLabel("Device center ID:")
        device_label.setMinimumWidth(130)
        device_layout.addWidget(device_label)

        self._device_edit = QLineEdit()
        self._device_edit.setPlaceholderText("Defaults to the selected BigQuery center")
        self._device_edit.setText(existing_name.strip())
        self._device_edit.setMinimumHeight(34)
        self._device_edit.returnPressed.connect(self._accept)
        device_layout.addWidget(self._device_edit)
        layout.addLayout(device_layout)

        hint = QLabel("Type to search the center list. Pick a catalog center before continuing.")
        hint.setObjectName("hintLabel")
        hint_font = QFont()
        hint_font.setPointSize(9)
        hint_font.setItalic(True)
        hint.setFont(hint_font)
        hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(hint)

        btn_layout = QHBoxLayout()
        btn_layout.addStretch()

        self._ok_btn = QPushButton("Save & Continue")
        self._ok_btn.setObjectName("primaryButton")
        self._ok_btn.setMinimumHeight(36)
        self._ok_btn.setMinimumWidth(150)
        self._ok_btn.clicked.connect(self._accept)
        btn_layout.addWidget(self._ok_btn)

        layout.addLayout(btn_layout)
        self._roster_combo.setFocus()

    def _accept(self) -> None:
        roster = resolve_center_name(self._roster_combo.currentText())
        if not roster:
            QMessageBox.warning(
                self,
                "Center Required",
                "Choose one of the Tata STRIVE centers from the list.",
            )
            return
        device = self._device_edit.text().strip() or roster
        self._roster_combo.setCurrentText(roster)
        self._device_edit.setText(device)
        self.accept()

    def center_name(self) -> str:
        """Device center_id used on uploaded analytics rows."""
        return self._device_edit.text().strip() or self.roster_center_name()

    def roster_center_name(self) -> str:
        """BigQuery center_name for intraining_students roster scope."""
        return resolve_center_name(self._roster_combo.currentText()) or ""
