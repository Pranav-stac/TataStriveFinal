"""
Center Setup Dialog.

Shown on first launch (or when center_id is missing from config).
Asks the operator to enter a Center Name that will tag every BigQuery row
from this device.
"""

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QFrame, QMessageBox
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont


class CenterDialog(QDialog):
    """
    Modal dialog that captures the center name on first launch.

    Usage::

        dlg = CenterDialog(parent=None, existing_name="")
        if dlg.exec() == QDialog.DialogCode.Accepted:
            center_id = dlg.center_name()
    """

    def __init__(self, parent=None, existing_name: str = ""):
        super().__init__(parent)
        self.setWindowTitle("TataStrive Analytics – Center Setup")
        self.setWindowFlags(
            Qt.WindowType.Dialog
            | Qt.WindowType.WindowTitleHint
            | Qt.WindowType.CustomizeWindowHint
        )
        self.setMinimumWidth(480)
        self.setModal(True)
        self._setup_ui(existing_name)

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _setup_ui(self, existing_name: str) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(32, 28, 32, 24)
        layout.setSpacing(18)

        # ── Header ────────────────────────────────────────────────────
        header_label = QLabel("Welcome to TataStrive Analytics")
        hfont = QFont()
        hfont.setPointSize(14)
        hfont.setBold(True)
        header_label.setFont(hfont)
        header_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(header_label)

        # Divider
        divider = QFrame()
        divider.setFrameShape(QFrame.Shape.HLine)
        divider.setFrameShadow(QFrame.Shadow.Sunken)
        layout.addWidget(divider)

        # ── Info text ─────────────────────────────────────────────────
        info_label = QLabel(
            "Please enter the name of this center / location.\n\n"
            "This value is used to identify all data uploaded from this\n"
            "device to the central BigQuery database, so every report\n"
            "will be tagged with this Center ID.\n\n"
            "You can change this later from Settings."
        )
        info_label.setWordWrap(True)
        info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(info_label)

        # ── Input ─────────────────────────────────────────────────────
        input_layout = QHBoxLayout()
        center_label = QLabel("Center Name:")
        center_label.setMinimumWidth(110)
        input_layout.addWidget(center_label)

        self._name_edit = QLineEdit()
        self._name_edit.setPlaceholderText("e.g.  Mumbai_Center_01")
        self._name_edit.setText(existing_name)
        self._name_edit.setMinimumHeight(34)
        self._name_edit.returnPressed.connect(self._accept)
        input_layout.addWidget(self._name_edit)
        layout.addLayout(input_layout)

        hint = QLabel("Tip: Use letters, numbers, or underscores only (no spaces).")
        hint.setObjectName("hintLabel")
        hint_font = QFont()
        hint_font.setPointSize(9)
        hint_font.setItalic(True)
        hint.setFont(hint_font)
        hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(hint)

        # ── Buttons ───────────────────────────────────────────────────
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()

        self._ok_btn = QPushButton("Save & Continue")
        self._ok_btn.setObjectName("primaryButton")
        self._ok_btn.setMinimumHeight(36)
        self._ok_btn.setMinimumWidth(150)
        self._ok_btn.clicked.connect(self._accept)
        btn_layout.addWidget(self._ok_btn)

        layout.addLayout(btn_layout)
        self._name_edit.setFocus()

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _accept(self) -> None:
        name = self._name_edit.text().strip()
        if not name:
            QMessageBox.warning(
                self,
                "Center Name Required",
                "Please enter a center name before continuing."
            )
            return
        # Sanitise: replace spaces with underscores
        sanitised = name.replace(" ", "_")
        if sanitised != name:
            self._name_edit.setText(sanitised)
        self.accept()

    # ------------------------------------------------------------------
    # Result
    # ------------------------------------------------------------------

    def center_name(self) -> str:
        """Return the entered (sanitised) center name."""
        return self._name_edit.text().strip().replace(" ", "_")
