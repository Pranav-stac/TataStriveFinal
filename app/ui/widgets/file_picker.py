"""
File and Folder Picker Widgets.
Styled file/folder selection components.
"""

from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QLineEdit, QPushButton,
    QLabel, QFileDialog, QFrame
)
from PyQt6.QtCore import pyqtSignal, Qt


class FilePicker(QFrame):
    """A styled file picker widget with label, text field, and browse button."""
    
    path_changed = pyqtSignal(str)
    
    def __init__(
        self,
        label: str = "File",
        placeholder: str = "Select a file...",
        file_filter: str = "All Files (*.*)",
        parent: QWidget = None
    ):
        super().__init__(parent)
        self.file_filter = file_filter
        self._setup_ui(label, placeholder)
        
    def _setup_ui(self, label: str, placeholder: str):
        """Setup the widget UI."""
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setObjectName("filePicker")
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(6)
        
        # Label
        self.label = QLabel(label)
        self.label.setObjectName("filePickerLabel")
        layout.addWidget(self.label)
        
        # Input row
        input_layout = QHBoxLayout()
        input_layout.setSpacing(8)
        
        self.path_edit = QLineEdit()
        self.path_edit.setPlaceholderText(placeholder)
        self.path_edit.setReadOnly(False)
        self.path_edit.textChanged.connect(self._on_text_changed)
        input_layout.addWidget(self.path_edit, 1)
        
        self.browse_btn = QPushButton("Browse...")
        self.browse_btn.setObjectName("browseButton")
        self.browse_btn.clicked.connect(self._browse)
        input_layout.addWidget(self.browse_btn)
        
        layout.addLayout(input_layout)
        
    def _browse(self):
        """Open file dialog."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            f"Select {self.label.text()}",
            self.path_edit.text(),
            self.file_filter
        )
        if file_path:
            self.path_edit.setText(file_path)
            
    def _on_text_changed(self, text: str):
        """Handle text change."""
        self.path_changed.emit(text)
        
    def get_path(self) -> str:
        """Get the current path."""
        return self.path_edit.text()
        
    def set_path(self, path: str):
        """Set the current path."""
        self.path_edit.setText(path)
        
    def setEnabled(self, enabled: bool):
        """Enable or disable the widget."""
        super().setEnabled(enabled)
        self.path_edit.setEnabled(enabled)
        self.browse_btn.setEnabled(enabled)


class FolderPicker(QFrame):
    """A styled folder picker widget with label, text field, and browse button."""
    
    path_changed = pyqtSignal(str)
    
    def __init__(
        self,
        label: str = "Folder",
        placeholder: str = "Select a folder...",
        parent: QWidget = None
    ):
        super().__init__(parent)
        self._setup_ui(label, placeholder)
        
    def _setup_ui(self, label: str, placeholder: str):
        """Setup the widget UI."""
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setObjectName("folderPicker")
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(6)
        
        # Label
        self.label = QLabel(label)
        self.label.setObjectName("folderPickerLabel")
        layout.addWidget(self.label)
        
        # Input row
        input_layout = QHBoxLayout()
        input_layout.setSpacing(8)
        
        self.path_edit = QLineEdit()
        self.path_edit.setPlaceholderText(placeholder)
        self.path_edit.setReadOnly(False)
        self.path_edit.textChanged.connect(self._on_text_changed)
        input_layout.addWidget(self.path_edit, 1)
        
        self.browse_btn = QPushButton("Browse...")
        self.browse_btn.setObjectName("browseButton")
        self.browse_btn.clicked.connect(self._browse)
        input_layout.addWidget(self.browse_btn)
        
        layout.addLayout(input_layout)
        
    def _browse(self):
        """Open folder dialog."""
        folder_path = QFileDialog.getExistingDirectory(
            self,
            f"Select {self.label.text()}",
            self.path_edit.text()
        )
        if folder_path:
            self.path_edit.setText(folder_path)
            
    def _on_text_changed(self, text: str):
        """Handle text change."""
        self.path_changed.emit(text)
        
    def get_path(self) -> str:
        """Get the current path."""
        return self.path_edit.text()
        
    def set_path(self, path: str):
        """Set the current path."""
        self.path_edit.setText(path)
        
    def setEnabled(self, enabled: bool):
        """Enable or disable the widget."""
        super().setEnabled(enabled)
        self.path_edit.setEnabled(enabled)
        self.browse_btn.setEnabled(enabled)
