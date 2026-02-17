"""
Progress Panel Widget.
Displays progress bar and log output.
"""

from datetime import datetime
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QProgressBar, QTextEdit,
    QLabel, QFrame, QPushButton, QSizePolicy
)
from PyQt6.QtCore import Qt, pyqtSlot
from PyQt6.QtGui import QTextCursor, QColor


class ProgressPanel(QFrame):
    """A panel showing progress bar and scrolling log output."""
    
    def __init__(self, parent: QWidget = None):
        super().__init__(parent)
        self._setup_ui()
        
    def _setup_ui(self):
        """Setup the widget UI."""
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setObjectName("progressPanel")
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)
        
        # Progress section
        progress_layout = QVBoxLayout()
        progress_layout.setSpacing(6)
        
        # Progress header
        header_layout = QHBoxLayout()
        self.progress_label = QLabel("Progress")
        self.progress_label.setObjectName("progressLabel")
        header_layout.addWidget(self.progress_label)
        
        self.status_label = QLabel("Ready")
        self.status_label.setObjectName("statusLabel")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        header_layout.addWidget(self.status_label)
        
        progress_layout.addLayout(header_layout)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("%p%")
        progress_layout.addWidget(self.progress_bar)
        
        layout.addLayout(progress_layout)
        
        # Log section
        log_layout = QVBoxLayout()
        log_layout.setSpacing(6)
        
        # Log header
        log_header = QHBoxLayout()
        log_label = QLabel("Log Output")
        log_label.setObjectName("logLabel")
        log_header.addWidget(log_label)
        
        self.clear_btn = QPushButton("Clear")
        self.clear_btn.setObjectName("clearLogButton")
        self.clear_btn.setFixedWidth(60)
        self.clear_btn.clicked.connect(self.clear_log)
        log_header.addWidget(self.clear_btn)
        
        log_layout.addLayout(log_header)
        
        # Log text area
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setObjectName("logTextArea")
        self.log_text.setMinimumHeight(150)
        self.log_text.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        log_layout.addWidget(self.log_text)
        
        layout.addLayout(log_layout)
        
    @pyqtSlot(int, str)
    def update_progress(self, percent: int, message: str = ""):
        """Update progress bar and status."""
        self.progress_bar.setValue(percent)
        if message:
            self.status_label.setText(message)
            
    @pyqtSlot(str)
    def log(self, message: str, level: str = "info"):
        """Add a log message with timestamp."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        # Color based on level
        if level == "error":
            color = "#d32f2f"
        elif level == "warning":
            color = "#f57c00"
        elif level == "success":
            color = "#388e3c"
        else:
            color = "#333333"
            
        html = f'<span style="color: #888;">[{timestamp}]</span> <span style="color: {color};">{message}</span><br>'
        
        cursor = self.log_text.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self.log_text.setTextCursor(cursor)
        self.log_text.insertHtml(html)
        
        # Auto-scroll to bottom
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
        
    def log_info(self, message: str):
        """Log an info message."""
        self.log(message, "info")
        
    def log_warning(self, message: str):
        """Log a warning message."""
        self.log(message, "warning")
        
    def log_error(self, message: str):
        """Log an error message."""
        self.log(message, "error")
        
    def log_success(self, message: str):
        """Log a success message."""
        self.log(message, "success")
        
    def clear_log(self):
        """Clear the log output."""
        self.log_text.clear()
        
    def reset(self):
        """Reset progress and log."""
        self.progress_bar.setValue(0)
        self.status_label.setText("Ready")
        self.clear_log()
        
    def set_indeterminate(self, indeterminate: bool):
        """Set progress bar to indeterminate mode."""
        if indeterminate:
            self.progress_bar.setMaximum(0)
        else:
            self.progress_bar.setMaximum(100)
