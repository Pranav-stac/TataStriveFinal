"""
Video Preview Widget.
Displays video frames during processing.
"""

import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QFrame, QSizePolicy, QCheckBox, QHBoxLayout,
    QScrollArea
)
from PyQt6.QtCore import Qt, pyqtSlot, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap, QPainter, QColor, QFont


class VideoPreview(QFrame):
    """A widget that displays video frames with optional toggle."""
    
    preview_toggled = pyqtSignal(bool)
    
    def __init__(self, parent: QWidget = None):
        super().__init__(parent)
        self._preview_enabled = False
        self._preview_busy = False  # Drop frames when still processing
        self._last_preview_size = (0, 0)  # Skip adjustSize when unchanged
        self._setup_ui()
        
    def _setup_ui(self):
        """Setup the widget UI."""
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setObjectName("videoPreview")
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 12)
        layout.setSpacing(8)
        
        # Header with toggle
        header_layout = QHBoxLayout()
        
        label = QLabel("Video Preview")
        label.setObjectName("videoPreviewLabel")
        header_layout.addWidget(label)
        
        header_layout.addStretch()
        
        self.toggle_checkbox = QCheckBox("Enable Preview")
        self.toggle_checkbox.setChecked(False)
        self.toggle_checkbox.toggled.connect(self._on_toggle)
        header_layout.addWidget(self.toggle_checkbox)
        
        layout.addLayout(header_layout)
        
        # Video display area wrapped in scroll area
        self.scroll_area = QScrollArea()
        # Don't resize widget - let label keep pixmap size so full frame is visible (scroll to pan)
        self.scroll_area.setWidgetResizable(False)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        self.scroll_area.setMinimumSize(320, 240)
        
        self.video_label = QLabel()
        self.video_label.setObjectName("videoDisplayArea")
        self.video_label.setMinimumSize(320, 240)
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        # Fixed size policy so label keeps pixmap dimensions, not forced to fill space
        self.video_label.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.video_label.setStyleSheet("""
            QLabel#videoDisplayArea {
                background-color: #1a1a1a;
                border: 1px solid #d0d0d0;
                border-radius: 4px;
            }
        """)
        
        # Set placeholder
        self._show_placeholder()
        
        self.scroll_area.setWidget(self.video_label)
        layout.addWidget(self.scroll_area)
        
    def _show_placeholder(self):
        """Show placeholder text when no video is playing."""
        self.video_label.setText("No preview available")
        self.video_label.setStyleSheet("""
            QLabel#videoDisplayArea {
                background-color: #1a1a1a;
                border: 1px solid #d0d0d0;
                border-radius: 4px;
                color: #888888;
                font-size: 14px;
            }
        """)
        
    def _on_toggle(self, enabled: bool):
        """Handle preview toggle."""
        self._preview_enabled = enabled
        self.preview_toggled.emit(enabled)
        if not enabled:
            self._show_placeholder()
            
    @pyqtSlot(np.ndarray, bool)
    def update_frame(self, frame: np.ndarray, is_rgb: bool = False):
        """Update the display with a new frame. Frame is pre-resized RGB when is_rgb=True."""
        if not self._preview_enabled:
            return
        
        # Drop frame if still processing previous (keeps UI responsive)
        if self._preview_busy:
            return
        
        self._preview_busy = True
        try:
            # Use frame directly if already RGB (from worker), else convert BGR→RGB
            if not is_rgb and len(frame.shape) == 3 and frame.shape[2] == 3:
                import cv2
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                rgb = frame
            
            h, w, ch = rgb.shape
            bytes_per_line = ch * w
            qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            scaled_pixmap = QPixmap.fromImage(qimg)
            
            self.video_label.setPixmap(scaled_pixmap)
            # Only trigger layout when size changes (reduces lag)
            if (w, h) != getattr(self, '_last_preview_size', (0, 0)):
                self._last_preview_size = (w, h)
                self.video_label.adjustSize()
        except Exception as e:
            print(f"Error updating frame: {e}")
        finally:
            self._preview_busy = False
            
    def is_enabled(self) -> bool:
        """Check if preview is enabled."""
        return self._preview_enabled
        
    def set_enabled(self, enabled: bool):
        """Set preview enabled state."""
        self.toggle_checkbox.setChecked(enabled)
        
    def clear(self):
        """Clear the video display."""
        self._last_preview_size = (0, 0)
        self._show_placeholder()
        
    def setEnabled(self, enabled: bool):
        """Enable or disable the widget."""
        super().setEnabled(enabled)
        self.toggle_checkbox.setEnabled(enabled)
        if not enabled:
            self._show_placeholder()
