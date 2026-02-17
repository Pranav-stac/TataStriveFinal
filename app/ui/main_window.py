"""
Main Window for TataStrive Analytics.
Contains the tabbed interface and menu bar.
"""

import os
import sys
from pathlib import Path

from PyQt6.QtWidgets import (
    QMainWindow, QTabWidget, QWidget, QVBoxLayout, QHBoxLayout,
    QMenuBar, QMenu, QStatusBar, QMessageBox, QFileDialog,
    QLabel, QApplication, QToolBar
)
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QAction, QIcon, QCloseEvent

from app.config import get_config
from app.ui.classroom_tab import ClassroomTab
from app.ui.crossday_tab import CrossDayTab
from app.ui.report_viewer import ReportViewer
from app.ui.settings_dialog import SettingsDialog


class MainWindow(QMainWindow):
    """Main application window with tabbed interface."""
    
    def __init__(self, torch_available: bool = True):
        super().__init__()
        self.torch_available = torch_available
        self.config = get_config()
        self._setup_ui()
        self._setup_menu()
        self._setup_statusbar()
        self._restore_geometry()
        
    def _setup_ui(self):
        """Setup the main UI components."""
        self.setWindowTitle("TataStrive Analytics")
        self.setMinimumSize(1000, 700)
        
        # Central widget with tab container
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        layout = QVBoxLayout(central_widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Tab widget
        self.tab_widget = QTabWidget()
        self.tab_widget.setDocumentMode(True)
        self.tab_widget.setTabPosition(QTabWidget.TabPosition.North)
        
        # Create tabs
        self.classroom_tab = ClassroomTab()
        self.crossday_tab = CrossDayTab()
        self.report_viewer = ReportViewer()
        
        # Add tabs
        self.tab_widget.addTab(self.classroom_tab, "Classroom Analysis")
        self.tab_widget.addTab(self.crossday_tab, "Attendance only")
        self.tab_widget.addTab(self.report_viewer, "Report Viewer")
        
        # Disable analysis tabs if PyTorch unavailable
        if not self.torch_available:
            self.tab_widget.setTabEnabled(0, False)
            self.tab_widget.setTabEnabled(1, False)
            self.tab_widget.setCurrentIndex(2)  # Switch to Report Viewer
        
        layout.addWidget(self.tab_widget)
        
        # Toolbar with Settings button
        toolbar = QToolBar()
        toolbar.setMovable(False)
        toolbar.setObjectName("mainToolbar")
        settings_action = QAction("Settings", self)
        settings_action.setShortcut("Ctrl+,")
        settings_action.setToolTip("Open Settings (Ctrl+,)")
        settings_action.triggered.connect(self._show_settings)
        toolbar.addAction(settings_action)
        self.addToolBar(toolbar)
        
        # Connect signals
        self.classroom_tab.analysis_complete.connect(self._on_classroom_complete)
        self.crossday_tab.analysis_complete.connect(self._on_crossday_complete)
        
    def _setup_menu(self):
        """Setup the menu bar."""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("&File")
        
        open_video_action = QAction("&Open Video...", self)
        open_video_action.setShortcut("Ctrl+O")
        open_video_action.triggered.connect(self._open_video)
        file_menu.addAction(open_video_action)
        
        open_report_action = QAction("Open &Report...", self)
        open_report_action.setShortcut("Ctrl+R")
        open_report_action.triggered.connect(self._open_report)
        file_menu.addAction(open_report_action)
        
        file_menu.addSeparator()
        
        settings_action = QAction("&Settings...", self)
        settings_action.setShortcut("Ctrl+,")
        settings_action.triggered.connect(self._show_settings)
        file_menu.addAction(settings_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("E&xit", self)
        exit_action.setShortcut("Alt+F4")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # View menu
        view_menu = menubar.addMenu("&View")
        
        classroom_action = QAction("&Classroom Analysis", self)
        classroom_action.setShortcut("Ctrl+1")
        classroom_action.triggered.connect(lambda: self.tab_widget.setCurrentIndex(0))
        view_menu.addAction(classroom_action)
        
        crossday_action = QAction("&Attendance only", self)
        crossday_action.setShortcut("Ctrl+2")
        crossday_action.triggered.connect(lambda: self.tab_widget.setCurrentIndex(1))
        view_menu.addAction(crossday_action)
        
        report_action = QAction("&Report Viewer", self)
        report_action.setShortcut("Ctrl+3")
        report_action.triggered.connect(lambda: self.tab_widget.setCurrentIndex(2))
        view_menu.addAction(report_action)
        
        # Help menu
        help_menu = menubar.addMenu("&Help")
        
        about_action = QAction("&About", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)
        
    def _setup_statusbar(self):
        """Setup the status bar."""
        self.statusbar = QStatusBar()
        self.setStatusBar(self.statusbar)
        if self.torch_available:
            self.statusbar.showMessage("Ready")
        else:
            self.statusbar.showMessage("Limited mode: PyTorch unavailable. Report Viewer only.")
        
        # Add permanent widgets
        self.status_label = QLabel("TataStrive Analytics v1.0")
        self.statusbar.addPermanentWidget(self.status_label)
        
    def _restore_geometry(self):
        """Restore window geometry from config."""
        window_config = self.config.get_section("window")
        self.resize(window_config.get("width", 1200), window_config.get("height", 800))
        self.move(window_config.get("x", 100), window_config.get("y", 100))
        
    def _save_geometry(self):
        """Save window geometry to config."""
        geo = self.geometry()
        self.config.set("window.width", geo.width(), save=False)
        self.config.set("window.height", geo.height(), save=False)
        self.config.set("window.x", geo.x(), save=False)
        self.config.set("window.y", geo.y(), save=True)
        
    def _open_video(self):
        """Open a video file."""
        last_path = self.config.get("last_video_path", "")
        start_dir = os.path.dirname(last_path) if last_path else ""
        
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Video File",
            start_dir,
            "Video Files (*.mp4 *.avi *.mkv *.mov);;All Files (*.*)"
        )
        
        if file_path:
            self.config.set("last_video_path", file_path)
            current_tab = self.tab_widget.currentWidget()
            if hasattr(current_tab, 'set_video_path'):
                current_tab.set_video_path(file_path)
            self.statusbar.showMessage(f"Loaded: {os.path.basename(file_path)}")
            
    def _open_report(self):
        """Open a report JSON file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Report File",
            "",
            "JSON Files (*.json);;All Files (*.*)"
        )
        
        if file_path:
            self.tab_widget.setCurrentIndex(2)  # Switch to Report Viewer
            self.report_viewer.load_report(file_path)
            self.statusbar.showMessage(f"Loaded report: {os.path.basename(file_path)}")
            
    def _show_settings(self):
        """Show the settings dialog."""
        dialog = SettingsDialog(self)
        if dialog.exec():
            self.statusbar.showMessage("Settings saved")
            # Notify tabs to reload config
            self.classroom_tab.reload_config()
            self.crossday_tab.reload_config()
            
    def _show_about(self):
        """Show the about dialog."""
        QMessageBox.about(
            self,
            "About TataStrive Analytics",
            "<h2>TataStrive Analytics</h2>"
            "<p>Version 1.0.0</p>"
            "<p>A professional desktop application for:</p>"
            "<ul>"
            "<li>Classroom engagement analysis</li>"
            "<li>Attendance tracking</li>"
            "</ul>"
            "<p>Built with PyQt6 and Python.</p>"
        )
        
    def _on_classroom_complete(self, report_path: str):
        """Handle classroom analysis completion."""
        self.statusbar.showMessage(f"Analysis complete: {report_path}")
        reply = QMessageBox.question(
            self,
            "Analysis Complete",
            "Classroom analysis completed successfully.\n\nWould you like to view the report?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if reply == QMessageBox.StandardButton.Yes:
            self.tab_widget.setCurrentIndex(2)
            self.report_viewer.load_report(report_path)
            
    def _on_crossday_complete(self, report_path: str):
        """Handle attendance analysis completion."""
        self.statusbar.showMessage(f"Analysis complete: {report_path}")
        reply = QMessageBox.question(
            self,
            "Analysis Complete",
            "Attendance analysis completed successfully.\n\nWould you like to view the report?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if reply == QMessageBox.StandardButton.Yes:
            self.tab_widget.setCurrentIndex(2)
            self.report_viewer.load_report(report_path)
            
    def closeEvent(self, event: QCloseEvent):
        """Handle window close event."""
        # Check if any analysis is running
        if self.classroom_tab.is_running() or self.crossday_tab.is_running():
            reply = QMessageBox.question(
                self,
                "Confirm Exit",
                "An analysis is currently running.\n\nAre you sure you want to exit?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.No:
                event.ignore()
                return
            # Stop running analyses
            self.classroom_tab.stop_analysis()
            self.crossday_tab.stop_analysis()
            
        self._save_geometry()
        event.accept()
