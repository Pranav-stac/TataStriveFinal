"""
Main Window for TataStrive Analytics.
Contains the tabbed interface, menu bar, and BigQuery sync integration.
"""

import os
import sys
import threading
from pathlib import Path
from datetime import date

from PyQt6.QtWidgets import (
    QMainWindow, QTabWidget, QWidget, QVBoxLayout, QHBoxLayout,
    QMenuBar, QMenu, QStatusBar, QMessageBox, QFileDialog,
    QLabel, QApplication, QToolBar, QInputDialog, QStackedWidget,
)
from PyQt6.QtCore import Qt, QSize, QMetaObject, Q_ARG, pyqtSlot, QTimer
from PyQt6.QtGui import QAction, QIcon, QCloseEvent

from app import __version__
from app.config import get_config
from app.ui.classroom_tab import ClassroomTab
from app.ui.crossday_tab import CrossDayTab
from app.ui.report_viewer import ReportViewer
from app.ui.settings_tab import SettingsTab
from app.ui.match_verifier_tab import MatchVerifierTab


class MainWindow(QMainWindow):
    """Main application window with tabbed interface and BigQuery sync."""

    def __init__(self, torch_available: bool = True, bq_service=None):
        super().__init__()
        self.torch_available = torch_available
        self.config = get_config()
        self._bq_service = bq_service          # BigQuerySyncService | None
        self._setup_ui()
        self._setup_menu()
        self._setup_statusbar()
        self._restore_geometry()

    # ──────────────────────────────────────────────────────────────────
    # UI setup
    # ──────────────────────────────────────────────────────────────────

    def _setup_ui(self):
        """Setup the main UI components."""
        center_id = self.config.get("center_id", "")
        roster_center = self.config.get("student_roster_center_name", "")
        if center_id:
            title = f"TataStrive Analytics v{__version__}  |  Center: {center_id}"
        elif roster_center:
            title = f"TataStrive Analytics v{__version__}  |  Center: {roster_center}"
        else:
            title = f"TataStrive Analytics v{__version__}"
        self.setWindowTitle(title)
        self.setMinimumSize(1000, 700)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        layout = QVBoxLayout(central_widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Tab widget (analysis + reports only)
        self.tab_widget = QTabWidget()
        self.tab_widget.setDocumentMode(True)
        self.tab_widget.setTabPosition(QTabWidget.TabPosition.North)

        self.classroom_tab = ClassroomTab()
        self.crossday_tab  = CrossDayTab()
        self.report_viewer = ReportViewer(config=self.config)
        self.match_verifier = MatchVerifierTab()
        self.settings_tab = SettingsTab()

        self.tab_widget.addTab(self.classroom_tab, "Classroom Analysis")
        self.tab_widget.addTab(self.crossday_tab,  "Attendance only")
        self.tab_widget.addTab(self.report_viewer, "Report Viewer")
        self.tab_widget.addTab(self.match_verifier, "Match Verifier")

        self._main_page_stack = QStackedWidget()
        self._main_page_stack.addWidget(self.tab_widget)
        self._main_page_stack.addWidget(self.settings_tab)
        self._previous_tab_index = 0

        if not self.torch_available:
            self.tab_widget.setTabEnabled(0, False)
            self.tab_widget.setTabEnabled(1, False)
            self.tab_widget.setCurrentIndex(2)

        layout.addWidget(self._main_page_stack)

        # ── Toolbar ──────────────────────────────────────────────────
        toolbar = QToolBar()
        toolbar.setMovable(False)
        toolbar.setObjectName("mainToolbar")

        settings_action = QAction("Settings", self)
        settings_action.setShortcut("Ctrl+,")
        settings_action.setToolTip("Open Settings page (Ctrl+,)")
        settings_action.triggered.connect(self._show_settings)
        toolbar.addAction(settings_action)

        toolbar.addSeparator()

        sync_action = QAction("☁  Sync to BigQuery", self)
        sync_action.setObjectName("bqSyncAction")
        sync_action.setShortcut("Ctrl+Shift+S")
        sync_action.setToolTip(
            "Manually sync all attendance & engagement reports\n"
            "from the last output directory to BigQuery (Ctrl+Shift+S)"
        )
        sync_action.triggered.connect(self._manual_bq_sync)
        toolbar.addAction(sync_action)
        self._sync_action = sync_action

        self.addToolBar(toolbar)

        # Connect analysis signals
        self.classroom_tab.analysis_complete.connect(self._on_classroom_complete)
        self.crossday_tab.analysis_complete.connect(self._on_crossday_complete)
        self.settings_tab.settings_saved.connect(self._on_settings_saved)
        self.settings_tab.back_requested.connect(self._show_analysis)

    def _setup_menu(self):
        """Setup the menu bar."""
        menubar = self.menuBar()

        # ── File ──────────────────────────────────────────────────────
        file_menu = menubar.addMenu("&File")

        open_video_action = QAction("&Open Input Folder...", self)
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

        # ── View ──────────────────────────────────────────────────────
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
        report_action.triggered.connect(self._show_report_viewer)
        view_menu.addAction(report_action)

        # ── BigQuery ──────────────────────────────────────────────────
        bq_menu = menubar.addMenu("&BigQuery")

        sync_now_action = QAction("Sync &Now  (Ctrl+Shift+S)", self)
        sync_now_action.setShortcut("Ctrl+Shift+S")
        sync_now_action.triggered.connect(self._manual_bq_sync)
        bq_menu.addAction(sync_now_action)

        bq_menu.addSeparator()

        change_center_action = QAction("Change &Center...", self)
        change_center_action.triggered.connect(self._change_center_id)
        bq_menu.addAction(change_center_action)

        bq_menu.addSeparator()

        view_sync_action = QAction("View BigQuery &Console", self)
        view_sync_action.triggered.connect(lambda: self._open_bq_console())
        bq_menu.addAction(view_sync_action)

        # ── Help ──────────────────────────────────────────────────────
        help_menu = menubar.addMenu("&Help")

        check_updates_action = QAction("Check for &Updates…", self)
        check_updates_action.triggered.connect(self._check_for_updates)
        help_menu.addAction(check_updates_action)

        help_menu.addSeparator()
        about_action = QAction("&About", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)

    def _setup_statusbar(self):
        """Setup the status bar."""
        self.statusbar = QStatusBar()
        self.setStatusBar(self.statusbar)

        center_id = self.config.get("center_id", "")
        roster_center = self.config.get("student_roster_center_name", "")
        center_text = center_id or roster_center
        base_msg = "Ready" if self.torch_available else "Limited mode: Report Viewer only."
        self.statusbar.showMessage(base_msg)

        if center_text and roster_center and roster_center != center_text:
            label = f"  Device: {center_id}  |  Roster: {roster_center}  "
        elif center_text:
            label = f"  Center: {center_text}  "
        else:
            label = "  Center: (not set)  "
        self._center_label = QLabel(label)
        self._center_label.setObjectName("centerLabel")
        self.statusbar.addPermanentWidget(self._center_label)

        self.status_label = QLabel(f"TataStrive Analytics v{__version__}")
        self.statusbar.addPermanentWidget(self.status_label)

    def _check_for_updates(self):
        """
        Manual update check (UI action).
        - If a newer GitHub Release exists, show UpdateDialog.
        - If up to date, show a status message.
        """
        from app.updater import UpdateChecker

        checker = getattr(self, "_update_checker", None)
        if checker is None:
            checker = UpdateChecker(current_version=__version__, silent_auto_apply=False)
            self._update_checker = checker

        self.statusbar.showMessage("Checking for updates…")

        def _worker():
            try:
                # Private but stable in our codebase; keeps manual button lightweight.
                info = checker._fetch_latest_info()  # type: ignore[attr-defined]
                err = ""
            except Exception as e:
                info = None
                err = str(e)

            def _on_ui():
                if err:
                    self.statusbar.showMessage(f"Update check failed: {err}")
                    QMessageBox.warning(self, "Update Check Failed", err)
                    return

                if info is None:
                    self.statusbar.showMessage("You’re up to date.")
                    QMessageBox.information(
                        self,
                        "No Updates",
                        f"You’re already on the latest version (v{__version__}).",
                    )
                    return

                from app.ui.update_dialog import UpdateDialog

                self.statusbar.showMessage(f"Update available: v{info.version}")
                dlg = UpdateDialog(info, parent=self)
                dlg.exec()

            QTimer.singleShot(0, _on_ui)

        threading.Thread(target=_worker, daemon=True, name="manual-update-check").start()

    # ──────────────────────────────────────────────────────────────────
    # BigQuery sync – manual & auto
    # ──────────────────────────────────────────────────────────────────

    def _manual_bq_sync(self):
        """User-triggered sync: scan last_output_dir and push to BigQuery."""
        if self._bq_service is None:
            QMessageBox.warning(
                self, "BigQuery Not Available",
                "BigQuery service is not initialised.\n"
                "Check that google-cloud-bigquery is installed:\n\n"
                "  pip install google-cloud-bigquery"
            )
            return

        output_dirs: list[str] = []
        for key in ("last_output_dir", "last_classroom_output_dir", "last_crossday_output_dir"):
            d = (self.config.get(key) or "").strip()
            if d and d not in output_dirs:
                output_dirs.append(d)
        if not output_dirs:
            output_dir_chosen = QFileDialog.getExistingDirectory(
                self, "Select Output Directory to Sync", ""
            )
            if not output_dir_chosen:
                return
            output_dirs = [output_dir_chosen]

        self.statusbar.showMessage("⏳ BigQuery sync in progress...")
        self._sync_action.setEnabled(False)

        self._bq_service.trigger_daily_sync(
            output_dirs=output_dirs,
            log_callback=lambda msg: self.statusbar.showMessage(f"[BQ] {msg}"),
            done_callback=self.on_bq_sync_done
        )

    def _sync_single_report(self, report_path: str, videos_in_queue: int | None = None):
        """Auto-sync a single newly completed report immediately."""
        if self._bq_service is None or not report_path:
            return

        import threading

        def _run():
            try:
                self._bq_service.ensure_tables()
                result = self._bq_service.sync_report(report_path, videos_in_queue=videos_in_queue)
                if result["status"] == "ok":
                    msg = (
                        f"✅ BigQuery: synced {result['rows_inserted']} rows "
                        f"from {Path(report_path).name}"
                    )
                elif (
                    result["status"] == "skipped"
                    and result.get("error_msg") == "already_synced"
                ):
                    msg = (
                        f"☁ BigQuery: skipped duplicate (same report already synced) — "
                        f"{Path(report_path).name}"
                    )
                else:
                    msg = f"⚠ BigQuery sync issue: {result.get('error_msg', '')}"
            except Exception as e:
                msg = f"⚠ BigQuery sync error: {e}"
            # Update status bar from main thread
            QMetaObject.invokeMethod(
                self.statusbar, "showMessage",
                Qt.ConnectionType.QueuedConnection,
                Q_ARG(str, msg)
            )

        t = threading.Thread(target=_run, daemon=True, name="bq-single-sync")
        t.start()

    def _sync_classroom_reports(self, report_path: str, videos_in_queue: int | None = None):
        """Push class dynamics and grouped management summary after classroom analysis."""
        self._sync_single_report(report_path, videos_in_queue=videos_in_queue)
        mgmt_path = Path(report_path).with_name("management_summary_report.json")
        if mgmt_path.is_file():
            self._sync_single_report(str(mgmt_path), videos_in_queue=videos_in_queue)

    @pyqtSlot(object)
    def on_bq_sync_done(self, summary: dict):
        """Slot called (from any thread) when a BQ sync completes."""
        if "error" in summary:
            msg = f"⚠ BigQuery sync failed: {summary['error']}"
        else:
            msg = (
                f"☁ BigQuery sync done — "
                f"synced: {summary.get('synced', 0)}, "
                f"skipped: {summary.get('skipped', 0)}, "
                f"errors: {summary.get('errors', 0)}"
            )
        # Must update UI from main thread
        QMetaObject.invokeMethod(
            self.statusbar, "showMessage",
            Qt.ConnectionType.QueuedConnection,
            Q_ARG(str, msg)
        )
        QMetaObject.invokeMethod(
            self._sync_action, "setEnabled",
            Qt.ConnectionType.QueuedConnection,
            Q_ARG(bool, True)
        )

    # ──────────────────────────────────────────────────────────────────
    # Center ID management
    # ──────────────────────────────────────────────────────────────────

    def _change_center_id(self):
        """Update device center_id and BigQuery roster center_name."""
        from app.ui.center_dialog import CenterDialog

        current = self.config.get("center_id", "")
        roster_current = self.config.get("student_roster_center_name", "")
        dlg = CenterDialog(
            parent=self,
            existing_name=current,
            existing_roster_center=roster_current,
        )
        if dlg.exec():
            new_id = dlg.center_name()
            new_roster = dlg.roster_center_name()
            if not new_roster:
                return
            changed = new_id != current or new_roster != roster_current
            if not changed:
                return
            self.config.set("center_id", new_id)
            self.config.set("student_roster_center_name", new_roster)
            from app.bigquery_sync import get_sync_service, _creds_path
            self._bq_service = get_sync_service(new_id, _creds_path())
            if new_roster and new_roster != new_id:
                label = f"  Device: {new_id}  |  Roster: {new_roster}  "
            else:
                label = f"  Center: {new_id}  "
            self._center_label.setText(label)
            self.setWindowTitle(f"TataStrive Analytics  |  Center: {new_id}")
            self.statusbar.showMessage(f"Center updated: roster {new_roster}")

    @staticmethod
    def _open_bq_console():
        """Open the BigQuery web console in the default browser."""
        import webbrowser
        webbrowser.open(
            "https://console.cloud.google.com/bigquery?"
            "project=tatastrive-269409&d=tatastrive_analytics&page=dataset"
        )

    # ──────────────────────────────────────────────────────────────────
    # Geometry
    # ──────────────────────────────────────────────────────────────────

    def _restore_geometry(self):
        """Restore window geometry from config."""
        window_config = self.config.get_section("window")
        self.resize(window_config.get("width", 1200), window_config.get("height", 800))
        self.move(window_config.get("x", 100), window_config.get("y", 100))

    def _save_geometry(self):
        """Save window geometry to config."""
        geo = self.geometry()
        self.config.set("window.width",  geo.width(),  save=False)
        self.config.set("window.height", geo.height(), save=False)
        self.config.set("window.x",      geo.x(),      save=False)
        self.config.set("window.y",      geo.y(),      save=True)

    # ──────────────────────────────────────────────────────────────────
    # File / settings helpers
    # ──────────────────────────────────────────────────────────────────

    def _open_video(self):
        """Open an input folder for whichever tab is currently active."""
        current_tab = self.tab_widget.currentWidget()
        # Use a per-tab config key so each tab remembers its own folder
        if current_tab is self.crossday_tab:
            cfg_key = "last_crossday_video_folder"
        else:
            cfg_key = "last_classroom_video_folder"
        last_folder = self.config.get(cfg_key, "")
        start_dir = last_folder if os.path.isdir(last_folder) else ""
        folder_path = QFileDialog.getExistingDirectory(
            self, "Open Input Folder", start_dir
        )
        if folder_path:
            self.config.set(cfg_key, folder_path)
            if hasattr(current_tab, 'set_video_folder'):
                current_tab.set_video_folder(folder_path)
            elif hasattr(current_tab, 'set_video_path'):
                current_tab.set_video_path(folder_path)
            self.statusbar.showMessage(f"Loaded folder: {os.path.basename(folder_path)}")

    def _open_report(self):
        """Open a report JSON file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Report File", "",
            "JSON Files (*.json);;All Files (*.*)"
        )
        if file_path:
            self._show_report_viewer()
            self.report_viewer.load_report(file_path)
            self.statusbar.showMessage(f"Loaded report: {os.path.basename(file_path)}")

    def _show_settings(self):
        """Open the full-page settings view."""
        self._previous_tab_index = self.tab_widget.currentIndex()
        self.settings_tab.activate()
        self._main_page_stack.setCurrentWidget(self.settings_tab)

    def _show_analysis(self):
        """Return to the analysis tab bar."""
        self._main_page_stack.setCurrentWidget(self.tab_widget)
        self.tab_widget.setCurrentIndex(self._previous_tab_index)

    def _show_report_viewer(self):
        """Open Report Viewer from the menu."""
        self._show_analysis()
        self.tab_widget.setCurrentWidget(self.report_viewer)

    def _on_settings_saved(self) -> None:
        self.statusbar.showMessage("Settings saved")
        self.classroom_tab.reload_config()
        self.crossday_tab.reload_config()

    def _show_about(self):
        """Show the about dialog."""
        center_id = self.config.get("center_id", "N/A")
        roster_center = self.config.get("student_roster_center_name", "N/A")
        QMessageBox.about(
            self, "About TataStrive Analytics",
            "<h2>TataStrive Analytics</h2>"
            "<p>Version 1.0.0</p>"
            f"<p><b>Device center ID:</b> {center_id}</p>"
            f"<p><b>BigQuery roster center:</b> {roster_center}</p>"
            "<p>A professional desktop application for:</p>"
            "<ul>"
            "<li>Classroom engagement analysis</li>"
            "<li>Attendance tracking</li>"
            "<li>Automatic BigQuery reporting</li>"
            "</ul>"
            "<p>Built with PyQt6 and Python.</p>"
        )

    # ──────────────────────────────────────────────────────────────────
    # Analysis completion handlers
    # ──────────────────────────────────────────────────────────────────

    def _on_classroom_complete(self, report_path: str):
        """Handle classroom analysis completion — view report + auto-sync."""
        self.statusbar.showMessage(f"Analysis complete: {report_path}")
        # Auto-sync the new report immediately (queue depth for BigQuery sync_log)
        q = self.classroom_tab.pending_video_queue_count()
        self._sync_classroom_reports(report_path, videos_in_queue=q)
        if not self.classroom_tab.is_monitoring():
            reply = QMessageBox.question(
                self, "Analysis Complete",
                "Classroom analysis completed successfully.\n\nWould you like to view the report?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.Yes:
                self.tab_widget.setCurrentIndex(2)
                self.report_viewer.load_report(report_path)

    def _on_crossday_complete(self, report_path: str):
        """Handle attendance analysis completion — view report + auto-sync."""
        self.statusbar.showMessage(f"Analysis complete: {report_path}")
        # Auto-sync the new report immediately (queue depth for BigQuery sync_log)
        q = self.crossday_tab.pending_video_queue_count()
        self._sync_single_report(report_path, videos_in_queue=q)
        if not self.crossday_tab.is_monitoring():
            reply = QMessageBox.question(
                self, "Analysis Complete",
                "Attendance analysis completed successfully.\n\nWould you like to view the report?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.Yes:
                self.tab_widget.setCurrentIndex(2)
                self.report_viewer.load_report(report_path)

    # ──────────────────────────────────────────────────────────────────
    # Close
    # ──────────────────────────────────────────────────────────────────

    def closeEvent(self, event: QCloseEvent):
        """Handle window close event."""
        if self.settings_tab.sync_running():
            QMessageBox.warning(
                self,
                "Roster sync running",
                "Wait for the student enrollment sync to finish before exiting.",
            )
            event.ignore()
            return
        if self.classroom_tab.is_running() or self.crossday_tab.is_running():
            reply = QMessageBox.question(
                self, "Confirm Exit",
                "An analysis is currently running.\n\nAre you sure you want to exit?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.No:
                event.ignore()
                return
            self.classroom_tab.stop_analysis()
            self.crossday_tab.stop_analysis()

        self._save_geometry()
        event.accept()

