"""
Report Viewer Tab.
Displays JSON reports in a formatted table view with export capability.
"""

import os
import json
import csv
import glob
from typing import Dict, Any, List, Optional

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QPushButton, QFrame, QSplitter, QTreeWidget, QTreeWidgetItem,
    QTableWidget, QTableWidgetItem, QHeaderView, QFileDialog,
    QMessageBox, QTabWidget, QSizePolicy, QComboBox
)
from PyQt6.QtCore import Qt


class ReportViewer(QWidget):
    """Tab for viewing and exporting analysis reports."""

    def __init__(self, parent: QWidget = None, config: Optional[Dict] = None):
        super().__init__(parent)
        self._config = config or {}
        self._report_data = None
        self._report_path = None
        self._report_paths: List[str] = []
        self._setup_ui()

    def _scan_reports(self) -> List[str]:
        """Scan known output directories for JSON reports."""
        paths = []
        seen = set()
        dirs_to_scan = []

        cfg = self._config or {}
        last_classroom = cfg.get("last_classroom_output_dir", "")
        last_crossday = cfg.get("last_crossday_output_dir", "")
        last_output = cfg.get("last_output_dir", "")

        for d in (last_classroom, last_crossday, last_output):
            if d and os.path.isdir(d):
                if d not in dirs_to_scan:
                    dirs_to_scan.append(d)
                parent = os.path.dirname(d)
                if parent and os.path.isdir(parent) and parent not in dirs_to_scan:
                    dirs_to_scan.append(parent)

        patterns = [
            "*attendance_report*.json",
            "*class_dynamics_report*.json",
            "*management_summary_report*.json",
            "*_report.json",
        ]

        for base_dir in dirs_to_scan:
            for pattern in patterns:
                for p in glob.glob(os.path.join(base_dir, pattern)):
                    if os.path.isfile(p) and p not in seen:
                        seen.add(p)
                        paths.append(p)
            for sub in glob.glob(os.path.join(base_dir, "*", "*.json")):
                if os.path.isfile(sub) and "report" in sub.lower() and sub not in seen:
                    seen.add(sub)
                    paths.append(sub)

        return sorted(paths, key=lambda p: (os.path.getmtime(p), p), reverse=True)

    def _refresh_dropdown(self):
        """Refresh the report dropdown with scanned reports."""
        self._report_paths = self._scan_reports()
        self.report_combo.clear()
        self.report_combo.addItem("— Select a report —", None)
        for path in self._report_paths:
            label = os.path.basename(path)
            parent = os.path.basename(os.path.dirname(path))
            if parent and parent != "Outputs":
                label = f"{parent} / {label}"
            self.report_combo.addItem(label, path)

    def _setup_ui(self):
        """Setup the tab UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        # Report selector: dropdown + refresh + browse
        picker_layout = QHBoxLayout()

        combo_label = QLabel("Report:")
        combo_label.setObjectName("filePickerLabel")
        picker_layout.addWidget(combo_label)

        self.report_combo = QComboBox()
        self.report_combo.setMinimumWidth(280)
        self.report_combo.setEditable(False)
        self.report_combo.currentIndexChanged.connect(self._on_combo_changed)
        picker_layout.addWidget(self.report_combo, 1)

        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.setToolTip("Rescan output directories for reports")
        self.refresh_btn.clicked.connect(self._refresh_dropdown)
        picker_layout.addWidget(self.refresh_btn)

        self.browse_btn = QPushButton("Browse...")
        self.browse_btn.clicked.connect(self._browse_report)
        picker_layout.addWidget(self.browse_btn)

        layout.addLayout(picker_layout)
        self._refresh_dropdown()
        
        # Summary cards
        self.summary_frame = QFrame()
        self.summary_frame.setObjectName("summaryFrame")
        self.summary_frame.setFrameShape(QFrame.Shape.StyledPanel)
        summary_layout = QHBoxLayout(self.summary_frame)
        summary_layout.setSpacing(16)
        
        self.summary_cards = {}
        for key, title in [("type", "Report Type"), ("count", "Total Count"), ("date", "Date"), ("duration", "Duration")]:
            card = self._create_summary_card(title, "-")
            self.summary_cards[key] = card
            summary_layout.addWidget(card)
        
        summary_layout.addStretch()
        layout.addWidget(self.summary_frame)
        
        # Content tabs
        self.content_tabs = QTabWidget()
        
        # Table view tab
        table_widget = QWidget()
        table_layout = QVBoxLayout(table_widget)
        table_layout.setContentsMargins(0, 8, 0, 0)
        
        self.data_table = QTableWidget()
        self.data_table.setAlternatingRowColors(True)
        self.data_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.data_table.horizontalHeader().setStretchLastSection(True)
        self.data_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        table_layout.addWidget(self.data_table)
        
        self.content_tabs.addTab(table_widget, "Data Table")
        
        # Tree view tab
        tree_widget = QWidget()
        tree_layout = QVBoxLayout(tree_widget)
        tree_layout.setContentsMargins(0, 8, 0, 0)
        
        self.json_tree = QTreeWidget()
        self.json_tree.setHeaderLabels(["Key", "Value"])
        self.json_tree.setAlternatingRowColors(True)
        self.json_tree.header().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        tree_layout.addWidget(self.json_tree)
        
        self.content_tabs.addTab(tree_widget, "JSON Tree")
        
        layout.addWidget(self.content_tabs, 1)
        
        # Export buttons
        export_layout = QHBoxLayout()
        export_layout.addStretch()
        
        self.export_csv_btn = QPushButton("Export to CSV")
        self.export_csv_btn.setEnabled(False)
        self.export_csv_btn.clicked.connect(self._export_csv)
        export_layout.addWidget(self.export_csv_btn)
        
        self.open_folder_btn = QPushButton("Open Folder")
        self.open_folder_btn.setEnabled(False)
        self.open_folder_btn.clicked.connect(self._open_folder)
        export_layout.addWidget(self.open_folder_btn)
        
        layout.addLayout(export_layout)
        
    def _create_summary_card(self, title: str, value: str) -> QFrame:
        """Create a summary card widget."""
        card = QFrame()
        card.setObjectName("summaryCard")
        card.setFrameShape(QFrame.Shape.StyledPanel)
        card.setMinimumWidth(150)
        
        layout = QVBoxLayout(card)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(4)
        
        title_label = QLabel(title)
        title_label.setObjectName("cardTitle")
        layout.addWidget(title_label)
        
        value_label = QLabel(value)
        value_label.setObjectName("cardValue")
        layout.addWidget(value_label)
        
        card.value_label = value_label
        return card
        
    def _on_combo_changed(self, index: int):
        """Handle dropdown selection change."""
        path = self.report_combo.currentData()
        if path and os.path.isfile(path):
            self.load_report(path)

    def _browse_report(self):
        """Open file dialog to select a report."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Report",
            "",
            "JSON Files (*.json);;All Files (*.*)"
        )
        if file_path:
            self.load_report(file_path)
            self._refresh_dropdown()
            
    def load_report(self, path: str):
        """Load and display a report file."""
        if not path or not os.path.isfile(path):
            if path:
                QMessageBox.warning(self, "Error", f"File not found: {path}")
            return

        try:
            with open(path, 'r', encoding='utf-8') as f:
                self._report_data = json.load(f)
            self._report_path = path

            idx = self.report_combo.findData(path)
            if idx >= 0:
                self.report_combo.blockSignals(True)
                self.report_combo.setCurrentIndex(idx)
                self.report_combo.blockSignals(False)
            else:
                self.report_combo.insertItem(1, os.path.basename(path), path)
                self.report_combo.setCurrentIndex(1)

            self._update_display()
            self.export_csv_btn.setEnabled(True)
            self.open_folder_btn.setEnabled(True)

        except json.JSONDecodeError as e:
            QMessageBox.critical(self, "Error", f"Invalid JSON file:\n{e}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not load report:\n{e}")
            
    def _update_display(self):
        """Update the display with loaded report data."""
        if not self._report_data:
            return
            
        # Detect report type and update summary
        if "sessions" in self._report_data:
            # Management summary report
            self._display_management_summary()
        elif "hourly_probes" in self._report_data:
            # Classroom report
            self._display_classroom_report()
        elif "People" in self._report_data:
            # Cross-day report
            self._display_crossday_report()
        else:
            # Generic JSON
            self._display_generic_json()
            
        # Update JSON tree
        self._populate_tree(self._report_data)
        
    def _display_classroom_report(self):
        """Display a classroom analysis report."""
        data = self._report_data
        
        # Update summary cards
        self.summary_cards["type"].value_label.setText("Classroom Analysis")
        self.summary_cards["count"].value_label.setText(f"{data.get('baseline_max_students', 0)} students")
        self.summary_cards["date"].value_label.setText(data.get("recording_date", "Unknown")[:10])
        self.summary_cards["duration"].value_label.setText(f"{len(data.get('hourly_probes', []))} probes")
        
        # Populate table
        probes = data.get("hourly_probes", [])
        
        self.data_table.clear()
        self.data_table.setRowCount(len(probes))
        self.data_table.setColumnCount(6)
        self.data_table.setHorizontalHeaderLabels([
            "Time Slice", "Real Time", "Students", "Engagement", "Class Mode", "Activities"
        ])
        
        for row, probe in enumerate(probes):
            self.data_table.setItem(row, 0, QTableWidgetItem(probe.get("time_slice", "")))
            self.data_table.setItem(row, 1, QTableWidgetItem(probe.get("real_world_time", "")))
            self.data_table.setItem(row, 2, QTableWidgetItem(str(probe.get("student_count_corrected", 0))))
            self.data_table.setItem(row, 3, QTableWidgetItem(f"{probe.get('avg_engagement', 0):.2f}"))
            self.data_table.setItem(row, 4, QTableWidgetItem(probe.get("class_mode", "")))
            
            # Format activities
            activities = probe.get("activity_distribution", {})
            act_str = ", ".join([f"{k}: {v}%" for k, v in activities.items()])
            self.data_table.setItem(row, 5, QTableWidgetItem(act_str))

    def _display_management_summary(self):
        """Display grouped classroom management summary report."""
        data = self._report_data
        sessions = data.get("sessions", [])

        self.summary_cards["type"].value_label.setText("Classroom Management Summary")
        self.summary_cards["count"].value_label.setText(f"{len(sessions)} sessions")
        self.summary_cards["date"].value_label.setText(data.get("recording_date", "Unknown")[:10])
        self.summary_cards["duration"].value_label.setText(data.get("classroom", "Unknown"))

        self.data_table.clear()
        self.data_table.setRowCount(len(sessions))
        self.data_table.setColumnCount(6)
        self.data_table.setHorizontalHeaderLabels([
            "Session Mode", "Time Window", "Avg Students", "Engagement", "Active/Passive", "Idle/Unobs"
        ])

        for row, session in enumerate(sessions):
            behavior = session.get("behavior_profile", {})
            active = behavior.get("active_participation", 0)
            passive = behavior.get("passive_focus", 0)
            idle = behavior.get("disengaged_idle", 0)
            unobs = behavior.get("unobservable", 0)

            self.data_table.setItem(row, 0, QTableWidgetItem(session.get("session_mode", "")))
            self.data_table.setItem(row, 1, QTableWidgetItem(session.get("time_window", "")))
            self.data_table.setItem(row, 2, QTableWidgetItem(str(session.get("avg_student_count", 0))))
            self.data_table.setItem(row, 3, QTableWidgetItem(f"{session.get('overall_engagement_score', 0):.2f}"))
            self.data_table.setItem(row, 4, QTableWidgetItem(f"active: {active}%, passive: {passive}%"))
            self.data_table.setItem(row, 5, QTableWidgetItem(f"idle: {idle}%, unobs: {unobs}%"))
            
    def _display_crossday_report(self):
        """Display an attendance report."""
        data = self._report_data
        
        # Update summary cards
        session = data.get("Session", {})
        counts = data.get("Counts", {})
        
        self.summary_cards["type"].value_label.setText("Attendance")
        self.summary_cards["count"].value_label.setText(f"{counts.get('unique_people', 0)} people")
        self.summary_cards["date"].value_label.setText(session.get("date", "Unknown"))
        self.summary_cards["duration"].value_label.setText(session.get("duration", "00:00:00"))
        
        # Populate table
        people = data.get("People", [])
        
        self.data_table.clear()
        self.data_table.setRowCount(len(people))
        self.data_table.setColumnCount(9)
        self.data_table.setHorizontalHeaderLabels([
            "ID", "Type", "Engagement ID", "Batch", "Match Confidence",
            "Entry", "Exit", "Duration", "Last 7 Days"
        ])
        
        for row, person in enumerate(people):
            self.data_table.setItem(row, 0, QTableWidgetItem(person.get("id", "")))
            
            person_type = person.get("type", "")
            type_item = QTableWidgetItem(person_type.capitalize())
            if person_type in {"returning", "returning_employee"}:
                type_item.setForeground(Qt.GlobalColor.darkGreen)
            elif person_type == "visitor":
                type_item.setForeground(Qt.GlobalColor.darkBlue)
            elif person_type == "enrolled_student":
                type_item.setForeground(Qt.GlobalColor.darkMagenta)
            self.data_table.setItem(row, 1, type_item)

            self.data_table.setItem(row, 2, QTableWidgetItem(str(person.get("engagement_id", "") or "")))
            self.data_table.setItem(row, 3, QTableWidgetItem(str(person.get("batch", "") or "")))
            confidence = person.get("confidence_score", 0.0)
            try:
                conf_text = f"{float(confidence):.3f}"
            except (TypeError, ValueError):
                conf_text = "0.000"
            self.data_table.setItem(row, 4, QTableWidgetItem(conf_text))
            
            self.data_table.setItem(row, 5, QTableWidgetItem(person.get("entry", "")))
            self.data_table.setItem(row, 6, QTableWidgetItem(person.get("exit", "")))
            
            duration_sec = person.get("duration_sec", 0)
            hours = duration_sec // 3600
            minutes = (duration_sec % 3600) // 60
            self.data_table.setItem(row, 7, QTableWidgetItem(f"{hours}h {minutes}m"))
            
            self.data_table.setItem(row, 8, QTableWidgetItem(str(person.get("present_last_7_days", 0))))
            
    def _display_generic_json(self):
        """Display generic JSON data."""
        self.summary_cards["type"].value_label.setText("JSON Report")
        self.summary_cards["count"].value_label.setText("-")
        self.summary_cards["date"].value_label.setText("-")
        self.summary_cards["duration"].value_label.setText("-")
        
        self.data_table.clear()
        self.data_table.setRowCount(0)
        
    def _populate_tree(self, data: Any, parent: QTreeWidgetItem = None):
        """Populate the JSON tree view."""
        self.json_tree.clear()
        self._add_tree_items(data, self.json_tree.invisibleRootItem())
        self.json_tree.expandToDepth(1)
        
    def _add_tree_items(self, data: Any, parent: QTreeWidgetItem, key: str = ""):
        """Recursively add items to tree."""
        if isinstance(data, dict):
            for k, v in data.items():
                if isinstance(v, (dict, list)):
                    item = QTreeWidgetItem(parent, [str(k), f"({type(v).__name__})"])
                    self._add_tree_items(v, item, k)
                else:
                    QTreeWidgetItem(parent, [str(k), str(v)])
        elif isinstance(data, list):
            for i, v in enumerate(data):
                if isinstance(v, (dict, list)):
                    item = QTreeWidgetItem(parent, [f"[{i}]", f"({type(v).__name__})"])
                    self._add_tree_items(v, item, f"[{i}]")
                else:
                    QTreeWidgetItem(parent, [f"[{i}]", str(v)])
        else:
            QTreeWidgetItem(parent, [key, str(data)])
            
    def _export_csv(self):
        """Export table data to CSV."""
        if not self._report_data:
            return
            
        default_name = os.path.splitext(os.path.basename(self._report_path))[0] + ".csv"
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export to CSV",
            default_name,
            "CSV Files (*.csv);;All Files (*.*)"
        )
        
        if not file_path:
            return
            
        try:
            with open(file_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                
                # Write headers
                headers = []
                for col in range(self.data_table.columnCount()):
                    headers.append(self.data_table.horizontalHeaderItem(col).text())
                writer.writerow(headers)
                
                # Write data
                for row in range(self.data_table.rowCount()):
                    row_data = []
                    for col in range(self.data_table.columnCount()):
                        item = self.data_table.item(row, col)
                        row_data.append(item.text() if item else "")
                    writer.writerow(row_data)
                    
            QMessageBox.information(self, "Export Complete", f"Data exported to:\n{file_path}")
            
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Could not export data:\n{e}")
            
    def _open_folder(self):
        """Open the folder containing the report."""
        if self._report_path:
            folder = os.path.dirname(self._report_path)
            os.startfile(folder)
