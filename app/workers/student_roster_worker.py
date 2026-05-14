"""Background worker for preparing the student enrollment SQLite database."""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import QObject, pyqtSignal

from app.student_embeddings_sync import SyncResult, sync_student_enrollments


class StudentRosterSyncWorker(QObject):
    log_message = pyqtSignal(str, str)
    progress = pyqtSignal(int, int)
    finished = pyqtSignal(object)

    def __init__(
        self,
        out_path: Path,
        *,
        center_name: str = "",
        force_all: bool = False,
    ):
        super().__init__()
        self._out_path = Path(out_path)
        self._center_name = center_name.strip()
        self._force_all = force_all

    def run(self) -> None:
        def log(message: str, level: str = "info") -> None:
            self.log_message.emit(message, level)

        def on_progress(done: int, total: int, _eng_id: str = "") -> None:
            self.progress.emit(done, total)

        result = sync_student_enrollments(
            self._out_path,
            log=log,
            progress=on_progress,
            force_all=self._force_all,
            center_name=self._center_name or None,
        )
        self.finished.emit(result)
