"""Sync enrolled-student embeddings: BigQuery roster + S3 photos -> SQLite."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import dotenv

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

dotenv.load_dotenv(_ROOT / ".env", override=False)

from app.student_embeddings_sync import sync_student_enrollments


def _log(message: str, level: str = "info") -> None:
    if level == "success":
        print(f"[OK] {message}")
    elif level == "warning":
        print(f"[WARN] {message}")
    elif level == "error":
        print(f"[ERROR] {message}")
    else:
        print(message)


def run_pipeline() -> int:
    db_path = Path(os.environ.get("STUDENT_DB_PATH", "student_enrollments.db"))
    if not db_path.is_absolute():
        db_path = _ROOT / db_path

    print("Loading InsightFace model for S3 images...")
    result = sync_student_enrollments(db_path, log=_log)

    if not result.ok:
        if result.message == "roster-empty":
            print("[WARN] No IDs fetched from BigQuery. Halting pipeline.")
        else:
            print(f"[WARN] Pipeline failed: {result.message}")
        return 1

    print()
    print(f"Pipeline complete. {result.output_path or db_path} is ready for CCTV inference.")
    print(f"   Updated: {result.processed} | Skipped: {result.skipped} | "
          f"No photo: {result.no_photo} | No face: {result.no_face} | Errors: {result.failed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run_pipeline())
