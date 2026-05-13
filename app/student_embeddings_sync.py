"""Sync enrolled-student face embeddings from BigQuery + S3 into SQLite."""

from __future__ import annotations

import os
import sqlite3
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

DET_SIZE = (640, 640)
BQ_PROJECT = "tatastrive-269409"
BQ_QUERY = """
    SELECT engagement_id, batch_name
    FROM `tatastrive-269409.student_intraining.intraining_students`
    WHERE student_engagement_status = 'intraining'
    LIMIT 2000
"""

LogCallback = Callable[[str, str], None]
_attendance_sync_done = False


def _load_env_files() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return

    load_dotenv()
    root = Path(__file__).resolve().parent.parent
    load_dotenv(root / ".env", override=False)
    if getattr(sys, "frozen", False):
        load_dotenv(Path(sys.executable).resolve().parent / ".env", override=False)


def _env_value(*names: str, default: str = "") -> str:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return default


def _bucket_name() -> str:
    return _env_value("ETL_S3_BUCKET", "BUCKET_NAME", default="bucket-name")


def _aws_region() -> str:
    return _env_value("ETL_AWS_REGION", "AWS_REGION", "AWS_DEFAULT_REGION", default="ap-south-1")


@dataclass
class SyncResult:
    ok: bool
    processed: int = 0
    skipped: int = 0
    no_photo: int = 0
    no_face: int = 0
    failed: int = 0
    output_path: Optional[Path] = None
    message: str = ""


def default_enrollments_db_path(roots: list[Path]) -> Path:
    for root in roots:
        for candidate in (root / "student_enrollments.db", root / "Models" / "student_enrollments.db"):
            if candidate.exists():
                return candidate
    if roots:
        return roots[0] / "student_enrollments.db"
    return Path.cwd() / "student_enrollments.db"


def _emit(log: Optional[LogCallback], message: str, level: str = "info") -> None:
    if log:
        log(message, level)
        return
    prefix = "[ETL] "
    if level == "error":
        print(f"{prefix}ERROR: {message}")
    else:
        print(f"{prefix}{message}")


def _bq_creds_path() -> str:
    try:
        from app.bigquery_sync import _creds_path

        return _creds_path()
    except Exception:
        pass

    root = Path(__file__).resolve().parent.parent
    app_dir = root / "app"
    for candidate in (
        root / "credentials.json",
        root / "Creds" / "credentials.json",
        app_dir / "Creds" / "credentials.json",
        app_dir / "Creds" / "credentials (1).json",
        app_dir / "creds" / "credentials.json",
    ):
        if candidate.exists():
            return str(candidate)
    return os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "")


def fetch_active_students(log: Optional[LogCallback] = None) -> dict[str, dict]:
    try:
        from google.cloud import bigquery
        from google.oauth2 import service_account
    except ImportError:
        _emit(log, "google-cloud-bigquery not installed.", "error")
        return {}

    creds_path = _bq_creds_path()
    if creds_path and Path(creds_path).exists():
        creds = service_account.Credentials.from_service_account_file(
            creds_path,
            scopes=["https://www.googleapis.com/auth/cloud-platform"],
        )
        client = bigquery.Client(project=BQ_PROJECT, credentials=creds)
    else:
        client = bigquery.Client(project=BQ_PROJECT)

    _emit(log, "Querying BigQuery for active students...")
    try:
        rows = list(client.query(BQ_QUERY).result())
    except Exception as exc:
        _emit(log, f"BigQuery query failed: {exc}", "error")
        return {}

    roster = {str(row["engagement_id"]): {"batch": row["batch_name"]} for row in rows}
    _emit(log, f"Found {len(roster)} active students in BigQuery.", "success")
    return roster


def _s3_client(log: Optional[LogCallback] = None):
    try:
        import boto3
    except ImportError:
        _emit(log, "boto3 not installed.", "error")
        return None
    try:
        return boto3.client(
            "s3",
            aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
            region_name=_aws_region(),
        )
    except Exception as exc:
        _emit(log, f"AWS S3 connection failed: {exc}", "error")
        return None


def latest_s3_image(s3, eng_id: str, log: Optional[LogCallback] = None) -> Optional[str]:
    try:
        resp = s3.list_objects_v2(Bucket=_bucket_name(), Prefix=f"{eng_id}/")
        contents = resp.get("Contents", [])
    except Exception as exc:
        _emit(log, f"S3 list failed for {eng_id}: {exc}", "warning")
        return None
    valid = [
        obj["Key"]
        for obj in contents
        if f"{eng_id}_StudentPicture_" in obj["Key"]
        and obj["Key"].lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    return sorted(valid, reverse=True)[0] if valid else None


def download_s3_image(s3, key: str, dest: str, log: Optional[LogCallback] = None) -> bool:
    try:
        s3.download_file(_bucket_name(), key, dest)
        return True
    except Exception as exc:
        _emit(log, f"Download failed ({key}): {exc}", "warning")
        return False


def _load_face_app(log: Optional[LogCallback] = None):
    try:
        import cv2  # noqa: F401
        from insightface.app import FaceAnalysis
    except ImportError as exc:
        _emit(log, f"insightface / opencv not available: {exc}", "error")
        return None

    import torch

    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if torch.cuda.is_available()
        else ["CPUExecutionProvider"]
    )
    root = Path(__file__).resolve().parent.parent
    for model_root in (root, root / "Models", Path(sys.executable).parent):
        if (model_root / "models" / "buffalo_l").exists():
            face_app = FaceAnalysis(name="buffalo_l", root=str(model_root), providers=providers)
            face_app.prepare(ctx_id=0, det_size=DET_SIZE)
            _emit(log, f"InsightFace loaded (buffalo_l from {model_root / 'models' / 'buffalo_l'})", "info")
            return face_app

    face_app = FaceAnalysis(name="buffalo_l", providers=providers)
    face_app.prepare(ctx_id=0, det_size=DET_SIZE)
    _emit(log, "InsightFace loaded (buffalo_l from default ~/.insightface)", "info")
    return face_app


def extract_embedding(face_app, image_path: str):
    import cv2

    img = cv2.imread(image_path)
    if img is None:
        return None
    faces = face_app.get(img)
    if not faces:
        return None
    faces.sort(key=lambda face: (face.bbox[2] - face.bbox[0]) * (face.bbox[3] - face.bbox[1]), reverse=True)
    return faces[0].embedding


def setup_db(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS enrolled_students (
            engagement_id      TEXT PRIMARY KEY,
            batch_name         TEXT,
            latest_s3_filename TEXT,
            embedding          BLOB
        )
        """
    )
    conn.commit()
    return conn


def already_processed(conn: sqlite3.Connection, eng_id: str, s3_key: str) -> bool:
    row = conn.execute(
        "SELECT latest_s3_filename FROM enrolled_students WHERE engagement_id = ?",
        (eng_id,),
    ).fetchone()
    return bool(row and row[0] == s3_key)


def upsert_student(conn: sqlite3.Connection, eng_id: str, batch: str, s3_key: str, emb) -> None:
    import numpy as np

    conn.execute(
        """
        REPLACE INTO enrolled_students
           (engagement_id, batch_name, latest_s3_filename, embedding)
           VALUES (?, ?, ?, ?)
        """,
        (eng_id, batch, s3_key, np.array(emb, dtype=np.float32).tobytes()),
    )
    conn.commit()


def sync_student_enrollments(
    out_path: Path,
    *,
    face_app=None,
    log: Optional[LogCallback] = None,
    dry_run: bool = False,
    force_all: bool = False,
    session_once: bool = False,
) -> SyncResult:
    global _attendance_sync_done

    _load_env_files()
    out_path = Path(out_path)
    if session_once and _attendance_sync_done and not force_all:
        _emit(log, "Student roster sync skipped (already ran this session).", "info")
        return SyncResult(ok=True, output_path=out_path, message="skipped-session")

    roster = fetch_active_students(log)
    if not roster:
        if out_path.exists():
            _emit(log, "BigQuery returned no students; using existing student_enrollments.db.", "warning")
            return SyncResult(ok=True, output_path=out_path, message="roster-empty-cached")
        _emit(log, "No students fetched and no local student_enrollments.db found.", "warning")
        return SyncResult(ok=False, output_path=out_path, message="roster-empty")

    s3 = _s3_client(log)
    if s3 is None:
        if out_path.exists():
            _emit(log, "S3 unavailable; using existing student_enrollments.db.", "warning")
            return SyncResult(ok=True, output_path=out_path, message="s3-unavailable-cached")
        return SyncResult(ok=False, output_path=out_path, message="s3-unavailable")

    if face_app is None:
        face_app = _load_face_app(log)
    if face_app is None:
        if out_path.exists():
            _emit(log, "InsightFace unavailable; using existing student_enrollments.db.", "warning")
            return SyncResult(ok=True, output_path=out_path, message="insightface-unavailable-cached")
        return SyncResult(ok=False, output_path=out_path, message="insightface-unavailable")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    conn = setup_db(str(out_path))
    processed = skipped = failed = no_photo = no_face = 0

    with tempfile.TemporaryDirectory(prefix="etl_student_") as tmpdir:
        for eng_id, info in roster.items():
            batch = info["batch"]
            s3_key = latest_s3_image(s3, eng_id, log)
            if not s3_key:
                no_photo += 1
                continue
            if not force_all and already_processed(conn, eng_id, s3_key):
                skipped += 1
                continue
            if dry_run:
                _emit(log, f"DRY-RUN: would process {eng_id} ({batch}) <- {s3_key}", "info")
                processed += 1
                continue

            local = os.path.join(tmpdir, f"{eng_id}_latest.jpg")
            if not download_s3_image(s3, s3_key, local, log):
                failed += 1
                continue

            emb = extract_embedding(face_app, local)
            if emb is None:
                _emit(log, f"No face detected: {eng_id}", "warning")
                no_face += 1
                continue

            upsert_student(conn, eng_id, batch, s3_key, emb)
            _emit(log, f"Saved {eng_id} | {batch}", "success")
            processed += 1

    conn.close()
    if session_once:
        _attendance_sync_done = True

    _emit(
        log,
        (
            "Student roster sync complete — "
            f"updated {processed}, skipped {skipped}, "
            f"no photo {no_photo}, no face {no_face}, download errors {failed}."
        ),
        "success",
    )
    return SyncResult(
        ok=True,
        processed=processed,
        skipped=skipped,
        no_photo=no_photo,
        no_face=no_face,
        failed=failed,
        output_path=out_path,
        message="complete",
    )
