"""Sync enrolled-student face embeddings from BigQuery + S3 into SQLite."""

from __future__ import annotations

import os
import sqlite3
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

DET_SIZE = (640, 640)
BQ_PROJECT = "tatastrive-269409"


@dataclass(frozen=True)
class RosterScope:
    table: str
    center_id: Optional[str]
    center_name: Optional[str]
    status: Optional[str]
    limit: Optional[int]


def _resolve_roster_scope(
    center_id: Optional[str] = None,
    center_name: Optional[str] = None,
) -> RosterScope:
    """Resolve intraining_students roster filters from args, .env, or app center_id."""
    _load_env_files()
    table = _env_value("GCP_BQ_TABLE", default="student_intraining.intraining_students").strip().strip("`")
    if table.count(".") < 2:
        table = f"{BQ_PROJECT}.{table}"

    cid = (center_id or _env_value("STUDENT_ROSTER_CENTER_ID", "GCP_CENTER_ID")).strip() or None
    cname = (center_name or _env_value("STUDENT_ROSTER_CENTER_NAME", "CENTER_NAME")).strip() or None
    if not cid and not cname:
        try:
            from app.center_catalog import resolve_center_name
            from app.config import get_config

            cfg = get_config()
            roster_center = (cfg.get("student_roster_center_name") or "").strip()
            if roster_center:
                cname = roster_center
            else:
                app_center = (cfg.get("center_id") or "").strip()
                if app_center:
                    cname = resolve_center_name(app_center) or app_center
        except Exception:
            pass

    status_raw = _env_value("STUDENT_ROSTER_STATUS")
    status = status_raw.strip() or None

    limit: Optional[int] = None
    limit_raw = _env_value("STUDENT_ROSTER_LIMIT", default="").strip()
    if limit_raw:
        try:
            limit = max(1, int(limit_raw))
        except ValueError:
            limit = None

    return RosterScope(
        table=table,
        center_id=cid,
        center_name=cname,
        status=status,
        limit=limit,
    )


def _roster_query(scope: RosterScope) -> tuple[str, Any]:
    from google.cloud import bigquery

    clauses = ["engagement_id IS NOT NULL"]
    params: list[Any] = []
    if scope.status:
        clauses.append("student_engagement_status = @status")
        params.append(bigquery.ScalarQueryParameter("status", "STRING", scope.status))
    if scope.center_id:
        clauses.append(
            "(CAST(center_id AS STRING) = @center_id OR center_id = SAFE_CAST(@center_id AS INT64))"
        )
        params.append(bigquery.ScalarQueryParameter("center_id", "STRING", scope.center_id))
    if scope.center_name:
        clauses.append("LOWER(TRIM(center_name)) = LOWER(TRIM(@center_name))")
        params.append(bigquery.ScalarQueryParameter("center_name", "STRING", scope.center_name))

    sql = (
        f"SELECT CAST(engagement_id AS STRING) AS engagement_id, batch_name "
        f"FROM `{scope.table}` WHERE {' AND '.join(clauses)}"
    )
    if scope.limit:
        sql += f" LIMIT {scope.limit}"
    job_config = bigquery.QueryJobConfig(query_parameters=params) if params else None
    return sql, job_config

LogCallback = Callable[[str, str], None]
ProgressCallback = Callable[[int, int, str], None]
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
    roster_total: int = 0
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


def runtime_roots() -> list[Path]:
    roots: list[Path] = []
    if getattr(sys, "frozen", False):
        roots.append(Path(sys.executable).resolve().parent)
    roots.append(Path(__file__).resolve().parent.parent)
    roots.append(Path.cwd())
    unique: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        key = str(root)
        if key not in seen:
            unique.append(root)
            seen.add(key)
    return unique


def resolve_enrollments_db_path(configured_path: str = "") -> Path:
    configured = (configured_path or "").strip()
    if configured:
        return Path(configured)
    return default_enrollments_db_path(runtime_roots())


def count_enrolled_students(db_path: Path) -> int:
    path = Path(db_path)
    if not path.is_file():
        return 0
    try:
        conn = sqlite3.connect(str(path))
        row = conn.execute("SELECT COUNT(*) FROM enrolled_students").fetchone()
        conn.close()
        return int(row[0]) if row else 0
    except sqlite3.Error:
        return 0


def _emit(
    log: Optional[LogCallback],
    message: str,
    level: str = "info",
    *,
    quiet: bool = False,
) -> None:
    if quiet and level not in ("error", "warning"):
        return
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


def load_enrolled_student_gallery(db_path: Path) -> dict[str, dict]:
    """Load enrolled-student embeddings from the SQLite gallery database."""
    import numpy as np

    gallery: dict[str, dict] = {}
    path = Path(db_path)
    if not path.is_file():
        return gallery
    conn = sqlite3.connect(str(path))
    try:
        rows = conn.execute(
            "SELECT engagement_id, batch_name, embedding FROM enrolled_students"
        ).fetchall()
    finally:
        conn.close()
    for engagement_id, batch_name, blob in rows:
        if not blob:
            continue
        embedding = np.frombuffer(blob, dtype=np.float32).copy()
        gallery[str(engagement_id)] = {
            "exemplars": [embedding],
            "batch": batch_name,
        }
    return gallery


def fetch_active_students(
    log: Optional[LogCallback] = None,
    *,
    center_id: Optional[str] = None,
    center_name: Optional[str] = None,
    quiet: bool = False,
) -> dict[str, dict]:
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

    scope = _resolve_roster_scope(center_id, center_name)
    if not scope.center_id and not scope.center_name:
        _emit(
            log,
            "No center_id/center_name set; roster is not scoped to a single center.",
            "warning",
            quiet=quiet,
        )
    elif scope.center_id and scope.center_name:
        _emit(
            log,
            f"BigQuery roster scoped to center_id={scope.center_id} and center_name={scope.center_name}.",
            "info",
            quiet=quiet,
        )
    elif scope.center_id:
        _emit(log, f"BigQuery roster scoped to center_id={scope.center_id}.", "info", quiet=quiet)
    else:
        _emit(log, f"BigQuery roster scoped to center_name={scope.center_name}.", "info", quiet=quiet)

    _emit(log, "Querying BigQuery for active students...", quiet=quiet)
    try:
        sql, job_config = _roster_query(scope)
        query_job = client.query(sql, job_config=job_config) if job_config else client.query(sql)
        rows = list(query_job.result())
    except Exception as exc:
        _emit(log, f"BigQuery query failed: {exc}", "error")
        return {}

    roster = {str(row["engagement_id"]): {"batch": row["batch_name"]} for row in rows}
    _emit(log, f"Found {len(roster)} active students in BigQuery.", "success", quiet=quiet)
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
    progress: Optional[ProgressCallback] = None,
    dry_run: bool = False,
    force_all: bool = False,
    session_once: bool = False,
    log_each_student: bool = True,
    quiet: bool = False,
    center_id: Optional[str] = None,
    center_name: Optional[str] = None,
) -> SyncResult:
    global _attendance_sync_done

    _load_env_files()
    out_path = Path(out_path)
    if session_once and _attendance_sync_done and not force_all:
        _emit(log, "Student roster sync skipped (already ran this session).", "info", quiet=quiet)
        return SyncResult(ok=True, output_path=out_path, message="skipped-session")

    roster = fetch_active_students(
        log,
        center_id=center_id,
        center_name=center_name,
        quiet=quiet,
    )
    if not roster:
        if out_path.exists():
            _emit(log, "BigQuery returned no students; using existing student_enrollments.db.", "warning", quiet=quiet)
            return SyncResult(ok=True, output_path=out_path, message="roster-empty-cached")
        _emit(log, "No students fetched and no local student_enrollments.db found.", "warning", quiet=quiet)
        return SyncResult(ok=False, output_path=out_path, message="roster-empty")

    s3 = _s3_client(log)
    if s3 is None:
        if out_path.exists():
            _emit(log, "S3 unavailable; using existing student_enrollments.db.", "warning", quiet=quiet)
            return SyncResult(ok=True, output_path=out_path, message="s3-unavailable-cached")
        return SyncResult(ok=False, output_path=out_path, message="s3-unavailable")

    if face_app is None:
        face_app = _load_face_app(log)
    if face_app is None:
        if out_path.exists():
            _emit(log, "InsightFace unavailable; using existing student_enrollments.db.", "warning", quiet=quiet)
            return SyncResult(ok=True, output_path=out_path, message="insightface-unavailable-cached")
        return SyncResult(ok=False, output_path=out_path, message="insightface-unavailable")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    conn = setup_db(str(out_path))
    processed = skipped = failed = no_photo = no_face = 0
    roster_total = len(roster)
    if progress:
        progress(0, roster_total, "")

    with tempfile.TemporaryDirectory(prefix="etl_student_") as tmpdir:
        for index, (eng_id, info) in enumerate(roster.items(), start=1):
            batch = info["batch"]
            if progress:
                progress(index - 1, roster_total, eng_id)
            s3_key = latest_s3_image(s3, eng_id, log)
            if not s3_key:
                no_photo += 1
                if progress:
                    progress(index, roster_total, eng_id)
                continue
            if not force_all and already_processed(conn, eng_id, s3_key):
                skipped += 1
                if progress:
                    progress(index, roster_total, eng_id)
                continue
            if dry_run:
                if log_each_student:
                    _emit(log, f"DRY-RUN: would process {eng_id} ({batch}) <- {s3_key}", "info", quiet=quiet)
                processed += 1
                if progress:
                    progress(index, roster_total, eng_id)
                continue

            local = os.path.join(tmpdir, f"{eng_id}_latest.jpg")
            if not download_s3_image(s3, s3_key, local, log):
                failed += 1
                if progress:
                    progress(index, roster_total, eng_id)
                continue

            emb = extract_embedding(face_app, local)
            if emb is None:
                if not quiet:
                    _emit(log, f"No face detected: {eng_id}", "warning")
                no_face += 1
                if progress:
                    progress(index, roster_total, eng_id)
                continue

            upsert_student(conn, eng_id, batch, s3_key, emb)
            if log_each_student:
                _emit(log, f"Saved {eng_id} | {batch}", "success", quiet=quiet)
            processed += 1
            if progress:
                progress(index, roster_total, eng_id)

    conn.close()
    if session_once:
        _attendance_sync_done = True

    if not quiet or processed or failed:
        _emit(
            log,
            (
                "Student roster sync complete — "
                f"updated {processed}, skipped {skipped}, "
                f"no photo {no_photo}, no face {no_face}, download errors {failed}."
            ),
            "success",
            quiet=quiet,
        )
    return SyncResult(
        ok=True,
        processed=processed,
        skipped=skipped,
        no_photo=no_photo,
        no_face=no_face,
        failed=failed,
        roster_total=roster_total,
        output_path=out_path,
        message="complete",
    )
