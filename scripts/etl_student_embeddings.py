"""
ETL: Sync enrolled-student face embeddings from BigQuery + AWS S3 → local SQLite.

What it does:
  1. Queries BigQuery for all active in-training students (engagement_id, batch_name).
  2. For each student, checks AWS S3 for their latest "*_StudentPicture_*" photo.
  3. Delta-skips anyone whose photo hasn't changed since the last run.
  4. Downloads new/updated photos, generates InsightFace buffalo_l embeddings, saves to SQLite.

The output SQLite file (`student_enrollments.db`) is automatically discovered and loaded
by the attendance worker (crossday_worker.py) — no manual configuration needed when
placed in the project root, Models/, or alongside the executable.

Usage:
    python scripts/etl_student_embeddings.py
    python scripts/etl_student_embeddings.py --out Models/student_enrollments.db
    python scripts/etl_student_embeddings.py --dry-run          # preview only, no downloads
    python scripts/etl_student_embeddings.py --force-all        # re-embed even unchanged photos

Credentials:
  BigQuery: resolved via the same logic as the attendance app
            (app/Creds/credentials.json, credentials.json next to exe, or
             GOOGLE_APPLICATION_CREDENTIALS env var).
  AWS S3:   set AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / AWS_DEFAULT_REGION as env vars,
            or configure ~/.aws/credentials.  Override BUCKET_NAME / AWS_REGION below.
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys
import tempfile
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# ---------------------------------------------------------------------------
# Config (override via env vars or edit here)
# ---------------------------------------------------------------------------
BUCKET_NAME = os.environ.get("ETL_S3_BUCKET", "bucket-name")
AWS_REGION  = os.environ.get("ETL_AWS_REGION", "ap-south-1")

BQ_PROJECT  = "tatastrive-269409"
BQ_QUERY    = """
    SELECT engagement_id, batch_name
    FROM `tatastrive-269409.student_intraining.intraining_students`
    WHERE student_engagement_status = 'intraining'
    LIMIT 2000
"""

DEFAULT_OUT = _ROOT / "student_enrollments.db"
DET_SIZE    = (640, 640)


# ---------------------------------------------------------------------------
# Credential helpers (reuse app logic for BigQuery)
# ---------------------------------------------------------------------------
def _bq_creds_path() -> str:
    try:
        from app.bigquery_sync import _creds_path
        return _creds_path()
    except Exception:
        pass
    app_dir = _ROOT / "app"
    for c in [
        _ROOT / "credentials.json",
        _ROOT / "Creds" / "credentials.json",
        app_dir / "Creds" / "credentials.json",
        app_dir / "Creds" / "credentials (1).json",
        app_dir / "creds" / "credentials.json",
    ]:
        if c.exists():
            return str(c)
    return os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "")


# ---------------------------------------------------------------------------
# BigQuery
# ---------------------------------------------------------------------------
def fetch_active_students() -> dict[str, dict]:
    """Returns {engagement_id: {"batch": batch_name}}."""
    try:
        from google.cloud import bigquery
        from google.oauth2 import service_account
    except ImportError:
        print("[ETL] ERROR: google-cloud-bigquery not installed.")
        print("      Run: pip install google-cloud-bigquery")
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

    print("[ETL] Querying BigQuery for active students...")
    try:
        rows = list(client.query(BQ_QUERY).result())
    except Exception as e:
        print(f"[ETL] ERROR: BigQuery query failed: {e}")
        return {}

    roster = {str(r["engagement_id"]): {"batch": r["batch_name"]} for r in rows}
    print(f"[ETL] Found {len(roster)} active students in BigQuery.")
    return roster


# ---------------------------------------------------------------------------
# AWS S3
# ---------------------------------------------------------------------------
def _s3_client():
    try:
        import boto3
    except ImportError:
        print("[ETL] ERROR: boto3 not installed.")
        print("      Run: pip install boto3")
        return None
    try:
        return boto3.client(
            "s3",
            aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
            region_name=AWS_REGION,
        )
    except Exception as e:
        print(f"[ETL] ERROR: AWS S3 connection failed: {e}")
        return None


def latest_s3_image(s3, eng_id: str) -> str | None:
    """Returns the S3 key of the newest StudentPicture_ photo, or None."""
    try:
        resp = s3.list_objects_v2(Bucket=BUCKET_NAME, Prefix=f"{eng_id}/")
        contents = resp.get("Contents", [])
    except Exception as e:
        print(f"[ETL]   S3 list failed for {eng_id}: {e}")
        return None
    valid = [
        obj["Key"] for obj in contents
        if f"{eng_id}_StudentPicture_" in obj["Key"]
        and obj["Key"].lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    return sorted(valid, reverse=True)[0] if valid else None


def download_s3_image(s3, key: str, dest: str) -> bool:
    try:
        s3.download_file(BUCKET_NAME, key, dest)
        return True
    except Exception as e:
        print(f"[ETL]   Download failed ({key}): {e}")
        return False


# ---------------------------------------------------------------------------
# InsightFace embedding
# ---------------------------------------------------------------------------
def _load_face_app():
    try:
        import cv2  # noqa: F401
        from insightface.app import FaceAnalysis
    except ImportError as e:
        print(f"[ETL] ERROR: insightface / opencv not available: {e}")
        return None

    import torch
    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if torch.cuda.is_available()
        else ["CPUExecutionProvider"]
    )

    # Resolve buffalo_l root (same logic as crossday_worker)
    for root in [_ROOT, _ROOT / "Models", Path(sys.executable).parent]:
        if (root / "models" / "buffalo_l").exists():
            fa = FaceAnalysis(name="buffalo_l", root=str(root), providers=providers)
            fa.prepare(ctx_id=0, det_size=DET_SIZE)
            print(f"[ETL] InsightFace loaded (buffalo_l from {root/'models'/'buffalo_l'})")
            return fa

    fa = FaceAnalysis(name="buffalo_l", providers=providers)
    fa.prepare(ctx_id=0, det_size=DET_SIZE)
    print("[ETL] InsightFace loaded (buffalo_l from default ~/.insightface)")
    return fa


def extract_embedding(fa, image_path: str):
    """Returns numpy embedding of the largest face in image, or None."""
    import cv2
    img = cv2.imread(image_path)
    if img is None:
        return None
    faces = fa.get(img)
    if not faces:
        return None
    # Pick the largest face (ID photos may have background people)
    faces.sort(key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]), reverse=True)
    return faces[0].embedding


# ---------------------------------------------------------------------------
# SQLite student DB
# ---------------------------------------------------------------------------
def setup_db(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS enrolled_students (
            engagement_id      TEXT PRIMARY KEY,
            batch_name         TEXT,
            latest_s3_filename TEXT,
            embedding          BLOB
        )
    """)
    conn.commit()
    return conn


def already_processed(conn: sqlite3.Connection, eng_id: str, s3_key: str) -> bool:
    row = conn.execute(
        "SELECT latest_s3_filename FROM enrolled_students WHERE engagement_id = ?",
        (eng_id,)
    ).fetchone()
    return bool(row and row[0] == s3_key)


def upsert_student(conn: sqlite3.Connection, eng_id: str, batch: str, s3_key: str, emb) -> None:
    import numpy as np
    conn.execute(
        """REPLACE INTO enrolled_students
           (engagement_id, batch_name, latest_s3_filename, embedding)
           VALUES (?, ?, ?, ?)""",
        (eng_id, batch, s3_key, np.array(emb, dtype=np.float32).tobytes()),
    )
    conn.commit()


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def run(out_path: Path, dry_run: bool = False, force_all: bool = False) -> int:
    roster = fetch_active_students()
    if not roster:
        print("[ETL] No students fetched — aborting.")
        return 1

    s3 = _s3_client()
    if s3 is None:
        return 1

    fa = _load_face_app()
    if fa is None:
        return 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    conn = setup_db(str(out_path))

    processed = skipped = failed = no_photo = no_face = 0

    with tempfile.TemporaryDirectory(prefix="etl_student_") as tmpdir:
        for eng_id, info in roster.items():
            batch = info["batch"]
            s3_key = latest_s3_image(s3, eng_id)
            if not s3_key:
                no_photo += 1
                continue

            if not force_all and already_processed(conn, eng_id, s3_key):
                skipped += 1
                continue

            if dry_run:
                print(f"[ETL] DRY-RUN: would process {eng_id} ({batch}) <- {s3_key}")
                processed += 1
                continue

            local = os.path.join(tmpdir, f"{eng_id}_latest.jpg")
            if not download_s3_image(s3, s3_key, local):
                failed += 1
                continue

            emb = extract_embedding(fa, local)
            if emb is None:
                print(f"[ETL]   No face detected: {eng_id}")
                no_face += 1
                continue

            upsert_student(conn, eng_id, batch, s3_key, emb)
            print(f"[ETL]   Saved {eng_id} | {batch}")
            processed += 1

    conn.close()

    print()
    print("=" * 55)
    print("ETL COMPLETE")
    print("=" * 55)
    print(f"  Processed / updated : {processed}")
    print(f"  Skipped (unchanged) : {skipped}")
    print(f"  No S3 photo found   : {no_photo}")
    print(f"  No face in photo    : {no_face}")
    print(f"  Download errors     : {failed}")
    print(f"  Output DB           : {out_path.resolve()}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Sync student embeddings: BigQuery + S3 → SQLite")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT,
                        help=f"Output SQLite path (default: {DEFAULT_OUT})")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview only — no downloads, no DB writes")
    parser.add_argument("--force-all", action="store_true",
                        help="Re-embed all students even if photo is unchanged")
    args = parser.parse_args()
    return run(args.out, dry_run=args.dry_run, force_all=args.force_all)


if __name__ == "__main__":
    raise SystemExit(main())
