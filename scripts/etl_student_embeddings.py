"""
ETL: Sync enrolled-student face embeddings from BigQuery + AWS S3 → local SQLite.

Usage:
    python scripts/etl_student_embeddings.py
    python scripts/etl_student_embeddings.py --out Models/student_enrollments.db
    python scripts/etl_student_embeddings.py --dry-run
    python scripts/etl_student_embeddings.py --force-all
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from app.student_embeddings_sync import (  # noqa: E402
    default_enrollments_db_path,
    sync_student_enrollments,
)

DEFAULT_OUT = _ROOT / "student_enrollments.db"


def main() -> int:
    parser = argparse.ArgumentParser(description="Sync student embeddings: BigQuery + S3 → SQLite")
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help=f"Output SQLite path (default: {DEFAULT_OUT})",
    )
    parser.add_argument("--dry-run", action="store_true", help="Preview only — no downloads, no DB writes")
    parser.add_argument("--force-all", action="store_true", help="Re-embed all students even if photo is unchanged")
    parser.add_argument("--center-id", default="", help="BigQuery center_id filter (intraining_students.center_id)")
    parser.add_argument("--center-name", default="", help="BigQuery center_name filter (matches app center_id by default)")
    parser.add_argument(
        "--status",
        default="",
        help="Optional student_engagement_status filter (omit for all statuses at the center)",
    )
    args = parser.parse_args()

    if args.status.strip():
        os.environ["STUDENT_ROSTER_STATUS"] = args.status.strip()

    result = sync_student_enrollments(
        args.out,
        dry_run=args.dry_run,
        force_all=args.force_all,
        center_id=args.center_id.strip() or None,
        center_name=args.center_name.strip() or None,
    )
    if not result.ok:
        print(f"[ETL] Sync failed: {result.message}")
        return 1

    out_path = result.output_path or args.out
    print()
    print("=" * 55)
    print("ETL COMPLETE")
    print("=" * 55)
    print(f"  Processed / updated : {result.processed}")
    print(f"  Skipped (unchanged) : {result.skipped}")
    print(f"  No S3 photo found   : {result.no_photo}")
    print(f"  No face in photo    : {result.no_face}")
    print(f"  Download errors     : {result.failed}")
    print(f"  Output DB           : {out_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
