#!/usr/bin/env python3
"""
Empty all TataStrive analytics tables in BigQuery (keeps dataset + table schemas).

Uses the same credentials as the desktop app (app/Creds/credentials.json or
GOOGLE_APPLICATION_CREDENTIALS).

Usage (from repository root):

    python scripts/clear_bigquery.py
    python scripts/clear_bigquery.py --yes

Requires: pip install google-cloud-bigquery google-auth
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.bigquery_sync import (  # noqa: E402
    ATTENDANCE_TABLE,
    DATASET_ID,
    ENGAGEMENT_TABLE,
    PROJECT_ID,
    SYNC_LOG_TABLE,
    BigQuerySyncService,
    _creds_path,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Truncate TataStrive BigQuery analytics tables.")
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Do not ask for confirmation.",
    )
    parser.add_argument(
        "--creds",
        default="",
        help="Path to service account JSON (default: same search as the app).",
    )
    args = parser.parse_args()

    print(f"Project: {PROJECT_ID}")
    print(f"Dataset: {DATASET_ID}")
    print(f"Tables:  {ATTENDANCE_TABLE}, {ENGAGEMENT_TABLE}, {SYNC_LOG_TABLE}")
    creds = args.creds or _creds_path()
    if creds:
        print(f"Creds:   {creds}")
    else:
        print("Creds:   (GOOGLE_APPLICATION_CREDENTIALS or default ADC)")

    if not args.yes:
        try:
            confirm = input("Type YES to truncate all rows in these tables: ").strip()
        except EOFError:
            confirm = ""
        if confirm != "YES":
            print("Aborted.")
            return 1

    try:
        svc = BigQuerySyncService(center_id="_script_", credentials_path=args.creds or "")
        result = svc.truncate_all_tables()
    except Exception as e:
        print(f"Error: {e}")
        return 2

    for name, status in result.items():
        print(f"  {name}: {status}")
    print("Done. Re-sync reports from the app when ready.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
