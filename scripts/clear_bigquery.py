#!/usr/bin/env python3
"""
Empty all TataStrive analytics tables in BigQuery (keeps dataset + table schemas).

Uses the same credentials as the desktop app (app/Creds/credentials.json or
GOOGLE_APPLICATION_CREDENTIALS).

Usage (from repository root):

    python scripts/clear_bigquery.py
    python scripts/clear_bigquery.py --yes
    python scripts/clear_bigquery.py --list-tables   # show all tables (e.g. manual copies like attendance_reports_*)
    python scripts/clear_bigquery.py --yes --drop    # DROP + recreate empty (if TRUNCATE still shows rows — streaming)

Only these tables are cleared: attendance_reports, engagement_reports,
management_summary_reports, sync_log.
If the UI still shows rows after TRUNCATE, use --drop (streaming inserts buffer).
Tables such as attendance_reports_31032026 are separate — not touched here.

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
    MANAGEMENT_SUMMARY_TABLE,
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
    parser.add_argument(
        "--list-tables",
        action="store_true",
        help="List all tables in the dataset, then exit (no truncate).",
    )
    parser.add_argument(
        "--drop",
        action="store_true",
        help="DROP the three tables and recreate empty (use if TRUNCATE leaves data; streaming buffer). Requires --yes.",
    )
    args = parser.parse_args()

    print(f"Project: {PROJECT_ID}")
    print(f"Dataset: {DATASET_ID}")
    print(f"Tables:  {ATTENDANCE_TABLE}, {ENGAGEMENT_TABLE}, {MANAGEMENT_SUMMARY_TABLE}, {SYNC_LOG_TABLE}")
    creds = args.creds or _creds_path()
    if creds:
        print(f"Creds:   {creds}")
    else:
        print("Creds:   (GOOGLE_APPLICATION_CREDENTIALS or default ADC)")

    if args.list_tables:
        try:
            svc = BigQuerySyncService(center_id="_script_", credentials_path=args.creds or "")
            client = svc._get_client()
        except Exception as e:
            print(f"Error: {e}")
            return 2
        ds_ref = f"{PROJECT_ID}.{DATASET_ID}"
        managed = {ATTENDANCE_TABLE, ENGAGEMENT_TABLE, MANAGEMENT_SUMMARY_TABLE, SYNC_LOG_TABLE}
        print(f"\nAll tables in `{ds_ref}`:")
        try:
            for t in sorted(client.list_tables(ds_ref), key=lambda x: x.table_id):
                extra = ""
                if t.table_id not in managed:
                    extra = "  ← not truncated by clear_bigquery.py (separate table / snapshot)"
                print(f"  {t.table_id}{extra}")
        except Exception as e:
            print(f"Error listing tables: {e}")
            return 2
        print(
            "\nThe app syncs only to: "
            f"{ATTENDANCE_TABLE}, {ENGAGEMENT_TABLE}, {MANAGEMENT_SUMMARY_TABLE}, {SYNC_LOG_TABLE}."
        )
        return 0

    if args.drop and not args.yes:
        print("ERROR: --drop is destructive. Run with --yes  (example: python scripts/clear_bigquery.py --yes --drop)")
        return 1

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
        if args.drop:
            print("Dropping and recreating tables (fixes streaming-insert buffer issues)...")
            result = svc.drop_and_recreate_tables()
        else:
            result = svc.truncate_all_tables()
    except Exception as e:
        print(f"Error: {e}")
        return 2

    fq_base = f"{PROJECT_ID}.{DATASET_ID}"
    print()
    for name, status in result.items():
        print(f"  {name}: {status}")
        if not str(status).startswith("truncated"):
            print(
                f"    WARNING: expected empty table. Run in BigQuery SQL (region must match dataset):"
            )
            print(f"    SELECT COUNT(*) FROM `{fq_base}.{name}`;")

    print()
    print("Verify in console: project =", PROJECT_ID, "| Explorer dataset =", DATASET_ID)
    print("Use a SQL query to confirm (Preview can lie for streamed tables): SELECT COUNT(*) FROM `...sync_log`")
    if not args.drop:
        print("If Preview still shows rows, run:  python scripts/clear_bigquery.py --yes --drop")
    print("(Looker Studio / Sheets caches separately — refresh there too.)")
    print()
    print("Done. Re-sync reports from the app when ready.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
