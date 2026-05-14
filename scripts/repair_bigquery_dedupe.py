#!/usr/bin/env python3
"""
Repair BigQuery analytics data:

1. **Deduplicate** `attendance_reports` and `engagement_reports` — keeps one row per
   natural key (same center, same report file, same logical row), preferring the
   latest `sync_timestamp`. Use when the same report was inserted more than once
   before client-side SHA256 dedupe (`~/.tatastrive/bq_synced_report_hashes.json`)
   was reliable.

2. **Fix `sync_log.rows_inserted`** — the app logs `rows_inserted` with a second
   `_now_ts()` call, so timestamps differ slightly from fact rows. This script
   recomputes counts by grouping fact tables on `(center_id, report_file, sync_timestamp)`
   and, for each `sync_log` row with `status = ok`, assigns the count from the
   nearest batch by timestamp (minimum |sync_ts − batch_ts|).

Usage (from repo root):

    python scripts/repair_bigquery_dedupe.py
    python scripts/repair_bigquery_dedupe.py --dry-run
    python scripts/repair_bigquery_dedupe.py --execute --yes

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


def _fq(table: str) -> str:
    return f"`{PROJECT_ID}.{DATASET_ID}.{table}`"


def sql_dedupe_attendance() -> str:
    t = _fq(ATTENDANCE_TABLE)
    return f"""
CREATE OR REPLACE TABLE {t} AS
SELECT * EXCEPT(rn) FROM (
  SELECT
    *,
    ROW_NUMBER() OVER (
      PARTITION BY
        center_id,
        report_file,
        COALESCE(person_id, ''),
        COALESCE(session_date, ''),
        COALESCE(entry_time, ''),
        COALESCE(exit_time, '')
      ORDER BY sync_timestamp DESC
    ) AS rn
  FROM {t}
)
WHERE rn = 1
"""


def sql_dedupe_engagement() -> str:
    t = _fq(ENGAGEMENT_TABLE)
    return f"""
CREATE OR REPLACE TABLE {t} AS
SELECT * EXCEPT(rn) FROM (
  SELECT
    *,
    ROW_NUMBER() OVER (
      PARTITION BY
        center_id,
        report_file,
        COALESCE(time_slice, ''),
        COALESCE(CAST(video_timestamp_sec AS STRING), ''),
        COALESCE(real_world_time, '')
      ORDER BY sync_timestamp DESC
    ) AS rn
  FROM {t}
)
WHERE rn = 1
"""


def sql_count_deduped_attendance() -> str:
    """Row count as if attendance dedupe were applied (read-only)."""
    t = _fq(ATTENDANCE_TABLE)
    return f"""
SELECT COUNT(1) AS c FROM (
  SELECT 1 AS x FROM (
    SELECT
      ROW_NUMBER() OVER (
        PARTITION BY
          center_id,
          report_file,
          COALESCE(person_id, ''),
          COALESCE(session_date, ''),
          COALESCE(entry_time, ''),
          COALESCE(exit_time, '')
        ORDER BY sync_timestamp DESC
      ) AS rn
    FROM {t}
  ) s
  WHERE s.rn = 1
)
"""


def sql_count_deduped_engagement() -> str:
    t = _fq(ENGAGEMENT_TABLE)
    return f"""
SELECT COUNT(1) AS c FROM (
  SELECT 1 AS x FROM (
    SELECT
      ROW_NUMBER() OVER (
        PARTITION BY
          center_id,
          report_file,
          COALESCE(time_slice, ''),
          COALESCE(CAST(video_timestamp_sec AS STRING), ''),
          COALESCE(real_world_time, '')
        ORDER BY sync_timestamp DESC
      ) AS rn
    FROM {t}
  ) s
  WHERE s.rn = 1
)
"""


def sql_merge_sync_log_attendance() -> str:
    """Set rows_inserted from deduped attendance batch counts nearest in time to each log row."""
    fa = _fq(ATTENDANCE_TABLE)
    sl = _fq(SYNC_LOG_TABLE)
    return f"""
MERGE {sl} T
USING (
  WITH batch_counts AS (
    SELECT center_id, report_file, sync_timestamp, COUNT(1) AS row_cnt
    FROM {fa}
    GROUP BY center_id, report_file, sync_timestamp
  ),
  ranked AS (
    SELECT
      s.sync_ts AS log_ts,
      s.center_id AS c_id,
      s.report_file AS r_file,
      b.row_cnt,
      ROW_NUMBER() OVER (
        PARTITION BY s.center_id, s.report_file, s.sync_ts
        ORDER BY ABS(TIMESTAMP_DIFF(s.sync_ts, b.sync_timestamp, MILLISECOND))
      ) AS rn
    FROM {sl} s
    INNER JOIN batch_counts b
      ON s.center_id = b.center_id
     AND s.report_file = b.report_file
    WHERE s.report_type = 'attendance'
      AND (s.status IS NULL OR s.status = 'ok')
  )
  SELECT log_ts, c_id, r_file, row_cnt
  FROM ranked
  WHERE rn = 1
) S
ON T.center_id = S.c_id
 AND T.report_file = S.r_file
 AND T.sync_ts = S.log_ts
 AND T.report_type = 'attendance'
WHEN MATCHED THEN
  UPDATE SET rows_inserted = S.row_cnt
"""


def sql_merge_sync_log_engagement() -> str:
    fe = _fq(ENGAGEMENT_TABLE)
    sl = _fq(SYNC_LOG_TABLE)
    return f"""
MERGE {sl} T
USING (
  WITH batch_counts AS (
    SELECT center_id, report_file, sync_timestamp, COUNT(1) AS row_cnt
    FROM {fe}
    GROUP BY center_id, report_file, sync_timestamp
  ),
  ranked AS (
    SELECT
      s.sync_ts AS log_ts,
      s.center_id AS c_id,
      s.report_file AS r_file,
      b.row_cnt,
      ROW_NUMBER() OVER (
        PARTITION BY s.center_id, s.report_file, s.sync_ts
        ORDER BY ABS(TIMESTAMP_DIFF(s.sync_ts, b.sync_timestamp, MILLISECOND))
      ) AS rn
    FROM {sl} s
    INNER JOIN batch_counts b
      ON s.center_id = b.center_id
     AND s.report_file = b.report_file
    WHERE s.report_type = 'engagement'
      AND (s.status IS NULL OR s.status = 'ok')
  )
  SELECT log_ts, c_id, r_file, row_cnt
  FROM ranked
  WHERE rn = 1
) S
ON T.center_id = S.c_id
 AND T.report_file = S.r_file
 AND T.sync_ts = S.log_ts
 AND T.report_type = 'engagement'
WHEN MATCHED THEN
  UPDATE SET rows_inserted = S.row_cnt
"""


def _scalar_count(client, sql: str, location: str) -> int:
    rows = list(client.query(sql, location=location).result())
    if not rows:
        return 0
    return int(rows[0][0] or 0)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Deduplicate BigQuery attendance/engagement tables and fix sync_log row counts."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show row counts and SQL only (default if --execute is not set).",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Run CREATE OR REPLACE dedupe and MERGE sync_log.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Required with --execute (confirmation).",
    )
    parser.add_argument(
        "--skip-dedupe",
        action="store_true",
        help="Only fix sync_log rows_inserted (no table rewrite).",
    )
    parser.add_argument(
        "--skip-sync-log",
        action="store_true",
        help="Only dedupe fact tables (no MERGE on sync_log).",
    )
    parser.add_argument(
        "--creds",
        default="",
        help="Path to service account JSON (default: same as app).",
    )
    args = parser.parse_args()
    if args.execute and args.dry_run:
        print("Use either --execute --yes or --dry-run, not both.")
        return 1

    creds = args.creds or _creds_path()
    print(f"Project: {PROJECT_ID}")
    print(f"Dataset: {DATASET_ID}")
    print(f"Creds:   {creds or '(ADC / env)'}")
    print()

    if args.execute and not args.yes:
        print("Refusing --execute without --yes. This rewrites tables and updates sync_log.")
        return 1

    try:
        svc = BigQuerySyncService(center_id="_script_", credentials_path=creds)
        client = svc._get_client()
        svc.ensure_tables()
        location = svc._dataset_location(client)
    except Exception as e:
        print(f"Error connecting: {e}")
        return 2

    fq_att = _fq(ATTENDANCE_TABLE)
    fq_eng = _fq(ENGAGEMENT_TABLE)
    fq_log = _fq(SYNC_LOG_TABLE)

    n_att_before = svc._table_row_count(client, fq_att, location)
    n_eng_before = svc._table_row_count(client, fq_eng, location)
    n_log = svc._table_row_count(client, fq_log, location)

    n_att_after_est = _scalar_count(client, sql_count_deduped_attendance(), location)
    n_eng_after_est = _scalar_count(client, sql_count_deduped_engagement(), location)

    print("Current row counts:")
    print(f"  {ATTENDANCE_TABLE}: {n_att_before}")
    print(f"  {ENGAGEMENT_TABLE}: {n_eng_before}")
    print(f"  {SYNC_LOG_TABLE}:    {n_log}")
    print()
    print("Estimated rows after dedupe (same keys, latest sync_timestamp kept):")
    print(f"  {ATTENDANCE_TABLE}: {n_att_after_est}  (remove {n_att_before - n_att_after_est})")
    print(f"  {ENGAGEMENT_TABLE}: {n_eng_after_est}  (remove {n_eng_before - n_eng_after_est})")
    print()

    if not args.execute:
        print("Dry-run only. To apply:")
        print("  python scripts/repair_bigquery_dedupe.py --execute --yes")
        print()
        print("SQL that would run:")
        if not args.skip_dedupe:
            print("--- dedupe attendance ---")
            print(sql_dedupe_attendance().strip())
            print("--- dedupe engagement ---")
            print(sql_dedupe_engagement().strip())
        if not args.skip_sync_log:
            print("--- merge sync_log (attendance) ---")
            print(sql_merge_sync_log_attendance().strip())
            print("--- merge sync_log (engagement) ---")
            print(sql_merge_sync_log_engagement().strip())
        return 0

    # --execute --yes
    try:
        if not args.skip_dedupe:
            print("Deduplicating attendance_reports...")
            svc._run_query(client, sql_dedupe_attendance(), location)
            print("Deduplicating engagement_reports...")
            svc._run_query(client, sql_dedupe_engagement(), location)
            na = svc._table_row_count(client, fq_att, location)
            ne = svc._table_row_count(client, fq_eng, location)
            print(f"  Done. attendance={na}, engagement={ne}")
        else:
            print("Skipping dedupe (--skip-dedupe).")

        if not args.skip_sync_log:
            print("Updating sync_log.rows_inserted (attendance)...")
            svc._run_query(client, sql_merge_sync_log_attendance(), location)
            print("Updating sync_log.rows_inserted (engagement)...")
            svc._run_query(client, sql_merge_sync_log_engagement(), location)
            nl = svc._table_row_count(client, fq_log, location)
            print(f"  sync_log rows (unchanged count): {nl}")
        else:
            print("Skipping sync_log (--skip-sync-log).")

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        return 2

    print()
    print("Done. Re-check counts in BigQuery console (region must match dataset).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
