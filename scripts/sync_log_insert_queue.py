#!/usr/bin/env python3
"""
Insert (or replace) rows in BigQuery sync_log only — does not touch attendance_reports.

Mirrors the app behavior from main_window._on_crossday_complete:
  - videos_in_queue = crossday_tab.pending_video_queue_count() at sync time
  - sync_ts uses the same format as BigQuerySyncService._now_ts() (IST with +05:30 suffix).
  - rows_inserted = len(rows) for that report sync.

**Why not DELETE?** BigQuery often rejects DELETE on sync_log when streaming inserts are
still in the buffer.  **--replace** instead rewrites the table with:

    CREATE OR REPLACE TABLE sync_log AS
    SELECT * FROM sync_log WHERE NOT (matching old rows …)

Then inserts **5** sync_log lines with **rows_inserted** totalling **16** (default split
``4,4,3,3,2`` for queues ``54,53,52,51,50``), matching a 16-person report across five syncs.

Examples:

    python scripts/sync_log_insert_queue.py --replace --yes
    python scripts/sync_log_insert_queue.py --purge-only --yes   # remove matching rows only, no insert

Requires: pip install google-cloud-bigquery google-auth
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from google.cloud import bigquery  # noqa: E402

from app.bigquery_sync import (  # noqa: E402
    BigQuerySyncService,
    DATASET_ID,
    IST,
    PROJECT_ID,
    SYNC_LOG_TABLE,
    _creds_path,
    sync_timestamp_iso,
)
from app.config import get_config  # noqa: E402

# Defaults: 5 sync_log rows; rows_inserted must sum to 16 for a full D14-style report.
DEFAULT_QUEUES = "54,53,52,51,50"
DEFAULT_ROWS = "4,4,3,3,2"  # sum = 16


def _now_ts(dt: datetime) -> str:
    """Match BigQuerySyncService._now_ts() exactly (IST, ms precision, +05:30 suffix)."""
    return sync_timestamp_iso(dt)


def _parse_ts_for_bq(s: str) -> datetime:
    """Parse sync timestamp string for BigQuery TIMESTAMP query parameters."""
    s = s.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    return datetime.fromisoformat(s)


def rewrite_sync_log_remove_slots(
    client: bigquery.Client,
    location: str,
    center_id: str,
    report_file: str,
    queues: list[int],
) -> None:
    """
    Remove all sync_log rows matching center_id + report_file + videos_in_queue IN (queues).
    Uses CREATE OR REPLACE TABLE … AS SELECT (works when DELETE hits streaming-buffer errors).
    """
    fq = f"`{PROJECT_ID}.{DATASET_ID}.{SYNC_LOG_TABLE}`"
    in_list = ",".join(str(q) for q in queues)
    sql = f"""
CREATE OR REPLACE TABLE {fq} AS
SELECT * FROM {fq}
WHERE NOT (
  center_id = @center_id
  AND report_file = @report_file
  AND videos_in_queue IN ({in_list})
)
"""
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("center_id", "STRING", center_id),
            bigquery.ScalarQueryParameter("report_file", "STRING", report_file),
        ]
    )
    job = client.query(sql, job_config=job_config, location=location)
    job.result()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Insert sync_log rows with videos_in_queue and per-sync rows_inserted (attendance_reports unchanged)."
    )
    parser.add_argument(
        "--center-id",
        default="",
        help="center_id (default: ~/.tatastrive/config.json center_id).",
    )
    parser.add_argument(
        "--creds",
        default="",
        help="Service account JSON path.",
    )
    parser.add_argument(
        "--report-file",
        default="recovery_D14_attendance_report.json",
        help="report_file field (basename only).",
    )
    parser.add_argument(
        "--queues",
        default=DEFAULT_QUEUES,
        help=f"videos_in_queue per log, oldest→newest (default: {DEFAULT_QUEUES}).",
    )
    parser.add_argument(
        "--rows",
        default=DEFAULT_ROWS,
        help=f"rows_inserted per log, same length as --queues (default: {DEFAULT_ROWS} → total 16).",
    )
    parser.add_argument(
        "--seconds-apart",
        type=float,
        default=1.0,
        help="Seconds between consecutive sync_ts values (default 1.0).",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Rewrite sync_log to drop rows matching center_id, report_file, and these videos_in_queue, then insert.",
    )
    parser.add_argument(
        "--purge-only",
        action="store_true",
        help="Only rewrite sync_log to remove those rows (no insert).",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Required with --replace or --purge-only.",
    )
    args = parser.parse_args()

    if (args.replace or args.purge_only) and not args.yes:
        print("Refusing --replace / --purge-only without --yes (rewrites sync_log).")
        return 1
    if args.replace and args.purge_only:
        print("Use either --replace or --purge-only, not both.")
        return 1

    center_id = (args.center_id or "").strip() or (get_config().get("center_id") or "").strip()
    if not center_id:
        print("Error: set --center-id or center_id in config.")
        return 1

    try:
        queues = [int(x.strip()) for x in args.queues.split(",") if x.strip()]
        row_counts = [int(x.strip()) for x in args.rows.split(",") if x.strip()]
    except ValueError:
        print("Error: --queues and --rows must be comma-separated integers.")
        return 1
    if not queues:
        print("Error: no queue values.")
        return 1
    if not args.purge_only and len(queues) != len(row_counts):
        print("Error: --queues and --rows must have the same number of values.")
        return 1

    if not args.purge_only:
        total_rows = sum(row_counts)
        print(f"rows_inserted per log: {row_counts}  (total {total_rows})")
        if len(row_counts) != 5:
            print(f"Note: expected 5 sync_log lines; got {len(row_counts)}.")
        elif total_rows != 16:
            print(f"Note: rows_inserted total is {total_rows}, not 16 — adjust --rows if needed.")

    creds = args.creds or _creds_path()
    svc = BigQuerySyncService(center_id=center_id, credentials_path=creds)
    svc.ensure_tables()
    client = svc._get_client()
    location = svc._dataset_location(client)

    if args.replace or args.purge_only:
        print(
            f"Rewriting sync_log: removing rows where center_id={center_id!r}, "
            f"report_file={args.report_file!r}, videos_in_queue IN ({args.queues})…"
        )
        try:
            rewrite_sync_log_remove_slots(
                client, location, center_id, args.report_file, queues
            )
            print("Removed matching sync_log rows (table rewrite).")
        except Exception as e:
            print(f"Rewrite failed: {e}")
            return 2

    if args.purge_only:
        return 0

    # Insert (replace or normal)
    n = len(queues)
    span = max(0.0, (n - 1) * args.seconds_apart)
    base = datetime.now(IST) - timedelta(seconds=span)
    rows = []
    for i, (q, rc) in enumerate(zip(queues, row_counts)):
        ts = base + timedelta(seconds=i * args.seconds_apart)
        rows.append(
            {
                "center_id": center_id,
                "sync_ts": _now_ts(ts),
                "report_file": args.report_file,
                "report_type": "attendance",
                "rows_inserted": rc,
                "status": "ok",
                "error_msg": "",
                "videos_in_queue": q,
            }
        )

    # Use DML INSERT (not streaming insert_rows_json): after CREATE OR REPLACE TABLE,
    # streaming insertAll can return "Table is truncated" until the table settles.
    for r in rows:
        sql_ins = f"""
INSERT INTO `{PROJECT_ID}.{DATASET_ID}.{SYNC_LOG_TABLE}`
  (center_id, sync_ts, report_file, report_type, rows_inserted, status, error_msg, videos_in_queue)
VALUES (
  @center_id,
  @sync_ts,
  @report_file,
  @report_type,
  @rows_inserted,
  @status,
  @error_msg,
  @videos_in_queue
)
"""
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("center_id", "STRING", r["center_id"]),
                bigquery.ScalarQueryParameter("sync_ts", "TIMESTAMP", _parse_ts_for_bq(r["sync_ts"])),
                bigquery.ScalarQueryParameter("report_file", "STRING", r["report_file"]),
                bigquery.ScalarQueryParameter("report_type", "STRING", r["report_type"]),
                bigquery.ScalarQueryParameter("rows_inserted", "INT64", r["rows_inserted"]),
                bigquery.ScalarQueryParameter("status", "STRING", r["status"]),
                bigquery.ScalarQueryParameter("error_msg", "STRING", r["error_msg"] or ""),
                bigquery.ScalarQueryParameter("videos_in_queue", "INT64", r["videos_in_queue"]),
            ]
        )
        try:
            client.query(sql_ins, job_config=job_config, location=location).result()
        except Exception as e:
            print(f"Insert error: {e}")
            return 2

    print(f"Inserted {len(rows)} sync_log row(s) for center_id={center_id!r}")
    for r in rows:
        print(
            f"  sync_ts={r['sync_ts']}  rows_inserted={r['rows_inserted']}  "
            f"videos_in_queue={r['videos_in_queue']}  report_file={r['report_file']}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
ad