#!/usr/bin/env python3
"""
Run read-only BigQuery summaries on attendance_reports and save CSV files.

Uses the same credential paths as the main app (app/Creds/credentials.json, etc.)
or GOOGLE_APPLICATION_CREDENTIALS.

Usage (from project root):
    python scripts/bq_attendance_summary.py
    python scripts/bq_attendance_summary.py --out ./bq_summary_output

Requires: pip install google-cloud-bigquery
"""

from __future__ import annotations

import argparse
import csv
import sys
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path

# Project root on path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from app.bigquery_sync import (  # noqa: E402
    ATTENDANCE_TABLE,
    DATASET_ID,
    PROJECT_ID,
    _creds_path,
)

try:
    from google.cloud import bigquery
    from google.oauth2 import service_account
except ImportError:
    print("Install: pip install google-cloud-bigquery", file=sys.stderr)
    sys.exit(1)


def _cell(v):
    if v is None:
        return ""
    if isinstance(v, (datetime, date)):
        return v.isoformat()
    if isinstance(v, Decimal):
        return str(v)
    if isinstance(v, bytes):
        return v.hex()
    return v


def write_query_csv(client: bigquery.Client, sql: str, out_path: Path) -> int:
    job = client.query(sql)
    it = job.result()
    rows = list(it)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        out_path.write_text("", encoding="utf-8")
        return 0
    fieldnames = list(rows[0].keys())
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: _cell(r[k]) for k in fieldnames})
    return len(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Export BigQuery attendance summary CSVs")
    parser.add_argument(
        "--out",
        type=Path,
        default=_ROOT / "bq_summary_output",
        help="Output directory for CSV files",
    )
    args = parser.parse_args()
    out_dir: Path = args.out

    creds_path = _creds_path()
    if creds_path and Path(creds_path).exists():
        creds = service_account.Credentials.from_service_account_file(
            creds_path,
            scopes=["https://www.googleapis.com/auth/cloud-platform"],
        )
        client = bigquery.Client(project=PROJECT_ID, credentials=creds)
    else:
        client = bigquery.Client(project=PROJECT_ID)

    fq = f"`{PROJECT_ID}.{DATASET_ID}.{ATTENDANCE_TABLE}`"

    queries = {
        "01_overall_summary.sql": f"""
SELECT
  COUNT(*) AS total_rows,
  COUNT(DISTINCT CONCAT(CAST(report_date AS STRING), '|', COALESCE(report_file, '')))
    AS distinct_reports,
  COUNT(DISTINCT person_id) AS distinct_person_ids_all_time,
  COUNT(DISTINCT center_id) AS centers,
  MIN(report_date) AS first_report_date,
  MAX(report_date) AS last_report_date,
  COUNTIF(person_type = 'enrolled_student') AS rows_enrolled,
  COUNTIF(person_type = 'visitor') AS rows_visitor,
  COUNTIF(person_type = 'returning_employee') AS rows_returning,
  COUNTIF(person_type = 'no_face_track') AS rows_no_face,
  COUNTIF(person_id IS NULL) AS rows_summary_only
FROM {fq}
""",
        "02_per_report_summary.sql": f"""
SELECT
  report_file,
  report_date,
  center_id,
  ANY_VALUE(source_video) AS source_video,
  COUNT(*) AS row_count,
  COUNT(DISTINCT person_id) AS distinct_person_ids,
  COUNT(*) - COUNT(DISTINCT person_id) AS duplicate_id_rows,
  COUNTIF(person_type = 'no_face_track') AS nf_rows,
  COUNTIF(person_type = 'visitor') AS visitor_rows,
  COUNTIF(person_type = 'returning_employee') AS returning_rows,
  COUNTIF(person_type = 'enrolled_student') AS enrolled_rows,
  MIN(entry_time) AS earliest_entry,
  MAX(exit_time) AS latest_exit,
  COUNT(DISTINCT sync_timestamp) AS distinct_sync_batches
FROM {fq}
WHERE person_id IS NOT NULL
GROUP BY report_file, report_date, center_id
ORDER BY report_date DESC, report_file
""",
        "03_same_person_id_duplicated_in_report.sql": f"""
SELECT
  center_id,
  report_date,
  report_file,
  person_id,
  person_type,
  COUNT(*) AS times_appearing,
  ARRAY_TO_STRING(
    ARRAY_AGG(DISTINCT CAST(sync_timestamp AS STRING) ORDER BY CAST(sync_timestamp AS STRING)),
    '; '
  ) AS sync_timestamps_concat
FROM {fq}
WHERE person_id IS NOT NULL
GROUP BY center_id, report_date, report_file, person_id, person_type
HAVING COUNT(*) > 1
ORDER BY report_date DESC, times_appearing DESC
""",
        "04_same_g_id_across_days.sql": f"""
SELECT
  person_id,
  ANY_VALUE(person_type) AS example_person_type,
  COUNT(DISTINCT report_date) AS days_seen,
  ARRAY_TO_STRING(
    ARRAY_AGG(DISTINCT CAST(report_date AS STRING) ORDER BY CAST(report_date AS STRING)),
    ', '
  ) AS report_dates_concat,
  COUNT(*) AS total_rows
FROM {fq}
WHERE person_id IS NOT NULL
  AND STARTS_WITH(person_id, 'G_')
GROUP BY person_id
HAVING COUNT(DISTINCT report_date) > 1
ORDER BY days_seen DESC, total_rows DESC
""",
        "05_overlapping_presence_same_report_heuristic.sql": f"""
WITH d AS (
  SELECT
    center_id,
    report_file,
    report_date,
    person_id,
    person_type,
    entry_time,
    exit_time,
    SAFE.PARSE_TIME('%H:%M:%S', entry_time) AS t_in,
    SAFE.PARSE_TIME('%H:%M:%S', exit_time) AS t_out
  FROM {fq}
  WHERE person_id IS NOT NULL
    AND entry_time IS NOT NULL
    AND exit_time IS NOT NULL
)
SELECT
  a.report_date,
  a.report_file,
  a.center_id,
  a.person_id AS id_a,
  a.person_type AS type_a,
  b.person_id AS id_b,
  b.person_type AS type_b,
  a.entry_time AS entry_a,
  a.exit_time AS exit_a,
  b.entry_time AS entry_b,
  b.exit_time AS exit_b
FROM d a
JOIN d b
  ON a.report_file = b.report_file
 AND a.report_date = b.report_date
 AND a.center_id = b.center_id
 AND a.person_id < b.person_id
WHERE a.t_in IS NOT NULL
  AND a.t_out IS NOT NULL
  AND b.t_in IS NOT NULL
  AND b.t_out IS NOT NULL
  AND a.t_in <= b.t_out
  AND b.t_in <= a.t_out
ORDER BY a.report_date DESC, a.report_file, id_a, id_b
LIMIT 2000
""",
        "06_person_type_counts_by_center_date.sql": f"""
SELECT
  center_id,
  report_date,
  person_type,
  COUNT(*) AS row_count,
  COUNT(DISTINCT person_id) AS distinct_ids
FROM {fq}
WHERE person_id IS NOT NULL
GROUP BY center_id, report_date, person_type
ORDER BY report_date DESC, center_id, person_type
""",
    }

    print(f"Project: {PROJECT_ID}  Table: {DATASET_ID}.{ATTENDANCE_TABLE}")
    print(f"Output:  {out_dir.resolve()}")
    total = 0
    for name, sql in queries.items():
        csv_name = name.replace(".sql", ".csv")
        path = out_dir / csv_name
        try:
            n = write_query_csv(client, sql, path)
            print(f"  OK {csv_name}  ({n} rows)")
            total += n
        except Exception as e:
            print(f"  FAIL {csv_name}: {e}", file=sys.stderr)
            return 1
    print(f"Done. {len(queries)} files written under {out_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
