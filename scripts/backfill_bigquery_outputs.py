#!/usr/bin/env python3
"""
Backfill local Outputs/*.json reports into BigQuery.

This script is for repair runs when reports exist under Outputs/ but were not
uploaded by the app's daily sync. It intentionally bypasses the local
~/.tatastrive/bq_synced_report_hashes.json dedupe file and instead compares
BigQuery row counts for each center_id + report_file group.

Examples:
    python scripts/backfill_bigquery_outputs.py --outputs Outputs
    python scripts/backfill_bigquery_outputs.py --outputs Outputs --execute
    python scripts/backfill_bigquery_outputs.py --outputs Outputs --execute --replace-mismatched
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from google.api_core.exceptions import NotFound  # noqa: E402
from google.cloud import bigquery  # noqa: E402

from app.bigquery_sync import (  # noqa: E402
    ATTENDANCE_TABLE,
    DATASET_ID,
    ENGAGEMENT_TABLE,
    MANAGEMENT_SUMMARY_TABLE,
    PROJECT_ID,
    BigQuerySyncService,
    _creds_path,
)
from app.config import get_config  # noqa: E402


REPORT_NAME_MARKERS = (
    "attendance_report",
    "class_dynamics_report",
    "management_summary_report",
)


@dataclass
class ReportPlan:
    path: Path
    report_file: str
    report_type: str
    table_name: str
    rows: list[dict[str, Any]]


def is_report_json(path: Path) -> bool:
    name = path.name.lower()
    return path.suffix.lower() == ".json" and any(marker in name for marker in REPORT_NAME_MARKERS)


def load_report(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("top-level JSON value is not an object")
    return data


def build_report_plan(svc: BigQuerySyncService, path: Path) -> ReportPlan | None:
    data = load_report(path)
    report_type = svc.detect_report_type(data)
    if report_type == "attendance":
        rows = svc._build_attendance_rows(data, str(path))
        table_name = ATTENDANCE_TABLE
    elif report_type == "engagement":
        rows = svc._build_engagement_rows(data, str(path))
        table_name = ENGAGEMENT_TABLE
    elif report_type == "management_summary":
        rows = svc._build_management_summary_rows(data, str(path))
        table_name = MANAGEMENT_SUMMARY_TABLE
    else:
        return None

    return ReportPlan(
        path=path,
        report_file=path.name,
        report_type=report_type,
        table_name=table_name,
        rows=rows,
    )


def count_existing_rows(
    client: bigquery.Client,
    location: str,
    table_name: str,
    center_id: str,
    report_file: str,
) -> int:
    sql = f"""
SELECT COUNT(1) AS row_count
FROM `{PROJECT_ID}.{DATASET_ID}.{table_name}`
WHERE center_id = @center_id
  AND report_file = @report_file
"""
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("center_id", "STRING", center_id),
            bigquery.ScalarQueryParameter("report_file", "STRING", report_file),
        ]
    )
    try:
        rows = list(client.query(sql, job_config=job_config, location=location).result())
    except NotFound:
        return 0
    return int(rows[0].row_count or 0) if rows else 0


def delete_report_rows(
    client: bigquery.Client,
    location: str,
    table_name: str,
    center_id: str,
    report_file: str,
) -> None:
    sql = f"""
DELETE FROM `{PROJECT_ID}.{DATASET_ID}.{table_name}`
WHERE center_id = @center_id
  AND report_file = @report_file
"""
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("center_id", "STRING", center_id),
            bigquery.ScalarQueryParameter("report_file", "STRING", report_file),
        ]
    )
    client.query(sql, job_config=job_config, location=location).result()


def chunked(items: list[dict[str, Any]], size: int) -> list[list[dict[str, Any]]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def insert_report_rows(svc: BigQuerySyncService, plan: ReportPlan, chunk_size: int) -> None:
    for chunk in chunked(plan.rows, chunk_size):
        svc._insert_rows(plan.table_name, chunk)
    svc._log_sync(str(plan.path), plan.report_type, len(plan.rows), "ok", "", None)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Backfill missed Outputs report JSON files into BigQuery."
    )
    parser.add_argument(
        "--outputs",
        default="Outputs",
        help="Outputs directory to scan recursively (default: Outputs).",
    )
    parser.add_argument(
        "--center-id",
        default="",
        help="center_id to write (default: ~/.tatastrive/config.json center_id).",
    )
    parser.add_argument(
        "--creds",
        default="",
        help="BigQuery service account JSON path (default: app credential search path).",
    )
    parser.add_argument(
        "--type",
        choices=["all", "attendance", "engagement"],
        default="all",
        help="Report type to backfill (default: all).",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually write to BigQuery. Without this, only prints a dry-run plan.",
    )
    parser.add_argument(
        "--replace-mismatched",
        action="store_true",
        help=(
            "If BigQuery has a non-zero row count that differs from local expected rows, "
            "delete that center_id + report_file group and reinsert local rows."
        ),
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=500,
        help="Rows per BigQuery streaming insert batch (default: 500).",
    )
    args = parser.parse_args()

    outputs_dir = (ROOT / args.outputs).resolve() if not Path(args.outputs).is_absolute() else Path(args.outputs)
    if not outputs_dir.exists():
        print(f"ERROR: outputs directory not found: {outputs_dir}")
        return 1

    center_id = (args.center_id or "").strip() or (get_config().get("center_id") or "").strip()
    if not center_id:
        print("ERROR: set --center-id or configure center_id in ~/.tatastrive/config.json.")
        return 1

    if args.chunk_size <= 0:
        print("ERROR: --chunk-size must be greater than zero.")
        return 1

    svc = BigQuerySyncService(center_id=center_id, credentials_path=args.creds or _creds_path())
    if args.execute:
        svc.ensure_tables()
    client = svc._get_client()
    location = svc._dataset_location(client)

    plans: list[ReportPlan] = []
    skipped: list[tuple[Path, str]] = []
    for path in sorted(outputs_dir.rglob("*.json")):
        if not is_report_json(path):
            continue
        try:
            plan = build_report_plan(svc, path)
            if plan is None:
                skipped.append((path, "unknown report shape"))
                continue
            if args.type != "all" and plan.report_type != args.type:
                continue
            plans.append(plan)
        except Exception as exc:
            skipped.append((path, str(exc)))

    if not plans:
        print(f"No matching report JSON files found under {outputs_dir}.")
        if skipped:
            print(f"Skipped {len(skipped)} unreadable/unknown report file(s).")
        return 0

    groups: dict[tuple[str, str], list[ReportPlan]] = {}
    for plan in plans:
        groups.setdefault((plan.table_name, plan.report_file), []).append(plan)

    mode = "EXECUTE" if args.execute else "DRY RUN"
    print(f"{mode}: center_id={center_id!r}, outputs={str(outputs_dir)!r}")
    print(f"Found {len(plans)} report file(s) in {len(groups)} BigQuery report_file group(s).")

    inserted_files = 0
    inserted_rows = 0
    skipped_groups = 0
    mismatch_groups = 0
    error_groups = 0

    for (table_name, report_file), group_plans in sorted(groups.items()):
        expected_rows = sum(len(plan.rows) for plan in group_plans)
        existing_rows = count_existing_rows(client, location, table_name, center_id, report_file)
        label = f"{table_name}/{report_file}"

        if existing_rows == expected_rows:
            skipped_groups += 1
            print(f"SKIP  {label}: BigQuery already has {existing_rows}/{expected_rows} rows.")
            continue

        if existing_rows and not args.replace_mismatched:
            mismatch_groups += 1
            print(
                f"WARN  {label}: BigQuery has {existing_rows} rows, local expects {expected_rows}. "
                "Use --execute --replace-mismatched to repair this group."
            )
            continue

        action = "REPLACE" if existing_rows else "INSERT"
        print(
            f"{action} {label}: {len(group_plans)} file(s), "
            f"BigQuery={existing_rows}, local={expected_rows} rows."
        )
        for plan in group_plans:
            print(f"      - {plan.path.relative_to(ROOT)} ({len(plan.rows)} rows)")

        if not args.execute:
            continue

        try:
            if existing_rows:
                delete_report_rows(client, location, table_name, center_id, report_file)
            for plan in group_plans:
                insert_report_rows(svc, plan, args.chunk_size)
                inserted_files += 1
                inserted_rows += len(plan.rows)
        except Exception as exc:
            error_groups += 1
            print(f"ERROR {label}: {exc}")

    print()
    print(
        "Summary: "
        f"inserted_files={inserted_files}, inserted_rows={inserted_rows}, "
        f"already_correct_groups={skipped_groups}, mismatched_groups={mismatch_groups}, "
        f"errors={error_groups}"
    )
    if skipped:
        print(f"Skipped {len(skipped)} unreadable/unknown report file(s).")

    if not args.execute:
        print("Dry run only. Re-run with --execute to upload missing zero-count groups.")
        print("For mismatched groups, add --replace-mismatched after verifying the dry-run output.")

    return 2 if error_groups else 0


if __name__ == "__main__":
    sys.exit(main())
