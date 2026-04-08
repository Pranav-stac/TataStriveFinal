#!/usr/bin/env python3
"""
Push a saved attendance report JSON to BigQuery (same path as the app uses).

Use after accidental BigQuery deletes or to backfill from a backup JSON.
If the same file was synced before, the client dedupe cache will skip it unless
you pass --force (removes this file's SHA-256 from ~/.tatastrive/bq_synced_report_hashes.json).

Usage (from repo root):

    python scripts/sync_attendance_json.py scripts/recovery_D14_attendance_report.json --center-id "YourCenterId"
    python scripts/sync_attendance_json.py path/to/report.json --force

Center ID defaults to `center_id` in ~/.tatastrive/config.json if set.

Requires: pip install google-cloud-bigquery google-auth
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.bigquery_sync import BigQuerySyncService, _creds_path  # noqa: E402
from app.config import get_config  # noqa: E402

DEDUPE_PATH = Path.home() / ".tatastrive" / "bq_synced_report_hashes.json"


def _clear_dedupe_for_file(report_path: str, center_id: str) -> bool:
    """Remove SHA-256 of file bytes from local dedupe map so sync_report will insert again."""
    path = os.path.abspath(os.path.normpath(report_path))
    if not os.path.isfile(path):
        return False
    with open(path, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    if not DEDUPE_PATH.is_file():
        return False
    try:
        with open(DEDUPE_PATH, "r", encoding="utf-8") as f:
            state = json.load(f)
    except (json.JSONDecodeError, OSError):
        return False
    cmap = state.setdefault("by_center", {}).setdefault(center_id, {})
    if digest not in cmap:
        print(f"No dedupe entry for this file hash (already clear or never synced).")
        return False
    del cmap[digest]
    DEDUPE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(DEDUPE_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)
    print(f"Cleared dedupe entry for hash {digest[:16]}… (center_id={center_id!r})")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Sync an attendance report JSON to BigQuery.")
    parser.add_argument(
        "json_path",
        nargs="?",
        default=str(ROOT / "scripts" / "recovery_D14_attendance_report.json"),
        help="Path to attendance report JSON (default: bundled recovery sample).",
    )
    parser.add_argument(
        "--center-id",
        default="",
        help="BigQuery center_id (default: from ~/.tatastrive/config.json).",
    )
    parser.add_argument(
        "--creds",
        default="",
        help="Service account JSON path (default: same search as the app).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Remove this file's hash from local dedupe cache before syncing.",
    )
    parser.add_argument(
        "--videos-in-queue",
        type=int,
        default=None,
        metavar="N",
        help="Optional: stored in sync_log.videos_in_queue for this sync (folder-listener depth).",
    )
    args = parser.parse_args()

    center_id = (args.center_id or "").strip() or (get_config().get("center_id") or "").strip()
    if not center_id:
        print("Error: set --center-id or center_id in ~/.tatastrive/config.json")
        return 1

    path = os.path.abspath(os.path.normpath(args.json_path))
    if not os.path.isfile(path):
        print(f"Error: file not found: {path}")
        return 1

    if args.force:
        _clear_dedupe_for_file(path, center_id)

    creds = args.creds or _creds_path()
    try:
        svc = BigQuerySyncService(center_id=center_id, credentials_path=creds)
        result = svc.sync_report(path, videos_in_queue=args.videos_in_queue)
    except Exception as e:
        print(f"Error: {e}")
        return 2

    print(f"status:       {result.get('status')}")
    print(f"rows_inserted:{result.get('rows_inserted')}")
    if result.get("error_msg"):
        print(f"message:      {result.get('error_msg')}")
    if result.get("status") == "skipped" and result.get("error_msg") == "already_synced":
        print("\nTip: run again with --force to allow re-insert after deleting BigQuery rows.")
    return 0 if result.get("status") == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
