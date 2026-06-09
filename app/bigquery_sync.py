"""
BigQuery Sync Service for TataStrive Analytics.

Handles:
 - Automatic BigQuery table creation (attendance, engagement probes, management summary)
 - Syncing completed report JSON files to BigQuery
 - Class dynamics probes and management summary sessions land in separate tables
 - Daily scheduled sync (runs once per day at startup / after analysis)
 - Center-ID isolation: every row carries the device's center_id
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import threading
import traceback
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# BigQuery client (google-cloud-bigquery + google-auth)
# Imported lazily so the app still starts if the library isn't installed.
# ---------------------------------------------------------------------------

_BQ_AVAILABLE = False

def _try_import_bq():
    global _BQ_AVAILABLE
    try:
        import google.auth  # noqa: F401
        from google.cloud import bigquery  # noqa: F401
        _BQ_AVAILABLE = True
    except ImportError:
        _BQ_AVAILABLE = False

_try_import_bq()


def _coerce_bq_date(value: Any) -> Optional[date]:
    """
    Parse a value into a calendar date for BigQuery DATE columns (date-only, no time).
    Accepts datetime.date, datetime.datetime, ISO date/datetime strings, and common JSON shapes.
    """
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    s = str(value).strip()
    if not s:
        return None
    try:
        core = s.replace("Z", "+00:00")
        if "+" in core:
            core = core.split("+", 1)[0].strip()
        return datetime.fromisoformat(core).date()
    except ValueError:
        pass
    if len(s) >= 10 and s[4:5] == "-" and s[7:8] == "-":
        try:
            return date.fromisoformat(s[:10])
        except ValueError:
            pass
    return None


def _date_only_string_field(value: Any) -> Optional[str]:
    """Normalize optional date-like strings to YYYY-MM-DD; keep other strings as-is."""
    if value is None:
        return None
    d = _coerce_bq_date(value)
    if d is not None:
        return d.isoformat()
    s = str(value).strip()
    return s or None


def _int_or_none(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _float_or_none(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


IST = timezone(timedelta(hours=5, minutes=30))


def sync_timestamp_iso(when: Optional[datetime] = None) -> str:
    """Return IST (+05:30) timestamp string for BigQuery TIMESTAMP fields."""
    ts = when or datetime.now(IST)
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=IST)
    else:
        ts = ts.astimezone(IST)
    return ts.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "+05:30"


def _json_safe_value(value: Any) -> Any:
    """Convert row values to types BigQuery streaming JSON accepts."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat(sep=" ", timespec="seconds")
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, dict):
        return {k: _json_safe_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe_value(v) for v in value]
    return value


def _json_safe_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [{k: _json_safe_value(v) for k, v in row.items()} for row in rows]


def _is_reportable_attendance_id(pid: Any) -> bool:
    """Per spec: only G_* and NF_* IDs are stored in attendance records.

    Mirrors the filter in the attendance worker so BigQuery rows can't be
    polluted by legacy DB prefixes loaded from chained pkl databases.
    """
    s = str(pid or "")
    if not s:
        return False
    if s.startswith("G_"):
        return True
    if s.startswith("NF_") or "_NF_" in s:
        return True
    return False


def _derive_video_id(video_path: Any) -> Optional[str]:
    """Stable video identifier for joining engagement and management summary rows."""
    if video_path is None:
        return None
    s = str(video_path).strip()
    if not s:
        return None
    stem = Path(s).stem
    return stem or Path(s).name or s


def _flatten_activity_distribution(activity: Any) -> Dict[str, Optional[float]]:
    dist = activity if isinstance(activity, dict) else {}
    return {
        "act_listening_pct": dist.get("listening"),
        "act_writing_pct": dist.get("writing"),
        "act_raising_hand_pct": dist.get("raising_hand"),
        "act_unknown_pct": dist.get("unknown"),
    }


def _flatten_attention_distribution(attention: Any) -> Dict[str, Optional[float]]:
    dist = attention if isinstance(attention, dict) else {}
    return {
        "att_focused_pct": dist.get("focused"),
        "att_partially_focused_pct": dist.get("partially_focused"),
        "att_distracted_pct": dist.get("distracted"),
        "att_not_visible_pct": dist.get("not_visible"),
    }


def _flatten_behavior_profile(profile: Any) -> Dict[str, Optional[float]]:
    dist = profile if isinstance(profile, dict) else {}
    return {
        "active_participation_pct": dist.get("active_participation"),
        "passive_focus_pct": dist.get("passive_focus"),
        "disengaged_idle_pct": dist.get("disengaged_idle"),
        "unobservable_pct": dist.get("unobservable"),
    }


# ---------------------------------------------------------------------------
# BigQuery dataset / table names
# ---------------------------------------------------------------------------
PROJECT_ID = "tatastrive-269409"
DATASET_ID   = "tatastrive_analytics"
ATTENDANCE_TABLE   = "attendance_reports"
ENGAGEMENT_TABLE   = "engagement_reports"
MANAGEMENT_SUMMARY_TABLE = "management_summary_reports"
SYNC_LOG_TABLE     = "sync_log"

# ---------------------------------------------------------------------------
# Table schemas
# ---------------------------------------------------------------------------

ATTENDANCE_SCHEMA = [
    # Partition / identity
    {"name": "center_id",          "type": "STRING",    "mode": "REQUIRED"},
    {"name": "sync_timestamp",     "type": "TIMESTAMP", "mode": "REQUIRED"},
    {"name": "report_date",        "type": "DATE",      "mode": "NULLABLE"},
    # Session
    {"name": "session_date",       "type": "STRING",    "mode": "NULLABLE"},
    {"name": "classroom",          "type": "STRING",    "mode": "NULLABLE"},
    {"name": "camera",             "type": "STRING",    "mode": "NULLABLE"},
    {"name": "source_video",       "type": "STRING",    "mode": "NULLABLE"},
    {"name": "source_video_path",  "type": "STRING",    "mode": "NULLABLE"},
    {"name": "session_duration",   "type": "STRING",    "mode": "NULLABLE"},
    {"name": "video_duration_sec", "type": "INTEGER",   "mode": "NULLABLE"},
    {"name": "video_processing_sec", "type": "FLOAT", "mode": "NULLABLE"},
    # Counts
    {"name": "unique_people",      "type": "INTEGER",   "mode": "NULLABLE"},
    {"name": "returning_count",    "type": "INTEGER",   "mode": "NULLABLE"},
    {"name": "visitor_count",      "type": "INTEGER",   "mode": "NULLABLE"},
    {"name": "identified_students","type": "INTEGER",   "mode": "NULLABLE"},
    {"name": "nf_presence",        "type": "INTEGER",   "mode": "NULLABLE"},
    # Per-person (repeated)
    {"name": "person_id",          "type": "STRING",    "mode": "NULLABLE"},
    {"name": "person_type",        "type": "STRING",    "mode": "NULLABLE"},
    {"name": "engagement_id",      "type": "STRING",    "mode": "NULLABLE"},
    {"name": "batch",              "type": "STRING",    "mode": "NULLABLE"},
    {"name": "entry_time",         "type": "STRING",    "mode": "NULLABLE"},
    {"name": "exit_time",          "type": "STRING",    "mode": "NULLABLE"},
    {"name": "duration_seconds",   "type": "INTEGER",   "mode": "NULLABLE"},
    {"name": "identity_confidence",      "type": "FLOAT", "mode": "NULLABLE"},
    {"name": "student_match_confidence", "type": "FLOAT", "mode": "NULLABLE"},
    {"name": "confidence_score",   "type": "FLOAT",     "mode": "NULLABLE"},
    {"name": "present_last_7_days","type": "INTEGER",   "mode": "NULLABLE"},
    {"name": "last_present_date",  "type": "STRING",    "mode": "NULLABLE"},
    # Source
    {"name": "report_file",        "type": "STRING",    "mode": "NULLABLE"},
]

ENGAGEMENT_SCHEMA = [
    # Partition / identity
    {"name": "center_id",            "type": "STRING",    "mode": "REQUIRED"},
    {"name": "sync_timestamp",       "type": "TIMESTAMP", "mode": "REQUIRED"},
    {"name": "report_date",          "type": "DATE",      "mode": "NULLABLE"},
    # Video / classroom metadata
    {"name": "video_id",             "type": "STRING",    "mode": "NULLABLE"},
    {"name": "video_path",           "type": "STRING",    "mode": "NULLABLE"},
    {"name": "classroom",            "type": "STRING",    "mode": "NULLABLE"},
    {"name": "recording_date_str",   "type": "STRING",    "mode": "NULLABLE"},
    {"name": "baseline_max_students","type": "INTEGER",   "mode": "NULLABLE"},
    {"name": "report_type",          "type": "STRING",    "mode": "NULLABLE"},
    # Per-probe (one row per probe)
    {"name": "time_slice",           "type": "STRING",    "mode": "NULLABLE"},
    {"name": "video_timestamp_sec",  "type": "FLOAT",     "mode": "NULLABLE"},
    {"name": "real_world_time",      "type": "STRING",    "mode": "NULLABLE"},
    {"name": "student_count",        "type": "INTEGER",   "mode": "NULLABLE"},
    {"name": "avg_engagement",       "type": "FLOAT",     "mode": "NULLABLE"},
    {"name": "class_mode",           "type": "STRING",    "mode": "NULLABLE"},
    {"name": "act_listening_pct",    "type": "FLOAT",     "mode": "NULLABLE"},
    {"name": "act_writing_pct",      "type": "FLOAT",     "mode": "NULLABLE"},
    {"name": "act_raising_hand_pct", "type": "FLOAT",     "mode": "NULLABLE"},
    {"name": "act_unknown_pct",      "type": "FLOAT",     "mode": "NULLABLE"},
    {"name": "att_focused_pct",      "type": "FLOAT",     "mode": "NULLABLE"},
    {"name": "att_partially_focused_pct", "type": "FLOAT", "mode": "NULLABLE"},
    {"name": "att_distracted_pct",   "type": "FLOAT",     "mode": "NULLABLE"},
    {"name": "att_not_visible_pct",  "type": "FLOAT",     "mode": "NULLABLE"},
    # Source
    {"name": "report_file",          "type": "STRING",    "mode": "NULLABLE"},
]

MANAGEMENT_SUMMARY_SCHEMA = [
    # Partition / identity
    {"name": "center_id",            "type": "STRING",    "mode": "REQUIRED"},
    {"name": "sync_timestamp",       "type": "TIMESTAMP", "mode": "REQUIRED"},
    {"name": "report_date",          "type": "DATE",      "mode": "NULLABLE"},
    # Video / classroom metadata
    {"name": "video_id",             "type": "STRING",    "mode": "NULLABLE"},
    {"name": "video_path",           "type": "STRING",    "mode": "NULLABLE"},
    {"name": "classroom",            "type": "STRING",    "mode": "NULLABLE"},
    {"name": "recording_date_str",   "type": "STRING",    "mode": "NULLABLE"},
    {"name": "baseline_max_students","type": "INTEGER",   "mode": "NULLABLE"},
    {"name": "report_type",          "type": "STRING",    "mode": "NULLABLE"},
    # Per-session (one row per grouped mode window)
    {"name": "session_mode",       "type": "STRING",    "mode": "NULLABLE"},
    {"name": "time_window",          "type": "STRING",    "mode": "NULLABLE"},
    {"name": "avg_student_count",    "type": "INTEGER",   "mode": "NULLABLE"},
    {"name": "overall_engagement_score", "type": "FLOAT", "mode": "NULLABLE"},
    {"name": "active_participation_pct", "type": "FLOAT", "mode": "NULLABLE"},
    {"name": "passive_focus_pct",    "type": "FLOAT",     "mode": "NULLABLE"},
    {"name": "disengaged_idle_pct",  "type": "FLOAT",     "mode": "NULLABLE"},
    {"name": "unobservable_pct",     "type": "FLOAT",     "mode": "NULLABLE"},
    # Source
    {"name": "report_file",          "type": "STRING",    "mode": "NULLABLE"},
]

SYNC_LOG_SCHEMA = [
    {"name": "center_id",     "type": "STRING",    "mode": "REQUIRED"},
    {"name": "sync_ts",       "type": "TIMESTAMP", "mode": "REQUIRED"},
    {"name": "report_file",   "type": "STRING",    "mode": "NULLABLE"},
    {"name": "report_type",   "type": "STRING",    "mode": "NULLABLE"},   # "attendance" | "engagement"
    {"name": "rows_inserted", "type": "INTEGER",   "mode": "NULLABLE"},
    {"name": "status",        "type": "STRING",    "mode": "NULLABLE"},   # "ok" | "error"
    {"name": "error_msg",     "type": "STRING",    "mode": "NULLABLE"},
    # Folder-listener: videos still waiting after this sync (auto-sync only; NULL for batch/manual)
    {"name": "videos_in_queue", "type": "INTEGER", "mode": "NULLABLE"},
    # Attendance: source video length and wall-clock analysis time (from report Session)
    {"name": "video_duration_sec",   "type": "INTEGER",   "mode": "NULLABLE"},
    {"name": "video_processing_sec", "type": "FLOAT",     "mode": "NULLABLE"},
    {"name": "source_video",         "type": "STRING",    "mode": "NULLABLE"},
    {"name": "session_date",         "type": "STRING", "mode": "NULLABLE"},
]

_TABLE_INSERT_FIELDS: Dict[str, frozenset[str]] = {
    ATTENDANCE_TABLE: frozenset(f["name"] for f in ATTENDANCE_SCHEMA),
    ENGAGEMENT_TABLE: frozenset(f["name"] for f in ENGAGEMENT_SCHEMA),
    MANAGEMENT_SUMMARY_TABLE: frozenset(f["name"] for f in MANAGEMENT_SUMMARY_SCHEMA),
    SYNC_LOG_TABLE: frozenset(f["name"] for f in SYNC_LOG_SCHEMA),
}


def _rows_for_table(table_name: str, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Drop keys outside the table schema (e.g. legacy probe JSON blobs)."""
    allowed = _TABLE_INSERT_FIELDS.get(table_name)
    if not allowed:
        return rows
    return [{k: v for k, v in row.items() if k in allowed} for row in rows]


# ---------------------------------------------------------------------------
# Helper: resolve credentials path
# ---------------------------------------------------------------------------

def _creds_path() -> str:
    """Return absolute path to the BigQuery service-account JSON."""
    app_dir = Path(__file__).resolve().parent
    exe_dir = Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else app_dir.parent
    candidates = [
        # Next to executable (for packaged installs on new PC — add credentials.json there)
        exe_dir / "credentials.json",
        exe_dir / "Creds" / "credentials.json",
        # Alongside this file (app/) - dev and bundled
        app_dir / "Creds" / "credentials.json",
        app_dir / "Creds" / "credentials (1).json",
        app_dir / "creds" / "credentials.json",
        app_dir.parent / "credentials.json",
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    # Fall back: let google-auth find it via env var
    return os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "")


# ---------------------------------------------------------------------------
# BigQuerySyncService
# ---------------------------------------------------------------------------

class BigQuerySyncService:
    """
    Thread-safe BigQuery sync service.

    Usage::

        svc = BigQuerySyncService(center_id="CenterAlpha")
        svc.ensure_tables()
        svc.sync_report("/path/to/report.json")
    """

    def __init__(self, center_id: str, credentials_path: str = ""):
        self.center_id = center_id.strip()
        self._creds_path = credentials_path or _creds_path()
        self._project_id = PROJECT_ID
        self._client = None
        self._lock = threading.Lock()
        self._last_sync_date: Optional[date] = None
        # Persisted map: center_id -> sha256(content) -> metadata (dedupe append-only inserts)
        self._dedupe_path = Path.home() / ".tatastrive" / "bq_synced_report_hashes.json"

    # ------------------------------------------------------------------
    # Client
    # ------------------------------------------------------------------

    def _get_client(self):
        """Lazily create and return the BigQuery client."""
        if self._client is not None:
            return self._client
        if not _BQ_AVAILABLE:
            raise RuntimeError(
                "google-cloud-bigquery is not installed.\n"
                "Run:  pip install google-cloud-bigquery"
            )
        from google.oauth2 import service_account
        from google.cloud import bigquery

        if self._creds_path and Path(self._creds_path).exists():
            creds = service_account.Credentials.from_service_account_file(
                self._creds_path,
                scopes=["https://www.googleapis.com/auth/cloud-platform"]
            )
            self._client = bigquery.Client(
                project=self._project_id,
                credentials=creds
            )
        elif os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
            # Use env var path
            self._client = bigquery.Client(project=self._project_id)
        else:
            # No explicit creds - will use ADC or fail with clear error
            try:
                self._client = bigquery.Client(project=self._project_id)
            except Exception as e:
                raise RuntimeError(
                    "BigQuery credentials not found. Place service account JSON at:\n"
                    f"  app/Creds/credentials.json\n"
                    "Or set GOOGLE_APPLICATION_CREDENTIALS to the JSON file path.\n"
                    f"Original error: {e}"
                )
        return self._client

    # ------------------------------------------------------------------
    # Table management
    # ------------------------------------------------------------------

    def ensure_tables(self) -> None:
        """Create the dataset and tables if they don't exist."""
        from google.cloud import bigquery
        from google.api_core.exceptions import Conflict

        client = self._get_client()

        # Dataset
        dataset_ref = f"{self._project_id}.{DATASET_ID}"
        dataset = bigquery.Dataset(dataset_ref)
        dataset.location = "US"
        try:
            client.create_dataset(dataset, exists_ok=True)
        except Exception as e:
            print(f"[BQ] Dataset create warning: {e}")

        # Tables
        for table_name, schema_dicts in [
            (ATTENDANCE_TABLE,  ATTENDANCE_SCHEMA),
            (ENGAGEMENT_TABLE,  ENGAGEMENT_SCHEMA),
            (MANAGEMENT_SUMMARY_TABLE, MANAGEMENT_SUMMARY_SCHEMA),
            (SYNC_LOG_TABLE,    SYNC_LOG_SCHEMA),
        ]:
            full_table = f"{dataset_ref}.{table_name}"
            schema = [
                bigquery.SchemaField(f["name"], f["type"], mode=f.get("mode", "NULLABLE"))
                for f in schema_dicts
            ]
            table = bigquery.Table(full_table, schema=schema)
            # Partition by report_date for attendance/engagement tables
            if table_name in (ATTENDANCE_TABLE, ENGAGEMENT_TABLE, MANAGEMENT_SUMMARY_TABLE):
                table.time_partitioning = bigquery.TimePartitioning(
                    type_=bigquery.TimePartitioningType.DAY,
                    field="report_date"
                )
            try:
                client.create_table(table, exists_ok=True)
                # Migrate: add any missing columns to existing tables (fixes "no such field" errors)
                self._add_missing_columns(client, full_table, schema_dicts)
                print(f"[BQ] Table ready: {full_table}")
            except Exception as e:
                print(f"[BQ] Table create warning for {table_name}: {e}")

    def _add_missing_columns(
        self, client, full_table: str, schema_dicts: List[Dict]
    ) -> None:
        """Add missing columns to an existing table (schema migration)."""
        _TYPE_MAP = {"INTEGER": "INT64", "FLOAT": "FLOAT64"}  # BigQuery DDL names
        try:
            table = client.get_table(full_table)
            existing = {f.name.lower() for f in table.schema}  # case-insensitive
            for f in schema_dicts:
                name = f.get("name")
                if name and name.lower() not in existing:
                    bq_type = _TYPE_MAP.get(f.get("type", "STRING"), f.get("type", "STRING"))
                    try:
                        q = f"ALTER TABLE `{full_table}` ADD COLUMN IF NOT EXISTS `{name}` {bq_type}"
                        client.query(q).result()
                        print(f"[BQ] Added column {name} to {full_table}")
                    except Exception as col_err:
                        if "already exists" not in str(col_err).lower():
                            print(f"[BQ] Could not add {name}: {col_err}")
        except Exception as e:
            print(f"[BQ] Column migration warning for {full_table}: {e}")

    def _dataset_location(self, client) -> str:
        """BigQuery dataset location (e.g. US). Queries must run in this region."""
        from google.cloud import bigquery

        ds_id = f"{self._project_id}.{DATASET_ID}"
        try:
            ds = client.get_dataset(ds_id)
            return ds.location or "US"
        except Exception:
            return "US"

    def _run_query(self, client, sql: str, location: str):
        # Pass location on client.query() — QueryJobConfig.location is not settable in some client versions.
        job = client.query(sql, location=location)
        job.result()
        return job

    def _table_row_count(self, client, fqtn: str, location: str) -> int:
        """COUNT(*) for a fully-qualified `project.dataset.table` identifier."""
        sql = f"SELECT COUNT(1) AS c FROM {fqtn}"
        rows = list(client.query(sql, location=location).result())
        if not rows:
            return 0
        return int(rows[0].c or 0)

    def truncate_all_tables(self) -> Dict[str, str]:
        """
        Remove all rows from attendance_reports, engagement_reports, and sync_log.
        Tables and schemas are kept; use before a full re-sync from local reports.

        Verifies row count after TRUNCATE; if rows remain, runs DELETE WHERE TRUE
        (handles edge cases with partitioned tables / buffered rows).
        """
        with self._lock:
            client = self._get_client()
            location = self._dataset_location(client)
            out: Dict[str, str] = {}
            for table_name in (
                ATTENDANCE_TABLE,
                ENGAGEMENT_TABLE,
                MANAGEMENT_SUMMARY_TABLE,
                SYNC_LOG_TABLE,
            ):
                fqtn = f"`{self._project_id}.{DATASET_ID}.{table_name}`"
                self._run_query(client, f"TRUNCATE TABLE {fqtn}", location)
                n = self._table_row_count(client, fqtn, location)
                if n > 0:
                    self._run_query(client, f"DELETE FROM {fqtn} WHERE TRUE", location)
                    n = self._table_row_count(client, fqtn, location)
                out[table_name] = "truncated" if n == 0 else f"rows_remain_{n}"
            return out

    def drop_and_recreate_tables(self) -> Dict[str, str]:
        """
        DROP attendance_reports, engagement_reports, sync_log and recreate empty.

        Use when TRUNCATE/DELETE still leave rows: the app uses **streaming inserts**
        (`insert_rows_json`), and BigQuery can keep a streaming buffer that does not
        always clear cleanly with TRUNCATE alone.
        """
        with self._lock:
            client = self._get_client()
            for table_name in (
                ATTENDANCE_TABLE,
                ENGAGEMENT_TABLE,
                MANAGEMENT_SUMMARY_TABLE,
                SYNC_LOG_TABLE,
            ):
                fq = f"{self._project_id}.{DATASET_ID}.{table_name}"
                client.delete_table(fq, not_found_ok=True)
        self.ensure_tables()
        with self._lock:
            client = self._get_client()
            location = self._dataset_location(client)
            out: Dict[str, str] = {}
            for table_name in (
                ATTENDANCE_TABLE,
                ENGAGEMENT_TABLE,
                MANAGEMENT_SUMMARY_TABLE,
                SYNC_LOG_TABLE,
            ):
                fqtn = f"`{self._project_id}.{DATASET_ID}.{table_name}`"
                n = self._table_row_count(client, fqtn, location)
                out[table_name] = "empty" if n == 0 else f"rows_remain_{n}"
            return out

    # ------------------------------------------------------------------
    # Dedupe: same report file must not insert twice (startup folder scan +
    # per-completion sync, or app relaunch re-scanning the same JSON).
    # ------------------------------------------------------------------

    def _load_dedupe_state(self) -> Dict[str, Any]:
        try:
            if self._dedupe_path.exists():
                with open(self._dedupe_path, "r", encoding="utf-8") as f:
                    return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
        return {"by_center": {}}

    def _save_dedupe_state(self, state: Dict[str, Any]) -> None:
        try:
            self._dedupe_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._dedupe_path, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2)
        except OSError as e:
            print(f"[BQ] Could not save dedupe state: {e}")

    def _prune_dedupe_center(self, center_map: Dict[str, Any], max_entries: int = 8000) -> None:
        if len(center_map) <= max_entries:
            return
        items = []
        for h, meta in center_map.items():
            ts = (meta or {}).get("synced_at") or ""
            items.append((ts, h))
        items.sort()
        for _, h in items[: len(center_map) - max_entries]:
            center_map.pop(h, None)

    # ------------------------------------------------------------------
    # Report detection & routing
    # ------------------------------------------------------------------

    def detect_report_type(self, report_data: Dict) -> str:
        """Return attendance, engagement, management_summary, or unknown."""
        if "hourly_probes" in report_data:
            return "engagement"
        if "sessions" in report_data:
            return "management_summary"
        if "People" in report_data and "Session" in report_data:
            return "attendance"
        return "unknown"

    def sync_report(
        self,
        report_path: str,
        videos_in_queue: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Parse a JSON report file and insert rows into BigQuery.
        Returns a dict with keys: status, rows_inserted, error_msg.

        status may be:
          - "ok" — rows inserted
          - "skipped" — unknown report type, or **already_synced** (same file bytes
            were pushed for this center_id; avoids doubling rows on re-scan / relaunch)
          - "error"

        videos_in_queue: optional count of videos still pending in the folder listener
        when this sync ran (stored in sync_log for ops visibility).
        """
        result: Dict[str, Any] = {"status": "ok", "rows_inserted": 0, "error_msg": ""}
        path = os.path.abspath(os.path.normpath(report_path))
        try:
            if not os.path.isfile(path):
                result["status"] = "error"
                result["error_msg"] = f"Report not found: {path}"
                return result

            self.ensure_tables()  # Guarantee migration runs (adds engagement_id, etc.)

            with open(path, "rb") as f:
                raw = f.read()
            digest = hashlib.sha256(raw).hexdigest()
            try:
                text = raw.decode("utf-8")
            except UnicodeDecodeError:
                result["status"] = "error"
                result["error_msg"] = "Report file is not valid UTF-8"
                return result
            data = json.loads(text)

            with self._lock:
                state = self._load_dedupe_state()
                cmap = state.setdefault("by_center", {}).setdefault(self.center_id, {})
                if digest in cmap:
                    result["status"] = "skipped"
                    result["error_msg"] = "already_synced"
                    return result

                rtype = self.detect_report_type(data)
                if rtype == "attendance":
                    rows = self._build_attendance_rows(data, path)
                    self._insert_rows(ATTENDANCE_TABLE, rows)
                elif rtype == "engagement":
                    rows = self._build_engagement_rows(data, path)
                    self._insert_rows(ENGAGEMENT_TABLE, rows)
                elif rtype == "management_summary":
                    rows = self._build_management_summary_rows(data, path)
                    self._insert_rows(MANAGEMENT_SUMMARY_TABLE, rows)
                else:
                    result["status"] = "skipped"
                    result["error_msg"] = "Unknown report type"
                    return result

                result["rows_inserted"] = len(rows)
                cmap[digest] = {
                    "synced_at": self._now_ts(),
                    "report_file": Path(path).name,
                }
                self._prune_dedupe_center(cmap)
                self._save_dedupe_state(state)
                self._log_sync(path, rtype, len(rows), "ok", "", videos_in_queue, data)

        except Exception as e:
            msg = f"{e}\n{traceback.format_exc()}"
            result["status"] = "error"
            result["error_msg"] = str(e)
            rtype = "unknown"
            try:
                with open(path, "rb") as f:
                    d = json.loads(f.read().decode("utf-8"))
                rtype = self.detect_report_type(d)
            except Exception:
                pass
            self._log_sync(path, rtype, 0, "error", str(e)[:1000], videos_in_queue, None)

        return result

    # ------------------------------------------------------------------
    # Row builders
    # ------------------------------------------------------------------

    def _now_ts(self) -> str:
        """Return IST (+05:30) timestamp in ISO 8601 format for BigQuery TIMESTAMP fields."""
        return sync_timestamp_iso()

    def _parse_report_date(self, data: Dict) -> Optional[date]:
        """Extract calendar date for BigQuery DATE column report_date (never a timestamp)."""
        session_date = (data.get("Session") or {}).get("date")
        if session_date is not None and str(session_date).strip() != "":
            d = _coerce_bq_date(session_date)
            if d is not None:
                return d
        rec_date = data.get("recording_date", "") or ""
        if rec_date:
            d = _coerce_bq_date(rec_date)
            if d is not None:
                return d
        return None

    def _build_attendance_rows(self, data: Dict, report_path: str) -> List[Dict]:
        """
        One BigQuery row per person in the attendance report (first row wins if the same
        person_id appears more than once in People).
        Plus always at least one summary row (even if no people).
        """
        ts = self._now_ts()
        session = data.get("Session", {})
        counts  = data.get("Counts", {})
        people  = data.get("People", [])
        rdate   = self._parse_report_date(data)
        fname   = Path(report_path).name

        base = {
            "center_id":          self.center_id,
            "sync_timestamp":     ts,
            "report_date":        _date_only_string_field(rdate),
            "session_date":       _date_only_string_field(session.get("date")),
            "classroom":          session.get("classroom"),
            "camera":             session.get("camera"),
            "source_video":       session.get("source_video"),
            "source_video_path":  session.get("source_video_path"),
            "session_duration":   session.get("duration"),
            "video_duration_sec": _int_or_none(session.get("duration_sec")),
            "video_processing_sec": _float_or_none(session.get("processing_time_sec")),
            "unique_people":      counts.get("unique_people", 0),
            "returning_count":    counts.get("returning", 0),
            "visitor_count":      counts.get("visitors", 0),
            "identified_students": counts.get("identified_students", 0),
            "nf_presence":        counts.get("nf_presence", 0),
            "report_file":        fname,
        }

        if not people:
            row = dict(base)
            row.update({
                "person_id": None, "person_type": None, "engagement_id": None, "batch": None, "entry_time": None,
                "exit_time": None, "duration_seconds": None,
                "identity_confidence": None, "student_match_confidence": None,
                "confidence_score": None, "present_last_7_days": None, "last_present_date": None,
            })
            return [row]

        rows = []
        seen_person_ids: set[str] = set()
        for person in people:
            pid = person.get("id")
            # Spec: only G_* / NF_* IDs reach BigQuery attendance rows.
            if not _is_reportable_attendance_id(pid):
                continue
            if pid is not None and str(pid).strip() != "":
                key = str(pid)
                if key in seen_person_ids:
                    continue
                seen_person_ids.add(key)
            row = dict(base)
            row.update({
                "person_id":           pid,
                "person_type":         person.get("type"),
                "engagement_id":       person.get("engagement_id"),
                "batch":               person.get("batch"),
                "entry_time":          person.get("entry"),
                "exit_time":           person.get("exit"),
                "duration_seconds":    person.get("duration_sec"),
                "identity_confidence":      person.get("identity_confidence"),
                "student_match_confidence": person.get("student_match_confidence"),
                "confidence_score":    person.get("confidence_score"),
                "present_last_7_days": person.get("present_last_7_days"),
                "last_present_date":   _date_only_string_field(person.get("last_present_date")),
            })
            rows.append(row)
        return rows

    def _engagement_report_base(self, data: Dict, report_path: str) -> Dict[str, Any]:
        ts = self._now_ts()
        rdate = self._parse_report_date(data)
        fname = Path(report_path).name
        video_path = data.get("video_path") or data.get("video") or data.get("source_video")
        return {
            "center_id":             self.center_id,
            "sync_timestamp":        ts,
            "report_date":           _date_only_string_field(rdate),
            "video_id":              _derive_video_id(video_path),
            "video_path":            video_path,
            "classroom":             data.get("classroom"),
            "recording_date_str":    _date_only_string_field(data.get("recording_date")),
            "baseline_max_students": data.get("baseline_max_students"),
            "report_type":           data.get("report_type"),
            "report_file":           fname,
        }

    def _build_engagement_rows(self, data: Dict, report_path: str) -> List[Dict]:
        """
        One BigQuery row per probe in class dynamics reports.
        Activity and attention distributions are flattened into scalar columns.
        """
        base = self._engagement_report_base(data, report_path)
        probes = data.get("hourly_probes", [])

        if not probes:
            row = dict(base)
            row.update({
                "time_slice": None,
                "video_timestamp_sec": None,
                "real_world_time": None,
                "student_count": None,
                "avg_engagement": None,
                "class_mode": None,
                **_flatten_activity_distribution({}),
                **_flatten_attention_distribution({}),
            })
            return [row]

        rows = []
        for probe in probes:
            row = dict(base)
            row.update({
                "time_slice": probe.get("time_slice"),
                "video_timestamp_sec": probe.get("video_timestamp_sec"),
                "real_world_time": probe.get("real_world_time"),
                "student_count": probe.get("student_count_corrected") or probe.get("student_count"),
                "avg_engagement": probe.get("avg_engagement"),
                "class_mode": probe.get("class_mode"),
                **_flatten_activity_distribution(probe.get("activity_distribution")),
                **_flatten_attention_distribution(probe.get("attention_distribution")),
            })
            rows.append(row)
        return rows

    def _build_management_summary_rows(self, data: Dict, report_path: str) -> List[Dict]:
        """One BigQuery row per grouped session in management summary reports."""
        base = self._engagement_report_base(data, report_path)

        sessions = data.get("sessions", [])
        if not sessions:
            row = dict(base)
            row.update({
                "session_mode": None,
                "time_window": None,
                "avg_student_count": None,
                "overall_engagement_score": None,
                **_flatten_behavior_profile({}),
            })
            return [row]

        rows = []
        for session in sessions:
            row = dict(base)
            row.update({
                "session_mode": session.get("session_mode"),
                "time_window": session.get("time_window"),
                "avg_student_count": session.get("avg_student_count"),
                "overall_engagement_score": session.get("overall_engagement_score"),
                **_flatten_behavior_profile(session.get("behavior_profile")),
            })
            rows.append(row)
        return rows

    # ------------------------------------------------------------------
    # Insert
    # ------------------------------------------------------------------

    def _insert_rows(self, table_name: str, rows: List[Dict]) -> None:
        if not rows:
            return
        client = self._get_client()
        full_table = f"{self._project_id}.{DATASET_ID}.{table_name}"
        payload = _json_safe_rows(_rows_for_table(table_name, rows))
        try:
            errors = client.insert_rows_json(full_table, payload)
            if errors:
                err_msgs = [str(e) for e in errors]
                raise RuntimeError(f"BigQuery insert errors for {table_name}: {err_msgs}")
        except Exception as e:
            if "Schema" in str(e) or "schema" in str(e).lower():
                raise RuntimeError(
                    f"BigQuery schema mismatch for {table_name}. "
                    "Tables may have been created with an older schema. "
                    "Try deleting and recreating the dataset in BigQuery console, or check field types. "
                    f"Details: {e}"
                ) from e
            raise

    # ------------------------------------------------------------------
    # Sync log
    # ------------------------------------------------------------------

    def _log_sync(
        self,
        report_path: str,
        rtype: str,
        rows: int,
        status: str,
        error_msg: str,
        videos_in_queue: Optional[int] = None,
        report_data: Optional[Dict[str, Any]] = None,
    ) -> None:
        try:
            queue_depth = 0 if videos_in_queue is None else max(0, int(videos_in_queue))
            session = (report_data or {}).get("Session") or {}
            row = {
                "center_id":     self.center_id,
                "sync_ts":       self._now_ts(),
                "report_file":   Path(report_path).name,
                "report_type":   rtype,
                "rows_inserted": rows,
                "status":        status,
                "error_msg":     error_msg,
                "videos_in_queue": queue_depth,
                "video_duration_sec": _int_or_none(session.get("duration_sec")),
                "video_processing_sec": _float_or_none(session.get("processing_time_sec")),
                "source_video": session.get("source_video"),
                "session_date": _date_only_string_field(session.get("date")),
            }
            self._insert_rows(SYNC_LOG_TABLE, [row])
        except Exception as e:
            print(f"[BQ] Could not write sync log: {e}")

    # ------------------------------------------------------------------
    # Batch / directory scan
    # ------------------------------------------------------------------

    def sync_directory(self, directory: str,
                       since_date: Optional[date] = None,
                       log_callback=None) -> Dict[str, Any]:
        """
        Scan *directory* recursively for JSON report files and sync them.
        Skips files already synced (tracked by file-modification date vs since_date).

        Returns summary dict.
        """
        summary = {"synced": 0, "skipped": 0, "errors": 0, "files": []}
        search_root = Path(directory)
        if not search_root.exists():
            return summary

        cutoff = since_date or (date.today() - timedelta(days=1))

        def _log(msg):
            if log_callback:
                log_callback(msg)
            print(f"[BQ-Sync] {msg}")

        for json_file in search_root.rglob("*.json"):
            # Filter: only attendance / engagement reports by name pattern
            name = json_file.name.lower()
            is_report = (
                "attendance_report" in name
                or "class_dynamics_report" in name
                or "management_summary_report" in name
            )
            if not is_report:
                continue

            # Only sync files modified on/after the cutoff
            mtime = date.fromtimestamp(json_file.stat().st_mtime)
            if mtime < cutoff:
                summary["skipped"] += 1
                continue

            _log(f"Syncing: {json_file.name}")
            result = self.sync_report(str(json_file))
            summary["files"].append({
                "file": json_file.name,
                "status": result["status"],
                "rows": result.get("rows_inserted", 0),
                "error": result.get("error_msg", ""),
            })
            if result["status"] == "ok":
                summary["synced"] += 1
            elif result["status"] == "error":
                summary["errors"] += 1
                _log(f"  ERROR: {result['error_msg']}")
            else:
                summary["skipped"] += 1

        return summary

    # ------------------------------------------------------------------
    # Daily auto-sync (call once per app session)
    # ------------------------------------------------------------------

    def trigger_daily_sync(self, output_dirs: List[str],
                           log_callback=None,
                           done_callback=None) -> None:
        """
        Trigger a background daily sync.
        Runs at most once per calendar day per process.

        Args:
            output_dirs:   List of output directory paths to scan.
            log_callback:  Optional callable(str) for status messages.
            done_callback: Optional callable(dict) called with the summary.
        """
        today = date.today()
        with self._lock:
            if self._last_sync_date == today:
                return          # Already synced today
            self._last_sync_date = today

        def _worker():
            def _log(msg):
                if log_callback:
                    log_callback(msg)
                print(f"[BQ-Daily] {msg}")

            _log("Starting daily BigQuery sync...")
            try:
                self.ensure_tables()
            except Exception as e:
                _log(f"Table setup failed: {e}")
                if done_callback:
                    done_callback({"error": str(e)})
                return

            total = {"synced": 0, "skipped": 0, "errors": 0, "files": []}
            since = today - timedelta(days=1)  # sync yesterday + today

            for d in output_dirs:
                if not d:
                    continue
                _log(f"Scanning: {d}")
                s = self.sync_directory(d, since_date=since, log_callback=_log)
                total["synced"]  += s["synced"]
                total["skipped"] += s["skipped"]
                total["errors"]  += s["errors"]
                total["files"].extend(s["files"])

            _log(
                f"Daily sync complete — "
                f"synced={total['synced']}, skipped={total['skipped']}, errors={total['errors']}"
            )
            files = total.get("files") or []
            if files:
                names = [f.get("file", "") for f in files if f.get("file")]
                preview = ", ".join(names[:10])
                if len(names) > 10:
                    preview += f" … (+{len(names) - 10} more)"
                _log(f"Sync summary: {len(names)} report file(s) — {preview}")
            else:
                _log("Sync summary: no report files matched this run (check folder and date cutoff).")
            if done_callback:
                done_callback(total)

        t = threading.Thread(target=_worker, daemon=True, name="bq-daily-sync")
        t.start()


# ---------------------------------------------------------------------------
# Module-level singleton (created lazily after center_id is known)
# ---------------------------------------------------------------------------

_service_instance: Optional[BigQuerySyncService] = None


def get_sync_service(center_id: str = "",
                     credentials_path: str = "") -> BigQuerySyncService:
    """Return (or create) the global BigQuerySyncService."""
    global _service_instance
    if _service_instance is None or (center_id and _service_instance.center_id != center_id):
        _service_instance = BigQuerySyncService(
            center_id=center_id or "UnknownCenter",
            credentials_path=credentials_path
        )
    return _service_instance
