"""
BigQuery Sync Service for TataStrive Analytics.

Handles:
 - Automatic BigQuery table creation (attendance + engagement)
 - Syncing completed report JSON files to BigQuery
 - Supports both class dynamics and management summary engagement reports
 - Daily scheduled sync (runs once per day at startup / after analysis)
 - Center-ID isolation: every row carries the device's center_id
"""

from __future__ import annotations

import json
import os
import sys
import threading
import traceback
from datetime import date, datetime, timedelta
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


# ---------------------------------------------------------------------------
# BigQuery dataset / table names
# ---------------------------------------------------------------------------
DATASET_ID   = "tatastrive_analytics"
ATTENDANCE_TABLE   = "attendance_reports"
ENGAGEMENT_TABLE   = "engagement_reports"
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
    {"name": "camera",             "type": "STRING",    "mode": "NULLABLE"},
    {"name": "source_video",       "type": "STRING",    "mode": "NULLABLE"},
    {"name": "session_duration",   "type": "STRING",    "mode": "NULLABLE"},
    # Counts
    {"name": "unique_people",      "type": "INTEGER",   "mode": "NULLABLE"},
    {"name": "returning_count",    "type": "INTEGER",   "mode": "NULLABLE"},
    {"name": "visitor_count",      "type": "INTEGER",   "mode": "NULLABLE"},
    {"name": "identified_students","type": "INTEGER",   "mode": "NULLABLE"},
    # Per-person (repeated)
    {"name": "person_id",          "type": "STRING",    "mode": "NULLABLE"},
    {"name": "person_type",        "type": "STRING",    "mode": "NULLABLE"},
    {"name": "engagement_id",      "type": "STRING",    "mode": "NULLABLE"},
    {"name": "batch",              "type": "STRING",    "mode": "NULLABLE"},
    {"name": "entry_time",         "type": "STRING",    "mode": "NULLABLE"},
    {"name": "exit_time",          "type": "STRING",    "mode": "NULLABLE"},
    {"name": "duration_seconds",   "type": "INTEGER",   "mode": "NULLABLE"},
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
    {"name": "classroom",            "type": "STRING",    "mode": "NULLABLE"},
    {"name": "recording_date_str",   "type": "STRING",    "mode": "NULLABLE"},
    {"name": "baseline_max_students","type": "INTEGER",   "mode": "NULLABLE"},
    {"name": "report_type",          "type": "STRING",    "mode": "NULLABLE"},
    # Event duration summary
    {"name": "lecture_sec",          "type": "FLOAT",     "mode": "NULLABLE"},
    {"name": "activity_sec",         "type": "FLOAT",     "mode": "NULLABLE"},
    {"name": "chaos_sec",            "type": "FLOAT",     "mode": "NULLABLE"},
    {"name": "break_sec",            "type": "FLOAT",     "mode": "NULLABLE"},
    # Per-probe (one row per probe)
    {"name": "probe_index",          "type": "STRING",    "mode": "NULLABLE"},
    {"name": "video_timestamp_sec",  "type": "FLOAT",     "mode": "NULLABLE"},
    {"name": "real_world_time",      "type": "STRING",    "mode": "NULLABLE"},
    {"name": "student_count",        "type": "INTEGER",   "mode": "NULLABLE"},
    {"name": "avg_engagement",       "type": "FLOAT",     "mode": "NULLABLE"},
    {"name": "class_mode",           "type": "STRING",    "mode": "NULLABLE"},
    {"name": "activity_distribution","type": "STRING",    "mode": "NULLABLE"},  # JSON string
    {"name": "attention_distribution","type": "STRING",   "mode": "NULLABLE"},  # JSON string
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
]


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
        self._project_id = "tatastrive-269409"
        self._client = None
        self._lock = threading.Lock()
        self._last_sync_date: Optional[date] = None

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
            (SYNC_LOG_TABLE,    SYNC_LOG_SCHEMA),
        ]:
            full_table = f"{dataset_ref}.{table_name}"
            schema = [
                bigquery.SchemaField(f["name"], f["type"], mode=f.get("mode", "NULLABLE"))
                for f in schema_dicts
            ]
            table = bigquery.Table(full_table, schema=schema)
            # Partition by report_date for attendance/engagement tables
            if table_name in (ATTENDANCE_TABLE, ENGAGEMENT_TABLE):
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

    # ------------------------------------------------------------------
    # Report detection & routing
    # ------------------------------------------------------------------

    def detect_report_type(self, report_data: Dict) -> str:
        """Return 'attendance', 'engagement', or 'unknown'."""
        if "hourly_probes" in report_data or "sessions" in report_data:
            return "engagement"
        if "People" in report_data and "Session" in report_data:
            return "attendance"
        return "unknown"

    def sync_report(self, report_path: str) -> Dict[str, Any]:
        """
        Parse a JSON report file and insert rows into BigQuery.
        Returns a dict with keys: status, rows_inserted, error_msg.
        """
        result = {"status": "ok", "rows_inserted": 0, "error_msg": ""}
        try:
            self.ensure_tables()  # Guarantee migration runs (adds engagement_id, etc.)
            with open(report_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            rtype = self.detect_report_type(data)
            if rtype == "attendance":
                rows = self._build_attendance_rows(data, report_path)
                self._insert_rows(ATTENDANCE_TABLE, rows)
            elif rtype == "engagement":
                rows = self._build_engagement_rows(data, report_path)
                self._insert_rows(ENGAGEMENT_TABLE, rows)
            else:
                result["status"] = "skipped"
                result["error_msg"] = "Unknown report type"
                return result

            result["rows_inserted"] = len(rows)
            self._log_sync(report_path, rtype, len(rows), "ok", "")

        except Exception as e:
            msg = f"{e}\n{traceback.format_exc()}"
            result["status"] = "error"
            result["error_msg"] = str(e)
            rtype = "unknown"
            try:
                with open(report_path) as f:
                    d = json.load(f)
                rtype = self.detect_report_type(d)
            except Exception:
                pass
            self._log_sync(report_path, rtype, 0, "error", str(e)[:1000])

        return result

    # ------------------------------------------------------------------
    # Row builders
    # ------------------------------------------------------------------

    def _now_ts(self) -> str:
        """Return UTC timestamp in ISO 8601 format for BigQuery TIMESTAMP fields."""
        return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"

    def _parse_report_date(self, data: Dict) -> Optional[str]:
        """Try to extract a YYYY-MM-DD date from the report."""
        # Attendance report
        session_date = (data.get("Session") or {}).get("date")
        if session_date:
            try:
                datetime.strptime(session_date, "%Y-%m-%d")
                return session_date
            except ValueError:
                pass
        # Engagement report: recording_date may be a datetime string
        rec_date = data.get("recording_date", "") or ""
        if rec_date and len(rec_date) >= 10:
            return rec_date[:10]
        return None

    def _build_attendance_rows(self, data: Dict, report_path: str) -> List[Dict]:
        """
        One BigQuery row per person in the attendance report.
        Plus always at least one summary row (even if no people).
        """
        ts = self._now_ts()
        session = data.get("Session", {})
        counts  = data.get("Counts", {})
        people  = data.get("People", [])
        rdate   = self._parse_report_date(data)
        fname   = Path(report_path).name

        base = {
            "center_id":        self.center_id,
            "sync_timestamp":   ts,
            "report_date":      rdate,
            "session_date":     session.get("date"),
            "camera":           session.get("camera"),
            "source_video":     session.get("source_video"),
            "session_duration": session.get("duration"),
            "unique_people":    counts.get("unique_people", 0),
            "returning_count":  counts.get("returning", 0),
            "visitor_count":    counts.get("visitors", 0),
            "identified_students": counts.get("identified_students", 0),
            "report_file":      fname,
        }

        if not people:
            row = dict(base)
            row.update({
                "person_id": None, "person_type": None, "engagement_id": None, "batch": None, "entry_time": None,
                "exit_time": None, "duration_seconds": None,
                "confidence_score": None, "present_last_7_days": None, "last_present_date": None,
            })
            return [row]

        rows = []
        for person in people:
            row = dict(base)
            row.update({
                "person_id":           person.get("id"),
                "person_type":         person.get("type"),
                "engagement_id":       person.get("engagement_id"),
                "batch":               person.get("batch"),
                "entry_time":          person.get("entry"),
                "exit_time":           person.get("exit"),
                "duration_seconds":    person.get("duration_sec"),
                "confidence_score":    person.get("confidence_score"),
                "present_last_7_days": person.get("present_last_7_days"),
                "last_present_date":   person.get("last_present_date"),
            })
            rows.append(row)
        return rows

    def _build_engagement_rows(self, data: Dict, report_path: str) -> List[Dict]:
        """
        One BigQuery row per probe/session in engagement reports.
        Supports both:
         - class_dynamics_report.json (hourly_probes)
         - management_summary_report.json (sessions)
        """
        ts     = self._now_ts()
        rdate  = self._parse_report_date(data)
        fname  = Path(report_path).name
        probes = data.get("hourly_probes", [])
        sessions = data.get("sessions", [])
        dur    = data.get("event_duration_summary", {})

        base = {
            "center_id":             self.center_id,
            "sync_timestamp":        ts,
            "report_date":           rdate,
            "classroom":             data.get("classroom"),
            "recording_date_str":    data.get("recording_date"),
            "baseline_max_students": data.get("baseline_max_students"),
            "report_type":           data.get("report_type"),
            "lecture_sec":           dur.get("Lecture_sec"),
            # Support old + new naming from classroom mode updates.
            "activity_sec":          dur.get("Activity_sec", dur.get("Interactive_sec")),
            "chaos_sec":             dur.get("Chaos_sec", dur.get("TransitionSparse_sec")),
            "break_sec":             dur.get("Break_sec"),
            "report_file":           fname,
        }

        if probes:
            rows = []
            for probe in probes:
                row = dict(base)
                row.update({
                    "probe_index":           probe.get("time_slice"),
                    "video_timestamp_sec":   probe.get("video_timestamp_sec"),
                    "real_world_time":       probe.get("real_world_time"),
                    "student_count":         probe.get("student_count_corrected") or probe.get("student_count"),
                    "avg_engagement":        probe.get("avg_engagement"),
                    "class_mode":            probe.get("class_mode"),
                    "activity_distribution": json.dumps(probe.get("activity_distribution", {})),
                    "attention_distribution":json.dumps(probe.get("attention_distribution", {})),
                })
                rows.append(row)
            return rows

        if sessions:
            rows = []
            for idx, session in enumerate(sessions, start=1):
                row = dict(base)
                behavior_profile = session.get("behavior_profile", {})
                row.update({
                    "probe_index":            f"Session {idx}",
                    "video_timestamp_sec":    None,
                    "real_world_time":        session.get("time_window"),
                    "student_count":          session.get("avg_student_count"),
                    "avg_engagement":         session.get("overall_engagement_score"),
                    "class_mode":             session.get("session_mode"),
                    "activity_distribution":  json.dumps(behavior_profile),
                    "attention_distribution": None,
                })
                rows.append(row)
            return rows

        if not probes and not sessions:
            row = dict(base)
            row.update({
                "probe_index": None, "video_timestamp_sec": None,
                "real_world_time": None, "student_count": None,
                "avg_engagement": None, "class_mode": None,
                "activity_distribution": None, "attention_distribution": None,
            })
            return [row]

    # ------------------------------------------------------------------
    # Insert
    # ------------------------------------------------------------------

    def _insert_rows(self, table_name: str, rows: List[Dict]) -> None:
        if not rows:
            return
        client = self._get_client()
        full_table = f"{self._project_id}.{DATASET_ID}.{table_name}"
        try:
            errors = client.insert_rows_json(full_table, rows)
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

    def _log_sync(self, report_path: str, rtype: str,
                  rows: int, status: str, error_msg: str) -> None:
        try:
            self._insert_rows(SYNC_LOG_TABLE, [{
                "center_id":     self.center_id,
                "sync_ts":       self._now_ts(),
                "report_file":   Path(report_path).name,
                "report_type":   rtype,
                "rows_inserted": rows,
                "status":        status,
                "error_msg":     error_msg,
            }])
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
