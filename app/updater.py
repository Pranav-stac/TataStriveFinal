"""
Auto-update system for TataStrive Analytics.

Checks GitHub Releases for newer versions and downloads only the changed files
(delta / patch ZIP) — never the full application.

How push-style updates work for a continuously running desktop app
------------------------------------------------------------------
Webhooks (GitHub → your URL) require the client machine to accept incoming
HTTP connections, which is impossible behind NAT/firewalls.  The equivalent
for a desktop app is a background polling loop:

  start_polling(interval_minutes=60)
      └── daemon thread wakes every 60 min
          └── hits GitHub Releases API  (fast, ~200 ms)
              └── new version found → download + apply patch in background
                  └── process restarts automatically (no clicks, no dialogs)

By default (silent_auto_apply=True) the user never interacts with the updater.
Failed installs are retried on the next poll.

Public API
----------
  UpdateChecker.start_polling()      – start the live background loop
  UpdateChecker.stop_polling()       – clean shutdown (called on app exit)
  UpdateChecker.check_once_async()   – fire a single one-shot check
  UpdateChecker.download_and_apply() – download + apply a patch in background
  UpdateChecker.restart_app()        – re-launch the process after patching
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import zipfile
from pathlib import Path
from typing import Callable, Dict, List, Optional
from urllib.request import Request, urlopen

# ─────────────────────────────────────────────────────────────────────────────
#  Configuration  ← set GITHUB_REPO to your "owner/repo" before deploying
# ─────────────────────────────────────────────────────────────────────────────
GITHUB_REPO          = os.getenv("TATASTRIVE_GITHUB_REPO", "OWNER/TataStriveFinal")
GITHUB_API           = "https://api.github.com"
UPDATE_CHECK_TIMEOUT = 10    # seconds – version check (fast, non-blocking)
DOWNLOAD_TIMEOUT     = 180   # seconds – patch ZIP download

# Default polling interval: check for updates every 60 minutes while the app
# is running.  Change to e.g. 30 for faster detection during testing.
DEFAULT_POLL_INTERVAL_MINUTES = 60

# How long to wait after startup before the first check (gives the app time
# to finish loading before hitting the network).
STARTUP_DELAY_SECONDS = 20

# Absolute path to the directory that contains run_app.py / app/
APP_ROOT = Path(__file__).resolve().parent.parent


# ─────────────────────────────────────────────────────────────────────────────
#  Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65_536), b""):
            h.update(chunk)
    return h.hexdigest()


def _get_json(url: str, timeout: int = UPDATE_CHECK_TIMEOUT) -> Optional[Dict]:
    try:
        req = Request(
            url,
            headers={
                "Accept":     "application/vnd.github+json",
                "User-Agent": "TataStriveUpdater/1.0",
            },
        )
        with urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        print(f"[Updater] fetch error: {exc}")
        return None


def _download_file(
    url: str,
    dest: Path,
    timeout: int = DOWNLOAD_TIMEOUT,
    progress_cb: Optional[Callable[[int, int], None]] = None,
) -> bool:
    try:
        req = Request(url, headers={"User-Agent": "TataStriveUpdater/1.0"})
        with urlopen(req, timeout=timeout) as resp:
            total = int(resp.headers.get("Content-Length", 0))
            done  = 0
            dest.parent.mkdir(parents=True, exist_ok=True)
            with open(dest, "wb") as fh:
                while True:
                    chunk = resp.read(65_536)
                    if not chunk:
                        break
                    fh.write(chunk)
                    done += len(chunk)
                    if progress_cb:
                        progress_cb(done, total)
        return True
    except Exception as exc:
        print(f"[Updater] download error: {exc}")
        return False


def _is_newer(remote: str, current: str) -> bool:
    def _parse(v: str):
        try:
            return tuple(int(x) for x in v.strip().lstrip("v").split("."))
        except ValueError:
            return (0,)
    return _parse(remote) > _parse(current)


# ─────────────────────────────────────────────────────────────────────────────
#  Data class
# ─────────────────────────────────────────────────────────────────────────────

class UpdateInfo:
    """Describes an available update fetched from GitHub Releases."""

    def __init__(
        self,
        version:   str,
        changelog: str,
        manifest:  Dict,
        patch_url: str,
    ) -> None:
        self.version   = version
        self.changelog = changelog
        self.manifest  = manifest
        self.patch_url = patch_url

    @property
    def changed_files(self) -> List[str]:
        return [f["path"] for f in self.manifest.get("files", [])]

    def __repr__(self) -> str:
        return f"<UpdateInfo v{self.version} — {len(self.changed_files)} file(s)>"


# ─────────────────────────────────────────────────────────────────────────────
#  Main class
# ─────────────────────────────────────────────────────────────────────────────

class UpdateChecker:
    """
    Continuously polls GitHub Releases for newer versions while the app runs.

    With ``silent_auto_apply=True`` (default), a new release is downloaded and
    applied without any UI; the process restarts when the patch is applied.

    With ``silent_auto_apply=False``, set ``on_update_found`` to show a dialog
    or custom UI.
    """

    def __init__(
        self,
        current_version: str,
        repo: str = GITHUB_REPO,
        poll_interval_minutes: int = DEFAULT_POLL_INTERVAL_MINUTES,
        silent_auto_apply: bool = True,
    ) -> None:
        self.current_version    = current_version
        self.repo               = repo
        self.poll_interval      = poll_interval_minutes * 60  # convert to seconds
        self.silent_auto_apply  = silent_auto_apply
        self.on_update_found:   Optional[Callable[[UpdateInfo], None]] = None
        self.on_error:          Optional[Callable[[str], None]]        = None

        self._stop_event = threading.Event()
        self._poll_thread: Optional[threading.Thread] = None
        self._install_lock = threading.Lock()
        self._apply_in_progress = False
        # When not using silent mode: notify at most once per version per run.
        self._notified_versions: set[str] = set()

    # ── Polling lifecycle ─────────────────────────────────────────────────────

    def start_polling(self) -> None:
        """
        Start the background polling loop.
        Waits STARTUP_DELAY_SECONDS before the first check so the app can
        finish initialising, then re-checks every poll_interval seconds.
        Safe to call from the Qt main thread.
        """
        if self._poll_thread and self._poll_thread.is_alive():
            return  # already running

        self._stop_event.clear()
        self._poll_thread = threading.Thread(
            target=self._poll_loop,
            daemon=True,
            name="TataStriveUpdatePoller",
        )
        self._poll_thread.start()
        print(
            f"[Updater] Polling started — "
            f"first check in {STARTUP_DELAY_SECONDS}s, "
            f"then every {self.poll_interval // 60} min"
        )

    def stop_polling(self) -> None:
        """Signal the background loop to stop cleanly (call on app exit)."""
        self._stop_event.set()

    # ── One-shot async check (kept for optional manual trigger) ───────────────

    def check_once_async(self) -> None:
        """Fire a single background check immediately, outside the polling loop."""
        threading.Thread(
            target=self._run_single_check,
            daemon=True,
            name="TataStriveUpdateOneShot",
        ).start()

    # ── Apply update ──────────────────────────────────────────────────────────

    def download_and_apply(
        self,
        info:        UpdateInfo,
        progress_cb: Optional[Callable[[int, int], None]]  = None,
        done_cb:     Optional[Callable[[bool, str], None]] = None,
    ) -> None:
        """Download patch.zip and overwrite only the changed files (background)."""
        threading.Thread(
            target=self._apply_worker,
            args=(info, progress_cb, done_cb),
            daemon=True,
            name="TataStriveUpdateApplier",
        ).start()

    @staticmethod
    def restart_app() -> None:
        """Re-launch the current Python / frozen-exe process."""
        exe  = sys.executable
        args = sys.argv[:]
        print("[Updater] Restarting application…")
        if getattr(sys, "frozen", False):
            os.execv(exe, [exe] + args)
        else:
            subprocess.Popen([exe] + args)
            sys.exit(0)

    # ── Background workers ────────────────────────────────────────────────────

    def _poll_loop(self) -> None:
        """
        Daemon thread body.
        Sleeps for STARTUP_DELAY_SECONDS, then checks every poll_interval.
        Uses threading.Event.wait() so stop_polling() wakes it immediately.
        """
        # Initial startup delay — don't hammer the network before the UI loads
        if self._stop_event.wait(timeout=STARTUP_DELAY_SECONDS):
            return  # stop was requested during startup delay

        while not self._stop_event.is_set():
            self._run_single_check()
            # Sleep in small increments so stop_polling() is responsive
            self._stop_event.wait(timeout=self.poll_interval)

    def _run_single_check(self) -> None:
        try:
            info = self._fetch_latest_info()
        except Exception as exc:
            if self.on_error:
                self.on_error(str(exc))
            return

        if info is None:
            return

        if self.silent_auto_apply:
            self._begin_silent_apply(info)
            return

        if info.version in self._notified_versions:
            return

        self._notified_versions.add(info.version)
        if self.on_update_found:
            self.on_update_found(info)

    def _begin_silent_apply(self, info: UpdateInfo) -> None:
        """Download and install without UI; restart on success; retry next poll on failure."""
        with self._install_lock:
            if self._apply_in_progress:
                return
            self._apply_in_progress = True

        print(
            f"[Updater] New version v{info.version} available — "
            f"downloading and installing automatically ({len(info.changed_files)} file(s))…"
        )
        self.download_and_apply(info, done_cb=self._silent_apply_done)

    def _silent_apply_done(self, success: bool, message: str) -> None:
        with self._install_lock:
            self._apply_in_progress = False

        if success:
            print(f"[Updater] {message} Restarting…")
            self.restart_app()
        else:
            print(f"[Updater] Automatic update failed — will retry on next poll: {message}")

    def _fetch_latest_info(self) -> Optional[UpdateInfo]:
        url  = f"{GITHUB_API}/repos/{self.repo}/releases/latest"
        data = _get_json(url)
        if not data or "tag_name" not in data:
            return None

        remote_ver = data["tag_name"].lstrip("v")
        if not _is_newer(remote_ver, self.current_version):
            print(f"[Updater] Up to date (local={self.current_version}, remote={remote_ver})")
            return None

        assets = {
            a["name"]: a["browser_download_url"]
            for a in data.get("assets", [])
        }
        manifest_url = assets.get("manifest.json")
        patch_url    = assets.get("patch.zip")

        if not manifest_url or not patch_url:
            print("[Updater] Release assets incomplete — skipping.")
            return None

        manifest = _get_json(manifest_url)
        if not manifest:
            return None

        # Compute true delta: skip files whose local hash already matches
        changed_entries = [
            entry for entry in manifest.get("files", [])
            if not (APP_ROOT / entry["path"]).exists()
            or _sha256(APP_ROOT / entry["path"]) != entry["sha256"]
        ]

        if not changed_entries:
            print("[Updater] Remote version newer but all local files already match.")
            return None

        return UpdateInfo(
            version=remote_ver,
            changelog=(data.get("body") or "").strip() or "No release notes.",
            manifest={**manifest, "files": changed_entries},
            patch_url=patch_url,
        )

    def _apply_worker(
        self,
        info:        UpdateInfo,
        progress_cb: Optional[Callable],
        done_cb:     Optional[Callable],
    ) -> None:
        tmp_dir    = Path(tempfile.mkdtemp(prefix="tatastrive_upd_"))
        patch_path = tmp_dir / "patch.zip"
        backup_dir = tmp_dir / "backup"

        try:
            ok = _download_file(info.patch_url, patch_path, progress_cb=progress_cb)
            if not ok:
                if done_cb:
                    done_cb(False, "Download failed. Check your internet connection.")
                return

            if not zipfile.is_zipfile(patch_path):
                if done_cb:
                    done_cb(False, "Downloaded file is corrupted (not a valid ZIP).")
                return

            # Back up files that will be overwritten
            with zipfile.ZipFile(patch_path, "r") as zf:
                for name in zf.namelist():
                    target = APP_ROOT / name
                    if target.exists():
                        bk = backup_dir / name
                        bk.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(target, bk)

            # Extract patch
            with zipfile.ZipFile(patch_path, "r") as zf:
                zf.extractall(APP_ROOT)

            n = len(info.changed_files)
            if done_cb:
                done_cb(True, f"Updated to v{info.version} ({n} file(s) patched).")

        except Exception as exc:
            # Rollback
            try:
                if backup_dir.exists():
                    for src in backup_dir.rglob("*"):
                        if src.is_file():
                            rel  = src.relative_to(backup_dir)
                            dest = APP_ROOT / rel
                            dest.parent.mkdir(parents=True, exist_ok=True)
                            dest.write_bytes(src.read_bytes())
                print("[Updater] Rollback complete.")
            except Exception as rb_exc:
                print(f"[Updater] Rollback failed: {rb_exc}")

            if done_cb:
                done_cb(False, f"Update failed: {exc}")
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)
