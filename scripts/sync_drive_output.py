#!/usr/bin/env python3
"""Download the full Drive output folder to LOCAL_OUTPUT_DIR for offline verification."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")
load_dotenv(ROOT / ".env.publish", override=False)


def _fmt_bytes(n: int) -> str:
    if n <= 0:
        return "0 B"
    units = ["B", "KB", "MB", "GB", "TB"]
    v = float(n)
    i = 0
    while v >= 1024 and i < len(units) - 1:
        v /= 1024
        i += 1
    return f"{v:.1f} {units[i]}"


def sync_output_folder(*, force: bool = False) -> None:
    from attendance_verify.config import DRIVE_OUTPUT_FOLDER_ID, LOCAL_OUTPUT_DIR
    from attendance_verify.drive_storage import (
        _is_video_file,
        _list_child_folders,
        download_drive_file,
        list_all_files_in_folder,
        local_file_complete,
    )

    LOCAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Syncing Drive folder {DRIVE_OUTPUT_FOLDER_ID}")
    print(f"  -> {LOCAL_OUTPUT_DIR}")

    run_folders = _list_child_folders(DRIVE_OUTPUT_FOLDER_ID)
    if not run_folders:
        print("No subfolders found under output folder (check sharing / folder ID).")
        return

    total_videos = 0
    skipped = 0
    downloaded = 0
    failed = 0

    for idx, folder in enumerate(run_folders, 1):
        folder_id = folder.get("id")
        name = folder.get("name") or folder_id
        if not folder_id:
            continue
        dest_dir = LOCAL_OUTPUT_DIR / name
        dest_dir.mkdir(parents=True, exist_ok=True)

        files = list_all_files_in_folder(folder_id)
        videos = [f for f in files if _is_video_file(f)]
        if not videos:
            continue

        for vf in videos:
            total_videos += 1
            file_id = vf["id"]
            fname = vf.get("name") or file_id
            dest = dest_dir / fname
            expected = int(vf.get("size") or 0)

            if not force and local_file_complete(dest, expected, file_id=file_id):
                skipped += 1
                size = dest.stat().st_size
                print(f"  [{idx}/{len(run_folders)}] skip {name}/{fname} ({_fmt_bytes(size)} already on disk)")
                continue

            print(f"  [{idx}/{len(run_folders)}] download {name}/{fname} ({_fmt_bytes(expected)})")

            def on_progress(done: int, total: int) -> None:
                total = total or expected
                if total <= 0:
                    return
                pct = min(100, int(100 * done / total))
                print(f"\r    {_fmt_bytes(done)} / {_fmt_bytes(total)} ({pct}%)", end="", flush=True)

            try:
                download_drive_file(file_id, dest, on_progress=on_progress)
                print()
                downloaded += 1
            except Exception as exc:
                print()
                failed += 1
                print(f"    FAILED: {exc}")

    print()
    print(f"Done: {downloaded} downloaded, {skipped} skipped, {failed} failed, {total_videos} videos total.")
    print(f"Local output: {LOCAL_OUTPUT_DIR}")
    print("Restart the portal — videos will play from disk (no Drive streaming).")


def main() -> None:
    force = "--force" in sys.argv
    try:
        sync_output_folder(force=force)
    except KeyboardInterrupt:
        print("\nInterrupted — partial files kept; re-run to resume.")
        sys.exit(130)
    except Exception as exc:
        print(f"Sync failed: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
