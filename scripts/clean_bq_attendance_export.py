"""
Deduplicate BigQuery attendance CSV exports where the same person/session
is repeated once per source_video. Writes a cleaned CSV + summary JSON.

Usage:
  python scripts/clean_bq_attendance_export.py bq_summary_output/bquxjob_5e83468_19d288ba089.csv
"""
from __future__ import annotations

import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path


def id_bucket(person_id: str) -> str:
    s = (person_id or "").strip()
    if re.match(r"^Day\d+_V_", s):
        return "Day*_V_* (visitor, eval-day id)"
    if re.match(r"^Day\d+_NF_", s) or ("_NF_" in s and s.startswith("Day")):
        return "Day*_NF_* (no face, eval-day id)"
    if s.startswith("G_"):
        return "G_* (gallery / build-db visitor)"
    if s.startswith("NF_"):
        return "NF_* (no face track)"
    return "other"


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python clean_bq_attendance_export.py <input.csv>", file=sys.stderr)
        return 1
    src = Path(sys.argv[1]).resolve()
    if not src.is_file():
        print(f"Not found: {src}", file=sys.stderr)
        return 1

    out_clean = src.with_name(src.stem + "_cleaned.csv")
    out_summary = src.with_name(src.stem + "_summary.json")

    with src.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    fields = rows[0].keys() if rows else []

    # Dedupe key: same logical attendance interval for that person on that date
    key_fn = lambda r: (
        r.get("center_id", ""),
        r.get("report_date", ""),
        r.get("person_id", ""),
        r.get("entry_time", ""),
        r.get("exit_time", ""),
        str(r.get("duration_seconds", "")),
    )

    merged: dict[tuple, dict] = {}
    video_refs: dict[tuple, set[str]] = defaultdict(set)

    for r in rows:
        k = key_fn(r)
        vid = (r.get("source_video") or "").strip()
        if k in merged:
            if vid:
                video_refs[k].add(vid)
        else:
            merged[k] = dict(r)
            if vid:
                video_refs[k].add(vid)

    cleaned = list(merged.values())
    for r in cleaned:
        k = key_fn(r)
        vids = sorted(video_refs[k])
        r["dedupe_collapsed_rows"] = str(
            sum(1 for x in rows if key_fn(x) == k)
        )
        r["distinct_source_videos"] = str(len(vids))
        r["source_videos_sample"] = "; ".join(vids[:5]) + ("; ..." if len(vids) > 5 else "")

    extra_fields = ["dedupe_collapsed_rows", "distinct_source_videos", "source_videos_sample"]
    out_fields = list(fields) + [f for f in extra_fields if f not in fields]

    with out_clean.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=out_fields, extrasaction="ignore")
        w.writeheader()
        for r in sorted(
            cleaned,
            key=lambda x: (
                x.get("report_date", ""),
                x.get("person_id", ""),
                x.get("entry_time", ""),
            ),
        ):
            row = {k: r.get(k, "") for k in out_fields}
            w.writerow(row)

    # Summary per report_date
    by_date: dict[str, list[dict]] = defaultdict(list)
    for r in cleaned:
        d = r.get("report_date") or ""
        by_date[d].append(r)

    summary: dict = {
        "input_file": str(src.name),
        "input_rows": len(rows),
        "cleaned_rows": len(cleaned),
        "rows_removed": len(rows) - len(cleaned),
        "per_report_date": {},
    }

    for date in sorted(by_date.keys()):
        day_rows = by_date[date]
        types = defaultdict(int)
        buckets = defaultdict(int)
        pids = set()
        collapsed = 0
        for r in day_rows:
            types[r.get("person_type") or "unknown"] += 1
            buckets[id_bucket(r.get("person_id", ""))] += 1
            pids.add(r.get("person_id", ""))
            collapsed += int(r.get("dedupe_collapsed_rows") or 0)

        summary["per_report_date"][date] = {
            "cleaned_attendance_rows": len(day_rows),
            "unique_person_id": len(pids),
            "by_person_type": dict(types),
            "by_id_bucket": dict(buckets),
            "raw_rows_that_collapsed_into_these": collapsed,
        }

    with out_summary.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"\nWrote: {out_clean}")
    print(f"Wrote: {out_summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
