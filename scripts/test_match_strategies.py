"""
Compare embedding match strategies: enrollment (BFSI) vs CCTV run gallery (master_database.db).

Strategies (probe = enrollment photo, gallery = per-G_id CCTV exemplars):
  max_exemplar  — current app style: max cosine probe vs any exemplar
  centroid      — cosine probe vs L2-normalized mean of exemplars
  top3_mean     — mean of top-3 probe-vs-exemplar cosines per G_id

Usage:
  python scripts/test_match_strategies.py
  python scripts/test_match_strategies.py --master path/to/master_database.db
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from scipy.spatial.distance import cosine

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv(_ROOT / ".env", override=False)
    if (_ROOT / "dist" / "TataStriveAnalytics" / ".env").is_file():
        load_dotenv(_ROOT / "dist" / "TataStriveAnalytics" / ".env", override=False)
except ImportError:
    pass

from app.student_embeddings_sync import _load_face_app, extract_embedding, load_enrolled_student_gallery


def l2norm(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n <= 0:
        return v
    return v / n


def cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(1 - cosine(l2norm(a), l2norm(b)))


def load_master_gallery(db_path: Path) -> dict[str, list[np.ndarray]]:
    gallery: dict[str, list[np.ndarray]] = {}
    conn = sqlite3.connect(str(db_path))
    try:
        for gid, blob in conn.execute("SELECT g_id, embedding FROM exemplars"):
            gallery.setdefault(str(gid), []).append(np.frombuffer(blob, dtype=np.float32).copy())
    finally:
        conn.close()
    return gallery


ScoreFn = Callable[[np.ndarray, list[np.ndarray]], float]

STRATEGIES: dict[str, ScoreFn] = {
    "max_exemplar": lambda probe, exs: max(cos_sim(probe, ex) for ex in exs),
    "centroid": lambda probe, exs: cos_sim(probe, np.mean(exs, axis=0)),
    "top3_mean": lambda probe, exs: float(
        np.mean(sorted((cos_sim(probe, ex) for ex in exs), reverse=True)[: min(3, len(exs))])
    ),
}


@dataclass
class MatchResult:
    g_id: str | None
    best_sim: float
    second_sim: float
    ambiguous: bool
    accepted: bool


def rank_gallery(
    probe: np.ndarray,
    gallery: dict[str, list[np.ndarray]],
    score_fn: ScoreFn,
) -> list[tuple[str, float]]:
    scores: list[tuple[str, float]] = []
    for g_id, exs in gallery.items():
        if not exs:
            continue
        scores.append((g_id, score_fn(probe, exs)))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores


def decide(
    scores: list[tuple[str, float]],
    *,
    threshold: float,
    ratio_margin: float,
) -> MatchResult:
    if not scores:
        return MatchResult(None, 0.0, 0.0, False, False)
    best_id, best_sim = scores[0]
    second_sim = scores[1][1] if len(scores) > 1 else 0.0
    ambiguous = (
        len(scores) > 1
        and (best_sim - second_sim) < ratio_margin
        and second_sim > threshold
    )
    accepted = best_sim > threshold and not ambiguous
    return MatchResult(
        best_id if accepted else None,
        best_sim,
        second_sim,
        ambiguous,
        accepted,
    )


def load_config() -> tuple[float, float, float]:
    t_gallery = 0.45
    t_student = 0.40
    t_ratio = 0.10
    cfg_path = Path.home() / ".tatastrive" / "config.json"
    if cfg_path.is_file():
        crossday = json.loads(cfg_path.read_text(encoding="utf-8")).get("crossday") or {}
        t_gallery = float(crossday.get("t_strict_merge", t_gallery))
        t_student = float(crossday.get("t_match_student", t_student))
        t_ratio = float(crossday.get("t_ratio_margin", t_ratio))
    return t_gallery, t_student, t_ratio


def embed_bfsi(face_app, bfsi_dir: Path) -> list[tuple[str, str | None, np.ndarray | None]]:
    rows: list[tuple[str, str | None, np.ndarray | None]] = []
    images = sorted(bfsi_dir.glob("*.jpeg")) + sorted(bfsi_dir.glob("*.jpg"))
    for img in images:
        m = re.match(r"^(\d+)_StudentPicture_", img.name)
        eng = m.group(1) if m else None
        emb = extract_embedding(face_app, str(img))
        rows.append((img.name, eng, emb))
    return rows


def summarize(
    label: str,
    probes: list[tuple[str, str | None, np.ndarray]],
    gallery: dict[str, list[np.ndarray]],
    strategy: str,
    threshold: float,
    ratio_margin: float,
) -> dict[str, int]:
    score_fn = STRATEGIES[strategy]
    accepted = ambiguous = below = 0
    sims: list[float] = []
    for _name, _eng, emb in probes:
        scores = rank_gallery(emb, gallery, score_fn)
        if not scores:
            below += 1
            continue
        sims.append(scores[0][1])
        r = decide(scores, threshold=threshold, ratio_margin=ratio_margin)
        if r.accepted:
            accepted += 1
        elif r.ambiguous:
            ambiguous += 1
        else:
            below += 1
    print(
        f"  {label:<22} thr={threshold:.2f}  accepted={accepted:2d}/{len(probes)}  "
        f"ambiguous={ambiguous:2d}  below={below:2d}  "
        f"max_sim={max(sims) if sims else 0:.3f}  median={np.median(sims) if sims else 0:.3f}"
    )
    return {"accepted": accepted, "ambiguous": ambiguous, "below": below}


def print_detail_table(
    probes: list[tuple[str, str | None, np.ndarray]],
    gallery: dict[str, list[np.ndarray]],
    strategy: str,
    threshold: float,
    ratio_margin: float,
) -> None:
    score_fn = STRATEGIES[strategy]
    print(f"\n--- Per-image ({strategy}, thr={threshold}, margin={ratio_margin}) ---")
    print(f"{'eng_id':<12} {'G_id':<10} {'sim':>7} {'2nd':>7}  status")
    print("-" * 50)
    for _name, eng, emb in probes:
        scores = rank_gallery(emb, gallery, score_fn)
        r = decide(scores, threshold=threshold, ratio_margin=ratio_margin)
        status = "match" if r.accepted else ("ambig" if r.ambiguous else "below")
        print(
            f"{str(eng or '-'):<12} {str(r.g_id or '-'):<10} {r.best_sim:7.3f} {r.second_sim:7.3f}  {status}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Test enrollment vs CCTV match strategies")
    parser.add_argument("--master", type=Path, default=_ROOT / "master_database.db")
    parser.add_argument("--bfsi", type=Path, default=_ROOT / "bfsi")
    parser.add_argument("--student-db", type=Path, default=_ROOT / "student_enrollments.db")
    parser.add_argument("--detail", action="store_true", help="Print per-image table for best strategy")
    args = parser.parse_args()

    t_gallery, t_student, t_ratio = load_config()

    if not args.bfsi.is_dir():
        print(f"BFSI folder missing: {args.bfsi}")
        return 1
    if not args.master.is_file():
        print(f"master DB missing: {args.master}")
        return 1

    face_app = _load_face_app()
    if face_app is None:
        print("InsightFace failed to load.")
        return 1

    master = load_master_gallery(args.master)
    student_db = load_enrolled_student_gallery(args.student_db) if args.student_db.is_file() else {}

    # Student DB as gallery: one exemplar per engagement_id (enrollment-to-enrollment baseline)
    student_gallery = {
        str(eid): list(data.get("exemplars") or [])
        for eid, data in student_db.items()
        if data.get("exemplars")
    }

    raw = embed_bfsi(face_app, args.bfsi)
    probes = [(n, e, emb) for n, e, emb in raw if emb is not None]
    no_face = len(raw) - len(probes)

    print("Enrollment (BFSI) vs CCTV run gallery — strategy comparison")
    print(f"BFSI embedded: {len(probes)} | no face: {no_face}")
    print(f"master_database: {len(master)} identities, {sum(len(v) for v in master.values())} exemplars")
    print(f"student_enrollments: {len(student_gallery)} students")
    print(f"Config: t_strict_merge={t_gallery}, t_match_student={t_student}, t_ratio_margin={t_ratio}")
    print()
    print("=== BFSI (enrollment) -> master_database.db (CCTV exemplars) ===")
    print("Threshold uses t_strict_merge (gallery) and looser cross-domain sweeps:")
    for strategy in STRATEGIES:
        print(f"\n[{strategy}]")
        for thr in (t_gallery, 0.40, 0.35, 0.30):
            summarize(f"{strategy}", probes, master, strategy, thr, t_ratio)

    print("\n=== BFSI -> student_enrollments.db (enrollment embeddings, sanity check) ===")
    print("(Same strategies; 1 exemplar/student so scores usually match max_exemplar.)")
    overlap = {e for _, e, _ in probes if e} & set(student_gallery)
    print(f"BFSI engagement_ids present in student DB: {len(overlap)} / {len(probes)}")
    if student_gallery:
        for strategy in STRATEGIES:
            print(f"\n[{strategy}]")
            summarize(f"{strategy}", probes, student_gallery, strategy, t_student, t_ratio)

    # Pick best strategy at cross-domain threshold 0.35 for detail view
    if args.detail:
        print_detail_table(probes, master, "centroid", 0.35, t_ratio)
        print_detail_table(probes, master, "top3_mean", 0.35, t_ratio)
        print_detail_table(probes, master, "max_exemplar", 0.35, t_ratio)

    print("\nDone. Compare 'accepted' counts across strategies at thr=0.35–0.40 for cross-domain.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
