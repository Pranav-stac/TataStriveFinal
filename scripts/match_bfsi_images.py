"""Embed BFSI enrollment photos and match to master_database.db (run gallery logic)."""
from __future__ import annotations

import json
import re
import sqlite3
import sys
from pathlib import Path

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

from app.student_embeddings_sync import extract_embedding, _load_face_app


def load_master_gallery(db_path: Path) -> dict[str, list[np.ndarray]]:
    gallery: dict[str, list[np.ndarray]] = {}
    conn = sqlite3.connect(str(db_path))
    try:
        for gid, blob in conn.execute("SELECT g_id, embedding FROM exemplars"):
            gallery.setdefault(str(gid), []).append(np.frombuffer(blob, dtype=np.float32))
    finally:
        conn.close()
    return gallery


def find_match_with_margin(
    track_emb: np.ndarray,
    global_gallery: dict[str, list[np.ndarray]],
    *,
    t_strict_merge: float,
    t_ratio_margin: float,
) -> tuple[str | None, float]:
    """Same scoring as crossday_worker.find_match_with_margin (no active_gids filter)."""
    scores: list[tuple[str, float]] = []
    for g_id, ex_list in global_gallery.items():
        if not ex_list:
            continue
        best_sim = max(float(1 - cosine(track_emb, ex)) for ex in ex_list)
        scores.append((g_id, best_sim))
    if not scores:
        return None, 0.0
    scores.sort(key=lambda x: x[1], reverse=True)
    best_id, best_sim = scores[0]
    if len(scores) > 1:
        second_sim = scores[1][1]
        if (best_sim - second_sim) < t_ratio_margin and second_sim > t_strict_merge:
            return None, best_sim
    if best_sim > t_strict_merge:
        return best_id, best_sim
    return None, best_sim


def main() -> int:
    bfsi_dir = _ROOT / "bfsi"
    master_db_path = _ROOT / "master_database.db"

    t_strict = 0.45
    t_ratio = 0.05
    cfg_path = Path.home() / ".tatastrive" / "config.json"
    if cfg_path.is_file():
        crossday = json.loads(cfg_path.read_text(encoding="utf-8")).get("crossday") or {}
        t_strict = float(crossday.get("t_strict_merge", t_strict))
        t_ratio = float(crossday.get("t_ratio_margin", t_ratio))

    if not bfsi_dir.is_dir():
        print(f"BFSI folder not found: {bfsi_dir}")
        return 1
    if not master_db_path.is_file():
        print(f"Run database not found: {master_db_path}")
        return 1

    face_app = _load_face_app()
    if face_app is None:
        print("InsightFace could not load. Install deps: pip install -r requirements_app.txt")
        return 1

    master_gallery = load_master_gallery(master_db_path)
    images = sorted(bfsi_dir.glob("*.jpeg")) + sorted(bfsi_dir.glob("*.jpg")) + sorted(bfsi_dir.glob("*.png"))

    print(f"BFSI images: {len(images)}")
    print(f"master_database.db identities with exemplars: {len(master_gallery)}")
    print(f"Gallery merge thresholds: t_strict_merge={t_strict}, t_ratio_margin={t_ratio}")
    print("(Same max-exemplar cosine + margin rules as attendance BUILD_DB gallery matching)")
    print()

    no_face = 0
    matched = 0
    ambiguous = 0
    below = 0
    rows: list[tuple] = []

    for img in images:
        m = re.match(r"^(\d+)_StudentPicture_", img.name)
        true_eng = m.group(1) if m else None
        emb = extract_embedding(face_app, str(img))
        if emb is None:
            no_face += 1
            rows.append((img.name, true_eng, None, 0.0, "no_face"))
            continue
        pred_gid, best_sim = find_match_with_margin(
            emb, master_gallery, t_strict_merge=t_strict, t_ratio_margin=t_ratio
        )
        if pred_gid:
            matched += 1
            note = "matched"
        elif best_sim > t_strict:
            ambiguous += 1
            note = "ambiguous"
        else:
            below += 1
            note = "below"
        rows.append((img.name, true_eng, pred_gid, best_sim, note))

    print(f"{'file':<52} {'eng_id':<12} {'G_id':<10} {'sim':>7}  note")
    print("-" * 90)
    for name, eng, gid, sim, note in rows:
        print(f"{name[:51]:<52} {str(eng or '-'):<12} {str(gid or '-'):<10} {sim:7.3f}  {note}")

    with_faces = len(images) - no_face
    print()
    print("=== Summary: BFSI photo -> master_database.db (run gallery) ===")
    print(f"Faces embedded: {with_faces} | no face: {no_face}")
    print(f"Matched to a G_* / NF_* (accepted merge): {matched} / {with_faces}")
    print(f"Ambiguous (margin failed, sim still high): {ambiguous} / {with_faces}")
    print(f"Below t_strict_merge: {below} / {with_faces}")

    unique_g = len({r[2] for r in rows if r[2]})
    print(f"Unique gallery IDs matched: {unique_g}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
