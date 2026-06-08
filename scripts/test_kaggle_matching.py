"""
Replicate Kaggle notebook matching on local BFSI + master_database.db.

Notebook patterns:
  - smart_enroll: enrollment photo vs G_* only (skip STU_*), T_MERGE=0.45
  - map_students_max: max(cctv_emb, stu_emb) over all pairs, T_MATCH=0.40
  - map_students_centroid: L2-normalized mean(CCTV exemplars) vs students, T_MATCH=0.40
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cosine

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(_ROOT / ".env", override=False)
except ImportError:
    pass

from app.student_embeddings_sync import _load_face_app, extract_embedding, load_enrolled_student_gallery
from scripts.test_match_strategies import load_master_gallery, embed_bfsi


def cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(1 - cosine(np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32)))


def gallery_g_only(gallery: dict[str, list[np.ndarray]]) -> dict[str, list[np.ndarray]]:
    return {k: v for k, v in gallery.items() if str(k).startswith("G_")}


def smart_enroll_merge(
    probe: np.ndarray,
    gallery: dict[str, list[np.ndarray]],
    threshold: float = 0.45,
) -> tuple[str | None, float]:
    """Notebook: compare enrollment to historical G_* only (skip STU_*)."""
    best_id, best_sim = None, -1.0
    for old_id, exs in gallery_g_only(gallery).items():
        if not exs:
            continue
        max_sim = max(cos_sim(probe, ex) for ex in exs)
        if max_sim > best_sim:
            best_sim = max_sim
            best_id = old_id
    if best_sim >= threshold and best_id:
        return best_id, best_sim
    return None, best_sim


def map_students_max_to_max(
    cctv_exemplars: list[np.ndarray],
    student_db: dict,
    threshold: float = 0.40,
) -> tuple[str | None, float]:
    """Notebook map_identities_to_students (final version)."""
    best_eng, best_sim = None, -1.0
    for eng_id, stu in student_db.items():
        for cctv_emb in cctv_exemplars:
            for stu_emb in (stu.get("exemplars") or []):
                s = cos_sim(cctv_emb, stu_emb)
                if s > best_sim:
                    best_sim = s
                    best_eng = eng_id
    if best_sim > threshold and best_eng:
        return str(best_eng), best_sim
    return None, best_sim


def map_students_centroid(
    cctv_exemplars: list[np.ndarray],
    student_db: dict,
    threshold: float = 0.40,
) -> tuple[str | None, float]:
    """Notebook earlier version: CCTV track centroid vs students."""
    if not cctv_exemplars:
        return None, -1.0
    g_centroid = np.mean(cctv_exemplars, axis=0)
    g_centroid = g_centroid / np.linalg.norm(g_centroid)
    best_eng, best_sim = None, -1.0
    for eng_id, stu in student_db.items():
        for stu_emb in (stu.get("exemplars") or []):
            s = cos_sim(g_centroid, stu_emb)
            if s > best_sim:
                best_sim = s
                best_eng = eng_id
    if best_sim > threshold and best_eng:
        return str(best_eng), best_sim
    return None, best_sim


def main() -> int:
    bfsi_dir = _ROOT / "bfsi"
    master_path = _ROOT / "master_database.db"
    student_path = _ROOT / "student_enrollments.db"

    T_MERGE_ENROLL = 0.45
    T_MATCH_STUDENT = 0.40

    face_app = _load_face_app()
    if face_app is None:
        print("InsightFace failed to load")
        return 1

    master_all = load_master_gallery(master_path)
    master_g = gallery_g_only(master_all)
    student_db = load_enrolled_student_gallery(student_path)
    probes = embed_bfsi(face_app, bfsi_dir)

    print("Kaggle-style matching on local data")
    print(f"BFSI photos: {len(probes)} | G_* in master: {len(master_g)} | students in DB: {len(student_db)}")
    print(f"Thresholds: T_MERGE(enrollment->G_*)={T_MERGE_ENROLL}, T_MATCH(student)={T_MATCH_STUDENT}")
    print()

    bfsi_ids = {e for _, e, _ in probes if e}
    overlap = bfsi_ids & set(student_db.keys())
    print(f"BFSI engagement_ids also in student_enrollments.db: {len(overlap)} / {len(bfsi_ids)}")
    print()

    merge_hits = max_hits = centroid_hits = 0
    print(f"{'eng_id':<12} {'file':<40} smart_merge->G_*  map_max->stu  map_centroid")
    print("-" * 95)

    for name, eng, emb in probes:
        gid, gsim = smart_enroll_merge(emb, master_all, T_MERGE_ENROLL)
        stu_max, ssim_max = map_students_max_to_max([emb], student_db, T_MATCH_STUDENT)
        stu_cent, ssim_cent = map_students_centroid([emb], student_db, T_MATCH_STUDENT)

        if gid:
            merge_hits += 1
        if stu_max:
            max_hits += 1
        if stu_cent:
            centroid_hits += 1

        g_str = gid or "-"
        sm = f"{stu_max or '-'} ({ssim_max:.3f})" if stu_max else "-"
        sc = f"{stu_cent or '-'} ({ssim_cent:.3f})" if stu_cent else "-"
        print(f"{eng or '-':<12} {name[:39]:<40} {g_str:<14} {sm:<22} {sc}")

    print()
    print("=== Summary ===")
    print(f"smart_enroll (BFSI photo -> merge into G_* @ {T_MERGE_ENROLL}): {merge_hits}/{len(probes)}")
    print(f"map_identities max-to-max @ {T_MATCH_STUDENT}:              {max_hits}/{len(probes)}")
    print(f"map_identities centroid @ {T_MATCH_STUDENT}:           {centroid_hits}/{len(probes)}")
    print()
    print("Why desktop app looked worse:")
    print("- student_enrollments.db is a DIFFERENT roster (0 BFSI overlap) -> map_identities finds nobody")
    print("- Gallery merge uses t_strict_merge ~0.55 on CCTV-only exemplars (stricter than 0.45)")
    print("- Notebook uses pliswork_4batch_master_db.pkl (BFSI+4 batches) and T_MATCH_STUDENT=0.40")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
