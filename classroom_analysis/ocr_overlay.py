"""Shared CCTV overlay OCR helpers for attendance and engagement pipelines."""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any, List, Optional, Sequence, Tuple

import cv2
import numpy as np

# Characters commonly seen in DVR overlays (allowlist was dropping "Live:" etc.)
_OCR_ALLOWLIST = (
    "0123456789:/.- "
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    "MonTueWedThuFriSatSun"
)


def scale_timestamp_coords(
    coords: Sequence[int],
    src_size: Tuple[int, int],
    dst_size: Tuple[int, int],
) -> Tuple[int, int, int, int]:
    """Scale [x, y, w, h] when the frame is resized before OCR."""
    sw, sh = src_size
    dw, dh = dst_size
    if sw <= 0 or sh <= 0:
        return tuple(int(v) for v in coords)  # type: ignore[return-value]
    x, y, w_roi, h_roi = (int(v) for v in coords)
    return (
        int(x * dw / sw),
        int(y * dh / sh),
        max(1, int(w_roi * dw / sw)),
        max(1, int(h_roi * dh / sh)),
    )


def parse_date_from_video_filename(name: str) -> Optional[str]:
    """
    Extract YYYY-MM-DD from NVR-style names, e.g. NVR_ch4_main_20260503072329.mp4.
    """
    if not name:
        return None
    for m in re.finditer(r"(20\d{2})(\d{2})(\d{2})", name):
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        try:
            return datetime(y, mo, d).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return None


def _try_build_date(y: int, mo: int, d: int) -> Optional[str]:
    try:
        return datetime(y, mo, d).strftime("%Y-%m-%d")
    except ValueError:
        return None


def parse_ocr_overlay_datetime(text: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Parse DVR on-screen text into (YYYY-MM-DD or None, HH:MM:SS or None).
    Tolerant of common EasyOCR mistakes (dots in time, spaces in ISO dates).
    """
    if not text:
        return None, None
    text = " ".join(text.split())
    d_iso = None

    m = re.search(r"(20\d{2})\s*[-/]\s*(\d{1,2})\s*[-/]\s*(\d{1,2})", text)
    if m:
        d_iso = _try_build_date(int(m.group(1)), int(m.group(2)), int(m.group(3)))

    if not d_iso:
        m = re.search(r"(\d{2})-(\d{2})-(\d{4})", text)
        if m:
            a, b, y = m.groups()
            ia, ib = int(a), int(b)
            if 1 <= ia <= 12 and 1 <= ib <= 31:
                d_iso = _try_build_date(int(y), ia, ib)
            elif 1 <= ib <= 12 and 1 <= ia <= 31:
                d_iso = _try_build_date(int(y), ib, ia)

    if not d_iso:
        m = re.search(r"(\d{2})/(\d{2})/(\d{4})", text)
        if m:
            a, b, y = m.groups()
            ia, ib = int(a), int(b)
            if 1 <= ib <= 12 and 1 <= ia <= 31:
                d_iso = _try_build_date(int(y), ib, ia)
            elif 1 <= ia <= 12 and 1 <= ib <= 31:
                d_iso = _try_build_date(int(y), ia, ib)

    tm = re.search(r"(\d{2})[:\.](\d{2})[:\.](\d{2})", text)
    t_str = f"{tm.group(1)}:{tm.group(2)}:{tm.group(3)}" if tm else None
    return d_iso, t_str


def _roi_candidates(
    frame_shape: Tuple[int, ...],
    timestamp_coords: Sequence[int],
) -> List[Tuple[int, int, int, int]]:
    fh, fw = int(frame_shape[0]), int(frame_shape[1])
    x, y, w_roi, h_roi = (int(v) for v in timestamp_coords)
    candidates: List[Tuple[int, int, int, int]] = []

    def _add(box: Tuple[int, int, int, int]) -> None:
        bx, by, bw, bh = box
        if by + bh <= fh and bx + bw <= fw and bw > 0 and bh > 0:
            if box not in candidates:
                candidates.append(box)

    _add((x, y, w_roi, h_roi))
    _add((0, 0, min(fw, max(w_roi, int(fw * 0.5))), max(h_roi, int(fh * 0.12))))
    _add((int(fw * 0.62), 0, max(1, fw - int(fw * 0.62)), max(h_roi, int(fh * 0.12))))
    _add((0, 0, fw, max(h_roi, int(fh * 0.08))))
    return candidates


def _ocr_roi(ocr_reader: Any, roi: np.ndarray) -> str:
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray_large = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    contrast_img = clahe.apply(gray_large)
    blurred = cv2.GaussianBlur(contrast_img, (3, 3), 0)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    final_img = cv2.filter2D(blurred, -1, kernel)

    chunks: List[str] = []
    for kwargs in (
        {},
        {"allowlist": _OCR_ALLOWLIST},
    ):
        try:
            chunks.extend(ocr_reader.readtext(final_img, detail=0, **kwargs))
        except TypeError:
            chunks.extend(ocr_reader.readtext(final_img, detail=0))
    try:
        chunks.extend(ocr_reader.readtext(roi, detail=0))
    except Exception:
        pass
    return " ".join(chunks)


def read_ocr_overlay_frame(
    frame: Any,
    ocr_reader: Any,
    timestamp_coords: Sequence[int],
) -> Tuple[Optional[str], Optional[str], str]:
    """OCR timestamp ROI(s): returns (date YYYY-MM-DD|None, time HH:MM:SS|None, raw_joined_text)."""
    if ocr_reader is None:
        return None, None, ""
    if frame is None or not hasattr(frame, "shape"):
        return None, None, ""

    best_date: Optional[str] = None
    best_time: Optional[str] = None
    best_raw = ""

    for x, y, w_roi, h_roi in _roi_candidates(frame.shape, timestamp_coords):
        roi = frame[y : y + h_roi, x : x + w_roi]
        text = _ocr_roi(ocr_reader, roi)
        if not text.strip():
            continue
        d_iso, t_str = parse_ocr_overlay_datetime(text)
        if d_iso and not best_date:
            best_date = d_iso
            best_raw = text
        if t_str:
            best_time = t_str
            if not best_raw:
                best_raw = text
        if d_iso and t_str:
            return d_iso, t_str, text
        if len(text) > len(best_raw):
            best_raw = text

    return best_date, best_time, best_raw


def apply_ocr_datetime_to_metadata(
    metadata: dict,
    date_str: Optional[str],
    time_str: Optional[str],
) -> bool:
    """Apply OCR date/time onto engagement metadata when both are available."""
    if not date_str or not time_str:
        return False
    dt_str = f"{date_str} {time_str}"
    try:
        metadata["base_datetime"] = datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return False
    metadata["base_datetime_str"] = dt_str
    return True
