"""Shared CCTV overlay OCR helpers for attendance and engagement pipelines."""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Optional, Sequence, Tuple

import cv2
import numpy as np


def parse_ocr_overlay_datetime(text: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Parse DVR on-screen text into (YYYY-MM-DD or None, HH:MM:SS or None).
    Supports: YYYY-MM-DD, MM-DD-YYYY / DD-MM-YYYY (heuristic), DD/MM/YYYY.
    """
    if not text:
        return None, None
    text = " ".join(text.split())
    d_iso = None
    m = re.search(r"(\d{4})-(\d{2})-(\d{2})", text)
    if m:
        y, mo, d = m.groups()
        try:
            d_iso = datetime(int(y), int(mo), int(d)).strftime("%Y-%m-%d")
        except ValueError:
            pass
    if not d_iso:
        m = re.search(r"(\d{2})-(\d{2})-(\d{4})", text)
        if m:
            a, b, y = m.groups()
            ia, ib = int(a), int(b)
            try:
                if 1 <= ia <= 12 and 1 <= ib <= 31:
                    d_iso = datetime(int(y), ia, ib).strftime("%Y-%m-%d")
                elif 1 <= ib <= 12 and 1 <= ia <= 31:
                    d_iso = datetime(int(y), ib, ia).strftime("%Y-%m-%d")
            except ValueError:
                pass
    if not d_iso:
        m = re.search(r"(\d{2})/(\d{2})/(\d{4})", text)
        if m:
            a, b, y = m.groups()
            ia, ib = int(a), int(b)
            try:
                if 1 <= ib <= 12 and 1 <= ia <= 31:
                    d_iso = datetime(int(y), ib, ia).strftime("%Y-%m-%d")
                elif 1 <= ia <= 12 and 1 <= ib <= 31:
                    d_iso = datetime(int(y), ia, ib).strftime("%Y-%m-%d")
            except ValueError:
                pass
    tm = re.search(r"(\d{2}:\d{2}:\d{2})", text)
    t_str = tm.group(1) if tm else None
    return d_iso, t_str


def read_ocr_overlay_frame(
    frame: Any,
    ocr_reader: Any,
    timestamp_coords: Sequence[int],
) -> Tuple[Optional[str], Optional[str], str]:
    """OCR timestamp ROI: returns (date YYYY-MM-DD|None, time HH:MM:SS|None, raw_joined_text)."""
    if ocr_reader is None:
        return None, None, ""
    if frame is None or not hasattr(frame, "shape"):
        return None, None, ""
    x, y, w_roi, h_roi = (int(v) for v in timestamp_coords)
    if y + h_roi > frame.shape[0] or x + w_roi > frame.shape[1]:
        return None, None, ""
    roi = frame[y:y + h_roi, x:x + w_roi]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray_large = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    contrast_img = clahe.apply(gray_large)
    blurred = cv2.GaussianBlur(contrast_img, (3, 3), 0)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    final_img = cv2.filter2D(blurred, -1, kernel)
    allowlist = "0123456789:/.- MonTueWedThuFriSatSun"
    results = ocr_reader.readtext(final_img, allowlist=allowlist, detail=0)
    text = " ".join(results)
    d_iso, t_str = parse_ocr_overlay_datetime(text)
    return d_iso, t_str, text


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
