"""Preflight checks before engagement / attendance processing."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional

LogFn = Callable[[str, str], None]


@dataclass
class PreflightResult:
    ok: bool
    failures: List[str] = field(default_factory=list)


def _tick(log: LogFn, label: str, ok: bool, *, required: bool = True) -> None:
    level = "success" if ok else ("error" if required else "warning")
    mark = "\u2713" if ok else "\u2717"
    log(f"{mark} {label}", level)


def _import_ok(module: str, log: Optional[LogFn] = None) -> bool:
    try:
        if module == "boxmot":
            from app.frozen_runtime import ensure_valid_stdio

            ensure_valid_stdio()
        __import__(module)
        return True
    except Exception as exc:
        if log is not None:
            log(f"{module} import failed: {exc}", "error")
        return False


def _runtime_roots() -> List[Path]:
    roots: List[Path] = []
    try:
        roots.append(Path(__file__).resolve().parent.parent)
    except Exception:
        pass
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        roots.append(Path(sys._MEIPASS))
    try:
        roots.append(Path(sys.executable).resolve().parent)
    except Exception:
        pass
    unique: List[Path] = []
    seen = set()
    for root in roots:
        key = str(root)
        if key not in seen:
            unique.append(root)
            seen.add(key)
    return unique


def _resolve_weight_file(weight_file: str) -> Optional[str]:
    try:
        from classroom_analysis.model_weights import model_weight_candidates, resolve_weights_dir

        weights_dir, base_path = resolve_weights_dir()
        for candidate in model_weight_candidates(weight_file, weights_dir, base_path):
            if os.path.isfile(candidate):
                return candidate
    except Exception:
        pass
    for root in _runtime_roots():
        for candidate in (
            root / "Models" / weight_file,
            root / weight_file,
        ):
            if candidate.is_file():
                return str(candidate)
    return None


def _face_model_ready() -> bool:
    if getattr(sys, "frozen", False):
        try:
            exe_dir = Path(sys.executable).resolve().parent
            if (exe_dir / "models" / "buffalo_l").is_dir():
                return True
        except Exception:
            pass
    for root in _runtime_roots():
        if (root / "models" / "buffalo_l").is_dir():
            return True
    return False


def run_classroom_preflight(log: LogFn) -> PreflightResult:
    failures: List[str] = []
    log("Preflight checks", "info")

    for module, label in (
        ("torch", "PyTorch"),
        ("cv2", "OpenCV"),
        ("numpy", "NumPy"),
        ("scipy", "SciPy"),
        ("ultralytics", "Ultralytics YOLO"),
        ("boxmot", "BoT-SORT tracker (boxmot)"),
    ):
        ok = _import_ok(module, log)
        _tick(log, label, ok)
        if not ok:
            failures.append(label)

    for weight_file, label in (
        ("yolov8m.pt", "Detection weights (yolov8m.pt)"),
        ("yolov8n-pose.pt", "Pose weights (yolov8n-pose.pt)"),
        ("osnet_x1_0_msmt17.pt", "Re-ID weights (osnet_x1_0_msmt17.pt)"),
    ):
        ok = _resolve_weight_file(weight_file) is not None
        _tick(log, label, ok)
        if not ok:
            failures.append(label)

    face_ok = _resolve_weight_file("yolov8n-face.pt") is not None
    _tick(log, "Face weights (yolov8n-face.pt)", face_ok, required=False)

    for module, label in (
        ("classroom_analysis.stitch_logic", "Stitching module"),
        ("classroom_analysis.vlm_metadata", "VLM metadata module"),
        ("classroom_analysis.group_by_session", "Management summary module"),
        ("classroom_analysis.ocr_overlay", "OCR overlay module"),
    ):
        _tick(log, label, _import_ok(module), required=False)

    _tick(log, "EasyOCR", _import_ok("easyocr"), required=False)

    ok = not failures
    if ok:
        log("Preflight complete — starting processing", "success")
    else:
        log("Preflight failed — fix the items marked with \u2717", "error")
    return PreflightResult(ok=ok, failures=failures)


def run_crossday_preflight(log: LogFn) -> PreflightResult:
    failures: List[str] = []
    log("Preflight checks", "info")

    for module, label in (
        ("torch", "PyTorch"),
        ("cv2", "OpenCV"),
        ("numpy", "NumPy"),
        ("scipy", "SciPy"),
        ("ultralytics", "Ultralytics YOLO"),
    ):
        ok = _import_ok(module, log)
        _tick(log, label, ok)
        if not ok:
            failures.append(label)

    yolo_ok = _resolve_weight_file("yolov8n.pt") is not None
    _tick(log, "Person detection weights (yolov8n.pt)", yolo_ok)
    if not yolo_ok:
        failures.append("Person detection weights (yolov8n.pt)")

    try:
        from classroom_analysis.model_weights import resolve_ultralytics_botsort_yaml

        botsort_ok = Path(resolve_ultralytics_botsort_yaml()).is_file()
    except Exception:
        botsort_ok = False
    _tick(log, "BoT-SORT tracker config", botsort_ok)
    if not botsort_ok:
        failures.append("BoT-SORT tracker config")

    ort_ok = _import_ok("onnxruntime")
    _tick(log, "ONNX Runtime", ort_ok, required=False)

    insight_ok = _import_ok("insightface")
    _tick(log, "InsightFace", insight_ok, required=False)

    _tick(log, "Face model bundle (buffalo_l)", _face_model_ready(), required=False)
    _tick(log, "EasyOCR", _import_ok("easyocr"), required=False)

    ok = not failures
    if ok:
        log("Preflight complete — starting processing", "success")
    else:
        log("Preflight failed — fix the items marked with \u2717", "error")
    return PreflightResult(ok=ok, failures=failures)
