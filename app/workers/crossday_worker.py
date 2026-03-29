"""
Attendance Worker Thread.
Runs the attendance pipeline in a background thread.
"""

import os
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, Optional, Callable

import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class CrossDayWorker(QThread):
    """Worker thread for attendance pipeline."""
    
    progress = pyqtSignal(int, str)  # percent, message
    log_message = pyqtSignal(str, str)  # message, level
    frame_ready = pyqtSignal(np.ndarray, bool)  # frame, is_rgb (for video preview)
    finished = pyqtSignal(str)  # report path
    error = pyqtSignal(str)  # error message
    
    def __init__(
        self,
        video_path: str,
        output_dir: str,
        db_path: str,
        config: Dict[str, Any],
        preview_enabled: bool = False,
        parent=None
    ):
        super().__init__(parent)
        self.video_path = video_path
        self.output_dir = output_dir
        self.db_path = db_path
        self.config = config
        self.preview_enabled = preview_enabled
        self._stop_requested = False
        
    def run(self):
        """Run the attendance analysis."""
        try:
            self.log_message.emit("Initializing attendance analysis...", "info")
            
            try:
                import torch
                _ = torch.__version__
            except (OSError, ImportError) as torch_err:
                err_msg = (
                    "PyTorch failed to load (DLL error). Try installing CPU-only PyTorch:\n\n"
                    "pip uninstall torch torchvision\n"
                    "pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu\n\n"
                    "Or install Microsoft Visual C++ Redistributable 2015-2022 from Microsoft's website."
                )
                self.error.emit(err_msg)
                return
            
            # Always use PyQt signal path for in-app preview (same as classroom worker)
            frame_cb = self._emit_frame if self.preview_enabled else None

            analyzer = CrossDayAnalyzerWithCallbacks(
                video_path=self.video_path,
                output_dir=self.output_dir,
                db_path=self.db_path,
                config=self.config,
                progress_callback=self._emit_progress,
                log_callback=self._emit_log,
                frame_callback=frame_cb,
                use_cv2_preview=False,
                stop_check=self._check_stop
            )
            
            # Run analysis
            report_path = analyzer.analyze_video()
            
            if self._stop_requested:
                self.log_message.emit("Analysis stopped by user.", "warning")
                self.finished.emit("")  # Notify UI to re-enable inputs
            else:
                self.finished.emit(report_path)
                
        except Exception as e:
            error_msg = f"{str(e)}\n\n{traceback.format_exc()}"
            self.error.emit(error_msg)
            
    def _emit_progress(self, percent: int, message: str):
        """Emit progress signal."""
        self.progress.emit(percent, message)
        
    def _emit_log(self, message: str, level: str = "info"):
        """Emit log message signal."""
        self.log_message.emit(message, level)
        
    def _emit_frame(self, frame: np.ndarray, is_rgb: bool = False):
        """Emit frame for preview."""
        if self.preview_enabled:
            self.frame_ready.emit(frame, is_rgb)
            
    def _check_stop(self) -> bool:
        """Check if stop was requested."""
        return self._stop_requested
        
    def stop(self):
        """Request the worker to stop."""
        self._stop_requested = True


class CrossDayAnalyzerWithCallbacks:
    """
    Attendance analyzer with GUI callbacks.
    Based on cross_day_code/main.py but with callback support.
    """
    
    def __init__(
        self,
        video_path: str,
        output_dir: str,
        db_path: str,
        config: Dict[str, Any],
        progress_callback: Optional[Callable] = None,
        log_callback: Optional[Callable] = None,
        frame_callback: Optional[Callable] = None,
        use_cv2_preview: bool = False,
        stop_check: Optional[Callable] = None
    ):
        self.video_path = video_path
        self.output_dir = output_dir
        self.db_path = db_path
        self.config = config
        self.progress_callback = progress_callback
        self.log_callback = log_callback
        self.frame_callback = frame_callback
        self.use_cv2_preview = use_cv2_preview
        self.stop_check = stop_check
        
    def _log(self, message: str, level: str = "info"):
        """Log a message."""
        if self.log_callback:
            self.log_callback(message, level)
        print(f"[{level.upper()}] {message}")
        
    def _progress(self, percent: int, message: str):
        """Report progress."""
        if self.progress_callback:
            self.progress_callback(percent, message)
            
    def _should_stop(self) -> bool:
        """Check if we should stop."""
        if self.stop_check:
            return self.stop_check()
        return False

    def _detect_motion_segments(
        self, cap, total_frames: int, fps: int, log_cb, progress_cb, stop_check
    ) -> list:
        """
        Scan video for motion using background subtraction.
        Returns list of (start_frame, end_frame) segments (inclusive) where motion was detected.
        """
        import cv2
        MOTION_SAMPLE_EVERY = 5  # Sample every Nth frame for speed
        MOTION_PIXEL_RATIO = 0.002  # Min fraction of pixels that must change
        MIN_SEGMENT_FRAMES = 15  # Minimum frames to form a segment
        GAP_MERGE_FRAMES = 90  # Merge segments within this gap (e.g. 3 sec at 30fps)
        
        bg_sub = cv2.createBackgroundSubtractorMOG2(history=120, varThreshold=25, detectShadows=False)
        segments = []
        in_segment = False
        segment_start = 0
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frame_idx = 0
        
        while frame_idx < total_frames:
            if stop_check and stop_check():
                return segments
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            if frame_idx % MOTION_SAMPLE_EVERY != 0:
                frame_idx += 1
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            fg = bg_sub.apply(gray)
            motion_pixels = np.count_nonzero(fg)
            total_pixels = fg.size
            has_motion = (total_pixels > 0 and motion_pixels / total_pixels >= MOTION_PIXEL_RATIO)
            
            if has_motion:
                if not in_segment:
                    in_segment = True
                    segment_start = max(0, frame_idx - MOTION_SAMPLE_EVERY)
            else:
                if in_segment:
                    end = frame_idx
                    if end - segment_start >= MIN_SEGMENT_FRAMES:
                        segments.append((segment_start, end))
                    in_segment = False
            
            if frame_idx % 500 == 0 and progress_cb:
                pct = min(15, int(frame_idx / max(1, total_frames) * 15))
                progress_cb(pct, f"Scanning for motion: {frame_idx}/{total_frames}")
            frame_idx += 1
        
        if in_segment and (frame_idx - segment_start) >= MIN_SEGMENT_FRAMES:
            segments.append((segment_start, min(frame_idx, total_frames - 1)))
        
        # Merge nearby segments
        merged = []
        for start, end in sorted(segments):
            if merged and (start - merged[-1][1]) <= GAP_MERGE_FRAMES:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))
        
        return merged
        
    def analyze_video(self) -> str:
        """Run the attendance analysis with callbacks."""
        import cv2
        import torch
        import numpy as np
        import json
        import pickle
        import sqlite3
        import re
        import shutil
        from datetime import datetime, timedelta
        from pathlib import Path
        from scipy.spatial.distance import cosine
        from ultralytics import YOLO
        
        # InsightFace + onnxruntime for face-based identity; fallback to simplified (track-only) mode
        face_app = None
        try:
            import onnxruntime as ort
            _ = ort.__version__
        except (ImportError, OSError) as e:
            self._log(
                f"onnxruntime not available ({e}). Using simplified mode (track-only, no face matching).",
                "warning"
            )
            self._log(
                "To enable face matching: pip uninstall onnxruntime-gpu -y; pip install onnxruntime",
                "info"
            )
        else:
            try:
                from insightface.app import FaceAnalysis
                face_app = "pending"  # Will load below
            except ImportError as e:
                self._log(f"InsightFace not available ({e}). Using simplified mode.", "warning")
        
        # Configuration
        RUN_MODE = self.config.get("run_mode", "BUILD_DB")
        CURRENT_DATE = self.config.get("current_date", datetime.now().strftime("%Y-%m-%d"))
        DAY_LABEL = self.config.get("day_label", "Day1")

        crossday_cfg = self.config.get("crossday") or {}
        T_STRICT_MERGE = float(crossday_cfg.get("t_strict_merge", 0.50))
        T_NEW_ID = float(crossday_cfg.get("t_new_id", 0.35))
        T_RATIO_MARGIN = float(crossday_cfg.get("t_ratio_margin", 0.10))
        MIN_SAMPLES = int(crossday_cfg.get("min_samples", 8))
        MAX_EXEMPLARS = 5
        T_OUTLIER = 0.6
        VISITOR_UPGRADE_DAYS = int(crossday_cfg.get("visitor_upgrade_days", 3))
        T_MATCH_STUDENT = float(crossday_cfg.get("t_match_student", 0.40))

        def runtime_roots():
            roots = []
            try:
                roots.append(Path(__file__).resolve().parent.parent.parent)
            except Exception:
                pass
            if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
                roots.append(Path(sys._MEIPASS))
            try:
                roots.append(Path(sys.executable).resolve().parent)
            except Exception:
                pass
            # Keep order while de-duplicating.
            unique = []
            seen = set()
            for root in roots:
                key = str(root)
                if key not in seen:
                    unique.append(root)
                    seen.add(key)
            return unique

        roots = runtime_roots()

        def resolve_existing_file(candidates):
            for c in candidates:
                if c and c.exists():
                    return str(c)
            return ""

        student_db_path = str(crossday_cfg.get("student_db_path", "") or "").strip()
        if student_db_path and not Path(student_db_path).exists():
            self._log(f"Configured student DB path not found: {student_db_path}", "warning")
            student_db_path = ""
        if not student_db_path:
            auto_candidates = []
            for root in roots:
                auto_candidates.extend([
                    # ETL-generated SQLite (preferred — always up to date from BQ+S3)
                    root / "student_enrollments.db",
                    root / "Models" / "student_enrollments.db",
                    # Legacy pickle formats
                    root / "pliswork_4batch_master_db.pkl",
                    root / "4batches_student_embeddings.pkl",
                    root / "Models" / "pliswork_4batch_master_db.pkl",
                    root / "Models" / "4batches_student_embeddings.pkl",
                    root / "app" / "Models" / "pliswork_4batch_master_db.pkl",
                    root / "app" / "Models" / "4batches_student_embeddings.pkl",
                ])
            student_db_path = resolve_existing_file(auto_candidates)
        enable_ocr_timestamp = bool(crossday_cfg.get("enable_ocr_timestamp", False))
        ocr_interval = max(1, int(crossday_cfg.get("ocr_interval", 30)))
        timestamp_coords = crossday_cfg.get("timestamp_coords", [0, 15, 600, 90])
        if not isinstance(timestamp_coords, (list, tuple)) or len(timestamp_coords) != 4:
            timestamp_coords = [0, 15, 600, 90]
        try:
            timestamp_coords = tuple(int(v) for v in timestamp_coords)
        except (TypeError, ValueError):
            timestamp_coords = (0, 15, 600, 90)
        
        # Device setup (allow forcing CPU from settings)
        force_cpu = bool((self.config.get("inference") or {}).get("force_cpu", False))
        device = 'cpu' if force_cpu else ('cuda' if torch.cuda.is_available() else 'cpu')
        if force_cpu:
            self._log("CPU-only mode enabled from settings")
        self._log(f"Running on: {device}")
        self._log(f"Mode: {RUN_MODE}, Date: {CURRENT_DATE}")
        
        # 100% match unique_and_recognition.py: hardcode params (ignore config)
        inference_cfg = self.config.get("inference") or {}
        use_openvino = False  # Standalone uses PyTorch only
        enable_motion_detection = False  # Standalone processes all frames
        enable_ocr_timestamp = True  # Standalone always uses EasyOCR for timestamp
        if use_openvino:
            try:
                v = tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:2])
                if v < (2, 1):
                    use_openvino = False
                    self._log("OpenVINO requires torch>=2.1 (you have torch {}). Using PyTorch.".format(torch.__version__.split("+")[0]), "info")
            except Exception:
                use_openvino = False
        yolo_imgsz = 640  # Standalone uses 640
        face_det_size = 640
        frame_skip = 1  # Standalone runs face every frame
        
        # Load person detection model (YOLO)
        self._log("Loading person detection model...")
        person_model = None
        if use_openvino:
            try:
                base_root = roots[0] if roots else Path.cwd()
                ov_cache = base_root / "Models" / f"yolov8n_ov_{yolo_imgsz}"
                ov_model_dir = ov_cache / "yolov8n_openvino_model"
                xml_files = list(ov_cache.glob("**/*.xml"))
                if xml_files:
                    ov_path = str(xml_files[0].parent)
                    person_model = YOLO(ov_path)
                    self._log(f"Using OpenVINO YOLO (imgsz={yolo_imgsz})", "success")
                else:
                    yolo_candidates = []
                    for root in roots:
                        yolo_candidates.extend([
                            root / "yolov8n.pt",
                            root / "Models" / "yolov8n.pt",
                            root / "app" / "Models" / "yolov8n.pt",
                        ])
                    yolo_weights = resolve_existing_file(yolo_candidates) or "yolov8n.pt"
                    base = YOLO(yolo_weights)
                    self._log("Exporting YOLO to OpenVINO (one-time, ~1 min)...", "info")
                    ov_cache.mkdir(parents=True, exist_ok=True)
                    orig_cwd = os.getcwd()
                    try:
                        os.chdir(str(ov_cache))
                        base.export(format="openvino", imgsz=yolo_imgsz, half=True)
                    finally:
                        os.chdir(orig_cwd)
                    xml_files = list(ov_cache.glob("**/*.xml"))
                    if xml_files:
                        person_model = YOLO(str(xml_files[0].parent))
                        self._log(f"OpenVINO export done. Using imgsz={yolo_imgsz}", "success")
            except Exception as e:
                self._log(f"OpenVINO YOLO failed ({e}), using PyTorch", "warning")
        if person_model is None:
            yolo_candidates = []
            for root in roots:
                yolo_candidates.extend([
                    root / "yolov8n.pt",
                    root / "Models" / "yolov8n.pt",
                    root / "app" / "Models" / "yolov8n.pt",
                ])
            yolo_weights = resolve_existing_file(yolo_candidates) or "yolov8n.pt"
            person_model = YOLO(yolo_weights)
        if device == 'cuda':
            self._log("YOLO using GPU (CUDA)", "success")
        
        # FaceAnalysis: load only if onnxruntime + InsightFace available
        # Uses buffalo_l; det_size from config (416=fast, 640=quality)
        # Match standalone script: 640 for best accuracy. root points to models folder.
        face_root = None
        if getattr(sys, "frozen", False):
            try:
                exe_dir = Path(sys.executable).resolve().parent
                if (exe_dir / "models" / "buffalo_l").exists():
                    face_root = str(exe_dir)
            except Exception:
                pass
        if face_root is None:
            for root in roots:
                if (root / "models" / "buffalo_l").exists():
                    face_root = str(root)
                    break

        if face_app == "pending":
            self._log("Loading face analysis model...")
            try:
                fa_kw = {"name": "buffalo_l"}
                if face_root:
                    fa_kw["root"] = face_root
                if device == 'cuda':
                    try:
                        face_app = FaceAnalysis(
                            providers=['CUDAExecutionProvider', 'CPUExecutionProvider'],
                            **fa_kw
                        )
                        face_app.prepare(ctx_id=0, det_size=(face_det_size, face_det_size))
                    except Exception as e:
                        self._log(f"CUDA failed ({e}), falling back to CPU...", "warning")
                        face_app = None
                if face_app is None or face_app == "pending":
                    face_providers = ['CPUExecutionProvider']
                    if use_openvino:
                        try:
                            import onnxruntime as _ort
                            if 'OpenVINOExecutionProvider' in _ort.get_available_providers():
                                face_providers = ['OpenVINOExecutionProvider', 'CPUExecutionProvider']
                                self._log("Using OpenVINO for face detection", "info")
                        except Exception:
                            pass
                    face_app = FaceAnalysis(providers=face_providers, **fa_kw)
                    face_app.prepare(ctx_id=0, det_size=(face_det_size, face_det_size))
                self._log("Face analysis model loaded", "success")
            except Exception as e:
                self._log(f"Face model failed ({e}), using simplified mode.", "warning")
                face_app = None
        if face_app is None:
            self._log("Running in simplified mode (track-only, no face matching)", "info")

        # Optional OCR model for camera timestamp extraction.
        ocr_reader = None
        if enable_ocr_timestamp:
            try:
                import easyocr
                ocr_reader = easyocr.Reader(['en'], gpu=False)
                self._log(f"OCR timestamp enabled (interval={ocr_interval}, roi={timestamp_coords})", "success")
            except Exception as e:
                self._log(f"EasyOCR unavailable ({e}). Falling back to video timeline timestamps.", "warning")
                enable_ocr_timestamp = False
        
        def _is_nf_id(gid) -> bool:
            if not gid:
                return False
            s = str(gid)
            return s.startswith("NF_") or "_NF_" in s

        # Data structures
        global_gallery = {}
        track_vault = {}
        next_global_id = 1
        next_nf_id = 1
        next_visitor_id = 1
        operational_dates = []
        student_db = {}
        
        # Load existing database — SQLite (.db) preferred, pickle (.pkl) as legacy fallback
        def _load_db(db_path):
            nonlocal global_gallery, operational_dates
            if not db_path or not os.path.exists(db_path):
                self._log("No existing database found, starting fresh")
                return
            self._log(f"Loading database from {db_path}...")
            if db_path.endswith(".db"):
                try:
                    conn = sqlite3.connect(db_path)
                    cur = conn.cursor()
                    cur.execute("SELECT date FROM operational_dates")
                    operational_dates = [r[0] for r in cur.fetchall()]
                    cur.execute("SELECT g_id, join_date, engagement_id, batch, confidence FROM identities")
                    for row in cur.fetchall():
                        g_id = row[0]
                        global_gallery[g_id] = {
                            "join_date": row[1], "engagement_id": row[2],
                            "batch": row[3], "confidence": row[4] or 0.0,
                            "exemplars": [], "attendance": {}
                        }
                    cur.execute("SELECT g_id, embedding FROM exemplars")
                    for row in cur.fetchall():
                        if row[0] in global_gallery:
                            global_gallery[row[0]]["exemplars"].append(
                                np.frombuffer(row[1], dtype=np.float32)
                            )
                    cur.execute("SELECT g_id, date, entry_time, exit_time FROM attendance")
                    for row in cur.fetchall():
                        if row[0] in global_gallery:
                            global_gallery[row[0]]["attendance"][row[1]] = {
                                "entry": row[2], "exit": row[3]
                            }
                    conn.close()
                except sqlite3.OperationalError:
                    self._log("SQLite DB exists but is empty, starting fresh", "warning")
            else:
                with open(db_path, 'rb') as f:
                    db_data = pickle.load(f)
                    global_gallery = db_data.get("gallery", {})
                    operational_dates = db_data.get("operational_dates", [])
            self._log(f"Loaded {len(global_gallery)} identities", "success")

        _load_db(self.db_path)

        g_ids = [k for k in global_gallery.keys() if str(k).startswith("G_")]
        if g_ids:
            next_global_id = len(g_ids) + 1
        for k in global_gallery.keys():
            s = str(k)
            try:
                if s.startswith("NF_"):
                    next_nf_id = max(next_nf_id, int(s.split("_")[-1]) + 1)
                elif "_NF_" in s:
                    next_nf_id = max(next_nf_id, int(s.split("_")[-1]) + 1)
            except ValueError:
                pass
        visitor_prefix = f"{DAY_LABEL}_V_"
        v_ids = [k for k in global_gallery.keys() if k.startswith(visitor_prefix)]
        if v_ids:
            try:
                nums = [int(k.split("_")[-1]) for k in v_ids]
                next_visitor_id = max(nums) + 1
            except (ValueError, IndexError):
                pass

        if student_db_path:
            if os.path.exists(student_db_path):
                try:
                    if student_db_path.endswith(".db"):
                        # ETL-generated SQLite: enrolled_students(engagement_id, batch_name, embedding BLOB)
                        _conn = sqlite3.connect(student_db_path)
                        _cur = _conn.cursor()
                        _cur.execute("SELECT engagement_id, batch_name, embedding FROM enrolled_students")
                        for _eid, _batch, _blob in _cur.fetchall():
                            if _blob:
                                _emb = np.frombuffer(_blob, dtype=np.float32).copy()
                                student_db[str(_eid)] = {
                                    "exemplars": [_emb],
                                    "batch": _batch,
                                }
                        _conn.close()
                    else:
                        with open(student_db_path, 'rb') as f:
                            student_db = pickle.load(f)
                    self._log(f"Loaded Student DB: {len(student_db)} enrolled faces from {Path(student_db_path).name}.", "success")
                except Exception as e:
                    self._log(f"Failed to load student DB ({e}). Student mapping disabled.", "warning")
            else:
                self._log(f"Student DB not found at: {student_db_path}", "warning")
        
        if RUN_MODE == "BUILD_DB" and CURRENT_DATE not in operational_dates:
            operational_dates.append(CURRENT_DATE)
            operational_dates.sort()
        
        # Create output directories
        crops_dir = os.path.join(self.output_dir, f"crops_{CURRENT_DATE.replace('-', '')}")
        verification_dir = os.path.join(self.output_dir, "Verification_Matches")
        os.makedirs(crops_dir, exist_ok=True)
        os.makedirs(verification_dir, exist_ok=True)
        
        # Open video
        self._log(f"Opening video: {self.video_path}")
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise ValueError("Could not open video file")
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w, h = int(cap.get(3)), int(cap.get(4))
        
        self._log(f"Video: {total_frames} frames, {fps} FPS, {w}x{h}")
        
        # Motion detection: always OFF for 100% match with unique_and_recognition.py
        motion_segments = []
        enable_motion_detection = False
        if enable_motion_detection:
            self._log("Motion detection enabled — scanning video for motion segments...")
            motion_segments = self._detect_motion_segments(
                cap, total_frames, fps,
                self._log, self._progress, self._should_stop
            )
            if motion_segments:
                motion_frames = sum(end - start + 1 for start, end in motion_segments)
                self._log(
                    f"Found {len(motion_segments)} motion segment(s), "
                    f"~{motion_frames} frames (of {total_frames}) — attendance will run only on these.",
                    "success"
                )
            else:
                self._log("No motion segments found. Processing entire video.", "warning")
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        def _in_motion_segment(idx):
            if not motion_segments:
                return True
            for start, end in motion_segments:
                if start <= idx <= end:
                    return True
            return False
        
        # Resize if needed
        target_w, target_h = (1280, 720) if w > 1920 else (w, h)
        
        # Output video (optional - skip for faster processing)
        # mp4v can truncate long videos on Windows; use AVI/MJPG for reliability
        save_output_video = crossday_cfg.get("save_output_video", self.config.get("save_output_video", False))
        use_avi = crossday_cfg.get("use_avi_output", False)
        if total_frames > 5000 and not use_avi:
            use_avi = True
            self._log("Long video detected. Using AVI format for reliable output (mp4v can truncate on Windows).", "info")
        ext = '.avi' if use_avi else '.mp4'
        output_video_path = os.path.join(self.output_dir, f"{CURRENT_DATE.replace('-', '_')}_output{ext}")
        out = None
        if save_output_video:
            fourcc = cv2.VideoWriter_fourcc(*'MJPG') if use_avi else cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_video_path, fourcc, fps, (target_w, target_h))
            if not out.isOpened():
                self._log("VideoWriter failed. Trying alternate format...", "warning")
                output_video_path = os.path.join(self.output_dir, f"{CURRENT_DATE.replace('-', '_')}_output.avi")
                out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'MJPG'), fps, (target_w, target_h))
            if not out.isOpened():
                self._log("VideoWriter failed. Output video disabled.", "warning")
                out = None
            else:
                self._log(f"Output video: {output_video_path}", "info")
        if not save_output_video:
            self._log("Output video disabled (report and DB only)", "info")
        
        # Helper functions
        def match_face_to_body(face_bbox, person_tracks):
            fx1, fy1, fx2, fy2 = face_bbox
            face_area = max(0, fx2 - fx1) * max(0, fy2 - fy1)
            face_cx = (fx1 + fx2) / 2
            face_cy = (fy1 + fy2) / 2
            
            best_match_id = None
            min_center_dist = float('inf')
            
            for pt in person_tracks:
                px1, py1, px2, py2 = pt['bbox']
                t_id = pt['track_id']
                
                if face_cy > py1 + (py2 - py1) * 0.5:
                    continue
                
                ix1 = max(fx1, px1)
                iy1 = max(fy1, py1)
                ix2 = min(fx2, px2)
                iy2 = min(fy2, py2)
                inter_area = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                
                if face_area > 0 and (inter_area / face_area) > 0.5:
                    person_cx = (px1 + px2) / 2
                    dist = abs(face_cx - person_cx)
                    if dist < min_center_dist:
                        min_center_dist = dist
                        best_match_id = t_id
            return best_match_id
        
        def find_match_with_margin(track_emb, active_gids):
            scores = []
            for g_id, data in global_gallery.items():
                if g_id in active_gids:
                    continue
                ex_list = data.get("exemplars") or []
                if not ex_list:
                    continue
                best_sim = max([1 - cosine(track_emb, ex) for ex in ex_list])
                scores.append((g_id, best_sim))
            
            if not scores:
                return None, 0
            scores.sort(key=lambda x: x[1], reverse=True)
            best_id, best_sim = scores[0]
            
            if len(scores) > 1:
                second_sim = scores[1][1]
                if (best_sim - second_sim) < T_RATIO_MARGIN:
                    return None, best_sim
            
            if best_sim > T_STRICT_MERGE:
                return best_id, best_sim
            return None, best_sim
        
        def log_attendance(g_id, timestamp):
            g_data = global_gallery[g_id]
            if "join_date" not in g_data:
                g_data["join_date"] = CURRENT_DATE
            if "attendance" not in g_data:
                g_data["attendance"] = {}
            
            if CURRENT_DATE not in g_data["attendance"]:
                g_data["attendance"][CURRENT_DATE] = {"entry": timestamp, "exit": timestamp}
            else:
                current_entry = g_data["attendance"][CURRENT_DATE]["entry"]
                current_exit = g_data["attendance"][CURRENT_DATE]["exit"]
                if timestamp < current_entry:
                    g_data["attendance"][CURRENT_DATE]["entry"] = timestamp
                if timestamp > current_exit:
                    g_data["attendance"][CURRENT_DATE]["exit"] = timestamp
        
        def update_exemplars(g_id, new_emb):
            # BIOMETRIC LOCK: Never add new face data to original Day 1 Employees during evaluation days
            if RUN_MODE == "EVAL_DAY" and str(g_id).startswith("G_"):
                return
            gallery = global_gallery[g_id]
            is_diverse = True
            for ex in gallery["exemplars"]:
                if (1 - cosine(new_emb, ex)) > 0.60:
                    is_diverse = False
                    break
            if is_diverse:
                if len(gallery["exemplars"]) >= MAX_EXEMPLARS:
                    gallery["exemplars"].pop(0)
                gallery["exemplars"].append(new_emb)

        def get_ocr_timestamp(frame):
            if ocr_reader is None:
                return None
            if frame is None or not hasattr(frame, "shape"):
                return None
            x, y, w_roi, h_roi = timestamp_coords
            if y + h_roi > frame.shape[0] or x + w_roi > frame.shape[1]:
                return None
            roi = frame[y:y + h_roi, x:x + w_roi]
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            gray_large = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            contrast_img = clahe.apply(gray_large)
            blurred = cv2.GaussianBlur(contrast_img, (3, 3), 0)
            kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
            final_img = cv2.filter2D(blurred, -1, kernel)
            allowlist = '0123456789:- MonTueWedThuFriSatSun'
            results = ocr_reader.readtext(final_img, allowlist=allowlist, detail=0)
            text = " ".join(results)
            match = re.search(r'(\d{2}:\d{2}:\d{2})', text)
            if match:
                return match.group(1)
            return None

        def map_identities_to_students():
            if not student_db:
                self._log("Student DB unavailable; skipping enrolled-student mapping.", "info")
                return
            self._log("Mapping global IDs to enrolled students (max-to-max)...")
            mapped = 0
            for g_id, g_data in global_gallery.items():
                g_data["engagement_id"] = None
                g_data["batch"] = None
                g_data["confidence"] = 0.0
                exemplars = g_data.get("exemplars") or []
                if not exemplars:
                    continue
                best_eng_id = None
                best_sim = -1.0
                for eng_id, stu_data in student_db.items():
                    stu_exemplars = (stu_data or {}).get("exemplars") or []
                    if not stu_exemplars:
                        continue
                    for cctv_emb in exemplars:
                        for stu_emb in stu_exemplars:
                            sim = 1 - cosine(cctv_emb, stu_emb)
                            if sim > best_sim:
                                best_sim = sim
                                best_eng_id = eng_id
                if best_eng_id is not None and best_sim > T_MATCH_STUDENT:
                    g_data["engagement_id"] = best_eng_id
                    g_data["batch"] = (student_db.get(best_eng_id) or {}).get("batch")
                    g_data["confidence"] = float(best_sim)
                    mapped += 1
            self._log(f"Student mapping complete. Identified {mapped} people.", "success")
        
        # Process frames
        frame_idx = 0
        frames_written = 0
        face_fail_streak = 0
        FACE_FAIL_FALLBACK = 30  # After this many consecutive face_app.get() failures, fall back to tracking-only
        effective_face_mode = face_app is not None  # Will switch to False if face_app.get fails repeatedly
        cv2_preview_window = "Attendance Preview" if self.use_cv2_preview else None
        if frame_skip > 1:
            self._log(f"Face detection every {frame_skip} frames (faster, slight accuracy trade-off)", "info")
        if self.use_cv2_preview:
            self._log("Using cv2.imshow preview (faster than PyQt)", "info")
        last_valid_timestamp = "00:00:00"
        executor = ThreadPoolExecutor(max_workers=2)
        try:
            while cap.isOpened():
                if self._should_stop():
                    break
                
                success, frame = cap.read()
                if not success:
                    break

                # Some deployed environments/video codecs can yield success=True but frame=None.
                # Skip these bad decodes so face/yolo inference never receives invalid frames.
                if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
                    if frame_idx % 100 == 0:
                        self._log(
                            f"Skipping invalid decoded frame at index {frame_idx}.",
                            "warning"
                        )
                    frame_idx += 1
                    continue
                
                if w != target_w:
                    frame = cv2.resize(frame, (target_w, target_h))
                if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
                    frame_idx += 1
                    continue

                # Skip frames outside motion segments (when motion detection enabled)
                if not _in_motion_segment(frame_idx):
                    frame_idx += 1
                    continue
                
                fallback_timestamp = datetime.fromtimestamp(frame_idx / max(1, fps)).strftime('%H:%M:%S')
                if enable_ocr_timestamp and frame_idx % ocr_interval == 0:
                    ocr_ts = get_ocr_timestamp(frame)
                    if ocr_ts:
                        last_valid_timestamp = ocr_ts
                if enable_ocr_timestamp:
                    timestamp = last_valid_timestamp if last_valid_timestamp != "00:00:00" else fallback_timestamp
                else:
                    timestamp = fallback_timestamp
                
                # Progress (0-90% for main loop; 90-100% reserved for save/report)
                if frame_idx % 100 == 0 and total_frames > 0:
                    progress = min(90, int(frame_idx / total_frames * 90))
                    self._progress(progress, f"Processing frame {frame_idx}/{total_frames}")
                
                # Parallel: YOLO tracking + Face detection (independent, run concurrently)
                def run_yolo():
                    return person_model.track(frame, persist=True, classes=[0], tracker="botsort.yaml", verbose=False, imgsz=yolo_imgsz, device=device)

                def run_face():
                    if face_app is None or not effective_face_mode or (frame_idx % frame_skip) != 0:
                        return []
                    if frame is None or not hasattr(frame, "shape") or getattr(frame, "shape", None) is None:
                        return []
                    try:
                        return face_app.get(frame)
                    except (AttributeError, ValueError, Exception):
                        raise

                person_tracks = []
                faces = []
                yolo_future = executor.submit(run_yolo)
                face_future = executor.submit(run_face)
                results = yolo_future.result()
                try:
                    faces = face_future.result()
                    face_fail_streak = 0
                except (AttributeError, ValueError, Exception) as e:
                    faces = []
                    face_fail_streak += 1
                    if face_fail_streak == 1:
                        self._log(f"Face detection failed: {e}", "warning")
                    if face_fail_streak >= FACE_FAIL_FALLBACK:
                        effective_face_mode = False
                        self._log(
                            "Face detection failing repeatedly. Switching to tracking-only mode.",
                            "warning"
                        )

                if results[0].boxes.id is not None:
                    boxes = results[0].boxes.xyxy.cpu().numpy()
                    track_ids = results[0].boxes.id.int().cpu().numpy()
                    for box, t_id in zip(boxes, track_ids):
                        t_id = int(t_id)
                        person_tracks.append({'track_id': t_id, 'bbox': box})
                        if t_id not in track_vault:
                            track_vault[t_id] = {
                                "embeddings": [], "global_id": None,
                                "first_seen": timestamp, "last_seen": timestamp,
                                "frames": 0, "bbox": box
                            }
                        track_vault[t_id]["last_seen"] = timestamp
                        track_vault[t_id]["frames"] += 1
                        track_vault[t_id]["bbox"] = box
                        gid = track_vault[t_id]["global_id"]
                        if gid:
                            log_attendance(gid, timestamp)
                assigned_tracks = set()

                # Skip face processing entirely if no person was detected this frame
                if not person_tracks:
                    faces = []

                for face in faces:
                    face_width = face.bbox[2] - face.bbox[0]
                    if face.det_score < 0.75 or face_width < 40:
                        continue
                    if face.kps is not None:
                        l_eye, r_eye, nose = face.kps[0], face.kps[1], face.kps[2]
                        eye_dist = np.linalg.norm(l_eye - r_eye)
                        if face_width > 0 and (eye_dist / face_width) < 0.35:
                            continue
                        eye_center_x = (l_eye[0] + r_eye[0]) / 2
                        nose_offset = abs(nose[0] - eye_center_x)
                        if nose_offset > (eye_dist * 0.5):
                            continue
                    matched_t_id = match_face_to_body(face.bbox, person_tracks)
                    if matched_t_id is not None and matched_t_id not in assigned_tracks:
                        # Early stopping: once we have enough samples the track already has
                        # an identity — skip costly embedding math for this track.
                        if len(track_vault[matched_t_id]["embeddings"]) >= MIN_SAMPLES:
                            continue
                        assigned_tracks.add(matched_t_id)
                        emb = face.embedding
                        should_add = True
                        if len(track_vault[matched_t_id]["embeddings"]) > 3:
                            track_avg = np.mean(track_vault[matched_t_id]["embeddings"], axis=0)
                            track_avg = track_avg / np.linalg.norm(track_avg)
                            if cosine(emb, track_avg) > T_OUTLIER:
                                should_add = False
                        if should_add:
                            track_vault[matched_t_id]["embeddings"].append(emb)
                            gid = track_vault[matched_t_id]["global_id"]
                            if gid and not _is_nf_id(gid):
                                update_exemplars(gid, emb)
                            if len(track_vault[matched_t_id]["embeddings"]) == 1:
                                    # Crops disabled - skip saving
                                    pass
                
                # Identity assignment
                active_gids_in_frame = set(
                    track_vault[pt['track_id']]["global_id"]
                    for pt in person_tracks
                    if track_vault[pt['track_id']]["global_id"] is not None
                )
                
                for pt in person_tracks:
                    t_id = pt['track_id']
                    t_data = track_vault[t_id]
                    x1, y1, x2, y2 = map(int, pt['bbox'])
                    cur_gid = t_data["global_id"]
                    # Promote body-only NF_* to face-based G_* when enough face embeddings exist
                    if _is_nf_id(cur_gid) and len(t_data["embeddings"]) >= MIN_SAMPLES:
                        track_centroid = np.mean(t_data["embeddings"], axis=0)
                        track_centroid = track_centroid / np.linalg.norm(track_centroid)
                        best_id, best_sim = find_match_with_margin(
                            track_centroid, active_gids_in_frame
                        )
                        nf_data = global_gallery.pop(cur_gid, {})
                        nf_att = nf_data.get("attendance", {})
                        if best_id:
                            ba = global_gallery[best_id].setdefault("attendance", {})
                            for dkey, slot in nf_att.items():
                                if dkey not in ba:
                                    ba[dkey] = dict(slot)
                                else:
                                    ex = ba[dkey]
                                    if slot["entry"] < ex["entry"]:
                                        ex["entry"] = slot["entry"]
                                    if slot["exit"] > ex["exit"]:
                                        ex["exit"] = slot["exit"]
                            t_data["global_id"] = best_id
                            if cur_gid in active_gids_in_frame:
                                active_gids_in_frame.discard(cur_gid)
                            active_gids_in_frame.add(best_id)
                            log_attendance(best_id, timestamp)
                            update_exemplars(best_id, track_centroid)
                        elif best_sim < T_NEW_ID or RUN_MODE == "EVAL_DAY":
                            if RUN_MODE == "BUILD_DB":
                                new_gid = f"G_{next_global_id:03d}"
                                next_global_id += 1
                            else:
                                new_gid = f"{DAY_LABEL}_V_{next_visitor_id:03d}"
                                next_visitor_id += 1
                            global_gallery[new_gid] = {
                                "exemplars": [track_centroid],
                                "attendance": dict(nf_att),
                                "join_date": nf_data.get("join_date", CURRENT_DATE),
                            }
                            t_data["global_id"] = new_gid
                            if cur_gid in active_gids_in_frame:
                                active_gids_in_frame.discard(cur_gid)
                            active_gids_in_frame.add(new_gid)
                            log_attendance(new_gid, timestamp)
                        else:
                            # Gray zone: restore NF slot
                            global_gallery[cur_gid] = nf_data
                    # Body-only: one NF_* per continuous BoT-SORT track (stable while track_id lives)
                    if (face_app is None or not effective_face_mode) and t_data["global_id"] is None and t_data["frames"] >= 5:
                        if RUN_MODE == "BUILD_DB":
                            new_gid = f"NF_{next_nf_id:03d}"
                            next_nf_id += 1
                        else:
                            new_gid = f"{DAY_LABEL}_NF_{next_nf_id:03d}"
                            next_nf_id += 1
                        global_gallery[new_gid] = {
                            "exemplars": [], "attendance": {}, "join_date": CURRENT_DATE
                        }
                        t_data["global_id"] = new_gid
                        active_gids_in_frame.add(new_gid)
                        log_attendance(new_gid, timestamp)
                    # Face on but no face crops on this track yet — still label body track (stable per track_id)
                    if (
                        face_app is not None
                        and effective_face_mode
                        and t_data["global_id"] is None
                        and t_data["frames"] >= 5
                        and len(t_data["embeddings"]) == 0
                    ):
                        if RUN_MODE == "BUILD_DB":
                            new_gid = f"NF_{next_nf_id:03d}"
                            next_nf_id += 1
                        else:
                            new_gid = f"{DAY_LABEL}_NF_{next_nf_id:03d}"
                            next_nf_id += 1
                        global_gallery[new_gid] = {
                            "exemplars": [], "attendance": {}, "join_date": CURRENT_DATE
                        }
                        t_data["global_id"] = new_gid
                        active_gids_in_frame.add(new_gid)
                        log_attendance(new_gid, timestamp)
                    if t_data["global_id"] is None and len(t_data["embeddings"]) >= MIN_SAMPLES:
                        track_centroid = np.mean(t_data["embeddings"], axis=0)
                        track_centroid = track_centroid / np.linalg.norm(track_centroid)
                        best_id, best_sim = find_match_with_margin(track_centroid, active_gids_in_frame)
                        if best_id:
                            t_data["global_id"] = best_id
                            active_gids_in_frame.add(best_id)
                            log_attendance(best_id, timestamp)
                        elif best_sim < T_NEW_ID or RUN_MODE == "EVAL_DAY":
                            # In EVAL_DAY mode, always create a new visitor when no confident
                            # match is found. T_NEW_ID only guards against duplicates in
                            # BUILD_DB; in EVAL_DAY with a pre-existing gallery, similarities
                            # often land in the 0.35–0.55 gray zone and would leave every
                            # track permanently unidentified (resulting in 0 detections).
                            if RUN_MODE == "BUILD_DB":
                                new_gid = f"G_{next_global_id:03d}"
                                next_global_id += 1
                            else:
                                new_gid = f"{DAY_LABEL}_V_{next_visitor_id:03d}"
                                next_visitor_id += 1
                            global_gallery[new_gid] = {"exemplars": [track_centroid]}
                            t_data["global_id"] = new_gid
                            active_gids_in_frame.add(new_gid)
                            log_attendance(new_gid, timestamp)
                        if t_data["global_id"]:
                            old_path = f"{crops_dir}/track_{t_id}.jpg"
                            new_path = f"{verification_dir}/{t_data['global_id']}_track_{t_id}.jpg"
                            if os.path.exists(old_path) and not os.path.exists(new_path):
                                shutil.copy(old_path, new_path)
                    # Draw annotations whenever the frame will be seen (preview) or saved
                    if save_output_video or self.frame_callback is not None or self.use_cv2_preview:
                        gid = t_data["global_id"]
                        if gid:
                            if str(gid).startswith("G_"):
                                color = (0, 255, 0)
                            elif _is_nf_id(gid):
                                color = (255, 255, 0)
                            else:
                                color = (255, 0, 0)
                            label = f"ID: {gid}"
                        else:
                            color = (0, 165, 255)
                            if effective_face_mode:
                                progress_count = min(len(t_data['embeddings']), MIN_SAMPLES)
                                label = f"Scanning: {progress_count}/{MIN_SAMPLES}"
                            else:
                                label = f"Scanning: {min(t_data['frames'], 5)}/5"
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                # Draw timestamp overlay once per frame (not per-person)
                if save_output_video or self.frame_callback is not None or self.use_cv2_preview:
                    cv2.putText(frame, f"Live: {timestamp}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                if out is not None:
                    out.write(frame)
                    frames_written += 1
                if self.use_cv2_preview and frame_idx % 5 == 0:
                    preview_frame = frame.copy()
                    h, w = preview_frame.shape[:2]
                    if w > 640:
                        preview_frame = cv2.resize(preview_frame, (640, int(h * 640 / w)))
                    cv2.imshow(cv2_preview_window, preview_frame)
                    cv2.waitKey(1)
                elif self.frame_callback and frame_idx % 5 == 0:
                    preview_frame = frame.copy()
                    h, w = preview_frame.shape[:2]
                    if w > 480:
                        preview_frame = cv2.resize(preview_frame, (480, int(h * 480 / w)))
                    preview_frame = cv2.cvtColor(preview_frame, cv2.COLOR_BGR2RGB)
                    self.frame_callback(preview_frame, True)
                frame_idx += 1
        finally:
            executor.shutdown(wait=True)
        if self.use_cv2_preview:
            try:
                cv2.destroyWindow(cv2_preview_window)
            except Exception:
                pass
        cap.release()
        if out is not None:
            out.release()
            self._log(f"Output video: {frames_written} frames written (expected {total_frames})", "info")
            if frames_written != total_frames:
                self._log(
                    f"WARNING: Frame count mismatch. If output is shorter than input, try Settings > Performance > save as AVI, or install FFmpeg for better MP4 support.",
                    "warning"
                )
        
        if self._should_stop():
            return ""

        # Enrolled student mapping before summary/reporting.
        map_identities_to_students()

        # Visitor upgrade
        self._log("Checking for visitor upgrades...")
        visitors_to_upgrade = []
        for g_id, g_data in global_gallery.items():
            if "_V_" in g_id:
                days_present = len(g_data.get("attendance", {}))
                if days_present >= VISITOR_UPGRADE_DAYS:
                    visitors_to_upgrade.append(g_id)
        
        for old_vid in visitors_to_upgrade:
            new_gid = f"G_{next_global_id:03d}"
            next_global_id += 1
            global_gallery[new_gid] = global_gallery.pop(old_vid)
            self._log(f"Upgraded {old_vid} to {new_gid}", "success")
        
        # Save database
        self._log("Saving database...")
        self._progress(92, "Saving database...")

        # Always persist DB under this run's output_dir. Using self.db_path here broke when
        # last_db_path pointed at a missing folder (other PC / cleared drive): video wrote to
        # F:\...\Outputs\... but SQLite targeted the stale path → "unable to open database file".
        if RUN_MODE == "BUILD_DB":
            save_path = os.path.join(self.output_dir, "master_database.db")
        else:
            save_path = os.path.join(self.output_dir, "updated_master_database.db")

        def _save_db(path):
            path = os.path.abspath(os.path.normpath(path))
            parent = os.path.dirname(path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            if path.endswith(".db"):
                try:
                    conn = sqlite3.connect(path)
                except sqlite3.OperationalError as e:
                    self._log(
                        f"SQLite could not open {path!r} ({e}). "
                        "Check disk space, permissions, and that the path is not on a read-only or problematic drive.",
                        "error",
                    )
                    raise
                cur = conn.cursor()
                cur.execute("CREATE TABLE IF NOT EXISTS operational_dates (date TEXT PRIMARY KEY)")
                cur.execute("""CREATE TABLE IF NOT EXISTS identities (
                    g_id TEXT PRIMARY KEY, join_date TEXT,
                    engagement_id TEXT, batch TEXT, confidence REAL)""")
                cur.execute("CREATE TABLE IF NOT EXISTS exemplars (g_id TEXT, embedding BLOB)")
                cur.execute("""CREATE TABLE IF NOT EXISTS attendance (
                    g_id TEXT, date TEXT, entry_time TEXT, exit_time TEXT,
                    PRIMARY KEY (g_id, date))""")
                # Full replace to prevent stale ghost identities after visitor upgrades
                cur.execute("DELETE FROM exemplars")
                cur.execute("DELETE FROM identities")
                cur.execute("DELETE FROM attendance")
                for date in operational_dates:
                    cur.execute("REPLACE INTO operational_dates (date) VALUES (?)", (date,))
                for g_id, data in global_gallery.items():
                    cur.execute(
                        "REPLACE INTO identities (g_id, join_date, engagement_id, batch, confidence) VALUES (?,?,?,?,?)",
                        (g_id, data.get("join_date"), data.get("engagement_id"),
                         data.get("batch"), data.get("confidence", 0.0))
                    )
                    for emb in data.get("exemplars", []):
                        cur.execute("INSERT INTO exemplars (g_id, embedding) VALUES (?,?)",
                                    (g_id, np.array(emb, dtype=np.float32).tobytes()))
                    for date, times in data.get("attendance", {}).items():
                        cur.execute(
                            "REPLACE INTO attendance (g_id, date, entry_time, exit_time) VALUES (?,?,?,?)",
                            (g_id, date, times["entry"], times["exit"])
                        )
                conn.commit()
                conn.close()
            else:
                db_data = {"gallery": global_gallery, "operational_dates": operational_dates}
                with open(path, 'wb') as f:
                    pickle.dump(db_data, f)

        _save_db(save_path)
        self._log(f"Database saved: {save_path}", "success")
        
        # Generate report
        self._log("Generating report...")
        self._progress(96, "Generating report...")
        
        def calculate_duration(entry_str, exit_str):
            fmt = '%H:%M:%S'
            try:
                td = datetime.strptime(exit_str, fmt) - datetime.strptime(entry_str, fmt)
                return int(td.total_seconds())
            except (ValueError, TypeError):
                return 0
        
        report = {
            "Session": {
                "date": CURRENT_DATE,
                "camera": "Cam_01",
                "source_video": os.path.basename(self.video_path),
                "duration": "00:00:00"
            },
            "Counts": {
                "unique_people": 0,
                "nf_presence": 0,
                "returning": 0,
                "visitors": 0,
                "identified_students": 0
            },
            "People": []
        }
        
        current_dt = datetime.strptime(CURRENT_DATE, "%Y-%m-%d")
        latest_exit_time = "00:00:00"
        
        for g_id, g_data in global_gallery.items():
            attendance = g_data.get("attendance", {})
            if CURRENT_DATE not in attendance:
                continue
            
            entry_time = attendance[CURRENT_DATE]["entry"]
            exit_time = attendance[CURRENT_DATE]["exit"]
            duration = calculate_duration(entry_time, exit_time)
            
            if exit_time > latest_exit_time:
                latest_exit_time = exit_time
            
            is_new_walk_in = g_data.get("join_date") == CURRENT_DATE
            
            person_dict = {
                "id": g_id,
                "engagement_id": g_data.get("engagement_id"),
                "batch": g_data.get("batch"),
                "entry": entry_time,
                "exit": exit_time,
                "duration_sec": duration,
                "confidence_score": g_data.get("confidence", 0.0)
            }
            
            if _is_nf_id(g_id):
                person_dict["type"] = "no_face_track"
                person_dict["last_present_date"] = None
                person_dict["present_last_7_days"] = 0
                report["Counts"]["nf_presence"] += 1
            elif g_data.get("engagement_id") is not None:
                person_dict["type"] = "enrolled_student"
                person_dict["last_present_date"] = None
                person_dict["present_last_7_days"] = 0
                report["Counts"]["identified_students"] += 1
            elif is_new_walk_in:
                person_dict["type"] = "visitor"
                person_dict["last_present_date"] = None
                person_dict["present_last_7_days"] = 0
                report["Counts"]["visitors"] += 1
            else:
                person_dict["type"] = "returning_employee"
                past_dates = [d for d in attendance.keys() if d < CURRENT_DATE]
                past_dates.sort(reverse=True)
                person_dict["last_present_date"] = past_dates[0] if past_dates else None
                
                present_last_7 = 0
                for i in range(1, 8):
                    check_date = (current_dt - timedelta(days=i)).strftime("%Y-%m-%d")
                    if check_date in attendance:
                        present_last_7 += 1
                person_dict["present_last_7_days"] = present_last_7
                
                report["Counts"]["returning"] += 1
            
            report["People"].append(person_dict)
        
        report["Counts"]["unique_people"] = (
            report["Counts"]["returning"] +
            report["Counts"]["visitors"] +
            report["Counts"].get("identified_students", 0)
        )
        report["Session"]["duration"] = latest_exit_time
        
        report_path = os.path.join(self.output_dir, f"{CURRENT_DATE.replace('-', '_')}_attendance_report.json")
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=4)
        
        self._log(f"Report saved: {report_path}", "success")
        self._log("=== FINAL DAILY SUMMARY ===\n" + json.dumps(report["Counts"], indent=4), "info")
        if save_output_video:
            self._log(f"Output video: {output_video_path}", "success")
        
        return report_path
