"""
Classroom Analysis Worker Thread.
Runs the classroom analysis pipeline in a background thread.
Full implementation matching classroom_activity_1.py (V14 - FINAL INTEGRATED PIPELINE).
"""

import os
import sys
import traceback
from pathlib import Path
from typing import Dict, Any, Optional, Callable, List

import numpy as np


def _purge_ultralytics_weight_cache(filename: str) -> None:
    """Remove cached .pt copies so YOLO(name) can download a clean checkpoint."""
    roots: List[Path] = [
        Path.home() / ".cache" / "ultralytics",
        Path.home() / ".cache" / "torch" / "hub",
    ]
    if sys.platform == "win32":
        la = os.environ.get("LOCALAPPDATA", "")
        if la:
            roots.append(Path(la) / "Ultralytics")
    for root in roots:
        if not root.is_dir():
            continue
        try:
            for p in root.rglob(filename):
                if p.is_file():
                    try:
                        p.unlink()
                    except OSError:
                        pass
        except OSError:
            pass
from PyQt6.QtCore import QThread, pyqtSignal

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class ClassroomWorker(QThread):
    """Worker thread for classroom analysis pipeline."""
    
    progress = pyqtSignal(int, str)  # percent, message
    log_message = pyqtSignal(str, str)  # message, level
    frame_ready = pyqtSignal(np.ndarray, bool)  # frame, is_rgb (for video preview)
    finished = pyqtSignal(str)  # report path
    error = pyqtSignal(str)  # error message
    
    def __init__(
        self,
        video_path: str,
        output_dir: str,
        config: Dict[str, Any],
        inference_config: Optional[Dict[str, Any]] = None,
        preview_enabled: bool = False,
        parent=None
    ):
        super().__init__(parent)
        self.video_path = video_path
        self.output_dir = output_dir
        self.config = config
        self.inference_config = inference_config or {}
        self.preview_enabled = preview_enabled
        self._stop_requested = False
        
    def run(self):
        """Run the classroom analysis."""
        try:
            try:
                import torch
                # Verify torch loads - if DLL fails, we catch it here
                _ = torch.__version__
            except (OSError, ImportError) as torch_err:
                err_msg = (
                    "PyTorch failed to load (DLL error).\n\n"
                    "Most common cause on Windows: Microsoft Visual C++ 2015-2022 "
                    "Redistributable (x64) is missing.\n"
                    "Download: https://aka.ms/vs/17/release/vc_redist.x64.exe\n"
                    "Install, restart the machine, then relaunch the app.\n\n"
                    "If the error persists, install CPU-only PyTorch:\n"
                    "  pip uninstall torch torchvision\n"
                    "  pip install torch torchvision --index-url "
                    "https://download.pytorch.org/whl/cpu\n\n"
                    f"Original error: {torch_err}"
                )
                self.error.emit(err_msg)
                return
            
            # Create analyzer with callbacks
            analyzer = FaceEngagementAnalyzerWithCallbacks(
                video_path=self.video_path,
                output_dir=self.output_dir,
                config=self.config,
                inference_config=self.inference_config,
                progress_callback=self._emit_progress,
                log_callback=self._emit_log,
                frame_callback=self._emit_frame if self.preview_enabled else None,
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


# --- RUNTIME ANCHOR MANAGER (from original) ---
class RuntimeAnchorManager:
    """
    Manages stable ID assignment by anchoring detections to spatial 'seats'.
    Prevents ID fragmentation when tracker loses/regains tracks.
    """
    def __init__(self, similarity_thresh=0.70, distance_thresh=120):
        self.seats = {}
        self.next_seat_uid = 0
        self.active_mapping = {}
        self.sim_thresh = similarity_thresh
        self.dist_thresh = distance_thresh
        self.lock_frames = 10
        from collections import defaultdict
        self.potential_seats = defaultdict(lambda: {'count': 0, 'centroids': []})

    def get_corrected_id(self, raw_id, bbox, embedding, frame_idx):
        from scipy.spatial.distance import cosine
        import math
        
        curr_centroid = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
        if raw_id in self.active_mapping:
            stable_id = self.active_mapping[raw_id]
            self._update_seat_position(stable_id, curr_centroid, embedding, frame_idx)
            return stable_id

        best_seat_id = None
        min_dist = float('inf')
        for s_uid, seat in self.seats.items():
            dist = math.sqrt((curr_centroid[0] - seat['centroid'][0])**2 + 
                           (curr_centroid[1] - seat['centroid'][1])**2)
            if dist < self.dist_thresh:
                sim = 0.0
                if embedding is not None and seat['embedding'] is not None:
                    sim = 1.0 - cosine(embedding, seat['embedding'])
                time_gap = frame_idx - seat['last_seen']
                is_match = False
                if time_gap < 90 and dist < 80:
                    is_match = True
                elif sim > self.sim_thresh:
                    is_match = True
                if is_match and dist < min_dist:
                    min_dist = dist
                    best_seat_id = s_uid

        if best_seat_id is not None:
            stable_id = self.seats[best_seat_id]['owner_id']
            self.active_mapping[raw_id] = stable_id
            self._update_seat_position(stable_id, curr_centroid, embedding, frame_idx)
            return stable_id

        self.active_mapping[raw_id] = raw_id
        self._check_create_seat(raw_id, curr_centroid, embedding, frame_idx)
        return raw_id

    def _update_seat_position(self, stable_id, centroid, embedding, frame_idx):
        for s_uid, seat in self.seats.items():
            if seat['owner_id'] == stable_id:
                seat['centroid'][0] = 0.9 * seat['centroid'][0] + 0.1 * centroid[0]
                seat['centroid'][1] = 0.9 * seat['centroid'][1] + 0.1 * centroid[1]
                seat['last_seen'] = frame_idx
                if embedding is not None and seat['embedding'] is None:
                    seat['embedding'] = embedding
                return

    def _check_create_seat(self, raw_id, centroid, embedding, frame_idx):
        import math
        data = self.potential_seats[raw_id]
        data['count'] += 1
        data['centroids'].append(centroid)
        if data['count'] == self.lock_frames:
            avg_x = sum(c[0] for c in data['centroids']) / len(data['centroids'])
            avg_y = sum(c[1] for c in data['centroids']) / len(data['centroids'])
            for s in self.seats.values():
                d = math.sqrt((avg_x - s['centroid'][0])**2 + (avg_y - s['centroid'][1])**2)
                if d < 50:
                    return
            self.seats[self.next_seat_uid] = {
                'owner_id': raw_id,
                'centroid': [avg_x, avg_y],
                'embedding': embedding,
                'last_seen': frame_idx
            }
            self.next_seat_uid += 1


class FaceEngagementAnalyzerWithCallbacks:
    """
    Full FaceEngagementAnalyzer with GUI callbacks.
    Matches original classroom_activity_1.py (V14 - FINAL INTEGRATED PIPELINE).
    """
    
    def __init__(
        self,
        video_path: str,
        output_dir: str,
        config: Dict[str, Any],
        inference_config: Optional[Dict[str, Any]] = None,
        progress_callback: Optional[Callable] = None,
        log_callback: Optional[Callable] = None,
        frame_callback: Optional[Callable] = None,
        stop_check: Optional[Callable] = None
    ):
        self.video_path = video_path
        self.output_dir = output_dir
        self.config = config
        self.inference_config = inference_config or {}
        self.progress_callback = progress_callback
        self.log_callback = log_callback
        self.frame_callback = frame_callback
        self.stop_check = stop_check
        
        # Video properties (set during analysis)
        self.fps = None
        self.width = None
        self.height = None
        self.total_frames = None
        
        # Stitch model reference (from tracker)
        self.stitch_model = None
        
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
    
    def get_embedding(self, frame, bbox, device):
        """Extract embedding for a person crop using the stitch model."""
        import cv2
        import torch
        
        if self.stitch_model is None:
            return None
        x1, y1, x2, y2 = map(int, bbox)
        h, w, _ = frame.shape
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 - x1 < 10 or y2 - y1 < 10:
            return None
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        crop = cv2.resize(crop, (128, 256))
        crop = crop.transpose(2, 0, 1)
        crop = np.ascontiguousarray(crop, dtype=np.float32)
        crop /= 255.0
        tensor = torch.from_numpy(crop).unsqueeze(0).to(device)
        try:
            with torch.no_grad():
                model_to_call = self.stitch_model
                if hasattr(model_to_call, 'model'):
                    model_to_call = model_to_call.model
                elif hasattr(model_to_call, 'net'):
                    model_to_call = model_to_call.net
                try:
                    if next(model_to_call.parameters()).dtype == torch.float16:
                        tensor = tensor.half()
                except:
                    pass
                feat = model_to_call(tensor)
                norm = torch.norm(feat, p=2, dim=1, keepdim=True)
                feat = feat.div(norm.expand_as(feat))
                return feat.cpu().numpy().flatten().tolist()
        except:
            return None
    
    def calculate_engagement_score(self, activity, attention, zone, confidence):
        """Calculate engagement score based on activity, attention, zone, and confidence."""
        score = 0.0
        act_s = {
            'raising_hand': 1.0, 'writing': 0.9, 'reading': 0.75, 
            'listening': 0.6, 'talking': 0.5, 'walking': 0.1, 
            'standing': 0.3, 'unknown': 0.2
        }
        score += act_s.get(activity, 0.2)
        att_s = {'focused': 0.3, 'partially_focused': 0.15, 'distracted': 0.0}
        score += att_s.get(attention, 0.0)
        score += 0.1 if zone == 'front' else 0.05
        score += confidence * 0.1
        state = 'engaged' if score >= 0.8 else 'partially_engaged' if score >= 0.5 else 'not_engaged'
        return min(1.0, score), state

    def match_face_to_person(self, face_bbox, persons, pose_data):
        """Match a face bounding box to a person and determine activity/attention."""
        import math
        
        if not persons:
            return {'activity': 'unknown', 'attention': 'not_visible', 'zone': 'middle', 'confidence': 0.0}
        
        fc = ((face_bbox[0] + face_bbox[2]) / 2, (face_bbox[1] + face_bbox[3]) / 2)
        min_d = float('inf')
        m_pose = None
        m_person = None
        
        for i, p in enumerate(persons):
            pc = p['center']
            d = math.sqrt((fc[0] - pc[0])**2 + (fc[1] - pc[1])**2)
            if d < min_d:
                min_d = d
                m_person = p
                m_pose = pose_data[i] if i < len(pose_data) else None
        
        if min_d > 200:
            return {'activity': 'unknown', 'attention': 'not_visible', 'zone': 'middle', 'confidence': 0.0}
        
        act = 'unknown'
        if m_pose:
            kp = m_pose['keypoints']
            # Check for raising hand (wrist above shoulder)
            if len(kp) >= 10 and kp[9][1] > 0 and kp[5][1] > 0 and kp[9][1] < kp[5][1]:
                act = 'raising_hand'
            # Check for writing (head below shoulders)
            elif len(kp) >= 6 and kp[0][1] > 0 and kp[5][1] > 0 and kp[0][1] > kp[5][1]:
                act = 'writing'
            elif kp[0][0] > 0:
                act = 'listening'
        
        att = 'distracted'
        if act in ['raising_hand', 'writing']:
            att = 'focused'
        elif act == 'listening':
            att = 'partially_focused'
        
        y = fc[1]
        zone = 'front' if y < self.height * 0.4 else 'middle' if y < self.height * 0.7 else 'back'
        
        return {
            'activity': act, 
            'attention': att, 
            'zone': zone, 
            'confidence': m_person.get('confidence', 0.0) if m_person else 0.0
        }

    def extract_pose_data(self, pose_results):
        """Extract pose keypoints from YOLO pose results."""
        data = []
        for r in pose_results:
            if r.keypoints is not None:
                for kp in r.keypoints.data:
                    data.append({'keypoints': kp.cpu().numpy()})
        return data
        
    def analyze_video(self) -> str:
        """Run the full analysis with callbacks (matching original V14 pipeline)."""
        import warnings
        import logging
        warnings.filterwarnings("ignore", category=FutureWarning)
        logging.getLogger("boxmot").setLevel(logging.ERROR)
        
        import cv2
        import json
        import torch
        from collections import defaultdict, Counter
        import math
        from pathlib import Path
        from scipy.spatial.distance import cosine
        from datetime import datetime, timedelta
        
        from ultralytics import YOLO
        
        os.makedirs(self.output_dir, exist_ok=True)

        from app.runtime_checks import run_classroom_preflight

        self._progress(0, "Preflight checks")
        preflight = run_classroom_preflight(self._log)
        if not preflight.ok:
            raise RuntimeError(
                "Preflight failed:\n" + "\n".join(preflight.failures)
            )
        
        # Import stitching and VLM
        try:
            from classroom_analysis.stitch_logic import perform_hierarchical_stitching
        except ImportError:
            def perform_hierarchical_stitching(json_path, **kwargs): 
                return {}
            
        try:
            from classroom_analysis.vlm_metadata import extract_camera_metadata_vlm
        except ImportError:
            def extract_camera_metadata_vlm(frame): 
                return {"classroom": "Unknown", "base_datetime": None, "base_datetime_str": "Unknown"}
        try:
            from classroom_analysis.group_by_session import generate_management_summary
        except ImportError:
            def generate_management_summary(*args, **kwargs):
                return None
        try:
            from classroom_analysis.ocr_overlay import (
                apply_ocr_datetime_to_metadata,
                read_ocr_overlay_frame,
            )
        except ImportError:
            def read_ocr_overlay_frame(*args, **kwargs):
                return None, None, ""
            def apply_ocr_datetime_to_metadata(*args, **kwargs):
                return False
        try:
            from classroom_analysis.model_weights import (
                model_weight_candidates,
                resolve_model_weight,
                resolve_weights_dir,
            )
        except ImportError:
            def resolve_weights_dir():
                base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                weights_dir = os.path.join(base_path, "Models")
                if not os.path.exists(weights_dir):
                    weights_dir = base_path
                return weights_dir, base_path
            def model_weight_candidates(weight_file, weights_dir=None, base_path=None):
                bp = base_path or os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                wd = weights_dir or os.path.join(bp, "Models")
                return [os.path.join(wd, weight_file), os.path.join(bp, weight_file)]
            def resolve_model_weight(weight_file, weights_dir=None, base_path=None):
                for candidate in model_weight_candidates(weight_file, weights_dir, base_path):
                    if os.path.isfile(candidate):
                        return candidate
                return None
        
        self._progress(5, "Loading models…")

        force_cpu = bool(self.inference_config.get("force_cpu", False))
        if force_cpu:
            self.device = 'cpu'
            self.use_fp16 = False
        elif torch.cuda.is_available():
            self.device = 'cuda:0'
            self.use_fp16 = True
        else:
            self.device = 'cpu'
            self.use_fp16 = False
        device = self.device
        
        # Resolve Models path (project root or frozen app)
        weights_dir, base_path = resolve_weights_dir()

        # Load models (check Models/ first; face model not in ultralytics hub — custom file)
        models = {}
        for name, file in {'detection': 'yolov8m.pt', 'pose': 'yolov8n-pose.pt', 'face': 'yolov8n-face.pt'}.items():
            candidates = model_weight_candidates(file, weights_dir, base_path)
            path = None
            for c in candidates:
                if os.path.isfile(c):
                    path = c
                    break
            load_name = path if path else file
            try:
                m = YOLO(load_name)
                m.to(device)
                if hasattr(m.model, 'fuse'):
                    m.model.fuse()
                models[name] = m
            except Exception as e:
                # Corrupt local .pt or corrupt hub cache — remove all known copies, purge cache, re-fetch
                if name in ('detection', 'pose'):
                    self._log(f"Weight load failed ({file}): {e}", "warning")
                    removed = 0
                    for c in candidates:
                        if os.path.isfile(c):
                            try:
                                os.remove(c)
                                removed += 1
                            except OSError:
                                pass
                    if removed:
                        self._log(f"Removed {removed} bad on-disk copy(s) of {file}", "warning")
                    _purge_ultralytics_weight_cache(file)
                    self._log(f"Purged Ultralytics/torch cache for {file}; re-downloading…", "warning")
                    try:
                        m = YOLO(file)
                        m.to(device)
                        if hasattr(m.model, 'fuse'):
                            m.model.fuse()
                        models[name] = m
                    except Exception as e2:
                        self._log(f"Warning: Could not load {name} model: {e2}", "warning")
                else:
                    self._log(f"Warning: Could not load {name} model: {e}", "warning")
        
        # Open video
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise ValueError("Could not open video file")
            
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # VLM metadata extraction
        ret, first_frame = cap.read()
        if ret and first_frame is not None and isinstance(first_frame, np.ndarray) and first_frame.size > 0:
            video_metadata = extract_camera_metadata_vlm(first_frame)
            self._log(f"Classroom: {video_metadata['classroom']}", "success")
        else:
            video_metadata = {"classroom": "Unknown", "base_datetime": None, "base_datetime_str": "Unknown"}
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        enable_ocr_timestamp = bool(self.config.get("enable_ocr_timestamp", True))
        timestamp_coords = self.config.get("timestamp_coords", [0, 15, 600, 90])
        if not isinstance(timestamp_coords, (list, tuple)) or len(timestamp_coords) != 4:
            timestamp_coords = [0, 15, 600, 90]
        try:
            timestamp_coords = tuple(int(v) for v in timestamp_coords)
        except (TypeError, ValueError):
            timestamp_coords = (0, 15, 600, 90)

        if enable_ocr_timestamp and ret and first_frame is not None:
            try:
                import easyocr
                easyocr_dir = os.path.join(weights_dir, "easyocr")
                os.makedirs(easyocr_dir, exist_ok=True)
                ocr_reader = easyocr.Reader(
                    ['en'],
                    gpu=False,
                    model_storage_directory=easyocr_dir,
                )
                ocr_d, ocr_t, ocr_raw = read_ocr_overlay_frame(
                    first_frame, ocr_reader, timestamp_coords
                )
                if apply_ocr_datetime_to_metadata(video_metadata, ocr_d, ocr_t):
                    self._log(
                        f"OCR recording datetime (applied): {video_metadata['base_datetime_str']}",
                        "success",
                    )
                elif ocr_raw:
                    self._log(
                        f"OCR overlay read but date/time incomplete: {ocr_raw[:220]!r}",
                        "warning",
                    )
            except Exception as ocr_err:
                self._log(f"OCR metadata unavailable ({ocr_err})", "warning")

        # Apply configured fallback when VLM returned no classroom — keeps the
        # ClassRoom Name consistent across runs even when the VLM is unavailable.
        classroom_fallback = str(self.config.get("classroom_name_fallback") or "").strip()
        if (
            not video_metadata.get("classroom")
            or str(video_metadata["classroom"]).strip().lower() in {"", "unknown", "none"}
        ) and classroom_fallback:
            video_metadata["classroom"] = classroom_fallback
            self._log(f"Classroom (fallback): {classroom_fallback}", "info")
        
        # Sampling config from GUI (defaults match original)
        PROBE_DURATION_SEC = self.config.get("probe_duration", 300)
        PROBE_INTERVAL_SEC = self.config.get("probe_interval", 3600)
        FRAME_SKIP = self.config.get("frame_skip", 3)
        
        probe_frames = int(PROBE_DURATION_SEC * self.fps)
        interval_frames = int(PROBE_INTERVAL_SEC * self.fps)
        
        start_frames = []
        curr = 0
        while curr < self.total_frames:
            start_frames.append(curr)
            curr += interval_frames
            
        # Tracker init (fp16 only on CUDA - CPU lacks half-precision conv support)
        tracker = None
        def _ensure_valid_stdio():
            if sys.stdout is None:
                sys.stdout = open(os.devnull, "w")
            if sys.stderr is None:
                sys.stderr = open(os.devnull, "w")

        def _stabilize_loguru_sink():
            try:
                from loguru import logger
                logger.remove()
                logger.add(sys.stderr if sys.stderr is not None else os.devnull, level="ERROR")
            except Exception:
                pass

        def _init_botsort_tracker():
            try:
                import pkg_resources  # noqa: F401 — required by boxmot
            except ImportError as import_err:
                raise RuntimeError(
                    "BoT-SORT requires setuptools with pkg_resources. "
                    "Install app dependencies with: pip install -r requirements_app.txt"
                ) from import_err
            from boxmot import BoTSORT

            reid_weights = resolve_model_weight(
                "osnet_x1_0_msmt17.pt", weights_dir, base_path
            )
            if not reid_weights:
                raise FileNotFoundError(
                    "Re-ID weights osnet_x1_0_msmt17.pt not found. "
                    "Place the file in Models/ or the project root."
                )
            tracker = BoTSORT(
                model_weights=Path(reid_weights),
                device=self.device,
                fp16=self.use_fp16,
                track_buffer=300,
                match_thresh=0.75,
            )
            if hasattr(tracker, "model"):
                self.stitch_model = tracker.model
            return tracker

        try:
            # boxmot/loguru can fail in GUI contexts when std streams are None.
            _ensure_valid_stdio()
            _stabilize_loguru_sink()
            tracker = _init_botsort_tracker()
        except Exception as e:
            # Retry once after forcing non-None streams (fixes loguru sink errors).
            try:
                _ensure_valid_stdio()
                _stabilize_loguru_sink()
                tracker = _init_botsort_tracker()
            except Exception as retry_err:
                self._log(f"BoT-SORT init failed: {e}", "error")
                self._log(f"BoT-SORT retry failed: {retry_err}", "error")
                raise RuntimeError(
                    "BoT-SORT / Re-ID model failed to load. "
                    "Ensure osnet_x1_0_msmt17.pt is present in Models/."
                ) from retry_err
        
        # Initialize RuntimeAnchorManager for ID correction
        anchor_manager = RuntimeAnchorManager(similarity_thresh=0.75, distance_thresh=120)
        
        # Data storage (matching original structure)
        face_engagement_data = defaultdict(lambda: {
            'face_id': None, 'total_frames': 0, 'best_embedding': None,
            'start_centroid': None, 'last_centroid': None,
            'engagement_scores': [], 'first_seen': None, 'last_seen': None,
            'activities': Counter(), 'attention_levels': Counter()
        })
        raw_probe_data = []
        
        # Process probes
        total_probes = len(start_frames)
        # Emit a progress update only when the integer percent changes, and only
        # send preview frames every Nth frame, to keep the Qt event loop responsive
        # on lower-end laptops. (Matches the attendance worker's _last_emit_pct
        # behaviour.)
        _last_emit_pct = -1
        PREVIEW_EVERY_N_FRAMES = 10
        for probe_idx, start_f in enumerate(start_frames):
            if self._should_stop():
                break
                
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
            
            probe_ids_seen = set()
            frames_processed = 0
            
            while frames_processed < probe_frames:
                if self._should_stop():
                    break
                    
                ret, frame = cap.read()
                if not ret:
                    break
                if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
                    frames_processed += 1
                    continue
                    
                curr_frame_global = start_f + frames_processed
                
                # Progress calculation (0-90% for main loop; 90-100% reserved for stitching/report)
                total_work = total_probes * probe_frames
                if total_work > 0:
                    overall_progress = min(
                        90,
                        int((probe_idx * probe_frames + frames_processed) / total_work * 90),
                    )
                    if overall_progress != _last_emit_pct:
                        _last_emit_pct = overall_progress
                        self._progress(
                            overall_progress,
                            f"Probe {probe_idx + 1}/{total_probes} - Frame {frames_processed}/{probe_frames}",
                        )
                
                if frames_processed % FRAME_SKIP == 0:
                    tracked_bodies = []  # Always init so frame callback can iterate
                    # Detection & Merge (matching original)
                    if 'detection' in models and 'face' in models:
                        body_results = models['detection'](frame, verbose=False, conf=0.40, classes=[0], imgsz=640)
                        face_results = models['face'](frame, verbose=False, conf=0.25, imgsz=640)
                        
                        raw_body_boxes = body_results[0].boxes.data.cpu().numpy() if len(body_results[0].boxes) > 0 else []
                        face_boxes = face_results[0].boxes.data.cpu().numpy() if len(face_results[0].boxes) > 0 else []
                        
                        # Merge detections (same logic as original)
                        final_dets = []
                        if len(raw_body_boxes) > 0:
                            for b in raw_body_boxes:
                                final_dets.append(b)
                        
                        for f_box in face_boxes:
                            fx1, fy1, fx2, fy2, f_conf, _ = f_box
                            is_matched = False
                            if len(raw_body_boxes) > 0:
                                for b_box in raw_body_boxes:
                                    bx1, by1, bx2, by2, _, _ = b_box
                                    if fx1 > bx1 - 50 and fx2 < bx2 + 50 and fy1 > by1 - 50 and fy2 < by2 + 50:
                                        is_matched = True
                                        break
                            if not is_matched:
                                f_w, f_h = fx2 - fx1, fy2 - fy1
                                synth_w, synth_h = f_w * 2.5, f_h * 4.0
                                sx1, sy1 = max(0, fx1 - (synth_w - f_w) / 2), max(0, fy1)
                                sx2, sy2 = min(self.width, sx1 + synth_w), min(self.height, sy1 + synth_h)
                                final_dets.append([sx1, sy1, sx2, sy2, f_conf, 0.0])
                        
                        # Tracking with RuntimeAnchorManager ID correction
                        if tracker and len(final_dets) > 0:
                            tracks = tracker.update(np.array(final_dets), frame)
                            for t in tracks:
                                raw_id = int(t[4])
                                bbox = t[:4]
                                
                                # Get embedding periodically for better stitching
                                curr_emb = None
                                if frames_processed % 30 == 0:
                                    curr_emb = self.get_embedding(frame, bbox, device)
                                
                                # Use RuntimeAnchorManager for stable ID assignment
                                final_id = anchor_manager.get_corrected_id(raw_id, bbox, curr_emb, curr_frame_global)
                                tracked_bodies.append({'id': final_id, 'bbox': bbox})
                                
                                # Data Update
                                probe_ids_seen.add(final_id)
                                fdata = face_engagement_data[final_id]
                                fdata['face_id'] = final_id
                                fdata['total_frames'] += 1
                                curr_center = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
                                fdata['last_centroid'] = curr_center
                                if not fdata['start_centroid']:
                                    fdata['start_centroid'] = curr_center
                                if not fdata['first_seen']:
                                    fdata['first_seen'] = curr_frame_global / self.fps
                                fdata['last_seen'] = curr_frame_global / self.fps
                                if curr_emb:
                                    fdata['best_embedding'] = curr_emb
                        elif tracker:
                            tracker.update(np.empty((0, 6)), frame)
                        
                        # Engagement analysis using pose (matching original)
                        if 'pose' in models and tracked_bodies:
                            pose_results = models['pose'](frame, verbose=False)
                            pose_data = self.extract_pose_data(pose_results)
                            detected_faces_fmt = [{'bbox': f[:4], 'confidence': f[4]} for f in face_boxes]
                            
                            for body in tracked_bodies:
                                matched_face = None
                                bx1, by1, bx2, by2 = body['bbox']
                                for face in detected_faces_fmt:
                                    fx = (face['bbox'][0] + face['bbox'][2]) / 2
                                    fy = (face['bbox'][1] + face['bbox'][3]) / 2
                                    if bx1 < fx < bx2 and by1 < fy < by2:
                                        matched_face = face
                                        break
                                
                                persons_list = [{'center': [(b['bbox'][0] + b['bbox'][2]) / 2, 
                                                          (b['bbox'][1] + b['bbox'][3]) / 2]} 
                                               for b in tracked_bodies]
                                person_data = self.match_face_to_person(body['bbox'], persons_list, pose_data)
                                
                                activity = person_data['activity']
                                attention = 'not_visible'
                                face_conf = 0.0
                                
                                if matched_face:
                                    face_conf = matched_face['confidence']
                                    attention = person_data['attention']
                                else:
                                    if activity == 'writing':
                                        attention = 'focused'
                                    elif activity == 'walking':
                                        attention = 'partially_focused'
                                
                                score, state = self.calculate_engagement_score(
                                    activity, attention, person_data['zone'], face_conf
                                )
                                
                                # Store in Global Data
                                face_engagement_data[body['id']]['engagement_scores'].append(score)
                                face_engagement_data[body['id']]['activities'][activity] += 1
                                face_engagement_data[body['id']]['attention_levels'][attention] += 1
                                body['activity'] = activity
                                body['state'] = state
                            # If no pose model, still draw boxes with ID only
                            for body in tracked_bodies:
                                if 'activity' not in body:
                                    body['activity'] = 'detected'
                                    body['state'] = ''
                    
                    # Draw annotations and send frame for real-time preview.
                    # Throttle to every Nth frame to reduce Qt event-loop pressure
                    # and avoid UI freezes on weaker hardware.
                    if self.frame_callback and (frames_processed % PREVIEW_EVERY_N_FRAMES == 0):
                        preview_frame = frame.copy()
                        for body in tracked_bodies:
                            x1, y1, x2, y2 = [int(v) for v in body['bbox']]
                            pid = body['id']
                            activity = body.get('activity', '')
                            state = body.get('state', '')
                            color = (0, 255, 0) if state == 'engaged' else (0, 165, 255) if state == 'partially_engaged' else (0, 0, 255)
                            cv2.rectangle(preview_frame, (x1, y1), (x2, y2), color, 2)
                            label = f"ID:{pid}" + (f" {activity}" if activity else "")
                            cv2.putText(preview_frame, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                        # Resize and convert to RGB in worker thread for responsive UI
                        h, w = preview_frame.shape[:2]
                        if w > 480:
                            preview_frame = cv2.resize(preview_frame, (480, int(h * 480 / w)))
                        preview_frame = cv2.cvtColor(preview_frame, cv2.COLOR_BGR2RGB)
                        self.frame_callback(preview_frame, True)
                
                frames_processed += 1
            
            # Save which IDs were in this probe for later report mapping
            raw_probe_data.append({
                "probe_index": probe_idx,
                "start_time": start_f / self.fps,
                "raw_ids": list(probe_ids_seen)
            })
        
        cap.release()
        
        if self._should_stop():
            return ""
        
        # Post-processing & Stitching (matching original)
        self._progress(90, "Post-processing…")
        self._progress(92, "Stitching tracks...")
        
        # 1. Export Raw Stitching Index
        stitch_index_path = os.path.join(self.output_dir, "stitching_index.json")
        stitch_data = []
        for fid, data in face_engagement_data.items():
            if data['total_frames'] < 5:
                continue
            stitch_data.append({
                'face_id': fid,
                'first_seen': data['first_seen'],
                'last_seen': data['last_seen'],
                'start_centroid': data['start_centroid'],
                'last_centroid': data['last_centroid'],
                'embedding': data['best_embedding']
            })
        
        with open(stitch_index_path, 'w') as f:
            json.dump(stitch_data, f)
        
        # 2. Run Self-Stitching
        id_map = perform_hierarchical_stitching(
            stitch_index_path,
            similarity_threshold=self.config.get("similarity_threshold", 0.75),
            max_time_gap=self.config.get("max_time_gap", 600),
            max_pixel_dist=self.config.get("max_pixel_dist", 200)
        )
        
        # 3. Pre-calculate Baseline
        probe_corrected_counts = {}
        for probe in raw_probe_data:
            unique_in_probe = set()
            for raw_id in probe['raw_ids']:
                root_id = id_map.get(raw_id, raw_id)
                unique_in_probe.add(root_id)
            probe_corrected_counts[probe['probe_index']] = len(unique_in_probe)
        
        max_students = max(probe_corrected_counts.values()) if probe_corrected_counts else 0
        
        # 4. Generate Final Corrected Report
        self._progress(96, "Generating report...")
        
        final_hourly_report = []
        for probe in raw_probe_data:
            unique_students = set()
            probe_scores = []
            probe_activities = Counter()
            probe_attention = Counter()
            
            for raw_id in probe['raw_ids']:
                root_id = id_map.get(raw_id, raw_id)
                unique_students.add(root_id)
                raw_data = face_engagement_data[raw_id]
                probe_scores.extend(raw_data['engagement_scores'])
                probe_activities.update(raw_data['activities'])
                probe_attention.update(raw_data['attention_levels'])
            
            student_count = len(unique_students)
            avg_score = sum(probe_scores) / len(probe_scores) if probe_scores else 0
            
            total_act = sum(probe_activities.values())
            act_dist = {k: round(v / total_act * 100, 1) for k, v in probe_activities.items()} if total_act else {}
            total_att = sum(probe_attention.values())
            att_dist = {k: round(v / total_att * 100, 1) for k, v in probe_attention.items()} if total_att else {}
            
            # Class mode logic (matching shared classroom_analysis package)
            if student_count < 5:
                mode = "Break"
            elif max_students > 0 and student_count < (max_students * 0.66):
                mode = "Transition/Sparse"
            else:
                mvmt = probe_activities.get('walking', 0) + probe_activities.get('talking', 0)
                if total_act > 0 and mvmt > (total_act * 0.3):
                    mode = "Interactive"
                else:
                    mode = "Lecture"
            
            # Real world time calculator (matching original)
            real_time_str = "Unknown"
            if video_metadata["base_datetime"] is not None:
                real_time_obj = video_metadata["base_datetime"] + timedelta(seconds=probe['start_time'])
                real_time_str = real_time_obj.strftime("%I:%M:%S %p")
            
            final_hourly_report.append({
                "time_slice": f"Probe {probe['probe_index']}",
                "video_timestamp_sec": round(probe['start_time'], 1),
                "real_world_time": real_time_str,
                "student_count_corrected": student_count,
                "avg_engagement": round(avg_score, 2),
                "class_mode": mode,
                "activity_distribution": act_dist,
                "attention_distribution": att_dist
            })
        
        # 1. Class start time: first probe where class is in session (not Break)
        class_start_time_sec = None
        class_start_real_time = "Unknown"
        for entry in final_hourly_report:
            if entry["class_mode"] != "Break" and entry["student_count_corrected"] >= 5:
                class_start_time_sec = entry["video_timestamp_sec"]
                class_start_real_time = entry["real_world_time"]
                break
        
        # 2. Event duration: merge consecutive probes with same mode into events
        events = []
        duration_by_type = {"Lecture": 0, "Interactive": 0, "Transition/Sparse": 0, "Break": 0}
        
        if final_hourly_report:
            current_event = {
                "type": final_hourly_report[0]["class_mode"],
                "start_time_sec": final_hourly_report[0]["video_timestamp_sec"],
                "start_real_time": final_hourly_report[0]["real_world_time"],
                "probe_indices": [final_hourly_report[0]["time_slice"]]
            }
            
            for entry in final_hourly_report[1:]:
                probe_start = entry["video_timestamp_sec"]
                probe_end = probe_start + PROBE_INTERVAL_SEC

                if entry["class_mode"] == current_event["type"]:
                    current_event["probe_indices"].append(entry["time_slice"])
                else:
                    # Close current event. Each probe represents PROBE_INTERVAL_SEC of
                    # wall-clock time (the sampling cadence), not PROBE_DURATION_SEC
                    # which is just the size of the sample window inside each probe.
                    current_event["end_time_sec"] = current_event["start_time_sec"] + (
                        len(current_event["probe_indices"]) * PROBE_INTERVAL_SEC
                    )
                    current_event["duration_sec"] = len(current_event["probe_indices"]) * PROBE_INTERVAL_SEC
                    if video_metadata["base_datetime"] is not None:
                        end_dt = video_metadata["base_datetime"] + timedelta(seconds=current_event["end_time_sec"])
                        current_event["end_real_time"] = end_dt.strftime("%I:%M:%S %p")
                    else:
                        current_event["end_real_time"] = "Unknown"
                    events.append(current_event)
                    duration_by_type[current_event["type"]] = duration_by_type.get(
                        current_event["type"], 0
                    ) + current_event["duration_sec"]
                    
                    # Start new event
                    current_event = {
                        "type": entry["class_mode"],
                        "start_time_sec": probe_start,
                        "start_real_time": entry["real_world_time"],
                        "probe_indices": [entry["time_slice"]]
                    }
            
            # Close last event
            current_event["end_time_sec"] = current_event["start_time_sec"] + (
                len(current_event["probe_indices"]) * PROBE_INTERVAL_SEC
            )
            current_event["duration_sec"] = len(current_event["probe_indices"]) * PROBE_INTERVAL_SEC
            if video_metadata["base_datetime"] is not None:
                end_dt = video_metadata["base_datetime"] + timedelta(seconds=current_event["end_time_sec"])
                current_event["end_real_time"] = end_dt.strftime("%I:%M:%S %p")
            else:
                current_event["end_real_time"] = "Unknown"
            events.append(current_event)
            duration_by_type[current_event["type"]] = duration_by_type.get(
                current_event["type"], 0
            ) + current_event["duration_sec"]
        
        # Save report (matching original structure)
        report_path = os.path.join(self.output_dir, "class_dynamics_report.json")
        report_data = {
            # Absolute path keeps every processed lecture traceable even when the
            # input folder moves or the report is copied across machines.
            "video_path": os.path.abspath(self.video_path),
            "classroom": video_metadata["classroom"],
            "recording_date": video_metadata["base_datetime_str"],
            "report_type": "Corrected (Stitched)",
            "baseline_max_students": max_students,
            "class_start_time": {
                "video_timestamp_sec": class_start_time_sec,
                "real_world_time": class_start_real_time
            },
            "event_duration_summary": {
                "Lecture_sec": round(duration_by_type.get("Lecture", 0), 1),
                "Interactive_sec": round(duration_by_type.get("Interactive", 0), 1),
                "TransitionSparse_sec": round(duration_by_type.get("Transition/Sparse", 0), 1),
                "Break_sec": round(duration_by_type.get("Break", 0), 1)
            },
            "events": [
                {
                    "type": e["type"],
                    "start_time_sec": e["start_time_sec"],
                    "end_time_sec": e["end_time_sec"],
                    "duration_sec": e["duration_sec"],
                    "start_real_time": e["start_real_time"],
                    "end_real_time": e["end_real_time"]
                }
                for e in events
            ],
            "hourly_probes": final_hourly_report
        }
        with open(report_path, 'w') as f:
            json.dump(report_data, f, indent=4)

        # Generate grouped management report (new shared format).
        # Pass PROBE_INTERVAL_SEC so time-window end times advance by the wall-clock
        # cadence (e.g. 30 min) rather than the 5-min sample window inside each probe.
        try:
            mgmt_path = os.path.join(self.output_dir, "management_summary_report.json")
            generated_path = generate_management_summary(
                report_path, mgmt_path, probe_interval_sec=PROBE_INTERVAL_SEC
            )
            if generated_path:
                self._log(f"Management summary saved: {generated_path}", "success")
        except Exception as mgmt_err:
            self._log(f"Management summary generation failed: {mgmt_err}", "warning")
        
        self._log(f"Report saved: {report_path}", "success")
        self._progress(100, "Analysis complete!")
        return report_path
