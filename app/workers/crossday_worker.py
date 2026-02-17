"""
Attendance Worker Thread.
Runs the attendance pipeline in a background thread.
"""

import os
import sys
import traceback
from typing import Dict, Any, Optional, Callable

import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class CrossDayWorker(QThread):
    """Worker thread for attendance pipeline."""
    
    progress = pyqtSignal(int, str)  # percent, message
    log_message = pyqtSignal(str, str)  # message, level
    frame_ready = pyqtSignal(np.ndarray)  # for video preview
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
            
            # Create analyzer with callbacks
            analyzer = CrossDayAnalyzerWithCallbacks(
                video_path=self.video_path,
                output_dir=self.output_dir,
                db_path=self.db_path,
                config=self.config,
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
        
    def _emit_frame(self, frame: np.ndarray):
        """Emit frame for preview."""
        if self.preview_enabled:
            self.frame_ready.emit(frame)
            
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
        stop_check: Optional[Callable] = None
    ):
        self.video_path = video_path
        self.output_dir = output_dir
        self.db_path = db_path
        self.config = config
        self.progress_callback = progress_callback
        self.log_callback = log_callback
        self.frame_callback = frame_callback
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
        
    def analyze_video(self) -> str:
        """Run the attendance analysis with callbacks."""
        import cv2
        import torch
        import numpy as np
        import json
        import pickle
        import shutil
        from datetime import datetime, timedelta
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
        
        T_STRICT_MERGE = self.config.get("t_strict_merge", 0.55)
        T_NEW_ID = self.config.get("t_new_id", 0.35)
        T_RATIO_MARGIN = self.config.get("t_ratio_margin", 0.10)
        MIN_SAMPLES = self.config.get("min_samples", 8)
        MAX_EXEMPLARS = 5
        T_OUTLIER = 0.6
        VISITOR_UPGRADE_DAYS = self.config.get("visitor_upgrade_days", 3)
        
        # Device setup (GPU when available, else CPU - same logic on both)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self._log(f"Running on: {device}")
        self._log(f"Mode: {RUN_MODE}, Date: {CURRENT_DATE}")
        
        # Load models
        self._log("Loading person detection model...")
        person_model = YOLO("yolov8n.pt")
        
        # FaceAnalysis: load only if onnxruntime + InsightFace available
        if face_app == "pending":
            self._log("Loading face analysis model...")
            try:
                if device == 'cuda':
                    try:
                        face_app = FaceAnalysis(
                            name='buffalo_l',
                            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
                        )
                        face_app.prepare(ctx_id=0, det_size=(640, 640))
                    except Exception as e:
                        self._log(f"CUDA failed ({e}), falling back to CPU...", "warning")
                        face_app = None
                if face_app is None or face_app == "pending":
                    face_app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
                    face_app.prepare(ctx_id=-1, det_size=(640, 640))
                self._log("Face analysis model loaded", "success")
            except Exception as e:
                self._log(f"Face model failed ({e}), using simplified mode.", "warning")
                face_app = None
        if face_app is None:
            self._log("Running in simplified mode (track-only, no face matching)", "info")
        
        # Data structures
        global_gallery = {}
        track_vault = {}
        next_global_id = 1
        next_visitor_id = 1
        operational_dates = []
        
        # Load existing database (same as cross_day_code - no try/except)
        if os.path.exists(self.db_path):
            self._log(f"Loading database from {self.db_path}...")
            with open(self.db_path, 'rb') as f:
                db_data = pickle.load(f)
                global_gallery = db_data.get("gallery", {})
                operational_dates = db_data.get("operational_dates", [])
            
            g_ids = [k for k in global_gallery.keys() if k.startswith('G_')]
            if g_ids:
                next_global_id = len(g_ids) + 1
            self._log(f"Loaded {len(global_gallery)} identities", "success")
        else:
            self._log("No existing database found, starting fresh")
        
        if RUN_MODE == "BUILD_DB" and CURRENT_DATE not in operational_dates:
            operational_dates.append(CURRENT_DATE)
            operational_dates.sort()
        
        # Create output directories
        crops_dir = os.path.join(self.output_dir, f"crops_{CURRENT_DATE.replace('-', '')}")
        verification_dir = os.path.join(self.output_dir, "Verification_Matches")
        os.makedirs(crops_dir, exist_ok=True)
        if RUN_MODE == "EVAL_DAY":
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
        
        # Resize if needed
        target_w, target_h = (1280, 720) if w > 1920 else (w, h)
        
        # Output video
        output_video_path = os.path.join(self.output_dir, f"{CURRENT_DATE.replace('-', '_')}_output.mp4")
        out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (target_w, target_h))
        
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
                best_sim = max([1 - cosine(track_emb, ex) for ex in data["exemplars"]])
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
        
        # Process frames
        frame_idx = 0
        while cap.isOpened():
            if self._should_stop():
                break
            
            success, frame = cap.read()
            if not success:
                break
            
            if w != target_w:
                frame = cv2.resize(frame, (target_w, target_h))
            
            timestamp = datetime.fromtimestamp(frame_idx / fps).strftime('%H:%M:%S')
            
            # Progress
            progress = int(frame_idx / total_frames * 100)
            if frame_idx % 100 == 0:
                self._progress(progress, f"Processing frame {frame_idx}/{total_frames}")
            
            # Person tracking
            results = person_model.track(frame, persist=True, classes=[0], tracker="botsort.yaml", verbose=False)
            person_tracks = []
            
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
            
            # Face detection and embedding (skip if simplified mode)
            if face_app is not None:
                faces = face_app.get(frame)
            else:
                faces = []
            assigned_tracks = set()
            
            for face in faces:
                face_width = face.bbox[2] - face.bbox[0]
                if face.det_score < 0.75 or face_width < 40:
                    continue
                
                # Face pose quality checks (same as cross_day_code)
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
                        if gid:
                            update_exemplars(gid, emb)
                        
                        if len(track_vault[matched_t_id]["embeddings"]) == 1:
                            fx1, fy1, fx2, fy2 = map(int, face.bbox)
                            face_img = frame[max(0, fy1):fy2, max(0, fx1):fx2]
                            if face_img.size > 0:
                                cv2.imwrite(f"{crops_dir}/track_{matched_t_id}.jpg", face_img)
            
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
                
                # Simplified mode: assign ID after a few frames (no face embedding)
                if face_app is None and t_data["global_id"] is None and t_data["frames"] >= 5:
                    if RUN_MODE == "BUILD_DB":
                        new_gid = f"G_{next_global_id:03d}"
                        next_global_id += 1
                    else:
                        new_gid = f"{DAY_LABEL}_V_{next_visitor_id:03d}"
                        next_visitor_id += 1
                    global_gallery[new_gid] = {
                        "exemplars": [], "attendance": {}, "join_date": CURRENT_DATE
                    }
                    t_data["global_id"] = new_gid
                    log_attendance(new_gid, timestamp)
                
                if t_data["global_id"] is None and len(t_data["embeddings"]) >= MIN_SAMPLES:
                    track_centroid = np.mean(t_data["embeddings"], axis=0)
                    track_centroid = track_centroid / np.linalg.norm(track_centroid)
                    
                    best_id, best_sim = find_match_with_margin(track_centroid, active_gids_in_frame)
                    
                    if best_id:
                        t_data["global_id"] = best_id
                        active_gids_in_frame.add(best_id)
                        log_attendance(best_id, timestamp)
                    elif best_sim < T_NEW_ID:
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
                    
                    if t_data["global_id"] and RUN_MODE == "EVAL_DAY":
                        old_path = f"{crops_dir}/track_{t_id}.jpg"
                        new_path = f"{verification_dir}/{t_data['global_id']}_track_{t_id}.jpg"
                        if os.path.exists(old_path) and not os.path.exists(new_path):
                            shutil.copy(old_path, new_path)
                
                # Draw bounding box
                gid = t_data["global_id"]
                if gid:
                    if gid.startswith("G_"):
                        color = (0, 255, 0)  # Green for returning
                    else:
                        color = (255, 0, 0)  # Blue for visitors
                    label = f"ID: {gid}"
                else:
                    color = (0, 165, 255)  # Orange for scanning
                    if face_app is not None:
                        progress_count = min(len(t_data['embeddings']), MIN_SAMPLES)
                        label = f"Scanning: {progress_count}/{MIN_SAMPLES}"
                    else:
                        label = f"Scanning: {min(t_data['frames'], 5)}/5"
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            out.write(frame)
            
            # Send frame for preview (every 25th to reduce lag)
            if self.frame_callback and frame_idx % 25 == 0:
                self.frame_callback(frame.copy())
            
            frame_idx += 1
        
        cap.release()
        out.release()
        
        if self._should_stop():
            return ""
        
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
        self._progress(95, "Saving database...")
        
        db_data = {
            "gallery": global_gallery,
            "operational_dates": operational_dates
        }
        
        if RUN_MODE == "BUILD_DB":
            save_path = self.db_path if self.db_path else os.path.join(self.output_dir, "master_database.pkl")
        else:
            save_path = os.path.join(self.output_dir, "updated_master_database.pkl")
        
        with open(save_path, 'wb') as f:
            pickle.dump(db_data, f)
        self._log(f"Database saved: {save_path}", "success")
        
        # Generate report
        self._log("Generating report...")
        self._progress(98, "Generating report...")
        
        def calculate_duration(entry_str, exit_str):
            fmt = '%H:%M:%S'
            try:
                td = datetime.strptime(exit_str, fmt) - datetime.strptime(entry_str, fmt)
                return int(td.total_seconds())
            except:
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
                "returning": 0,
                "visitors": 0
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
                "entry": entry_time,
                "exit": exit_time,
                "duration_sec": duration
            }
            
            if is_new_walk_in:
                person_dict["type"] = "visitor"
                person_dict["last_present_date"] = None
                person_dict["present_last_7_days"] = 0
                report["Counts"]["visitors"] += 1
            else:
                person_dict["type"] = "returning"
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
        
        report["Counts"]["unique_people"] = report["Counts"]["returning"] + report["Counts"]["visitors"]
        report["Session"]["duration"] = latest_exit_time
        
        report_path = os.path.join(self.output_dir, f"{CURRENT_DATE.replace('-', '_')}_attendance_report.json")
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=4)
        
        self._log(f"Report saved: {report_path}", "success")
        self._log(f"Output video: {output_video_path}", "success")
        
        return report_path
