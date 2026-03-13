import cv2
import torch
import numpy as np
import json
import os
import pickle
import shutil
import re
import easyocr  # <-- NEW IMPORT
from datetime import datetime, timedelta
from tqdm import tqdm
from ultralytics import YOLO
from scipy.spatial.distance import cosine

# InsightFace Imports
import insightface
from insightface.app import FaceAnalysis

# ================= CONFIGURATION =================

# RUN_MODE = "BUILD_DAY"  

# CURRENT_DATE = "2026-02-16" 
# VIDEO_PATH = "/kaggle/input/datasets/titikshabhavsar2/engagement/16feb_1.mp4" 

# OUTPUT_VIDEO = "/kaggle/working/feb_16_output_final.mp4"
# REPORT_JSON = "/kaggle/working/feb_16_attendance_report_final.json"
# CROPS_DIR = "/kaggle/working/crops_16feb_final"
# DB_PATH = "/kaggle/input/datasets/titikshabhavsar2/crossdaydata/master_database.pkl"
# STUDENT_DB_PATH = "/kaggle/input/datasets/titikshabhavsar2/crossdaydata/4batches_student_embeddings.pkl"
# # --- OCR TIMESTAMPS ---
# # Coordinates: (x, y, w, h)
# TIMESTAMP_COORDS = (0, 15, 600, 90)  

# # 2. OCR INTERVAL: Run OCR once every 30 frames (approx. 1 second) 
# # instead of every single frame.
# OCR_INTERVAL = 30

# DAY_LABEL = "Day1" 
# VERIFICATION_DIR = "/kaggle/working/16feb_Verification_Matches_final" 
# VISITOR_UPGRADE_DAYS = 3  

# # ArcFace Specific Thresholds
# T_STRICT_MERGE = 0.55  
# T_NEW_ID = 0.35        
# T_RATIO_MARGIN = 0.10 
# MIN_SAMPLES = 8       
# MAX_EXEMPLARS = 5     
# T_OUTLIER = 0.6     
# T_MATCH_STUDENT = 0.45
# ================= CONFIGURATION =================

RUN_MODE = "BUILD_DB"  # Fixed typo

CURRENT_DATE = "2026-02-16" 
# VIDEO_PATH = "/kaggle/input/datasets/titikshabhavsar2/engagement/16feb_1.mp4" 
# VIDEO_PATH = "/kaggle/input/datasets/titikshabhavsar2/engagement/16feb_1.mp4" 

# --- NEW PLAYLIST FEATURE ---
VIDEO_PATHS = [
    "/kaggle/input/datasets/titikshabhavsar2/engagement/16feb_1.mp4",
    "/kaggle/input/datasets/titikshabhavsar2/engagement/16feb_2.mp4" # Add your second part here!
]

OUTPUT_VIDEO = "/kaggle/working/feb_16_output_final.mp4"
REPORT_JSON = "/kaggle/working/feb_16_attendance_report_final.json"
CROPS_DIR = "/kaggle/working/crops_16feb_final"

# START FRESH: Point this to the working directory so it creates a brand new base file
DB_PATH = "/kaggle/working/16feb_master_database_base.pkl" 
# STUDENT_DB_PATH = "/kaggle/input/datasets/titikshabhavsar2/crossdaydata/4batches_student_embeddings.pkl"
# STUDENT_DB_PATH = "/kaggle/working/pliswork_4batch_master_db.pkl"
STUDENT_DB_PATH = "/kaggle/input/datasets/titikshabhavsar2/crossdaydata/pliswork_4batch_master_db.pkl"

# --- OCR TIMESTAMPS ---
TIMESTAMP_COORDS = (0, 15, 600, 90)  
OCR_INTERVAL = 30

DAY_LABEL = "Day1" 
VERIFICATION_DIR = "/kaggle/working/16feb_Verification_Matches_final" 
VISITOR_UPGRADE_DAYS = 3  

# ArcFace Specific Thresholds
T_STRICT_MERGE = 0.55  
T_NEW_ID = 0.35        
T_RATIO_MARGIN = 0.10 
MIN_SAMPLES = 8       
MAX_EXEMPLARS = 5     
T_OUTLIER = 0.6     
T_MATCH_STUDENT = 0.40  
# Lowered to bridge the CCTV angle gap
# T_MATCH_STUDENT = 0.30 # Lowered to fix the Domain Gap!
# =================================================
# =================================================

# --- HELPER: PURE OCR EXTRACTOR ---
def get_ocr_timestamp(frame, reader):
    """
    Extracts the timestamp from a frame using EasyOCR.
    Returns a string like '09:24:18' or None if it fails.
    """
    x, y, w, h = TIMESTAMP_COORDS
    
    # Safety bounds check
    if y+h > frame.shape[0] or x+w > frame.shape[1]:
        return None

    roi = frame[y:y+h, x:x+w]

    # Preprocessing (CLAHE + Upscale) for CCTV noise
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray_large = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    contrast_img = clahe.apply(gray_large)
    blurred = cv2.GaussianBlur(contrast_img, (3, 3), 0)
    kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
    final_img = cv2.filter2D(blurred, -1, kernel)

    # Run OCR
    allowlist = '0123456789:- MonTueWedThuFriSatSun'
    results = reader.readtext(final_img, allowlist=allowlist, detail=0)
    text = " ".join(results)

    # Extract exactly HH:MM:SS
    match = re.search(r'(\d{2}:\d{2}:\d{2})', text)
    if match:
        return match.group(1)
    return None

# =================================================

class UniversalMultiRepSystem:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Initializing Universal System on {self.device} in {RUN_MODE} mode...")
        
        self.person_model = YOLO("yolov8n.pt") 
        self.app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider'])
        self.app.prepare(ctx_id=0, det_size=(640, 640))

        # self.global_gallery = {} 
        # self.track_vault = {}    
        # self.next_global_id = 1
        # self.next_visitor_id = 1
        # self.operational_dates = []
        self.global_gallery = {} 
        self.track_vault = {}    
        self.next_global_id = 1
        self.next_visitor_id = 1
        self.track_offset = 0  # <--- YOU MUST ADD THIS LINE
        self.operational_dates = []
        
        if os.path.exists(DB_PATH):
            print(f"Loading Master Database from {DB_PATH}...")
            with open(DB_PATH, 'rb') as f:
                db_data = pickle.load(f)
                self.global_gallery = db_data.get("gallery", {})
                self.operational_dates = db_data.get("operational_dates", [])
            
            g_ids = [k for k in self.global_gallery.keys() if k.startswith('G_')]
            if g_ids: self.next_global_id = len(g_ids) + 1
            print(f"Loaded {len(self.global_gallery)} total identities.")
        else:
            print("No database found. Starting fresh.")
        # --- NEW: LOAD STUDENT DATABASE ---
        self.student_db = {}
        if os.path.exists(STUDENT_DB_PATH):
            with open(STUDENT_DB_PATH, 'rb') as f:
                self.student_db = pickle.load(f)
            print(f"Loaded Student DB: {len(self.student_db)} enrolled faces.")
        else:
            print("⚠️ WARNING: Student DB not found at path!")

        if RUN_MODE == "BUILD_DB" and CURRENT_DATE not in self.operational_dates:
            self.operational_dates.append(CURRENT_DATE)
            self.operational_dates.sort()
            
        if not os.path.exists(CROPS_DIR): os.makedirs(CROPS_DIR)
        # if RUN_MODE == "EVAL_DAY" and not os.path.exists(VERIFICATION_DIR): 
        #     os.makedirs(VERIFICATION_DIR)
        if not os.path.exists(VERIFICATION_DIR): 
            os.makedirs(VERIFICATION_DIR)

    def match_face_to_body(self, face_bbox, person_tracks):
        fx1, fy1, fx2, fy2 = face_bbox
        face_area = max(0, fx2 - fx1) * max(0, fy2 - fy1)
        face_cx = (fx1 + fx2) / 2
        face_cy = (fy1 + fy2) / 2
        
        best_match_id = None
        min_center_dist = float('inf') 
        
        for pt in person_tracks:
            px1, py1, px2, py2 = pt['bbox']
            t_id = pt['track_id']
            
            if face_cy > py1 + (py2 - py1) * 0.5: continue
                
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

    def find_match_with_margin(self, track_emb, active_gids):
        scores = []
        for g_id, data in self.global_gallery.items():
            if g_id in active_gids: continue 
            best_sim_for_person = max([1 - cosine(track_emb, ex) for ex in data["exemplars"]])
            scores.append((g_id, best_sim_for_person))
        
        if not scores: return None, 0
        scores.sort(key=lambda x: x[1], reverse=True)
        best_id, best_sim = scores[0]
        
        if len(scores) > 1:
            second_sim = scores[1][1]
            if (best_sim - second_sim) < T_RATIO_MARGIN:
                return None, best_sim 
        
        if best_sim > T_STRICT_MERGE: return best_id, best_sim
        return None, best_sim

    def update_exemplars(self, g_id, new_emb):
        if RUN_MODE == "EVAL_DAY" and str(g_id).startswith("G_"):
            return 
            
        gallery = self.global_gallery[g_id]
        is_diverse = True
        for ex in gallery["exemplars"]:
            if (1 - cosine(new_emb, ex)) > 0.60: 
                is_diverse = False
                break
        if is_diverse:
            if len(gallery["exemplars"]) >= MAX_EXEMPLARS: gallery["exemplars"].pop(0)
            gallery["exemplars"].append(new_emb)

    def log_attendance(self, g_id, timestamp):
        g_data = self.global_gallery[g_id]
        if "join_date" not in g_data: g_data["join_date"] = CURRENT_DATE
        if "attendance" not in g_data: g_data["attendance"] = {}
            
        if CURRENT_DATE not in g_data["attendance"]:
            g_data["attendance"][CURRENT_DATE] = {"entry": timestamp, "exit": timestamp}
        else:
            g_data["attendance"][CURRENT_DATE]["exit"] = timestamp

    def process_frame(self, frame, timestamp):
        results = self.person_model.track(frame, persist=True, classes=[0], tracker="botsort.yaml", verbose=False)
        person_tracks = []
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().numpy()
            
            for box, t_id in zip(boxes, track_ids):
                # t_id = int(t_id)
                t_id = int(t_id) + self.track_offset  # <--- YOU MUST ADD THE OFFSET HERE
                person_tracks.append({'track_id': t_id, 'bbox': box})
                if t_id not in self.track_vault:
                    self.track_vault[t_id] = {
                        "embeddings": [], "global_id": None, 
                        "first_seen": timestamp, "last_seen": timestamp, "frames": 0, "bbox": box
                    }
                self.track_vault[t_id]["last_seen"] = timestamp
                self.track_vault[t_id]["frames"] += 1
                self.track_vault[t_id]["bbox"] = box

                gid = self.track_vault[t_id]["global_id"]
                if gid: self.log_attendance(gid, timestamp)

        faces = self.app.get(frame)
        assigned_tracks = set() 
        
        for face in faces:
            face_width = face.bbox[2] - face.bbox[0]
            if face.det_score < 0.75 or face_width < 40: continue

            if face.kps is not None:
                l_eye, r_eye, nose = face.kps[0], face.kps[1], face.kps[2]
                eye_dist = np.linalg.norm(l_eye - r_eye)
                if face_width > 0 and (eye_dist / face_width) < 0.35: continue 
                eye_center_x = (l_eye[0] + r_eye[0]) / 2
                nose_offset = abs(nose[0] - eye_center_x)
                if nose_offset > (eye_dist * 0.5): continue

            matched_t_id = self.match_face_to_body(face.bbox, person_tracks)
            
            if matched_t_id is not None and matched_t_id not in assigned_tracks:
                assigned_tracks.add(matched_t_id) 
                emb = face.embedding
                
                should_add = True
                if len(self.track_vault[matched_t_id]["embeddings"]) > 3:
                    track_avg = np.mean(self.track_vault[matched_t_id]["embeddings"], axis=0)
                    track_avg = track_avg / np.linalg.norm(track_avg)
                    if cosine(emb, track_avg) > T_OUTLIER: should_add = False
                
                if should_add:
                    self.track_vault[matched_t_id]["embeddings"].append(emb)
                    gid = self.track_vault[matched_t_id]["global_id"]
                    if gid: self.update_exemplars(gid, emb)
                    
                    if len(self.track_vault[matched_t_id]["embeddings"]) == 1:
                        fx1, fy1, fx2, fy2 = map(int, face.bbox)
                        face_img = frame[max(0, fy1):fy2, max(0, fx1):fx2]
                        if face_img.size > 0:
                            cv2.imwrite(f"{CROPS_DIR}/track_{matched_t_id}.jpg", face_img)

        active_gids_in_frame = set(
            self.track_vault[pt['track_id']]["global_id"] 
            for pt in person_tracks 
            if self.track_vault[pt['track_id']]["global_id"] is not None
        )

        for pt in person_tracks:
            t_id = pt['track_id']
            t_data = self.track_vault[t_id]
            x1, y1, x2, y2 = map(int, pt['bbox'])
            
            if t_data["global_id"] is None and len(t_data["embeddings"]) >= MIN_SAMPLES:
                track_centroid = np.mean(t_data["embeddings"], axis=0)
                track_centroid = track_centroid / np.linalg.norm(track_centroid)
                
                best_id, best_sim = self.find_match_with_margin(track_centroid, active_gids_in_frame)

                if best_id:
                    t_data["global_id"] = best_id
                    active_gids_in_frame.add(best_id)
                    self.log_attendance(best_id, timestamp)
                elif best_sim < T_NEW_ID:
                    if RUN_MODE == "BUILD_DB":
                        new_gid = f"G_{self.next_global_id:03d}"
                        self.next_global_id += 1
                    else:
                        new_gid = f"{DAY_LABEL}_V_{self.next_visitor_id:03d}"
                        self.next_visitor_id += 1
                        
                    self.global_gallery[new_gid] = {"exemplars": [track_centroid]}
                    t_data["global_id"] = new_gid
                    active_gids_in_frame.add(new_gid)
                    self.log_attendance(new_gid, timestamp)
                    
                # if t_data["global_id"] and RUN_MODE == "EVAL_DAY":
                if t_data["global_id"]:
                    old_path = f"{CROPS_DIR}/track_{t_id}.jpg"
                    new_path = f"{VERIFICATION_DIR}/{t_data['global_id']}_track_{t_id}.jpg"
                    if os.path.exists(old_path) and not os.path.exists(new_path):
                        shutil.copy(old_path, new_path)

            gid = t_data["global_id"]
            color = (0, 165, 255) 
            
            if gid:
                if gid.startswith("G_"):
                    color = (0, 255, 0) 
                else:
                    color = (255, 0, 0) 
                
                label = f"ID: {gid}"
            else:
                progress = min(len(t_data['embeddings']), MIN_SAMPLES)
                label = f"Scanning: {progress}/{MIN_SAMPLES}"
                
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            # RENDER THE OCR TIMESTAMP LIVE ON SCREEN
            cv2.putText(frame, f"Live: {timestamp}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        return frame

    def calculate_duration(self, entry_str, exit_str):
        fmt = '%H:%M:%S'
        try:
            td = datetime.strptime(exit_str, fmt) - datetime.strptime(entry_str, fmt)
            return int(td.total_seconds())
        except:
            return 0

    def upgrade_eligible_visitors(self):
        visitors_to_upgrade = []
        for g_id, g_data in self.global_gallery.items():
            if "_V_" in g_id: 
                days_present = len(g_data.get("attendance", {}))
                if days_present >= VISITOR_UPGRADE_DAYS:
                    visitors_to_upgrade.append(g_id)
        
        for old_vid in visitors_to_upgrade:
            new_gid = f"G_{self.next_global_id:03d}"
            self.next_global_id += 1
            self.global_gallery[new_gid] = self.global_gallery.pop(old_vid)
            print(f"*** SYSTEM UPGRADE: {old_vid} has attended {VISITOR_UPGRADE_DAYS} days and is now Permanent ID: {new_gid} ***")

    def map_identities_to_students(self):
        print("\nMapping Global IDs to Enrolled Students (Max-to-Max)...")
        
        for g_id, g_data in self.global_gallery.items():
            g_data["engagement_id"] = None
            g_data["batch"] = None
            
            if not g_data.get("exemplars"): continue
            
            best_eng_id = None
            best_sim = -1
            
            # --- MAX-TO-MAX LOGIC ---
            # Compare EVERY CCTV frame against EVERY ID photo for that student
            for eng_id, stu_data in self.student_db.items():
                for cctv_emb in g_data["exemplars"]:
                    for stu_emb in stu_data["exemplars"]:
                        sim = 1 - cosine(cctv_emb, stu_emb)
                        if sim > best_sim:
                            best_sim = sim
                            best_eng_id = eng_id
                            
            # If the absolute best frame crosses the threshold, map them!
            if best_sim > T_MATCH_STUDENT and best_eng_id is not None:
                g_data["engagement_id"] = best_eng_id
                g_data["batch"] = self.student_db[best_eng_id]["batch"]
                g_data["confidence"] = float(best_sim)  # <--- ADD THIS
                print(f"✅ Mapped {g_id} -> Student {best_eng_id} (Batch: {g_data['batch']}) | Peak Confidence: {best_sim:.3f}")
            
    # def map_identities_to_students(self):
    #     print("\nMapping Global IDs to Enrolled Students...")
        
    #     for g_id, g_data in self.global_gallery.items():
    #         g_data["engagement_id"] = None
    #         g_data["batch"] = None
            
    #         if not g_data.get("exemplars"): continue
                
    #         g_centroid = np.mean(g_data["exemplars"], axis=0)
    #         g_centroid = g_centroid / np.linalg.norm(g_centroid)
            
    #         best_eng_id = None
    #         best_sim = -1
            
    #         for eng_id, stu_data in self.student_db.items():
    #             sims = [1 - cosine(g_centroid, ex) for ex in stu_data["exemplars"]]
    #             max_sim = max(sims) if sims else -1
                
    #             if max_sim > best_sim:
    #                 best_sim = max_sim
    #                 best_eng_id = eng_id
                    
    #         if best_sim > T_MATCH_STUDENT and best_eng_id is not None:
    #             g_data["engagement_id"] = best_eng_id
    #             g_data["batch"] = self.student_db[best_eng_id]["batch"]
    #             print(f"✅ Mapped {g_id} -> Student {best_eng_id} (Batch: {g_data['batch']}) | Confidence: {best_sim:.2f}")

    def generate_report(self):
        report = {
            # "Session": {
            #     "date": CURRENT_DATE,
            #     "camera": "Cam_01",
            #     "source_video": os.path.basename(VIDEO_PATH),
            #     "duration": "00:00:00" 
            # },
            "Session": {
                "date": CURRENT_DATE,
                "camera": "Cam_01",
                "source_video": ", ".join([os.path.basename(p) for p in VIDEO_PATHS]), # <--- FIXED FOR PLAYLISTS
                "duration": "00:00:00" 
            },
            "Counts": {
                "unique_people": 0,
                "returning": 0,
                "visitors": 0,
                "identified_students": 0  
            },
            "People": []
        }
        
        current_dt = datetime.strptime(CURRENT_DATE, "%Y-%m-%d")
        latest_exit_time = "00:00:00"
        
        for g_id, g_data in self.global_gallery.items():
            attendance = g_data.get("attendance", {})
            if CURRENT_DATE not in attendance: continue 
                
            entry_time = attendance[CURRENT_DATE]["entry"]
            exit_time = attendance[CURRENT_DATE]["exit"]
            duration = self.calculate_duration(entry_time, exit_time)
            
            if exit_time > latest_exit_time: latest_exit_time = exit_time

            is_new_walk_in = g_data.get("join_date") == CURRENT_DATE
            
            person_dict = {
                "id": g_id,
                "engagement_id": g_data.get("engagement_id"),
                "batch": g_data.get("batch"),
                "entry": entry_time,
                "exit": exit_time,
                "duration_sec": duration,
                "confidence_score": g_data.get("confidence", 0.0), # <--- ADD THIS
            }
            
            if g_data.get("engagement_id") is not None:
                person_dict["type"] = "enrolled_student"
                # Initialize these so the JSON doesn't crash if they are requested later
                if "identified_students" not in report["Counts"]: report["Counts"]["identified_students"] = 0
                report["Counts"]["identified_students"] += 1
            elif is_new_walk_in:
                person_dict["type"] = "visitor"
                report["Counts"]["visitors"] += 1
            else:
                person_dict["type"] = "returning_employee"
                report["Counts"]["returning"] += 1
            
            # is_new_walk_in = g_data.get("join_date") == CURRENT_DATE
            
            # person_dict = {
            #     "id": g_id,
            #     "entry": entry_time,
            #     "exit": exit_time,
            #     "duration_sec": duration
            # }
            
            # if is_new_walk_in:
            #     person_dict["type"] = "visitor"
            #     person_dict["last_present_date"] = None
            #     person_dict["present_last_7_days"] = 0
            #     report["Counts"]["visitors"] += 1
            # else:
            #     person_dict["type"] = "returning"
            #     past_dates = [d for d in attendance.keys() if d < CURRENT_DATE]
            #     past_dates.sort(reverse=True)
            #     person_dict["last_present_date"] = past_dates[0] if past_dates else None
                
            #     present_last_7 = 0
            #     for i in range(1, 8):
            #         check_date = (current_dt - timedelta(days=i)).strftime("%Y-%m-%d")
            #         if check_date in attendance: present_last_7 += 1
            #     person_dict["present_last_7_days"] = present_last_7
                
            #     report["Counts"]["returning"] += 1

            report["People"].append(person_dict)

        # report["Counts"]["unique_people"] = report["Counts"]["returning"] + report["Counts"]["visitors"]
        report["Counts"]["unique_people"] = (
            report["Counts"]["returning"] + 
            report["Counts"]["visitors"] + 
            report["Counts"].get("identified_students", 0)
        )
        report["Session"]["duration"] = latest_exit_time
            
        return report
def main():
    system = UniversalMultiRepSystem()
    
    print("Loading EasyOCR Model...")
    ocr_reader = easyocr.Reader(['en'], gpu=False)
    
    # --- THE PLAYLIST LOOP ---
    for vid_idx, current_video_path in enumerate(VIDEO_PATHS):
        print(f"\n🎬 Starting Video Part {vid_idx + 1} of {len(VIDEO_PATHS)}...")
        
        cap = cv2.VideoCapture(current_video_path)
        if not cap.isOpened():
            print(f"Error opening video: {current_video_path}")
            continue

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w, h = int(cap.get(3)), int(cap.get(4))
        
        target_w, target_h = (1280, 720) if w > 1920 else (w, h)
        
        # Give the output video a unique part number (e.g. feb_16_output_final_part1.mp4)
        part_suffix = f"_part{vid_idx + 1}.mp4"
        current_out_path = OUTPUT_VIDEO.replace(".mp4", part_suffix)
        
        out = cv2.VideoWriter(current_out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (target_w, target_h))
        
        pbar = tqdm(total=total_frames, desc=f"Processing Part {vid_idx + 1} in {RUN_MODE}")

        last_valid_timestamp = "00:00:00"
        frame_count = 0

        while cap.isOpened():
            success, frame = cap.read()
            if not success: break
            
            # # --- FRAME SKIP OPTIMIZATION (Uncommented for speed!) ---
            # if frame_count % 3 != 0: 
            #     frame_count += 1
            #     pbar.update(1)
            #     continue
            
            if w != target_w: frame = cv2.resize(frame, (target_w, target_h))

            if frame_count % OCR_INTERVAL == 0:
                ocr_result = get_ocr_timestamp(frame, ocr_reader)
                if ocr_result:
                    last_valid_timestamp = ocr_result
                    
            timestamp = last_valid_timestamp
            
            annotated_frame = system.process_frame(frame, timestamp)
            out.write(annotated_frame)
            
            frame_count += 1
            pbar.update(1)

        pbar.close()
        cap.release()
        out.release()
        
        if system.track_vault:
            system.track_offset = max(system.track_vault.keys())
    # --- END OF PLAYLIST: RUN THE FINAL CALCULATIONS ONCE ---
    print("\nAll video parts processed! Running final analytics...")
    
    system.map_identities_to_students()
    system.upgrade_eligible_visitors()
    
    db_data = {
        "gallery": system.global_gallery,
        "operational_dates": system.operational_dates
    }
    
    save_path = DB_PATH
    print(f"\nProcessing Complete.")
    print(f"-> Saved the Master Database to {save_path}.")
        
    with open(save_path, 'wb') as f:
        pickle.dump(db_data, f)
    
    report_data = system.generate_report()
    with open(REPORT_JSON, "w") as f:
        json.dump(report_data, f, indent=4)
        
    print("\n=== FINAL DAILY SUMMARY ===")
    print(json.dumps(report_data["Counts"], indent=4))

if __name__ == "__main__":
    main()
# def main():
#     system = UniversalMultiRepSystem()
    
#     # --- INITIALIZE EASYOCR ---
#     print("Loading EasyOCR Model...")
#     # ocr_reader = easyocr.Reader(['en'])
#     ocr_reader = easyocr.Reader(['en'], gpu=False)
    
#     cap = cv2.VideoCapture(VIDEO_PATH)
#     if not cap.isOpened():
#         print("Error opening video.")
#         return

#     fps = int(cap.get(cv2.CAP_PROP_FPS))
#     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     w, h = int(cap.get(3)), int(cap.get(4))
    
#     target_w, target_h = (1280, 720) if w > 1920 else (w, h)
#     out = cv2.VideoWriter(OUTPUT_VIDEO, cv2.VideoWriter_fourcc(*'mp4v'), fps, (target_w, target_h))
    
#     pbar = tqdm(total=total_frames, desc=f"Processing {CURRENT_DATE} in {RUN_MODE}")


#     last_valid_timestamp = "00:00:00"
#     frame_count = 0

#     while cap.isOpened():
#         success, frame = cap.read()
#         if not success: break
        
#         if w != target_w: frame = cv2.resize(frame, (target_w, target_h))

#         # --- OPTIMIZATION: SELECTIVE OCR PROCESSING ---
#         # Run OCR only once every 30 frames
#         if frame_count % OCR_INTERVAL == 0:
#             ocr_result = get_ocr_timestamp(frame, ocr_reader)
#             if ocr_result:
#                 last_valid_timestamp = ocr_result
                
#         # Use the "Sticky" timestamp for the current frame
#         timestamp = last_valid_timestamp
            
#         annotated_frame = system.process_frame(frame, timestamp)
#         out.write(annotated_frame)
        
#         frame_count += 1
#         pbar.update(1)
#     # last_valid_timestamp = "00:00:00"
#     # frame_count = 0

#     # while cap.isOpened():
#     #     success, frame = cap.read()
#     #     if not success: break

#     #     if frame_count % GLOBAL_FRAME_SKIP != 0:
#     #         frame_count += 1
#     #         pbar.update(1)
#     #         continue
        
#     #     if w != target_w: frame = cv2.resize(frame, (target_w, target_h))

#     #     # Run OCR only once every 30 frames (approx once per second)
#     #     if frame_count % OCR_INTERVAL == 0:
#     #         ocr_result = get_ocr_timestamp(frame, ocr_reader)
#     #         if ocr_result:
#     #             last_valid_timestamp = ocr_result
#     #     # Use the "Sticky" timestamp for the current frame
#     #     timestamp = last_valid_timestamp
        
#     #     # # --- NEW LOGIC: Pure Frame-by-Frame OCR ---
#     #     # ocr_result = get_ocr_timestamp(frame, ocr_reader)
    
        
#     #     if ocr_result:
#     #         timestamp = ocr_result
#     #         last_valid_timestamp = ocr_result
#     #     else:
#     #         # If blurry/unreadable, just hold the last known time
#     #         timestamp = last_valid_timestamp
            
#     #     annotated_frame = system.process_frame(frame, timestamp)
#     #     out.write(annotated_frame)
        
#     #     pbar.update(1)

#     pbar.close()
#     cap.release()
#     out.release()
    
#     # --- NEW: RUN THE MAPPER ---
#     system.map_identities_to_students()

#     system.upgrade_eligible_visitors()
    
#     db_data = {
#         "gallery": system.global_gallery,
#         "operational_dates": system.operational_dates
#     }
    
#     # Force saving to the working directory to avoid Kaggle read-only crashes
#     save_path = DB_PATH
#     # save_path = "/kaggle/working/master_database_base.pkl"
#     print(f"\nProcessing Complete.")
#     print(f"-> Saved the database to {save_path}.")
#     # if RUN_MODE == "BUILD_DB":
#     #     save_path = DB_PATH
#     #     print(f"\nSaved Baseline Master Database to {save_path}.")
#     # else:
#     #     save_path = "/kaggle/working/updated_master_database.pkl"
#     #     print(f"\nEVAL_DAY Complete.")
#     #     print(f"-> Saved the updated attendance ledger and new visitors to {save_path}.")
        
#     with open(save_path, 'wb') as f:
#         pickle.dump(db_data, f)
    
#     report_data = system.generate_report()
#     with open(REPORT_JSON, "w") as f:
#         json.dump(report_data, f, indent=4)
        
#     print("\n=== FINAL DAILY SUMMARY ===")
#     print(json.dumps(report_data["Counts"], indent=4))

# if __name__ == "__main__":
#     main()