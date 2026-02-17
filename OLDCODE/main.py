import cv2
import torch
import numpy as np
import json
import os
import pickle
import shutil
from datetime import datetime, timedelta
from tqdm import tqdm
from ultralytics import YOLO
from scipy.spatial.distance import cosine

# InsightFace Imports
import insightface
from insightface.app import FaceAnalysis

# ================= CONFIGURATION =================
# ================= CONFIGURATION =================
RUN_MODE = "EVAL_DAY"  
CURRENT_DATE = "2026-02-05" 

# Local Relative Paths
VIDEO_PATH = "./Input_Data/5feb_merged.mp4" 
DB_PATH = "./Input_Data/master_database.pkl"

OUTPUT_VIDEO = "./Outputs/feb_05_output_final.mp4"
REPORT_JSON = "./Outputs/feb_05_attendance_report_final.json"
CROPS_DIR = "./Outputs/crops_5feb_final"
VERIFICATION_DIR = "./Outputs/Verification_Matches_final" 
# =================================================

# # Set to "BUILD_DB" for Day 1. Set to "EVAL_DAY" for Day 2 and beyond.
# RUN_MODE = "EVAL_DAY"  

# CURRENT_DATE = "2026-02-05" 
# # VIDEO_PATH = "/kaggle/input/datasets/your_folder/feb_05_motion.mp4" 
# VIDEO_PATH = "/kaggle/input/datasets/titikshabhavsar2/crossdaydata/5feb_merged.mp4" 


# OUTPUT_VIDEO = "/kaggle/working/feb_05_output_final.mp4"
# REPORT_JSON = "/kaggle/working/feb_05_attendance_report_final.json"
# CROPS_DIR = "/kaggle/working/crops_5feb_final"
# # DB_PATH = "/kaggle/working/master_database.pkl" 
# DB_PATH = "/kaggle/input/datasets/titikshabhavsar2/crossdaydata/master_database.pkl"

# # New Evaluation Specific Variables
# DAY_LABEL = "Day2" 
# VERIFICATION_DIR = "/kaggle/working/Verification_Matches_final" 
VISITOR_UPGRADE_DAYS = 3  # How many days a visitor must be seen to become a G_ID
# ArcFace Specific Thresholds
T_STRICT_MERGE = 0.55  
T_NEW_ID = 0.35        
T_RATIO_MARGIN = 0.10 
MIN_SAMPLES = 8      
MAX_EXEMPLARS = 5     
T_OUTLIER = 0.6       
# =================================================

class UniversalMultiRepSystem:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Initializing Universal System on {self.device} in {RUN_MODE} mode...")
        
        self.person_model = YOLO("yolov8n.pt") 
        self.app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider'])
        self.app.prepare(ctx_id=0, det_size=(640, 640))

        self.global_gallery = {} 
        self.track_vault = {}    
        self.next_global_id = 1
        self.next_visitor_id = 1
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

        if RUN_MODE == "BUILD_DB" and CURRENT_DATE not in self.operational_dates:
            self.operational_dates.append(CURRENT_DATE)
            self.operational_dates.sort()
            
        if not os.path.exists(CROPS_DIR): os.makedirs(CROPS_DIR)
        if RUN_MODE == "EVAL_DAY" and not os.path.exists(VERIFICATION_DIR): 
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
        # --- NEW: BIOMETRIC LOCK ---
        # Never add new face data to original Day 1 Employees during evaluation days
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
                t_id = int(t_id)
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

        # IDENTITY ASSIGNMENT & VISUAL VERIFICATION
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

                if t_data["global_id"] and RUN_MODE == "EVAL_DAY":
                    old_path = f"{CROPS_DIR}/track_{t_id}.jpg"
                    new_path = f"{VERIFICATION_DIR}/{t_data['global_id']}_track_{t_id}.jpg"
                    if os.path.exists(old_path) and not os.path.exists(new_path):
                        shutil.copy(old_path, new_path)

            # --- BOUNDING BOX COLOR LOGIC UPGRADE ---
            gid = t_data["global_id"]
            color = (0, 165, 255) # Default: Orange for scanning
            
            if gid:
                if gid.startswith("G_"):
                    color = (0, 255, 0) # Green for Returning Employees
                else:
                    color = (255, 0, 0) # Blue for Visitors
                
                label = f"ID: {gid}"
            else:
                progress = min(len(t_data['embeddings']), MIN_SAMPLES)
                label = f"Scanning: {progress}/{MIN_SAMPLES}"
                
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        return frame

    # --- HELPER FUNCTION: DURATION CALCULATOR ---
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
            # Find anyone with a Visitor tag
            if "_V_" in g_id: 
                days_present = len(g_data.get("attendance", {}))
                
                # If they hit the threshold, mark them for an upgrade!
                if days_present >= VISITOR_UPGRADE_DAYS:
                    visitors_to_upgrade.append(g_id)
        
        # Perform the actual upgrades
        for old_vid in visitors_to_upgrade:
            new_gid = f"G_{self.next_global_id:03d}"
            self.next_global_id += 1
            
            # Transfer all their face math and historical attendance to the new ID
            self.global_gallery[new_gid] = self.global_gallery.pop(old_vid)
            print(f"*** SYSTEM UPGRADE: {old_vid} has attended {VISITOR_UPGRADE_DAYS} days and is now Permanent ID: {new_gid} ***")

    # --- UPGRADED JSON REPORT GENERATOR ---
    def generate_report(self):
        report = {
            "Session": {
                "date": CURRENT_DATE,
                "camera": "Cam_01",
                "source_video": os.path.basename(VIDEO_PATH),
                "duration": "00:00:00" 
            },
            "Counts": {
                "unique_people": 0,
                "returning": 0,
                "visitors": 0
                # "fragments": 0
            },
            "People": []
        }
        
        current_dt = datetime.strptime(CURRENT_DATE, "%Y-%m-%d")
        latest_exit_time = "00:00:00"
        
        # 1. Process Valid IDs (Returning & Visitors)
        for g_id, g_data in self.global_gallery.items():
            attendance = g_data.get("attendance", {})
            if CURRENT_DATE not in attendance: continue 
                
            entry_time = attendance[CURRENT_DATE]["entry"]
            exit_time = attendance[CURRENT_DATE]["exit"]
            duration = self.calculate_duration(entry_time, exit_time)
            
            # Track latest exit for session duration
            if exit_time > latest_exit_time: latest_exit_time = exit_time
            
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
                # person_dict["confidence"] = "medium"
                
                report["Counts"]["visitors"] += 1
            else:
                person_dict["type"] = "returning"
                # person_dict["confidence"] = "high"
                
                past_dates = [d for d in attendance.keys() if d < CURRENT_DATE]
                past_dates.sort(reverse=True)
                person_dict["last_present_date"] = past_dates[0] if past_dates else None
                
                present_last_7 = 0
                for i in range(1, 8):
                    check_date = (current_dt - timedelta(days=i)).strftime("%Y-%m-%d")
                    if check_date in attendance: present_last_7 += 1
                person_dict["present_last_7_days"] = present_last_7
                
                report["Counts"]["returning"] += 1

            report["People"].append(person_dict)

        # # 2. Process Fragments (Temporary)
        # if RUN_MODE == "EVAL_DAY":
        #     temp_id_counter = 1
        #     for t_id, t_data in self.track_vault.items():
        #         if t_data["global_id"] is None and t_data["frames"] > 90:
        #             entry_time = t_data["first_seen"]
        #             exit_time = t_data["last_seen"]
        #             duration = self.calculate_duration(entry_time, exit_time)
                    
        #             if exit_time > latest_exit_time: latest_exit_time = exit_time
                    
        #             frag_dict = {
        #                 "id": f"Frag_{temp_id_counter:03d}",
        #                 "type": "fragment",
        #                 "entry": entry_time,
        #                 "exit": exit_time,
        #                 "duration_sec": duration,
        #                 "frames": t_data["frames"],
        #                 "reason": "no_frontal_face",
        #                 "last_present_date": None,
        #                 "present_last_7_days": 0,
        #                 #"confidence": "low"
        #             }
                    
        #             report["People"].append(frag_dict)
        #             report["Counts"]["fragments"] += 1
        #             temp_id_counter += 1
        
        # 3. Finalize Totals
        report["Counts"]["unique_people"] = report["Counts"]["returning"] + report["Counts"]["visitors"]
        report["Session"]["duration"] = latest_exit_time
            
        return report

def main():
    system = UniversalMultiRepSystem()
    cap = cv2.VideoCapture(VIDEO_PATH)
    
    if not cap.isOpened():
        print("Error opening video.")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w, h = int(cap.get(3)), int(cap.get(4))
    
    target_w, target_h = (1280, 720) if w > 1920 else (w, h)
    out = cv2.VideoWriter(OUTPUT_VIDEO, cv2.VideoWriter_fourcc(*'mp4v'), fps, (target_w, target_h))
    
    pbar = tqdm(total=total_frames, desc=f"Processing {CURRENT_DATE} in {RUN_MODE}")
    frame_idx = 0
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success: break
        
        if w != target_w: frame = cv2.resize(frame, (target_w, target_h))
        
        timestamp = str(datetime.fromtimestamp(frame_idx / fps).strftime('%H:%M:%S'))
        annotated_frame = system.process_frame(frame, timestamp)
        out.write(annotated_frame)
        
        frame_idx += 1
        pbar.update(1)

    pbar.close()
    cap.release()
    out.release()

    # --- TRIGGER THE VISITOR UPGRADE SWEEP ---
    system.upgrade_eligible_visitors()
    
    # --- UPGRADED DATABASE SAVING LOGIC ---
    db_data = {
        "gallery": system.global_gallery,
        "operational_dates": system.operational_dates
    }
    
    if RUN_MODE == "BUILD_DB":
        save_path = DB_PATH
        print(f"\nSaved Baseline Master Database to {save_path}.")
    # else:
    #     # Save to the working directory to protect the original Kaggle input file
    #     save_path = "/kaggle/working/updated_master_database.pkl"
    #     print(f"\nEVAL_DAY Complete.")
    #     print(f"-> Saved the updated attendance ledger and new visitors to {save_path}.")
    #     print(f"-> Your original baseline at {DB_PATH} remains completely untouched.")
    else:
        # Save to the local Outputs folder
        save_path = "./Outputs/updated_master_database.pkl"
        print(f"\nEVAL_DAY Complete.")
        print(f"-> Saved the updated attendance ledger and new visitors to {save_path}.")
        
    with open(save_path, 'wb') as f:
        pickle.dump(db_data, f)
    
    report_data = system.generate_report()
    with open(REPORT_JSON, "w") as f:
        json.dump(report_data, f, indent=4)
        
    print("\n=== FINAL DAILY SUMMARY ===")
    print(json.dumps(report_data["Counts"], indent=4))

if __name__ == "__main__":
    main()