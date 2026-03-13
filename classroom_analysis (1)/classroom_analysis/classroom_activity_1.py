"""
Face Tracking V14 - FINAL INTEGRATED PIPELINE
- Logic: Sampling (5min/hr) -> Extraction -> Stitching -> Reporting
- Input: Video File
- Output: Corrected Class Dynamics Report
"""

import warnings
import logging
warnings.filterwarnings("ignore", category=FutureWarning)
logging.getLogger("boxmot").setLevel(logging.ERROR)

import cv2
import json
import numpy as np
import os
import sys
import torch
from collections import defaultdict, Counter
import math
from tqdm import tqdm
from pathlib import Path
from scipy.spatial.distance import cosine
import re
from datetime import datetime, timedelta


# AI Libraries
from ultralytics import YOLO

# --- IMPORT STITCHING LOGIC ---
try:
    from stitch_logic import perform_hierarchical_stitching
except ImportError:
    # Fallback if file missing (dev safety)
    print("⚠️ 'stitch_logic.py' not found. Creating merged logic internally.")
    def perform_hierarchical_stitching(json_path, **kwargs): return {}

try:
    from vlm_metadata import extract_camera_metadata_vlm
except ImportError:
    print("⚠️ 'vlm_metadata.py' not found.")
    def extract_camera_metadata_vlm(frame): return {"classroom": "Unknown", "base_datetime": None, "base_datetime_str": "Unknown"}

# --- RUNTIME ANCHOR MANAGER ---
class RuntimeAnchorManager:
    def __init__(self, similarity_thresh=0.70, distance_thresh=120): 
        self.seats = {} 
        self.next_seat_uid = 0
        self.active_mapping = {}
        self.sim_thresh = similarity_thresh
        self.dist_thresh = distance_thresh
        self.lock_frames = 10  
        self.potential_seats = defaultdict(lambda: {'count': 0, 'centroids': []})

    def get_corrected_id(self, raw_id, bbox, embedding, frame_idx):
        curr_centroid = [(bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2]
        if raw_id in self.active_mapping:
            stable_id = self.active_mapping[raw_id]
            self._update_seat_position(stable_id, curr_centroid, embedding, frame_idx)
            return stable_id

        best_seat_id = None
        min_dist = float('inf')
        for s_uid, seat in self.seats.items():
            dist = math.sqrt((curr_centroid[0]-seat['centroid'][0])**2 + (curr_centroid[1]-seat['centroid'][1])**2)
            if dist < self.dist_thresh:
                sim = 0.0
                if embedding is not None and seat['embedding'] is not None:
                    sim = 1.0 - cosine(embedding, seat['embedding'])
                time_gap = frame_idx - seat['last_seen']
                is_match = False
                if time_gap < 90 and dist < 80: is_match = True 
                elif sim > self.sim_thresh: is_match = True
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
        data = self.potential_seats[raw_id]
        data['count'] += 1
        data['centroids'].append(centroid)
        if data['count'] == self.lock_frames:
            avg_x = sum(c[0] for c in data['centroids']) / len(data['centroids'])
            avg_y = sum(c[1] for c in data['centroids']) / len(data['centroids'])
            for s in self.seats.values():
                d = math.sqrt((avg_x - s['centroid'][0])**2 + (avg_y - s['centroid'][1])**2)
                if d < 50: return 
            self.seats[self.next_seat_uid] = {
                'owner_id': raw_id,
                'centroid': [avg_x, avg_y],
                'embedding': embedding,
                'last_seen': frame_idx
            }
            self.next_seat_uid += 1

class FaceEngagementAnalyzer:
    def __init__(self, video_path, output_dir):
        self.video_path = video_path
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"⚙️ Running on: {self.device}")
        
        self.models = self.load_models()
        self.stitch_model = None 
        self.anchor_manager = RuntimeAnchorManager(similarity_thresh=0.75, distance_thresh=120)
        
        # Data Storage
        self.face_engagement_data = defaultdict(lambda: {
            'face_id': None, 'total_frames': 0, 'best_embedding': None,
            'start_centroid': None, 'last_centroid': None,
            'engagement_scores': [], 'first_seen': None, 'last_seen': None,
            'activities': Counter(), 'attention_levels': Counter()
        })
        
        # To store raw probe stats before merging
        self.raw_probe_data = [] 
        
        self.fps = None
        self.width = None
        self.height = None
        self.total_frames = None

    def load_models(self):
        print("🔄 Loading Detection Models...")
        models = {}
        try: base_path = sys._MEIPASS if getattr(sys, 'frozen', False) else os.path.dirname(os.path.abspath(__file__))
        except: base_path = os.getcwd()
        weights_dir = os.path.join(base_path, "Models")
        if not os.path.exists(weights_dir): weights_dir = base_path 
        for name, file in {'detection':'yolov8m.pt', 'pose':'yolov8n-pose.pt', 'face':'yolov8n-face.pt'}.items():
            path = os.path.join(weights_dir, file)
            load_name = path if os.path.exists(path) else file 
            try:
                m = YOLO(load_name)
                m.to(self.device)
                if hasattr(m.model, 'fuse'): m.model.fuse()
                models[name] = m
            except Exception as e: print(f"⚠️ Warning: Could not load {name} ({file}). Error: {e}")
        return models

    def get_embedding(self, frame, bbox):
        if self.stitch_model is None: return None
        x1, y1, x2, y2 = map(int, bbox)
        h, w, _ = frame.shape
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2-x1 < 10 or y2-y1 < 10: return None
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0: return None
        crop = cv2.resize(crop, (128, 256))
        crop = crop.transpose(2, 0, 1) 
        crop = np.ascontiguousarray(crop, dtype=np.float32)
        crop /= 255.0
        tensor = torch.from_numpy(crop).unsqueeze(0).to(self.device)
        try:
            with torch.no_grad():
                model_to_call = self.stitch_model
                if hasattr(model_to_call, 'model'): model_to_call = model_to_call.model
                elif hasattr(model_to_call, 'net'): model_to_call = model_to_call.net
                try: 
                    if next(model_to_call.parameters()).dtype == torch.float16: tensor = tensor.half()
                except: pass
                feat = model_to_call(tensor)
                norm = torch.norm(feat, p=2, dim=1, keepdim=True)
                feat = feat.div(norm.expand_as(feat))
                return feat.cpu().numpy().flatten().tolist()
        except: return None

    
    def analyze_video(self):
        print(f"🎬 Starting Analysis (V14 - FINAL INTEGRATED): {self.video_path}")
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened(): raise ValueError("Video not found")
        
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # --- VLM METADATA EXTRACTION ---
        ret, first_frame = cap.read()
        if ret:
            self.video_metadata = extract_camera_metadata_vlm(first_frame)
        else:
            self.video_metadata = {"classroom": "Unknown", "base_datetime": None, "base_datetime_str": "Unknown"}
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        # --- SAMPLING CONFIG ---
        PROBE_DURATION_SEC = 300 
        PROBE_INTERVAL_SEC = 3600
        FRAME_SKIP = 3
        
        probe_frames = int(PROBE_DURATION_SEC * self.fps)
        interval_frames = int(PROBE_INTERVAL_SEC * self.fps)
        start_frames = []
        curr = 0
        while curr < self.total_frames:
            start_frames.append(curr)
            curr += interval_frames
            
        print(f"📊 Sampling Strategy: {len(start_frames)} probes of 5-mins each.")

        # --- TRACKER INIT ---
        try: from boxmot import BoTSORT
        except ImportError: from boxmot import BoTSORT
        bot_weights = Path('Models\\osnet_x1_0_msmt17.pt')
        if not bot_weights.exists(): bot_weights = Path('osnet_x1_0_msmt17.pt')

        tracker = BoTSORT(model_weights=bot_weights, device='cuda:0', fp16=True, track_buffer=300, match_thresh=0.75)
        if hasattr(tracker, 'model'): 
            print("✅ Stitcher linked")
            self.stitch_model = tracker.model

        # --- MAIN LOOP ---
        for probe_idx, start_f in enumerate(start_frames):
            print(f"\n🔹 Processing Probe {probe_idx+1}/{len(start_frames)}")
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
            
            # Temporary store for this probe's RAW IDs to calculate per-probe stats
            probe_ids_seen = set()
            
            frames_processed = 0
            pbar = tqdm(total=probe_frames, desc=f"Probe {probe_idx+1}")
            
            while frames_processed < probe_frames:
                ret, frame = cap.read()
                if not ret: break
                curr_frame_global = start_f + frames_processed
                
                if frames_processed % FRAME_SKIP == 0:
                    # Detection & Merge (Same as V13)
                    body_results = self.models['detection'](frame, verbose=False, conf=0.40, classes=[0], imgsz=640)
                    face_results = self.models['face'](frame, verbose=False, conf=0.25, imgsz=640)
                    raw_body_boxes = body_results[0].boxes.data.cpu().numpy() if len(body_results[0].boxes) > 0 else []
                    face_boxes = face_results[0].boxes.data.cpu().numpy() if len(face_results[0].boxes) > 0 else []
                    final_dets = []
                    if len(raw_body_boxes) > 0:
                        for b in raw_body_boxes: final_dets.append(b)
                    for f_box in face_boxes:
                        fx1, fy1, fx2, fy2, f_conf, _ = f_box
                        is_matched = False
                        if len(raw_body_boxes) > 0:
                            for b_box in raw_body_boxes:
                                bx1, by1, bx2, by2, _, _ = b_box
                                if fx1 > bx1 - 50 and fx2 < bx2 + 50 and fy1 > by1 - 50 and fy2 < by2 + 50:
                                    is_matched = True; break
                        if not is_matched:
                            f_w, f_h = fx2 - fx1, fy2 - fy1
                            synth_w, synth_h = f_w * 2.5, f_h * 4.0
                            sx1, sy1 = max(0, fx1 - (synth_w - f_w)/2), max(0, fy1)
                            sx2, sy2 = min(self.width, sx1 + synth_w), min(self.height, sy1 + synth_h)
                            final_dets.append([sx1, sy1, sx2, sy2, f_conf, 0.0])

                    # Tracker
                    tracked_bodies = []
                    if len(final_dets) > 0:
                        tracks = tracker.update(np.array(final_dets), frame)
                        for t in tracks:
                            raw_id = int(t[4])
                            bbox = t[:4]
                            curr_emb = None
                            if frames_processed % 30 == 0: curr_emb = self.get_embedding(frame, bbox)
                            final_id = self.anchor_manager.get_corrected_id(raw_id, bbox, curr_emb, curr_frame_global)
                            tracked_bodies.append({'id': final_id, 'bbox': bbox})
                            
                            # Data Update
                            probe_ids_seen.add(final_id)
                            fdata = self.face_engagement_data[final_id]
                            fdata['face_id'] = final_id
                            fdata['total_frames'] += 1
                            curr_center = [(bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2]
                            fdata['last_centroid'] = curr_center
                            if not fdata['start_centroid']: fdata['start_centroid'] = curr_center
                            if not fdata['first_seen']: fdata['first_seen'] = curr_frame_global / self.fps
                            fdata['last_seen'] = curr_frame_global / self.fps
                            if curr_emb: fdata['best_embedding'] = curr_emb
                    else:
                        tracker.update(np.empty((0, 6)), frame)

                    # Engagement
                    pose_results = self.models['pose'](frame, verbose=False)
                    pose_data = self.extract_pose_data(pose_results)
                    detected_faces_fmt = [{'bbox': f[:4], 'confidence': f[4]} for f in face_boxes]
                    for body in tracked_bodies:
                        matched_face = None
                        bx1, by1, bx2, by2 = body['bbox']
                        for face in detected_faces_fmt:
                            fx, fy = (face['bbox'][0]+face['bbox'][2])/2, (face['bbox'][1]+face['bbox'][3])/2
                            if bx1 < fx < bx2 and by1 < fy < by2: matched_face = face; break
                        
                        person_data = self.match_face_to_person(body['bbox'], [{'center':[(b['bbox'][0]+b['bbox'][2])/2, (b['bbox'][1]+b['bbox'][3])/2]} for b in tracked_bodies], pose_data)
                        activity = person_data['activity']
                        if activity == 'unknown': activity = 'unknown'
                        attention = 'not_visible'
                        face_conf = 0.0
                        if matched_face:
                            face_conf = matched_face['confidence']
                            attention = person_data['attention']
                        else:
                            if activity == 'writing': attention = 'focused'
                            elif activity == 'walking': attention = 'partially_focused'
                        score, state = self.calculate_engagement_score(activity, attention, person_data['zone'], face_conf)
                        
                        # Store in Global Data
                        self.face_engagement_data[body['id']]['engagement_scores'].append(score)
                        self.face_engagement_data[body['id']]['activities'][activity] += 1
                        self.face_engagement_data[body['id']]['attention_levels'][attention] += 1

                frames_processed += 1
                pbar.update(1)
            pbar.close()
            
            # Save which IDs were in this probe for later report mapping
            self.raw_probe_data.append({
                "probe_index": probe_idx,
                "start_time": start_f / self.fps,
                "raw_ids": list(probe_ids_seen)
            })

        cap.release()
        self.finalize_and_report()

    def finalize_and_report(self):
        print("\n🔄 Starting Post-Processing & Stitching...")
        
        # 1. Export Raw Stitching Index
        stitch_index_path = os.path.join(self.output_dir, "stitching_index.json")
        stitch_data = []
        for fid, data in self.face_engagement_data.items():
            if data['total_frames'] < 5: continue
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
            
        # 2. RUN SELF-STITCHING
        id_map = perform_hierarchical_stitching(stitch_index_path)
        
        # 3. PRE-CALCULATE BASELINE
        probe_corrected_counts = {}
        for probe in self.raw_probe_data:
            unique_in_probe = set()
            for raw_id in probe['raw_ids']:
                root_id = id_map.get(raw_id, raw_id)
                unique_in_probe.add(root_id)
            probe_corrected_counts[probe['probe_index']] = len(unique_in_probe)
            
        max_students = max(probe_corrected_counts.values()) if probe_corrected_counts else 0
        print(f"📊 Baseline Class Size determined as: {max_students} students")

        # 4. Generate Final CORRECTED Report
        print("📊 Generating Corrected Dynamics Report...")
        final_hourly_report = []
        
        for probe in self.raw_probe_data:
            unique_students = set()
            probe_scores = []
            probe_activities = Counter()
            probe_attention = Counter()
            
            for raw_id in probe['raw_ids']:
                root_id = id_map.get(raw_id, raw_id)
                unique_students.add(root_id)
                raw_data = self.face_engagement_data[raw_id]
                probe_scores.extend(raw_data['engagement_scores'])
                probe_activities.update(raw_data['activities'])
                probe_attention.update(raw_data['attention_levels'])
            
            student_count = len(unique_students)
            avg_score = sum(probe_scores)/len(probe_scores) if probe_scores else 0
            
            total_act = sum(probe_activities.values())
            act_dist = {k: round(v/total_act*100, 1) for k, v in probe_activities.items()} if total_act else {}
            total_att = sum(probe_attention.values())
            att_dist = {k: round(v/total_att*100, 1) for k, v in probe_attention.items()} if total_att else {}
            
            # --- CLASS MODE LOGIC ---
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
                    
            # --- REAL WORLD TIME CALCULATOR ---
            real_time_str = "Unknown"
            if self.video_metadata["base_datetime"] is not None:
                real_time_obj = self.video_metadata["base_datetime"] + timedelta(seconds=probe['start_time'])
                real_time_str = real_time_obj.strftime("%I:%M:%S %p")
            
            final_hourly_report.append({
                "time_slice": f"Probe {probe['probe_index']}",
                "video_timestamp_sec": round(probe['start_time'], 1),
                "real_world_time": real_time_str,  # <-- Added Here
                "student_count_corrected": student_count,
                "avg_engagement": round(avg_score, 2),
                "class_mode": mode,
                "activity_distribution": act_dist,
                "attention_distribution": att_dist
            })
            
        final_path = os.path.join(self.output_dir, "class_dynamics_report.json")
        with open(final_path, 'w') as f:
            json.dump({
                "video_path": self.video_path,
                "classroom": self.video_metadata["classroom"],          # <-- Added Here
                "recording_date": self.video_metadata["base_datetime_str"], # <-- Added Here
                "report_type": "Corrected (Stitched)",
                "baseline_max_students": max_students,
                "hourly_probes": final_hourly_report
            }, f, indent=4)
            
        print(f"✅ FINAL REPORT SAVED: {final_path}")

    # --- HELPERS (SAME AS BEFORE) ---
    def calculate_engagement_score(self, activity, attention, zone, confidence):
        score = 0.0
        act_s = {'raising_hand': 1.0, 'writing': 0.9, 'reading': 0.75, 'listening': 0.6, 'talking': 0.5, 'walking': 0.1, 'standing': 0.3, 'unknown': 0.2}
        score += act_s.get(activity, 0.2)
        att_s = {'focused': 0.3, 'partially_focused': 0.15, 'distracted': 0.0}
        score += att_s.get(attention, 0.0)
        score += 0.1 if zone == 'front' else 0.05
        score += confidence * 0.1
        return min(1.0, score), 'engaged' if score >= 0.8 else 'partially_engaged' if score >= 0.5 else 'not_engaged'

    def match_face_to_person(self, face_bbox, persons, pose_data):
        if not persons: return {'activity': 'unknown', 'attention': 'not_visible', 'zone': 'middle', 'confidence': 0.0}
        fc = ((face_bbox[0]+face_bbox[2])/2, (face_bbox[1]+face_bbox[3])/2)
        min_d = float('inf')
        m_pose = None
        m_person = None
        for i, p in enumerate(persons):
            pc = p['center']
            d = math.sqrt((fc[0]-pc[0])**2 + (fc[1]-pc[1])**2)
            if d < min_d:
                min_d = d
                m_person = p
                m_pose = pose_data[i] if i < len(pose_data) else None
        if min_d > 200: return {'activity': 'unknown', 'attention': 'not_visible', 'zone': 'middle', 'confidence': 0.0}
        act = 'unknown'
        if m_pose:
            kp = m_pose['keypoints']
            if len(kp) >= 7 and kp[9][1] > 0 and kp[9][1] < kp[5][1]: act = 'raising_hand'
            elif len(kp) >= 7 and kp[0][1] > kp[5][1]: act = 'writing'
            elif kp[0][0] > 0: act = 'listening'
        att = 'distracted'
        if act in ['raising_hand', 'writing']: att = 'focused'
        elif act == 'listening': att = 'partially_focused'
        y = fc[1]
        zone = 'front' if y < self.height*0.4 else 'middle' if y < self.height*0.7 else 'back'
        return {'activity': act, 'attention': att, 'zone': zone, 'confidence': m_person.get('confidence', 0.0)}

    def extract_pose_data(self, pose_results):
        data = []
        for r in pose_results:
            if r.keypoints is not None:
                for kp in r.keypoints.data:
                    data.append({'keypoints': kp.cpu().numpy()})
        return data

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('video_path', type=str)
    parser.add_argument('--output-dir', type=str, default='Progressive_results\\classroom\\analysis_results_v15_2_7nov')
    args = parser.parse_args()
    
    if not os.path.exists(args.video_path): return
    analyzer = FaceEngagementAnalyzer(args.video_path, args.output_dir)
    analyzer.analyze_video()

if __name__ == "__main__":
    main()