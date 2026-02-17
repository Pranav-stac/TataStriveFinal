# Comprehensive Code Documentation
## TataStriveFinal – Classroom Analysis & Cross-Day Attendance

---

# PART 1: CLASSROOM ANALYSIS (`classroom_analysis/`)

---

## 1.1 Project Overview

**Purpose:** Analyze single-classroom CCTV footage to produce engagement metrics, class dynamics, and hourly activity reports. Designed for educational settings (schools, colleges).

**Use Case:** "How engaged are students? What is the class mode (Lecture/Interactive/Break)? How many unique students per hour?"

**Key Design:** Sampling-based (5 min every hour) to reduce compute; two-stage ID correction (runtime anchors + post-stitching).

---

## 1.2 File Structure

```
classroom_analysis/
├── vlm_metadata.py             # Groq VLM metadata extraction (used by app)
├── stitch_logic.py             # Post-processing track merging (used by app)
└── requirements_classroom_activity.txt
```

**Note:** The GUI app uses `app/workers/classroom_worker.py` which implements the full pipeline (aligned with the original classroom_activity_1.py logic).

**Required External:**
- `Models/` folder with: `yolov8m.pt`, `yolov8n-pose.pt`, `yolov8n-face.pt`, `osnet_x1_0_msmt17.pt`
- `.env` file with `GROQ_API_KEY` (for vlm_metadata)

---

## 1.3 Installation & Usage

### Installation

```bash
cd classroom_analysis
pip install -r requirements_classroom_activity.txt
```

### Usage

Run via the **TataStrive Analytics** GUI app (`run_app.py` → Classroom tab). The app uses `app/workers/classroom_worker.py` which imports `stitch_logic` and `vlm_metadata` from this folder.

**Pre-requisites:**
- Video file must exist (script exits silently if not)
- GPU recommended (CUDA) for YOLO + BoTSORT
- Internet for Groq API (first frame only)

---

## 1.4 Dependencies (Detailed)

| Package | Version | Purpose |
|---------|---------|---------|
| numpy | 1.24.4 | Array ops, embeddings |
| opencv-python | - | Video I/O, image processing |
| torch | ≥2.0 | YOLO, BoTSORT, embeddings |
| tqdm | - | Progress bars |
| scipy | - | cosine distance |
| ultralytics | ≥8.0 | YOLO detection/pose |
| boxmot | 10.0.45 | BoTSORT tracker |
| groq | ≥0.4 | VLM API |
| python-dotenv | - | Load GROQ_API_KEY |

---

## 1.5 Module-by-Module Deep Dive

---

### 1.5.1 `vlm_metadata.py`

**Purpose:** Extract date, time, and room name from the first CCTV frame using a Vision Language Model.

#### Functions

**`encode_image(frame)`**
- Input: OpenCV BGR frame (numpy array)
- Process: `cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 100])` → base64
- Quality 100 to preserve overlay text (date/time/room) on CCTV
- Output: Base64 string

**`extract_camera_metadata_vlm(frame)`**
- Input: Single OpenCV frame
- Output: `{"classroom": str, "base_datetime": datetime|None, "base_datetime_str": str}`

**Flow:**
1. Load `GROQ_API_KEY` from `.env`
2. Encode frame to base64
3. Call Groq API: `meta-llama/llama-4-scout-17b-16e-instruct`, temperature=0
4. Prompt asks for JSON: `{"date": "YYYY-MM-DD", "time": "HH:MM:SS", "room": "..."}`
5. Clean response: strip ` ```json ` and ` ``` `
6. Parse JSON, build `datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")`
7. Return metadata dict

**Error Handling:**
- API error → returns `{"classroom": "Unknown", "base_datetime": None, "base_datetime_str": "Unknown"}`
- Date parse error → logs warning, keeps "Unknown"

**Standalone Test:**
```bash
python vlm_metadata.py   # Uses do_ocr_init2.jpg in cwd
```

---

### 1.5.2 `stitch_logic.py`

**Purpose:** Merge fragmented tracks (same person, different IDs) into one identity after video processing.

#### Algorithm: Union-Find Stitching

**Input:** JSON file path with list of track objects:
```json
[
  {
    "face_id": 5,
    "first_seen": 120.5,
    "last_seen": 180.2,
    "start_centroid": [320, 240],
    "last_centroid": [350, 260],
    "embedding": [0.1, -0.2, ...]
  }
]
```

**Parameters:**

| Param | Default | Meaning |
|-------|---------|---------|
| similarity_threshold | 0.75 | Min cosine similarity to merge |
| max_time_gap | 600 | Max seconds between track A end and track B start |
| max_pixel_dist | 200 | Max pixel distance between A's last centroid and B's start centroid |

**Merge Conditions (all must hold):**
1. **Time:** `0 <= track_b.first_seen - track_a.last_seen <= 600`
2. **Space:** `distance(track_a.last_centroid, track_b.start_centroid) <= 200`
3. **Visual:** `1 - cosine(emb_a, emb_b) > 0.75`

**Logic:**
- Tracks sorted by `first_seen`
- For each pair (A, B) where A ends before B: check conditions
- If merge: `parent_map[root_b] = root_a` (Union-Find)
- Early break: if `time_gap > max_time_gap`, no need to check later B's (sorted)
- Final pass: flatten so every ID → ultimate root

**Output:** `{original_id: root_id}` mapping

**Filtering:** Tracks without `embedding` or `last_centroid` are skipped.

---

### 1.5.3 `app/workers/classroom_worker.py` (Pipeline Implementation)

The worker implements the full pipeline. Key components:

#### 1.5.3.1 `RuntimeAnchorManager`

**Purpose:** Stabilize tracker IDs within a probe. When BoTSORT re-IDs the same person (e.g., after occlusion), map them back to a stable "seat" identity.

**Data Structures:**

```python
seats = {
  seat_uid: {
    "owner_id": stable_id,      # The canonical ID for this seat
    "centroid": [x, y],         # Running average position
    "embedding": [...],         # Re-ID embedding (or None)
    "last_seen": frame_idx      # Last frame this seat was seen
  }
}
active_mapping = { raw_tracker_id: stable_id }  # Current frame mapping
potential_seats = { raw_id: { "count": n, "centroids": [...] } }  # Before seat creation
```

**Parameters:**
- `similarity_thresh=0.70` (used 0.75 in analyzer)
- `distance_thresh=120` px
- `lock_frames=10` – frames at same position before creating new seat

**`get_corrected_id(raw_id, bbox, embedding, frame_idx)` Logic:**

1. **Already mapped:** If `raw_id in active_mapping` → return `stable_id`, update seat.
2. **Search seats:** For each seat within `distance_thresh`:
   - Match if: `(time_gap < 90 AND dist < 80)` OR `sim > similarity_thresh`
   - Pick closest matching seat
3. **Match found:** Map `raw_id` → seat's `owner_id`, update seat, return.
4. **No match:** `active_mapping[raw_id] = raw_id`, call `_check_create_seat`, return `raw_id`.

**`_check_create_seat`:** After 10 frames, if avg centroid is >50px from all existing seats, create new seat.

**`_update_seat_position`:** EMA: `centroid = 0.9 * old + 0.1 * new`

---

#### 1.5.3.2 `FaceEngagementAnalyzer`

**Initialization:**
- `video_path`, `output_dir`
- Device: CUDA if available else CPU
- Loads: YOLOv8m (detection), YOLOv8n-pose, YOLOv8n-face
- `RuntimeAnchorManager(similarity_thresh=0.75, distance_thresh=120)`
- `face_engagement_data`: defaultdict per ID with engagement stats

**Model Loading (`load_models`):**
- Base path: `sys._MEIPASS` if frozen (PyInstaller), else script dir
- Weights: `Models/yolov8m.pt`, etc. or fallback to ultralytics download
- `model.fuse()` for speed if available

---

#### 1.5.3.3 Embedding Extraction (`get_embedding`)

**Source:** BoTSORT's OSNet re-ID model (body crop, not face).

**Process:**
1. Crop bbox from frame, clamp to image bounds
2. Resize to 128×256 (H×W for re-ID)
3. Normalize: `/255.0`
4. Forward through `stitch_model` (BoTSORT's `.model` or `.net`)
5. L2-normalize output
6. Return flattened list

**Called:** Every 30 frames per track (to save compute).

**Returns:** `None` if stitch_model missing, crop too small, or forward fails.

---

#### 1.5.3.4 Main Analysis Loop (`analyze_video`)

**Step 1: Video Setup**
- Open video, get fps, total_frames, width, height
- Read first frame → `extract_camera_metadata_vlm(first_frame)`
- Reset to frame 0

**Step 2: Sampling Strategy**
- `PROBE_DURATION_SEC = 300` (5 min per probe)
- `PROBE_INTERVAL_SEC = 3600` (1 hour between probe starts)
- `FRAME_SKIP = 3` (process every 3rd frame)
- `start_frames = [0, interval_frames, 2*interval_frames, ...]`

**Step 3: Tracker Init**
- BoTSORT with `osnet_x1_0_msmt17.pt`, `track_buffer=300`, `match_thresh=0.75`
- `self.stitch_model = tracker.model` for embeddings

**Step 4: Per-Probe Loop**
For each probe:
- Seek to `start_f`
- Process `probe_frames` frames (5 min worth)
- Every `FRAME_SKIP` frames:

  **Detection:**
  - YOLO detection: persons (class 0, conf 0.40)
  - YOLO face: conf 0.25
  - Merge: bodies + unmatched faces (synthetic body: 2.5×w, 4×h, centered)

  **Tracking:**
  - `tracker.update(detections, frame)`
  - For each track: `get_embedding` every 30 frames, `anchor_manager.get_corrected_id`
  - Update `face_engagement_data[final_id]`: centroids, first/last seen, embedding

  **Engagement:**
  - YOLO pose on full frame
  - Match each tracked body to nearest pose, infer activity
  - Match face to body (center-in-bbox)
  - Compute engagement score, store

- Store `raw_probe_data`: `{probe_index, start_time, raw_ids}`

**Step 5: Finalize**
- `finalize_and_report()`

---

#### 1.5.3.5 Detection Merge Logic (Body + Face)

```python
# Body boxes from YOLO
raw_body_boxes = [...]

# Face boxes from YOLO face
for face_box in face_boxes:
  # Check if face is inside any body (with 50px margin)
  is_matched = any(face inside body_bbox + margin for body in raw_body_boxes)
  if not is_matched:
    # Create synthetic body: 2.5× face width, 4× face height
    synth_bbox = [fx1 - margin, fy1, fx2 + margin, fy2 + extended_height]
    final_dets.append(synth_bbox)
```

**Rationale:** Catch faces without full body (e.g., back of head, partial view).

---

#### 1.5.3.6 Pose-Based Activity (`match_face_to_person`)

**COCO Keypoint Indices (YOLO pose):**
- 0: nose
- 5: left shoulder
- 6: right shoulder
- 9: left wrist (index 9 in 0-based; YOLO uses 17 kpts)

**Activity Rules:**
- `raising_hand`: `kp[9].y < kp[5].y` (left wrist above left shoulder)
- `writing`: `kp[0].y > kp[5].y` (nose below shoulder – head tilted down)
- `listening`: nose visible (`kp[0].x > 0`)
- Default: `unknown`

**Attention:**
- `focused`: raising_hand or writing
- `partially_focused`: listening
- `distracted`: else
- `not_visible`: no face in body bbox

**Zone:**
- `y < 0.4*height` → front
- `y < 0.7*height` → middle
- else → back

---

#### 1.5.3.7 Engagement Score (`calculate_engagement_score`)

```
score = activity_score + attention_score + zone_bonus + (face_conf * 0.1)
```

| Activity | Score |
|----------|-------|
| raising_hand | 1.0 |
| writing | 0.9 |
| reading | 0.75 |
| listening | 0.6 |
| talking | 0.5 |
| standing | 0.3 |
| unknown | 0.2 |
| walking | 0.1 |

| Attention | Score |
|-----------|-------|
| focused | 0.3 |
| partially_focused | 0.15 |
| distracted | 0.0 |

| Zone | Score |
|------|-------|
| front | 0.1 |
| middle/back | 0.05 |

**State:** `engaged` (≥0.8), `partially_engaged` (≥0.5), `not_engaged` (<0.5)

---

#### 1.5.3.8 Class Mode Logic

- **Break:** `student_count < 5`
- **Transition/Sparse:** `student_count < 0.66 * max_students`
- **Interactive:** `(walking + talking) / total_activities > 0.3`
- **Lecture:** else

---

#### 1.5.3.9 Finalize & Report (`finalize_and_report`)

1. **Export stitching index:** All IDs with `total_frames >= 5`, with `first_seen`, `last_seen`, centroids, embedding
2. **Run stitching:** `perform_hierarchical_stitching(stitch_index_path)` → `id_map`
3. **Baseline:** `max_students = max(unique count per probe after stitching)`
4. **Per-probe report:** For each probe, apply `id_map` to get unique students, aggregate scores/activities/attention, compute class mode, real-world time
5. **Write** `class_dynamics_report.json`

**Real-world time:** `base_datetime + timedelta(seconds=probe_start_time)` → `"%I:%M:%S %p"`

---

## 1.6 Output Files

| File | Location | Content |
|------|----------|---------|
| `stitching_index.json` | output_dir | Raw tracks for stitching |
| `class_dynamics_report.json` | output_dir | Final report |

**Report Schema:**
```json
{
  "video_path": "...",
  "classroom": "Room 101",
  "recording_date": "2025-11-07 09:00:00",
  "report_type": "Corrected (Stitched)",
  "baseline_max_students": 25,
  "hourly_probes": [
    {
      "time_slice": "Probe 0",
      "video_timestamp_sec": 0,
      "real_world_time": "09:00:00 AM",
      "student_count_corrected": 24,
      "avg_engagement": 0.72,
      "class_mode": "Lecture",
      "activity_distribution": {"listening": 60, "writing": 30, ...},
      "attention_distribution": {"focused": 45, ...}
    }
  ]
}
```

---

## 1.7 Edge Cases & Limitations

- **No video:** Script exits without error message
- **Missing Models:** Falls back to ultralytics download; OSNet may fail if path wrong
- **No GROQ key:** VLM returns "Unknown" for metadata
- **Short video:** Fewer probes; last probe may be truncated
- **Pose order:** `match_face_to_person` assumes pose order matches tracked body order (nearest-centroid match)
- **Stitching:** Tracks with <5 frames or no embedding are excluded

---

# PART 2: ATTENDANCE ONLY (Cross-Day) – `app/workers/crossday_worker.py`

---

## 2.1 Project Overview

**Purpose:** Multi-day attendance and identity tracking. Distinguish returning people (employees) from visitors; build a persistent gallery; upgrade frequent visitors to permanent IDs.

**Use Case:** Office/ workplace CCTV: "Who was here today? Returning or new? How long did they stay?"

**Key Design:** Two modes (BUILD_DB / EVAL_DAY); InsightFace for face embeddings; gallery matching with margin; visitor upgrade after N days.

---

## 2.2 Implementation

The logic is implemented in **`app/workers/crossday_worker.py`** (based on the original `cross_day_code/main.py`). Run via the **TataStrive Analytics** GUI app (`run_app.py` → Attendance only tab).

**Required:**
- Video input and (for EVAL_DAY) existing database path
- Output directory (chosen in the app)

---

## 2.4 Dependencies

| Package | Purpose |
|---------|---------|
| opencv-python | Video I/O, drawing |
| torch, torchvision | YOLO backend |
| lapx | BoTSORT association |
| numpy (<2) | Arrays |
| scipy | cosine |
| ultralytics | YOLO |
| insightface | Face detection + ArcFace embeddings |
| onnxruntime, onnxruntime-gpu | InsightFace inference |

---

## 2.5 Configuration Parameters (Detailed)

| Parameter | Value | Meaning |
|-----------|-------|---------|
| T_STRICT_MERGE | 0.55 | Min similarity to match to existing gallery ID |
| T_NEW_ID | 0.35 | Max similarity to create new ID (below = new person) |
| T_RATIO_MARGIN | 0.10 | Min gap between best and second-best to avoid ambiguous match |
| MIN_SAMPLES | 8 | Min face embeddings before identity assignment |
| MAX_EXEMPLARS | 5 | Max exemplars per gallery ID |
| T_OUTLIER | 0.6 | Max cosine distance to accept new embedding (outlier rejection) |
| VISITOR_UPGRADE_DAYS | 3 | Days present to upgrade visitor → G_ID |

**Identity Assignment Bands:**
- `best_sim > 0.55` → match to gallery
- `best_sim < 0.35` → create new ID
- `0.35 <= best_sim <= 0.55` → uncertain, no assignment (track stays "Scanning")
- If `best_sim - second_sim < 0.10` → ambiguous, no match

---

## 2.6 UniversalMultiRepSystem – Deep Dive

### 2.6.1 Data Structures

**`global_gallery`:**
```python
{
  "G_001": {
    "exemplars": [emb1, emb2, ...],  # L2-normalized, max 5
    "join_date": "2026-02-01",
    "attendance": {
      "2026-02-01": {"entry": "09:00:00", "exit": "17:30:00"},
      "2026-02-05": {"entry": "09:15:00", "exit": "18:00:00"}
    }
  },
  "Day2_V_001": { ... }  # Visitors
}
```

**`track_vault`** (per-session, keyed by tracker ID):
```python
{
  track_id: {
    "embeddings": [emb1, emb2, ...],
    "global_id": "G_001" | None,
    "first_seen": "09:00:00",
    "last_seen": "17:30:00",
    "frames": 1234,
    "bbox": [x1,y1,x2,y2]
  }
}
```

---

### 2.6.2 Face-to-Body Association (`match_face_to_body`)

**Constraints:**
1. Face center Y must be in upper half of body: `face_cy <= py1 + 0.5*(py2-py1)`
2. Overlap: `intersection_area / face_area > 0.5`
3. Among matches: pick body with smallest `|face_cx - person_cx|`
4. One face → one body (`assigned_tracks`)

---

### 2.6.3 Face Quality Filters

- `face.det_score >= 0.75`
- `face_width >= 40` px
- Frontal check:
  - `eye_dist / face_width >= 0.35` (not too profile)
  - `|nose.x - eye_center_x| <= 0.5 * eye_dist` (nose centered)

---

### 2.6.4 Outlier Rejection

After 3+ embeddings per track:
- Compute track centroid (mean, L2-norm)
- If `cosine(new_emb, centroid) > 0.6` → reject (possible impostor or bad detection)

---

### 2.6.5 Identity Assignment Flow

**When:** `len(embeddings) >= 8` and `global_id is None`

1. Track centroid = mean(embeddings), L2-normalized
2. `find_match_with_margin(centroid, active_gids_in_frame)`
3. **Match:** Assign `global_id`, log attendance, add to `active_gids_in_frame`
4. **New person (best_sim < 0.35):**
   - BUILD_DB: `G_{next_global_id}`
   - EVAL_DAY: `{DAY_LABEL}_V_{next_visitor_id}` ← **DAY_LABEL must be defined!**
5. **Uncertain:** Leave as Scanning
6. **EVAL_DAY + assigned:** Copy crop to `VERIFICATION_DIR/{gid}_track_{tid}.jpg`

**`active_gids_in_frame`:** Prevents same gallery ID being assigned to two tracks in one frame.

---

### 2.6.6 Exemplar Update (`update_exemplars`)

**Biometric lock:** In EVAL_DAY, never update exemplars for `G_*` IDs.

**Diversity:** Add only if `1 - cosine(new, ex) <= 0.60` for all existing exemplars.

**Capacity:** FIFO, max 5 exemplars.

---

### 2.6.7 Attendance Logging (`log_attendance`)

- First seen: `attendance[date] = {"entry": ts, "exit": ts}`
- Each update: `attendance[date]["exit"] = ts`
- Entry = first timestamp, exit = last timestamp per day

---

### 2.6.8 Visitor Upgrade (`upgrade_eligible_visitors`)

**After video processing:**
- Find IDs with `_V_` in name
- If `len(attendance) >= VISITOR_UPGRADE_DAYS` (3):
  - Create new `G_{next_global_id}`
  - Move gallery entry from visitor ID to new G_ID
  - Remove old visitor ID

---

### 2.6.9 Process Frame – Full Flow

1. YOLO track (BoTSORT) → person tracks
2. Update track_vault (last_seen, frames, bbox)
3. If already has global_id → log_attendance
4. InsightFace `app.get(frame)` → faces
5. For each face: quality filter → match_face_to_body → add embedding (with outlier check)
6. For tracks with ≥8 embeddings and no global_id: identity assignment
7. Draw bboxes (Green=G_, Blue=Visitor, Orange=Scanning)
8. Return annotated frame

---

## 2.7 Main Loop & Outputs

**Main:**
- Open video, get fps, dimensions
- Resize to 1280×720 if width > 1920
- Process every frame, write to OUTPUT_VIDEO
- After processing: `upgrade_eligible_visitors()`
- Save DB: BUILD_DB → DB_PATH; EVAL_DAY → `./Outputs/updated_master_database.pkl`
- Generate and save report JSON

**Output Files:**

| File | Content |
|------|---------|
| OUTPUT_VIDEO | Annotated video with colored bboxes |
| REPORT_JSON | Attendance report |
| CROPS_DIR | First face crop per track |
| VERIFICATION_DIR | Crops copied for assigned IDs (EVAL_DAY) |
| master_database.pkl | Updated gallery + operational_dates |

---

## 2.8 Report Schema

```json
{
  "Session": {
    "date": "2026-02-05",
    "camera": "Cam_01",
    "source_video": "5feb_merged.mp4",
    "duration": "18:00:00"
  },
  "Counts": {
    "unique_people": 12,
    "returning": 10,
    "visitors": 2
  },
  "People": [
    {
      "id": "G_001",
      "type": "returning",
      "entry": "09:00:00",
      "exit": "17:30:00",
      "duration_sec": 30600,
      "last_present_date": "2026-02-04",
      "present_last_7_days": 5
    },
    {
      "id": "Day2_V_001",
      "type": "visitor",
      "entry": "10:15:00",
      "exit": "12:00:00",
      "duration_sec": 6300,
      "last_present_date": null,
      "present_last_7_days": 0
    }
  ]
}
```

**Type logic:**
- `visitor`: `join_date == CURRENT_DATE`
- `returning`: else

---

## 2.9 Known Bug

**`DAY_LABEL` is undefined** when EVAL_DAY creates a visitor. Line 256 uses `DAY_LABEL` but it's commented out. This causes `NameError` when a new visitor is created in EVAL_DAY.

**Fix:** Add before use, e.g.:
```python
DAY_LABEL = "Day2"  # Or: f"Day{CURRENT_DATE.replace('-','')}"
```

---

# PART 3: SIDE-BY-SIDE COMPARISON

| Aspect | Classroom (classroom_worker) | Attendance only (crossday_worker) |
|--------|-------------------|----------------|
| **Input** | Single video | Single video per run |
| **Temporal scope** | Single session | Multi-day (persistent DB) |
| **Sampling** | 5 min every hour | Full video |
| **Identity** | Anonymous (per-video) | Persistent (G_*, Visitor) |
| **Face model** | None (body re-ID) | InsightFace (buffalo_l) |
| **Embedding** | BoTSORT OSNet (body) | InsightFace ArcFace (face) |
| **ID correction** | RuntimeAnchor + stitch | Gallery matching |
| **Output** | Engagement, class mode | Attendance, entry/exit |
| **Database** | None | master_database.pkl |
| **Visitor logic** | N/A | Upgrade after 3 days |
| **Config** | CLI args | In-file constants |
| **VLM** | Groq (metadata) | None |

---

# PART 4: TYPICAL WORKFLOWS

## Classroom Analysis (Single Day)

1. Record classroom CCTV (e.g., 8 hours)
2. Run the **TataStrive Analytics** app → Classroom tab → select video → Start
3. Use `class_dynamics_report.json` for engagement and class mode per hour

## Attendance only (Multi-Day)

**Day 1 (BUILD_DB):**
1. Run app → Attendance only tab → select Day 1 video
2. Choose output directory
3. Start analysis; `master_database.pkl` created

**Day 2+ (EVAL_DAY):**
1. Run app → Attendance only tab → select Day 2+ video
2. Point to existing database from Day 1
3. Start analysis; check updated database and report

---

*End of Documentation*
