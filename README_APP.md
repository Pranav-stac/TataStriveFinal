# TataStrive Analytics

A professional desktop application for classroom engagement analysis and attendance tracking.

## Features

- **Classroom Analysis**: Analyze CCTV footage to measure student engagement, class dynamics, and activity patterns
- **Attendance**: Track people across multiple days, distinguish returning employees from visitors
- **Report Viewer**: View and export analysis reports in table and tree formats
- **Video Preview**: Optional real-time video preview during processing
- **Persistent Settings**: Configuration saved between sessions

## Installation

### Prerequisites

- Python 3.9 or higher
- CUDA-capable GPU (recommended for faster processing)

### Install Dependencies

```bash
pip install -r requirements_app.txt
```

### Required Model Files

Place the following model files in a `Models/` folder:

- `yolov8m.pt` - Person detection
- `yolov8n-pose.pt` - Pose estimation
- `yolov8n-face.pt` - Face detection
- `osnet_x1_0_msmt17.pt` - Re-ID model (for BoTSORT)

- `yolov8m.pt` and `yolov8n-pose.pt` are downloaded automatically by ultralytics if not present.
- `yolov8n-face.pt` is **not** in standard Ultralytics. Run the download script:
  ```bash
  python download_face_model.py
  ```
- `osnet_x1_0_msmt17.pt` is downloaded by boxmot on first run.

For cross-day attendance, InsightFace models will be downloaded automatically on first run.

### Groq API Key (Optional - for VLM metadata)

To extract classroom name, date, and time from CCTV frames via Groq's vision model:

1. Get an API key from [Groq Console](https://console.groq.com)
2. Copy `.env.example` to `.env` in the project root
3. Set `GROQ_API_KEY=your_key` in `.env`

Without the key, metadata will show "Unknown" for classroom and datetime.

## Running the Application

### Option 1: Python Script

```bash
python run_app.py
```

Or:

```bash
python -m app.main
```

### Option 2: Build Standalone Executable

```bash
python build_exe.py
```

This creates a standalone executable in `dist/TataStriveAnalytics/`.

### Option 3: Double-Click Installer for End Users (Windows)

If you want non-technical users to install a compiled version without Python:

1. Build executable with `python build_exe.py`
2. Build installer EXE with `installer/build_installer.ps1`
3. Share generated Setup EXE with client
4. Client double-clicks installer and completes wizard

Detailed steps are in `installer/README.md`.

## Usage

### Classroom Analysis Tab

1. Select a video file (MP4, AVI, MKV, MOV)
2. Choose an output directory
3. Configure analysis parameters:
   - **Probe Duration**: Length of each sampling probe (default: 300 sec = 5 min)
   - **Probe Interval**: Time between probe starts (default: 3600 sec = 1 hour)
   - **Frame Skip**: Process every Nth frame (default: 3)
   - **Similarity Threshold**: For track stitching (default: 0.75)
4. Optionally enable video preview
5. Click "Start Analysis"

### Cross-Day Attendance Tab

1. Select run mode:
   - **BUILD_DB**: First day - creates baseline database
   - **EVAL_DAY**: Subsequent days - matches against existing database
2. Set the current date
3. Select video and database files
4. Configure thresholds
5. Click "Start Analysis"

### Report Viewer Tab

1. Load a JSON report file
2. View data in table or tree format
3. Export to CSV if needed

## Configuration

Settings are stored in `~/.tatastrive/config.json` and persist between sessions.

Access settings via **File > Settings** or `Ctrl+,`.

## Keyboard Shortcuts

- `Ctrl+O` - Open video file
- `Ctrl+R` - Open report file
- `Ctrl+,` - Open settings
- `Ctrl+1` - Switch to Classroom Analysis tab
- `Ctrl+2` - Switch to Cross-Day Attendance tab
- `Ctrl+3` - Switch to Report Viewer tab

## Output Files

### Classroom Analysis

- `class_dynamics_report.json` - Main report with hourly probes
- `stitching_index.json` - Raw track data

### Cross-Day Attendance

- `{date}_attendance_report.json` - Daily attendance report
- `{date}_output.mp4` - Annotated video
- `master_database.pkl` or `updated_master_database.pkl` - Face database
- `crops_*/` - Face crop images
- `Verification_Matches/` - Matched face images (EVAL_DAY)

## Troubleshooting

### PyTorch DLL Error (WinError 1114, c10.dll)

If you see: `A dynamic link library (DLL) initialization routine failed. Error loading c10.dll`:

**Solution 1 - Install CPU-only PyTorch (recommended):**
```bash
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

**Solution 2 - Install Visual C++ Redistributable:**
Download and install [Microsoft Visual C++ Redistributable 2015-2022](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist) (64-bit).

**Solution 3 - CUDA version mismatch:**
If you need GPU support, ensure your CUDA version matches PyTorch. Check [PyTorch installation guide](https://pytorch.org/get-started/locally/).

### "Numpy is not available" (RuntimeError in cross-day or classroom analysis)

PyTorch 2.0.x does not support NumPy 2.0+. Downgrade NumPy:
```bash
pip install "numpy>=1.21,<2.0"
```
Or upgrade PyTorch to 2.2+ for NumPy 2.x support.

### "Could not load model" errors

Make sure YOLO model files are in the `Models/` folder or let ultralytics download them automatically.

### onnxruntime DLL error / "No module named 'onnxruntime'"

Cross-day attendance uses onnxruntime for face matching. If you see DLL errors:
```bash
pip uninstall onnxruntime-gpu -y
pip install onnxruntime
```
Or run `.\fix_onnxruntime.bat` (PowerShell) or `fix_onnxruntime.bat` (Command Prompt) from the project folder. The app will run in **simplified mode** (track-only) if onnxruntime fails.

### CUDA out of memory

- Reduce video resolution
- Increase frame skip
- Disable video preview

### Application won't start

Check that all dependencies are installed:
```bash
pip install -r requirements_app.txt --upgrade
```

## License

Proprietary - TataStrive
