"""
Download yolov8n-face.pt from Hugging Face (deepghs/yolo-face).
The face model is not in standard Ultralytics - run this script to enable face detection.
"""

import os
import sys
import urllib.request

# Hugging Face direct download (model.pt from yolov8n-face folder)
HF_URL = "https://huggingface.co/deepghs/yolo-face/resolve/main/yolov8n-face/model.pt"


def main():
    # Resolve Models directory
    base = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(base, "Models")
    os.makedirs(models_dir, exist_ok=True)
    out_path = os.path.join(models_dir, "yolov8n-face.pt")

    if os.path.exists(out_path):
        print(f"yolov8n-face.pt already exists at {out_path}")
        return 0

    print("Downloading yolov8n-face.pt from Hugging Face...")
    try:
        urllib.request.urlretrieve(HF_URL, out_path)
        print(f"Saved to {out_path}")
        return 0
    except Exception as e:
        print(f"Download failed: {e}")
        print("\nManual download:")
        print(f"  1. Open {HF_URL}")
        print(f"  2. Save as Models/yolov8n-face.pt")
        return 1


if __name__ == "__main__":
    sys.exit(main())
