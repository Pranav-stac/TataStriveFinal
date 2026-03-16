import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import json
import cv2
import base64
from datetime import datetime
from groq import Groq

def _load_env_for_frozen_and_dev() -> None:
    # 1) Normal dev flow: .env in current working directory/project root
    load_dotenv()

    # 2) Frozen app flow: .env next to exe and inside bundle directory
    candidates = []
    exe_dir = Path(sys.executable).parent
    candidates.append(exe_dir / ".env")
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        candidates.append(Path(meipass) / ".env")

    for env_path in candidates:
        if env_path.exists():
            load_dotenv(dotenv_path=env_path, override=False)


_load_env_for_frozen_and_dev()
groq_key = os.getenv("GROQ_API_KEY")

def encode_image(frame):
    """Encodes an OpenCV frame to base64 with MAXIMUM quality (100) to prevent blurry CCTV text."""
    # Force 100% quality to prevent JPEG compression artifacts
    _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
    return base64.b64encode(buffer).decode('utf-8')

def extract_camera_metadata_vlm(frame):
    metadata = {
        "classroom": "Unknown",
        "base_datetime": None,
        "base_datetime_str": "Unknown"
    }

    if not groq_key:
        print("[*] GROQ_API_KEY not set. Skipping VLM metadata extraction.")
        return metadata

    print("[*] Sending frame to Groq VLM (llama-4-scout-17b)...")

    try:
        # 1. Initialize Groq Client
        client = Groq(api_key=groq_key) 
        
        # 2. Encode the frame
        base64_image = encode_image(frame)
        
        # 3. Strict Prompting - EXACTLY MATCHING YOUR PLAYGROUND
        prompt = """
        Analyze this CCTV camera frame. Extract the following information:
        1. The Date (format: YYYY-MM-DD)
        2. The Time (format: HH:MM:SS)
        3. The Classroom Name (the text near the right bottom corner)
        
        Return ONLY a valid JSON object with the keys "date", "time", and "room". 
        Do not include markdown formatting, backticks, or any other conversational text.
        """
        
        # 4. Call Groq Vision Model
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                            },
                        },
                    ],
                }
            ],
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            temperature=0.0, # Zero temperature for deterministic extraction
        )
        
        # 5. Parse Response
        response_text = chat_completion.choices[0].message.content
        
        # Safety cleanup: Llama sometimes wraps JSON in markdown blocks
        clean_text = response_text.replace("```json", "").replace("```", "").strip()
        
        data = json.loads(clean_text)
        date_val = data.get("date", "")
        time_val = data.get("time", "")
        room_val = data.get("room", "Unknown")
        
        metadata["classroom"] = room_val
        
        # 6. Create Datetime Object for Math Later
        if date_val and time_val:
            dt_str = f"{date_val} {time_val}"
            metadata["base_datetime_str"] = dt_str
            try:
                metadata["base_datetime"] = datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")
                print(f"[OK] Groq VLM Success: {metadata['classroom']} at {metadata['base_datetime_str']}")
            except ValueError as e:
                print(f"[!] Groq Date Parsing Error: {e} - Raw strings: {date_val}, {time_val}")
        
    except Exception as e:
        print(f"[!] Groq API Error: {e}")
        
    return metadata

# Quick testing block
if __name__ == "__main__":
    test_img = cv2.imread("do_ocr_init2.jpg")
    if test_img is not None:
        res = extract_camera_metadata_vlm(test_img)
        print(res)
    else:
        print("Please provide a valid test image to test locally.")