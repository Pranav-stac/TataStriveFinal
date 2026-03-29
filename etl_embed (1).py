import boto3
import os
import cv2
import sqlite3
import numpy as np
from google.cloud import bigquery
from insightface.app import FaceAnalysis

# AWS Credentials
AWS_ACCESS_KEY = 'AWS_ACCESS_KEY'
AWS_SECRET_KEY = 'AWS_SECRET_KEY'
AWS_REGION = 'ap-south-1' # change accordingly
BUCKET_NAME = 'bucket-name' 
DOWNLOAD_DIR = './latest_student_downloads' 

# Google Cloud Credentials
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "./gcp-key.json"

# Local Edge Database 
STUDENT_DB_PATH = "student_enrollments.db"

# ===========================================================
print("Loading InsightFace Model for S3 Images...")
app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640)) 

# ===========================================================
def fetch_active_students_from_bq():
    print("⏳ Connecting to Google BigQuery...")
    try:
        client = bigquery.Client()
        query = """
            SELECT engagement_id, batch_name 
            FROM `tatastrive-269409.student_intraining.intraining_students`
            WHERE student_engagement_status = 'intraining'
            LIMIT 1000
        """
        
        query_job = client.query(query)
        results = query_job.result()
        
        student_roster = {}
        for row in results:
            eng_id = str(row["engagement_id"])
            student_roster[eng_id] = {"batch": row["batch_name"]}
            
        print(f"✅ Successfully fetched {len(student_roster)} active students!\n")
        return student_roster
        
    except Exception as e:
        print(f"❌ BigQuery Connection Failed.\nError: {e}")
        return {}

def generate_student_embedding(image_path):
    """Extracts the buffalo_l embedding from the downloaded image."""
    img = cv2.imread(image_path)
    if img is None:
        return None

    faces = app.get(img)
    if len(faces) == 0:
        return None

    # Grab the largest face (in case there are people in the background of the ID photo)
    faces = sorted(faces, key=lambda x: (x.bbox[2]-x.bbox[0]) * (x.bbox[3]-x.bbox[1]), reverse=True)
    return faces[0].embedding

def setup_sqlite_db():
    """Creates the local database table if it doesn't exist."""
    conn = sqlite3.connect(STUDENT_DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS enrolled_students (
            engagement_id TEXT PRIMARY KEY,
            batch_name TEXT,
            latest_s3_filename TEXT,
            embedding BLOB
        )
    ''')
    conn.commit()
    return conn

# ====================================================================
# 4. THE MASTER PIPELINE
# ====================================================================
def run_pipeline():
    # 1. Get dictionary of { ID: {"batch": "Java"} } from BQ
    bq_student_roster = fetch_active_students_from_bq()
    
    if not bq_student_roster:
        print("⚠️ No IDs fetched from BigQuery. Halting pipeline.")
        return

    # 2. Connect to AWS and Local SQLite
    print("⏳ Connecting to AWS S3 & Local SQLite Database...")
    try:
        s3_client = boto3.client('s3', aws_access_key_id=AWS_ACCESS_KEY, aws_secret_access_key=AWS_SECRET_KEY, region_name=AWS_REGION)
    except Exception as e:
        print(f"❌ AWS Connection failed.\nError: {e}")
        return

    if not os.path.exists(DOWNLOAD_DIR):
        os.makedirs(DOWNLOAD_DIR)

    conn = setup_sqlite_db()
    cursor = conn.cursor()

    # 3. Process the Data (The Delta Sync)
    print("\n🔍 Starting S3 Sync & Embedding Generation...")
    
    for engagement_id, student_info in bq_student_roster.items():
        batch_name = student_info["batch"]
        prefix = f"{engagement_id}/"

        try:
            # Check S3 for files
            response = s3_client.list_objects_v2(Bucket=BUCKET_NAME, Prefix=prefix)
            if 'Contents' not in response: 
                continue

            valid_images = [obj['Key'] for obj in response['Contents'] if f"{engagement_id}_StudentPicture_" in obj['Key'] and obj['Key'].lower().endswith(('.jpeg', '.jpg', '.png'))]
            if not valid_images: 
                continue

            # Find the newest image in S3
            valid_images.sort(reverse=True)
            latest_image_key = valid_images[0]

            # --- THE DELTA CHECK ---
            # Do we already have this exact file processed in our SQLite DB?
            cursor.execute("SELECT latest_s3_filename FROM enrolled_students WHERE engagement_id = ?", (engagement_id,))
            row = cursor.fetchone()
            
            if row and row[0] == latest_image_key:
                # We already have this image downloaded and embedded! Skip to save compute.
                continue

            # --- DOWNLOAD & EMBED ---
            print(f"   ⬇️ Downloading NEW/UPDATED photo for {engagement_id}...")
            local_path = os.path.join(DOWNLOAD_DIR, f"{engagement_id}_latest.jpeg")
            s3_client.download_file(BUCKET_NAME, latest_image_key, local_path)
            
            embedding = generate_student_embedding(local_path)
            
            if embedding is not None:
                # Save to database (REPLACE INTO automatically updates old photos or inserts new students)
                cursor.execute('''
                    REPLACE INTO enrolled_students (engagement_id, batch_name, latest_s3_filename, embedding)
                    VALUES (?, ?, ?, ?)
                ''', (engagement_id, batch_name, latest_image_key, embedding.astype(np.float32).tobytes()))
                conn.commit()
                print(f"   ✅ Processed & Saved {engagement_id} (Batch: {batch_name})")
            else:
                print(f"   ⚠️ No face detected in downloaded photo for {engagement_id}")

        except Exception as e:
            print(f"   ❌ Error processing {engagement_id}: {e}")

    conn.close()
    print(f"\n🎉 Pipeline Complete! {STUDENT_DB_PATH} is fully synced and ready for tomorrow's CCTV inference.")

if __name__ == "__main__":
    run_pipeline()