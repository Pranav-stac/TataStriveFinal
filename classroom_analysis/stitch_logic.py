import json
import math
import numpy as np
from scipy.spatial.distance import cosine
from collections import defaultdict

def calculate_pixel_distance(p1, p2):
    if not p1 or not p2: return 9999.0
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def perform_hierarchical_stitching(json_path, similarity_threshold=0.75, max_time_gap=600, max_pixel_dist=200):
    """
    Performs post-processing stitching on the generated index file.
    Returns: A mapping dict {original_id: merged_root_id}
    """
    print(f"\n[*] Loading stitching data from {json_path}...")
    if not json_path or not hasattr(json_path, 'read') and not isinstance(json_path, str):
        # Handle case where file might not exist yet
        return {}

    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"[!] Stitching Load Error: {e}")
        return {}

    # Filter invalid
    valid_tracks = [d for d in data if d.get('embedding') and d.get('last_centroid')]
    valid_tracks.sort(key=lambda x: x['first_seen'])
    
    print(f"[*] Stitching Analysis: {len(valid_tracks)} fragments...")
    
    parent_map = {t['face_id']: t['face_id'] for t in valid_tracks}
    
    for i in range(len(valid_tracks)):
        track_a = valid_tracks[i]
        id_a = track_a['face_id']
        
        # Find root
        root_a = parent_map[id_a]
        while parent_map[root_a] != root_a:
            root_a = parent_map[root_a]
            
        for j in range(i + 1, len(valid_tracks)):
            track_b = valid_tracks[j]
            id_b = track_b['face_id']
            
            # Time Check
            time_gap = track_b['first_seen'] - track_a['last_seen']
            if time_gap < 0: continue # Overlap
            if time_gap > max_time_gap: break 
            
            # SPATIAL CHECK
            dist = calculate_pixel_distance(track_a['last_centroid'], track_b['start_centroid'])
            if dist > max_pixel_dist: continue 
            
            # VISUAL CHECK
            sim = 1.0 - cosine(track_a['embedding'], track_b['embedding'])
            
            if sim > similarity_threshold:
                # Merge logic
                root_b = id_b
                while parent_map[root_b] != root_b:
                    root_b = parent_map[root_b]
                
                if root_b != root_a:
                    parent_map[root_b] = root_a # Point B's root to A's root
                    print(f"[+] MERGE: ID {id_b} -> ID {root_a} (Sim: {sim:.2f}, Dist: {dist:.0f}px, Gap: {time_gap:.0f}s)")

    # Flatten the map so every ID points directly to its ultimate root
    final_map = {}
    unique_roots = set()
    for original_id in parent_map.keys():
        root = original_id
        while parent_map[root] != root:
            root = parent_map[root]
        final_map[original_id] = root
        unique_roots.add(root)

    print(f"[OK] STITCHING COMPLETE: {len(valid_tracks)} IDs -> {len(unique_roots)} Unique Students")
    return final_map