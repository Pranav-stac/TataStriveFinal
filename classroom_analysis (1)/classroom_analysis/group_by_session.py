import json
from datetime import datetime, timedelta

def fuse_behavior_profile(activity_dict, attention_dict):
    """Merges detailed body activity and face attention into 4 high-level categories."""
    act = lambda k: activity_dict.get(k, 0.0)
    att = lambda k: attention_dict.get(k, 0.0)
    
    active = act("writing") + act("raising_hand")
    passive = act("listening") + att("focused") + att("partially_focused")
    idle = act("sitting") + act("standing") + att("distracted")
    unobservable = act("unknown") + att("not_visible")
    
    total = active + passive + idle + unobservable
    if total == 0: 
        return {"active_participation": 0, "passive_focus": 0, "disengaged_idle": 0, "unobservable": 100}
        
    return {
        "active_participation": round((active / total) * 100, 1),
        "passive_focus": round((passive / total) * 100, 1),
        "disengaged_idle": round((idle / total) * 100, 1),
        "unobservable": round((unobservable / total) * 100, 1)
    }

def get_probe_end_time(time_str):
    """Parses the probe start time, adds 30 minutes, and returns the true end time."""
    # Convert string (e.g. "09:37:32 AM") to a math-able datetime object
    dt = datetime.strptime(time_str, "%I:%M:%S %p")
    # Add exactly 30 minutes
    dt_end = dt + timedelta(minutes=30)
    # Convert back to clean string
    return dt_end.strftime("%I:%M:%S %p")

def generate_management_summary(input_json_path, output_json_path):
    print(f"📖 Reading raw probe data from {input_json_path}...")
    
    with open(input_json_path, 'r') as f:
        data = json.load(f)
        
    probes = data.get("hourly_probes", [])
    if not probes:
        print("⚠️ No probe data found.")
        return

    management_sessions = []
    current_session = None

    for i, probe in enumerate(probes):
        mode = probe["class_mode"]
        
        # 1. If this is the very first probe, start a new session
        if current_session is None:
            current_session = {
                "session_mode": mode,
                "start_time": probe["real_world_time"],
                "end_time": get_probe_end_time(probe["real_world_time"]), # <-- FIX
                "duration_probes": 1,
                "student_counts": [probe["student_count_corrected"]],
                "avg_engagements": [probe["avg_engagement"]],
                "raw_activities": [probe.get("activity_distribution", {})],
                "raw_attentions": [probe.get("attention_distribution", {})]
            }
        
        # 2. If the mode matches the current session, ADD it to the session
        elif mode == current_session["session_mode"]:
            current_session["end_time"] = get_probe_end_time(probe["real_world_time"]) # <-- FIX
            current_session["duration_probes"] += 1
            current_session["student_counts"].append(probe["student_count_corrected"])
            current_session["avg_engagements"].append(probe["avg_engagement"])
            current_session["raw_activities"].append(probe.get("activity_distribution", {}))
            current_session["raw_attentions"].append(probe.get("attention_distribution", {}))
            
        # 3. If the mode CHANGED, save the old session and start a new one
        else:
            management_sessions.append(current_session)
            current_session = {
                "session_mode": mode,
                "start_time": probe["real_world_time"],
                "end_time": get_probe_end_time(probe["real_world_time"]), # <-- FIX
                "duration_probes": 1,
                "student_counts": [probe["student_count_corrected"]],
                "avg_engagements": [probe["avg_engagement"]],
                "raw_activities": [probe.get("activity_distribution", {})],
                "raw_attentions": [probe.get("attention_distribution", {})]
            }
            
    # Don't forget to append the very last session!
    if current_session:
        management_sessions.append(current_session)

    # --- FORMATTING THE FINAL OUTPUT ---
    final_report = {
        "video_path": data["video_path"],
        "classroom": data.get("classroom", "Unknown"),
        "recording_date": data.get("recording_date", "Unknown"),
        "report_type": "Management Summary (Grouped by Mode)",
        "baseline_max_students": data.get("baseline_max_students", 0),
        "sessions": []
    }

    for session in management_sessions:
        avg_student_count = round(sum(session["student_counts"]) / len(session["student_counts"]))
        avg_engagement = round(sum(session["avg_engagements"]) / len(session["avg_engagements"]), 2)
        
        merged_act = {}
        merged_att = {}
        for act_dict in session["raw_activities"]:
            for k, v in act_dict.items(): merged_act[k] = merged_act.get(k, 0) + v
        for att_dict in session["raw_attentions"]:
            for k, v in att_dict.items(): merged_att[k] = merged_att.get(k, 0) + v
            
        num_probes = session["duration_probes"]
        merged_act = {k: v / num_probes for k, v in merged_act.items()}
        merged_att = {k: v / num_probes for k, v in merged_att.items()}
        
        behavior_profile = fuse_behavior_profile(merged_act, merged_att)
        
        final_report["sessions"].append({
            "session_mode": session["session_mode"],
            "time_window": f"{session['start_time']} to {session['end_time']}",
            "avg_student_count": avg_student_count,
            "overall_engagement_score": avg_engagement,
            "behavior_profile": behavior_profile
        })

    with open(output_json_path, 'w') as f:
        json.dump(final_report, f, indent=4)
        
    print(f"✅ Management Summary saved to {output_json_path}")

# Run it on your current file
if __name__ == "__main__":
    generate_management_summary("Progressive_results\\classroom\\analysis_results_v15_2_7nov_30min_probe\\class_dynamics_report.json", "Progressive_results\\classroom\\analysis_results_v15_2_7nov_30min_probe\\management_summary_report.json")