import json
from datetime import datetime, timedelta


def fuse_behavior_profile(activity_dict, attention_dict):
    """Merge low-level activity/attention percentages into 4 management buckets."""
    act = lambda k: activity_dict.get(k, 0.0)
    att = lambda k: attention_dict.get(k, 0.0)

    active = act("writing") + act("raising_hand")
    passive = act("listening") + att("focused") + att("partially_focused")
    idle = act("sitting") + act("standing") + att("distracted")
    unobservable = act("unknown") + att("not_visible")

    total = active + passive + idle + unobservable
    if total == 0:
        return {
            "active_participation": 0,
            "passive_focus": 0,
            "disengaged_idle": 0,
            "unobservable": 100,
        }

    return {
        "active_participation": round((active / total) * 100, 1),
        "passive_focus": round((passive / total) * 100, 1),
        "disengaged_idle": round((idle / total) * 100, 1),
        "unobservable": round((unobservable / total) * 100, 1),
    }


def _get_probe_end_time(time_str, fallback_minutes=30):
    """Parse probe start time and return end time string after fallback_minutes."""
    try:
        dt = datetime.strptime(time_str, "%I:%M:%S %p")
        dt_end = dt + timedelta(minutes=fallback_minutes)
        return dt_end.strftime("%I:%M:%S %p")
    except Exception:
        return "Unknown"


def generate_management_summary(
    input_json_path,
    output_json_path,
    probe_interval_sec=1800,
    probe_duration_sec=None,
):
    """
    Generate grouped-by-mode management summary from class dynamics report.

    ``probe_interval_sec`` is the wall-clock cadence between probes (default
    1800s / 30 min). Time-window end times advance by this value, NOT by the
    5-minute sample window inside each probe.

    ``probe_duration_sec`` is accepted as a deprecated alias for backwards
    compatibility with callers that previously passed the sample-window length.
    """
    # Backwards compatibility: old callers passed probe_duration_sec meaning the
    # sampling cadence (1800) rather than the sample window (300). Honour the
    # value but log only via the parameter alias.
    if probe_duration_sec is not None:
        probe_interval_sec = probe_duration_sec

    with open(input_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    probes = data.get("hourly_probes", [])
    if not probes:
        return None

    management_sessions = []
    current_session = None
    fallback_minutes = max(1, int(probe_interval_sec / 60))

    for probe in probes:
        mode = probe.get("class_mode", "Unknown")
        rt = probe.get("real_world_time", "Unknown")

        if current_session is None:
            current_session = {
                "session_mode": mode,
                "start_time": rt,
                "end_time": _get_probe_end_time(rt, fallback_minutes),
                "duration_probes": 1,
                "student_counts": [probe.get("student_count_corrected", 0)],
                "avg_engagements": [probe.get("avg_engagement", 0.0)],
                "raw_activities": [probe.get("activity_distribution", {})],
                "raw_attentions": [probe.get("attention_distribution", {})],
            }
            continue

        if mode == current_session["session_mode"]:
            current_session["end_time"] = _get_probe_end_time(rt, fallback_minutes)
            current_session["duration_probes"] += 1
            current_session["student_counts"].append(probe.get("student_count_corrected", 0))
            current_session["avg_engagements"].append(probe.get("avg_engagement", 0.0))
            current_session["raw_activities"].append(probe.get("activity_distribution", {}))
            current_session["raw_attentions"].append(probe.get("attention_distribution", {}))
            continue

        management_sessions.append(current_session)
        current_session = {
            "session_mode": mode,
            "start_time": rt,
            "end_time": _get_probe_end_time(rt, fallback_minutes),
            "duration_probes": 1,
            "student_counts": [probe.get("student_count_corrected", 0)],
            "avg_engagements": [probe.get("avg_engagement", 0.0)],
            "raw_activities": [probe.get("activity_distribution", {})],
            "raw_attentions": [probe.get("attention_distribution", {})],
        }

    if current_session:
        management_sessions.append(current_session)

    final_report = {
        "video_path": data.get("video_path", ""),
        "classroom": data.get("classroom", "Unknown"),
        "recording_date": data.get("recording_date", "Unknown"),
        "report_type": "Management Summary (Grouped by Mode)",
        "baseline_max_students": data.get("baseline_max_students", 0),
        "sessions": [],
    }

    for session in management_sessions:
        avg_student_count = round(sum(session["student_counts"]) / len(session["student_counts"]))
        avg_engagement = round(sum(session["avg_engagements"]) / len(session["avg_engagements"]), 2)

        merged_act = {}
        merged_att = {}
        for act_dict in session["raw_activities"]:
            for k, v in act_dict.items():
                merged_act[k] = merged_act.get(k, 0) + v
        for att_dict in session["raw_attentions"]:
            for k, v in att_dict.items():
                merged_att[k] = merged_att.get(k, 0) + v

        num_probes = session["duration_probes"]
        merged_act = {k: v / num_probes for k, v in merged_act.items()}
        merged_att = {k: v / num_probes for k, v in merged_att.items()}
        behavior_profile = fuse_behavior_profile(merged_act, merged_att)

        final_report["sessions"].append({
            "session_mode": session["session_mode"],
            "time_window": f"{session['start_time']} to {session['end_time']}",
            "avg_student_count": avg_student_count,
            "overall_engagement_score": avg_engagement,
            "behavior_profile": behavior_profile,
        })

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=4)

    return output_json_path
