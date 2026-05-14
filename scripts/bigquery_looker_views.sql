-- Optional BigQuery views for Looker / Looker Studio dashboards.
-- Run in project tatastrive-269409, dataset tatastrive_analytics (or change names below).
-- Then point Looker sql_table_name at these views instead of raw tables if you prefer
-- warehouse-defined grain (same logic as Looker NDTs in looker/views/*).

CREATE OR REPLACE VIEW `tatastrive-269409.tatastrive_analytics.v_attendance_session` AS
SELECT
  center_id,
  report_file,
  ANY_VALUE(report_date) AS report_date,
  ANY_VALUE(session_date) AS session_date,
  ANY_VALUE(camera) AS camera,
  ANY_VALUE(source_video) AS source_video,
  ANY_VALUE(session_duration) AS session_duration,
  MAX(unique_people) AS unique_people,
  MAX(returning_count) AS returning_count,
  MAX(visitor_count) AS visitor_count,
  MAX(identified_students) AS identified_students,
  MAX(nf_presence) AS nf_presence,
  COUNT(*) AS source_row_count,
  COUNTIF(person_id IS NOT NULL) AS person_detail_rows
FROM `tatastrive-269409.tatastrive_analytics.attendance_reports`
GROUP BY center_id, report_file;

CREATE OR REPLACE VIEW `tatastrive-269409.tatastrive_analytics.v_engagement_report` AS
SELECT
  center_id,
  report_file,
  ANY_VALUE(report_date) AS report_date,
  ANY_VALUE(classroom) AS classroom,
  ANY_VALUE(recording_date_str) AS recording_date_str,
  ANY_VALUE(baseline_max_students) AS baseline_max_students,
  ANY_VALUE(report_type) AS report_type,
  COUNT(*) AS probe_or_session_row_count,
  AVG(avg_engagement) AS avg_engagement_across_detail_rows,
  AVG(student_count) AS avg_student_count_across_detail_rows
FROM `tatastrive-269409.tatastrive_analytics.engagement_reports`
GROUP BY center_id, report_file;
