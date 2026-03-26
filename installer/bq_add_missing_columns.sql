-- BigQuery schema migration: add missing columns to attendance_reports
-- Run this in BigQuery Console if you get "no such field: engagement_id" error.
-- https://console.cloud.google.com/bigquery?project=tatastrive-269409

ALTER TABLE `tatastrive-269409.tatastrive_analytics.attendance_reports`
  ADD COLUMN IF NOT EXISTS engagement_id STRING,
  ADD COLUMN IF NOT EXISTS batch STRING,
  ADD COLUMN IF NOT EXISTS confidence_score FLOAT64,
  ADD COLUMN IF NOT EXISTS present_last_7_days INT64,
  ADD COLUMN IF NOT EXISTS last_present_date STRING,
  ADD COLUMN IF NOT EXISTS nf_presence INT64;
