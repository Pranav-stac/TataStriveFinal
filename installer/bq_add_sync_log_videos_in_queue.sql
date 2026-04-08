-- Add videos_in_queue to sync_log (folder-listener queue depth at sync time)
-- Run in BigQuery Console if sync_log was created before this column existed.
-- https://console.cloud.google.com/bigquery?project=tatastrive-269409

ALTER TABLE `tatastrive-269409.tatastrive_analytics.sync_log`
  ADD COLUMN IF NOT EXISTS videos_in_queue INT64;
