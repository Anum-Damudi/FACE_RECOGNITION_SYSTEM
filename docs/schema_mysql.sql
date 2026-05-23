-- Optional MySQL / MariaDB schema equivalent to MongoDB detection_logs documents.
-- Use if your org standardizes on SQL instead of pymongo.

CREATE DATABASE IF NOT EXISTS surveillance_alerts
  CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

USE surveillance_alerts;

CREATE TABLE IF NOT EXISTS detection_logs (
  id            BIGINT UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY,
  matched       TINYINT(1) NOT NULL,
  criminal_name VARCHAR(255) NOT NULL DEFAULT '',
  crime_type    VARCHAR(255) NOT NULL DEFAULT '',
  confidence    DOUBLE NOT NULL,
  distance      DOUBLE NOT NULL,
  alert_level   VARCHAR(32) NOT NULL,
  screenshot_path VARCHAR(512) NOT NULL DEFAULT '',
  detected_at   DATETIME(6) NOT NULL,
  source        VARCHAR(64) NOT NULL DEFAULT 'api',
  extra_json    JSON NULL,
  KEY idx_detected_at (detected_at),
  KEY idx_matched (matched),
  KEY idx_alert (alert_level)
) ENGINE=InnoDB;

-- Example row (HIGH_ALERT after cosine distance < threshold)
-- INSERT INTO detection_logs
-- (matched, criminal_name, crime_type, confidence, distance, alert_level, screenshot_path, detected_at, source)
-- VALUES (1, 'Jane Doe', 'Armed robbery', 0.87, 0.13, 'HIGH_ALERT', '/path/logs/MATCH_...jpg', UTC_TIMESTAMP(6), 'api');
