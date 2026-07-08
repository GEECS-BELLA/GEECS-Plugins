-- Migration: add optional curated value-alarm limits to the existing GEECS
-- MySQL database used by GeecsCAGateway.
--
-- Rollout is non-destructive for gateway startup: the gateway treats this table
-- as optional and starts with no value alarms when it is absent.  To remove the
-- overlay after testing:
--
--   DROP TABLE ca_alarm_limits;
--
-- Apply from a host with access to the GEECS DB, for example:
--
--   mysql --host=<db-host> --user=<db-user> --password <db-name> \
--     < deploy/ca_alarm_limits.sql

CREATE TABLE IF NOT EXISTS ca_alarm_limits (
  experiment VARCHAR(128) NOT NULL,
  device VARCHAR(128) NOT NULL,
  `variable` VARCHAR(128) NOT NULL,

  lolo DOUBLE NULL,
  low DOUBLE NULL,
  high DOUBLE NULL,
  hihi DOUBLE NULL,

  lolo_severity ENUM('MINOR', 'MAJOR', 'INVALID') NOT NULL DEFAULT 'MAJOR',
  low_severity  ENUM('MINOR', 'MAJOR', 'INVALID') NOT NULL DEFAULT 'MINOR',
  high_severity ENUM('MINOR', 'MAJOR', 'INVALID') NOT NULL DEFAULT 'MINOR',
  hihi_severity ENUM('MINOR', 'MAJOR', 'INVALID') NOT NULL DEFAULT 'MAJOR',

  hysteresis DOUBLE NULL,
  enabled BOOLEAN NOT NULL DEFAULT TRUE,
  description TEXT NULL,

  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    ON UPDATE CURRENT_TIMESTAMP,

  PRIMARY KEY (experiment, device, `variable`),
  CONSTRAINT fk_ca_alarm_limits_device
    FOREIGN KEY (device) REFERENCES device(name)
);
