-- Migration: archive deadband (DBE_LOG) for GeecsCAGateway.
--
-- The gateway flags a float readback's change DBE_LOG -- the event the
-- Archiver Appliance subscribes to -- only when it moves more than the
-- variable's `archive_deadband` from the last archived value (PV_CONTRACT.md
-- §4).  Live (DBE_VALUE) clients see every change regardless.  OPTIONAL and
-- non-destructive: a database without the column reads every variable as
-- "no archive deadband", the behaviour before it existed.
--
-- VARCHAR, like `tolerance` beside it: the LabVIEW editor writes '' into
-- instance rows, which GEECS-Core reads as unset -- an instance row's unset
-- value falls back to its devicetype row (the one field that does not
-- inherit wholesale).  NULL, '' and a non-finite value all mean unset;
-- '0' on an instance row means "archive every change" for that device.
--
-- Apply from a host with access to the GEECS DB, for example:
--
--   mysql --host=<db-host> --user=<db-user> --password <db-name> \
--     < deploy/archive_deadband.sql
--
-- [APPLIED 2026-10-19 on the reference deployment's DB.]  Not idempotent:
-- check first, and skip a table that already has the column:
--   SELECT TABLE_NAME FROM information_schema.COLUMNS
--     WHERE TABLE_SCHEMA = DATABASE() AND COLUMN_NAME = 'archive_deadband';
ALTER TABLE devicetype_variable
  ADD COLUMN archive_deadband VARCHAR(20) DEFAULT NULL
  COMMENT 'Archive deadband (DBE_LOG): post to the archiver only when |change| exceeds this; NULL = unset',
  ALGORITHM=INSTANT;
ALTER TABLE variable
  ADD COLUMN archive_deadband VARCHAR(20) DEFAULT NULL
  COMMENT 'Archive deadband (DBE_LOG): post to the archiver only when |change| exceeds this; NULL = unset',
  ALGORITHM=INSTANT;

-- Setting a value: at the devicetype level, for the variables that log
-- noise profusely (idle supplies, analog inputs).  Size it from archived
-- data: a few ADC counts above the step-to-step noise, far below any real
-- setting change.  Example (the reference deployment's idle CaenELS noise is
-- ~0.4 mV steps):
--   UPDATE devicetype_variable SET archive_deadband = '0.002'
--     WHERE devicetype = 'CaenELSEasyDriver' AND name = 'Voltage';
-- Restart the gateway afterwards (`…:cagateway:restart`) to pick it up.
