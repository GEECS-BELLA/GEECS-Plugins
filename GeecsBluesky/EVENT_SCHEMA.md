# GEECS Bluesky documents — what a run carries

The native shape (phase 1 of the rebuild, GEECS-Plugins#807; plan of
record `Planning/native_bluesky/03_clean_room_rebuild.md` §4): a run is
whatever the stock `bluesky.plans` verb emits over the namespace's
devices, and every GEECS fact rides in the places Bluesky already has for
it.  There is no GEECS schema version any more — consumers read the
documents as Bluesky documents.  This file lists the GEECS-specific keys
those documents carry.

The v1 schema (funnel-era: `geecs_event_schema`, `acquisition_mode`,
`shot_id` / `shot_offset` / `valid` companion columns, `bin_number`) was
deleted with the funnel in phase 1 PR 1; scans before that carry it and
`geecs_data_utils.tiled_schema` still reads them.

## Start document

The stock keys (`plan_name`, `detectors`, `motors`, `num_points`,
`plan_args`, `hints`, …) come from the plan.  GEECS adds, from the plan
layer (phase 1 PR 2, the `claim_scan` preprocessor and the ScanInfo
callback — **not yet emitted** on this branch):

| Key | Meaning |
|---|---|
| `experiment` | GEECS experiment name |
| `scan_number` / `scan_id` | Day-scoped GEECS scan number (`scan_id` is the Bluesky display field) |
| `scan_folder` | Absolute path of the claimed `scans/ScanNNN/` folder |
| `geecs` | Provenance only: the client's request (preset name, plan call) — never a worker instruction |

## Descriptor: configuration

Every `GeecsDetector` records its drain offset
(`<det>-drain_offset`, seconds — the calibrated edge-to-stamp latency, `0.0`
until measured, §4.F); the `ShotControl` device records the standing
trigger state (`shot_control-state`) when it is read.

## Event stream `primary`

| Column | Meaning |
|---|---|
| `<det>-<variable>` | The detector's DB-subscribed scalars, one column each (`safe_name`-mangled: `uc_amp4_ir_input-meancounts`) |
| `<det>-acq_timestamp` | The shot stamp: the join key for that detector's files and for cross-device alignment after the drain offset (§11.3) |
| `<det>-nonscalar_save_path` | The directory the detector's native files landed in this run — present only when the detector saved natively (`geecs_data_utils.tiled_schema.COMPANION_SUFFIXES` names the suffix) |
| `<device>-<variable>` | A scalar-only device's subscribed readbacks (`CaSnapshotReadable`) |
| `<device>-<settable>-position` / `-readback` | A settable child's readback when the DB subscribes it (`CaMotor` / `CaSettable`), and the scan motor's column |

Native files are named with the row's stamp
(`<Device>_<acq_timestamp>.png`, `geecs_data_utils.native_files`) and join
to rows by that stamp — never by position.

## Telemetry (phase 1 PR 2)

Every subscribed scalar of the experiment rides in the run through
`SupplementalData` (`baseline` at open/close, `monitors` for the changing
few) — an experiment-config fact, derived from measurement (§10.4).

## Legacy `Device Variable` headers

Every device carries `_column_headers` — event data-key → the GEECS
`Device Variable` header (`UC_Wavemeter Wavelength (nm)`) — for the s-file
exporter (`sfile_callback.py`, from Tiled at the stop document).  The
start-document map the exporter reads (`geecs_scalar_headers`) is emitted
by the plan layer (PR 2).
