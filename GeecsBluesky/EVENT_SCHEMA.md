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
`plan_args`, `plan_pattern_args`, `hints`, …) come from the plan.  GEECS
adds (phase 1 PR 2):

| Key | Emitted by | Meaning |
|---|---|---|
| `experiment` | `claim_scan` preprocessor | GEECS experiment name |
| `scan_number` / `scan_id` | `claim_scan` preprocessor | Day-scoped GEECS scan number (`scan_id` is the Bluesky display field) |
| `scan_folder` | `claim_scan` preprocessor | Absolute path of the claimed `scans/ScanNNN/` folder |
| `scan_tag` | `claim_scan` preprocessor | `{year, month, day, number, experiment}` — the `geecs_data_utils.ScanTag` |
| `geecs_scalar_headers` | `scalar_headers` preprocessor | Event key → legacy `Device Variable` header for every staged device (the s-file and the browser's display names) |
| `trigger_profile` | the bound plan | The trigger profile that drove the shots |
| `shots_per_step` | the bound plan | Rows per position (`1` for `count`, whose `num` is the shot count) |
| `description`, `background` | the client (`md`) | The preset's description (ScanInfo's `ScanStartInfo`) and background flag |
| `geecs` | the client (`md`) | Provenance only: `{preset, submission}` — never a worker instruction |

Every run the worker opens claims a scan number; a run without one is not a
GEECS scan.

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
| `bin_number` | The scan step the row belongs to, from 1 (the GEECS `per_step`; every row of a `count` is bin 1) — the s-file's `Bin #` |

A detector listed as `X.scalars` (the scalars-only view, `save_images:
false` in a preset) contributes the same `<det>-<variable>` and
`<det>-acq_timestamp` columns and no `-nonscalar_save_path`.

Native files are named with the row's stamp
(`<Device>_<acq_timestamp>.png`, `geecs_data_utils.native_files`) and join
to rows by that stamp — never by position.

## Event stream `baseline`

Every subscribed scalar of the experiment, read at the open and the close
of every run (`SupplementalData(baseline=namespace.telemetry())`): each
scalar-only device's columns and each detector's scalar signals, under the
same keys as in `primary`.  Two rows per run.  Which of them should be
per-event monitors instead is decided from measurement (§10.4), not up
front.

## Legacy `Device Variable` headers

Every device carries `_column_headers` — event data-key → the GEECS
`Device Variable` header (`UC_Wavemeter Wavelength (nm)`); the
`scalar_headers` preprocessor merges the staged devices' maps into the
start document's `geecs_scalar_headers`, which the s-file callback
(`callbacks.py`, from the run's own events at the stop document) and the
offline re-export (`geecs_data_utils.write_scalar_files_from_tiled`) read.
