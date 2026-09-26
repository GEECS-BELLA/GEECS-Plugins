# GEECS Bluesky documents — what a run carries

The native shape (phase 1 of the rebuild, GEECS-Plugins#807): a run is
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
| `shot_clock` / `shot_clock_column` | the bound plan (gated) | The device whose `acq_timestamp` is the shot id, and the row column carrying it — what the s-file's join keys on |
| `trigger_profile` | the bound plan | The trigger profile that drove the shots |
| `native_image_save` | the bound plan | The run's LabVIEW-files **switch** — the preset's value, else the experiment default (#738) — not a record of what was written. It reaches a plugin-backed camera only as a strict full detector (a `.scalars` view, a non-essential stream and a gated batch's plugin-backed cameras write no native files whatever it says; a device without a file plugin always writes them, as a strict detector or a gated essential). Whether a device wrote is the presence of its `<det>-nonscalar_save_path` column, in `primary` (strict) or `shots` (gated) |
| `shots_per_step` | the bound plan | Rows per position (`1` for `count`, whose `num` is the shot count) |
| `description`, `background` | the client (`md`) | The preset's description (ScanInfo's `ScanStartInfo`) and background flag |
| `geecs` | the client (`md`) | Provenance only: `{preset, submission}` — never a worker instruction |

Every run the worker opens claims a scan number; a run without one is not a
GEECS scan.

## Predetermined moving scans: `sweep`

The public moving plan is `sweep`; `count` remains motionless and `optimize`
is adaptive. A sweep start carries `sweep`, the validated JSON Sweep payload
with expanded namespace bindings. This is execution metadata, distinct from
`geecs` provenance. Relative coordinates in this payload remain offsets from
the readback after staging; events contain actual readbacks. Relative axes
restore before the run closes and before unstage; a failed restore fails the
run. A process kill or loss of hardware communication cannot guarantee reset.

`motors` preserves authored axis order; `num_points` is the full trajectory
length, with `num_intervals = num_points - 1`. `shape` is the per-axis lengths
for an axes/product grid, otherwise one trajectory dimension. `snaking`
has one flag per motor (all false outside a grid). No rectilinear plot hint is
emitted: arbitrary spacing and multiple shots per cell do not satisfy
Bluesky LiveGrid's uniform, single-event cell assumptions.
`extents` gives each axis's numeric bounds in its
requested frame. `hints.dimensions` groups correlated axes into one dimension.
`shots_per_step` retains the acquisition meaning above.

`sweep_first_axis` is `[start, end, first_increment]`, the legacy ScanInfo
projection: the first axis's spacing list for axes sweeps, otherwise its
ordered pattern points. Repeated values are retained; a singleton's increment
is zero. It is a lossy projection, not a recipe for replaying a nonuniform or
multidimensional scan. Relative projections are offsets, as for old rel_scan.
A pattern may revisit its initial first-axis coordinate, so equal legacy
Start/End values do not imply a motionless run; the Sweep payload is authoritative.

Data Utils classifies multi-axis axes/product as GRID, every other moving trajectory as
1D regardless of axis count. It retains stock `plan_pattern` and older funnel
readers for historical runs. No existing event column changes meaning.

## Descriptor: configuration

Every `GeecsDetector` records its drain offset
(`<det>-drain_offset`, seconds — the calibrated edge-to-stamp latency, `0.0`
until measured); the `ShotControl` device records the standing
trigger state (`shot_control-state`) when it is read.

## Event stream `primary`

| Column | Meaning |
|---|---|
| `<det>-<variable>` | The detector's DB-subscribed scalars, one column each (`safe_name`-mangled: `uc_amp4_ir_input-meancounts`) |
| `<det>-acq_timestamp` | The shot stamp: the join key for that detector's files and for cross-device alignment after the drain offset |
| `<det>` | A plugin-backed camera's frames (#806): an external `STREAM:` key — the row's frame is the stream datum's index into `ScanNNN/<device>/<device>.h5` (`/entry/data/data`); `<det>-<variable>` for a second image variable. Absent from a partial row (see below) |
| `<det>-nonscalar_save_path` | The directory the detector's native files landed in this run — present only when the detector saved natively (`geecs_data_utils.tiled_schema.COMPANION_SUFFIXES` names the suffix) |
| `<device>-<variable>` | A scalar-only device's subscribed readbacks (`CaSnapshotReadable`) |
| `<device>-<settable>-position` / `-readback` | A settable child's readback when the DB subscribes it (`CaMotor` / `CaSettable`), and the scan motor's column |
| `bin_number` | The scan step the row belongs to, from 1 (the GEECS `per_step`; every row of a `count` is bin 1) — the s-file's `Bin #` |

A device listed as `X.scalars` (the scalars-only view every namespace
device carries; `save_images: false` in a preset) contributes the same
columns as `X` — for a detector `<det>-<variable>` and
`<det>-acq_timestamp` with no `-nonscalar_save_path`.

**Partial rows.** A shot on which a detector produced no frame within the
timeout is still a row: every other device's columns are real, the
frameless detector's `<det>-<variable>` and `<det>-acq_timestamp` read
`NaN` (its monitor cache would otherwise carry the previous shot), and
the row references **no** frames for any plugin-backed camera (one datum
per external key per event, or none). The plan then takes one more shot
for the step, so every step has its full quota of complete rows; the
s-file carries the `NaN` holes.

Native files are named with the row's stamp
(`<Device>_<acq_timestamp>.png`, `geecs_data_utils.native_files`) and join
to rows by that stamp — never by position.

## Event stream `shots` (a gated run)

A gated run has no `primary` events at all: the box free-runs and the
plugin-backed cameras count the frames they write, so `primary` carries
only their stacks as stream datums.  The per-shot record is the sampler's
`shots` stream instead — **one event per shot**,
arriving as event *pages* from a `collect`: the latest value of every
non-plugin subscribed signal, the scanned motors' readbacks,
`bin_number`, and the clock device's `acq_timestamp`, which is the shot id
the sampler ticked on.  A plugin-backed camera's own scalars are **not**
repeated here; they ride in its stack as per-frame attributes
(`<ophyd>-hdf-<variable>-<scalar>`, GeecsPvaGateway >= 0.9).  A member
with a stamp of its own (a triggered device without a plugin, or its
view) is read once its `<det>-acq_timestamp` lands within half a period
of the clock's, not at the tick (its stamp lands after the clock's
whenever its device is slower); on a shot it missed, its numeric columns
read **`NaN`** and its `<det>-acq_timestamp` is `NaN` too — never the
previous shot's values (Scan015 of 26_0925 recorded every HASO row one
frame late before this rule).  A string column of such a member (the
save path below) is a run-long constant and stays.

| Column | Meaning |
|---|---|
| `<det>-nonscalar_save_path` | A **native-saving essential**'s save directory (a device without a file plugin — a LabVIEW-native camera, a DAQ or wavefront sensor with its own file writer, a scope with every capture channel disabled; admitted as a gated essential 2026-09-25): the same companion column a strict row carries, here a **run-long constant** — LabVIEW's saving is switched on at the run's first prepare and off at `unstage`, never per step. Its scalars and its own `<det>-acq_timestamp` ride in the row like any non-plugin device's (it may be the clock). Its files are named by that stamp and join by it (`geecs_data_utils.native_files`); a shot on which it dropped a frame is a row with no file — **no retake**, as the LabVIEW scanner had it. The stack check matches every row's stamp to a file in that directory at the stop (`geecs_data_utils.native_files.native_file_keys`) and appends a `native files check` line to `scan.log` — rows without a file and file stamps without a row counted apart (WARNING on either, never a failure). Absent for a plugin-backed camera (its stack is its record) and for a `.scalars` view |

An additive column convention, not a schema change: a reader that never
looked for the column in `shots` sees the rows it saw before.

The s-file of such a run is the `shots` rows with each stack's per-frame
columns joined on by offset-corrected stamp
(`geecs_data_utils.shot_join`): the attribute
`<ophyd>-hdf-<variable>-frame_acq_timestamp` becomes the column
`<ophyd>-acq_timestamp` and a subscribed scalar becomes
`<ophyd>-<scalar>` — the same spellings a strict row uses, so one header
map renames both.  One row per essential shot: a frame with no shot inside
the join window stays in the stack and in Tiled and is left out of the
s-file.  The same join adds a **non-essential** camera's columns
(`<name>_stream`) to either mode's rows.

## Event stream `baseline`

Every subscribed scalar of the experiment, read at the open and the close
of every run (`SupplementalData(baseline=namespace.telemetry())`): each
scalar-only device's columns and each detector's scalar signals, under the
same keys as in `primary`.  Two rows per run.  Which of them should be
per-event monitors instead is decided from measurement (#929), not up
front.

## Legacy `Device Variable` headers

Every device carries `_column_headers` — event data-key → the GEECS
`Device Variable` header (`UC_Wavemeter Wavelength (nm)`); the
`scalar_headers` preprocessor merges the staged devices' maps into the
start document's `geecs_scalar_headers`, which the s-file callback
(`callbacks.py`, from the run's own rows at the stop document) and the
offline re-export (`geecs_data_utils.write_scalar_files_from_tiled`) read.
Both take their rows from `primary` when it has events and from `shots`
otherwise, and both run the same join, so a re-export checks the live path
rather than re-implementing it.


## Native optimization

`plan_name: optimize` opens one run. `primary` stays one row per strict shot;
`bin_number` is the one-based adaptive iteration. Start metadata adds
`optimizer_config`, `max_iterations`, `optimization_variables`,
`optimization_objectives`, `optimization_move_targets`, and the resolved
`OptimizerConfig` as JSON text under `geecs.optimizer_json` (provenance only).
Bluesky forbids dots and slashes in document keys at any depth; event name
components use URI escaping with `~` as the escape marker (Tiled SQL forbids
`%`): `.`, `/`, `%`, `~`, `-` become `~2E`, `~2F`, `~25`, `~7E`, `~2D`.
Colons and underscores remain literal. The shared `optimization_events` codec
serves the worker and scanner. Columns must fit 60 characters after escaping
(Tiled adds `ts_` timestamp columns with a 63-character SQL limit) and be distinct
ignoring case; violations refuse before scan claim. The complete config retains
its original names in JSON text.

The shared `OptimizationRole` enum defines the role prefixes below.
The fixed `optimization` stream emits once per evaluated iteration, all
numeric columns: `iteration`, `proposal:<variable>`, `measured:<variable>`,
`output:<measurement-or-derived-name>`, `n_valid_shots:<measurement>`,
`best:<variable-or-objective>`, and `best_move:<Device:Variable>`.
Missing/failed measurements and absent best points are NaN; the scanner
serializes these as JSON null. `best_move` records physical positions while
relative pseudos still have their staged offsets. Relative pseudos restore
on unstage; the operator can later submit these physical positions through
Set to best. The scanner offers this only after success, for fifteen minutes,
and invalidates it after a service-submitted move or action. It does not monitor
out-of-band gateway writes. Observables-only BAX and multiobjective problems have no single
best point, so `on_finish: best` restores initial positions instead.

Acquisition is rewindable up to the bin's evaluation boundary. An immediate
pause during acquisition replays the acquired part of the bin; only the
latest complete quota feeds evaluation. Evaluation and Xopt tell are not
replayed. Deferred pause lands at the next iteration checkpoint. Failed
measurement rows stay in the Xopt dump with `xopt_error: true` and are excluded
from model fitting. The dump is written before run close into the folder
from the emitted start document, including abort cleanup after an in-flight
calculation has finished.
