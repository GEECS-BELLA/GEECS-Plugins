# GEECS Bluesky documents — what a run carries

A run is whatever the registered plan (`count`, `sweep` or `optimize` —
`plan_names.GEECS_PLAN_NAMES`) emits over the namespace's devices, and
every GEECS fact rides in the places Bluesky already has for it.  There
is no GEECS schema version — consumers read the documents as Bluesky
documents.  This file lists the GEECS-specific keys those documents
carry.

Older runs carry the v1 schema (`geecs_event_schema`, `acquisition_mode`,
`shot_id` / `shot_offset` / `valid` companion columns);
`geecs_data_utils.tiled_schema` reads them.

## Start document

The stock keys (`plan_name`, `detectors`, `motors`, `num_points`,
`plan_args`, `plan_pattern_args`, `hints`, …) come from the plan.  GEECS
adds:

| Key | Emitted by | Meaning |
|---|---|---|
| `experiment` | `claim_scan` preprocessor | GEECS experiment name |
| `scan_number` / `scan_id` | `claim_scan` preprocessor | Day-scoped GEECS scan number (`scan_id` is the Bluesky display field) |
| `scan_folder` | `claim_scan` preprocessor | Absolute path of the claimed `scans/ScanNNN/` folder |
| `scan_tag` | `claim_scan` preprocessor | `{year, month, day, number, experiment}` — the `geecs_data_utils.ScanTag` |
| `geecs_scalar_headers` | `scalar_headers` preprocessor | Event key → legacy `Device Variable` header for every staged device (the s-file and the browser's display names) |
| `shot_clock` / `shot_clock_column` | the bound plan (gated) | The device whose `acq_timestamp` is the shot id, and the row column carrying it — what the s-file's join keys on |
| `layout` | the bound plan (strict or fly `count` / `sweep`; `optimize`) | `"shot-table/1"`: `primary` is the shot table and every per-shot device is in its own stream (below). Absent on a gated run and every earlier run (an earlier `optimize` run included); readers branch on it (`geecs_data_utils.shot_table`) |
| `per_shot_streams` | the bound plan (shot-table) | Every per-shot device's stream, `<device>_stream` (ophyd name), in the run's device order |
| `required` | the bound plan (shot-table) | The per-shot devices (ophyd names) whose record makes a shot complete: waited on and refired for. Every per-shot device unless the request said `required=` (an empty list: none) |
| `shot_clock_column` | the bound plan (shot-table) | The row clock: `witness_stamp` when the witness is read (or the run is fly), else `fire_stamp` (both Unix seconds) |
| `stream_meta` | the bound plan (shot-table) | `{<stream>: {sync_class, stale_after}}`: every strict device stream is `synchronous` (it acquires on the fire), `stale_after` `null`. Fly: a Software-source native camera's stream (pressed at each tick) is `synchronous`, `stale_after` `null`; every other stream `asynchronous`, `stale_after` the tick period (free-running; the join's rule is in "A fly run" below) |
| `clock` | the bound plan (fly) | `"soft-tick"`: the rows are a software timer's ticks (`acquisition: fly`), not counted shots — the `witness_*` columns are the tick's (below). Absent on every other run |
| `shot_period` | the bound plan | Strict: the requested rep-rate throttle, seconds (absent when none). Fly: the tick period, always written (default 1 s) |
| `non_essential` | the bound plan (gated; an earlier `optimize`) | The devices streamed for the run beside the rows, by owner name (`<name>_stream`, below); a gated run's are the per-shot devices not `required` |
| `trigger_profile` | the bound plan | The trigger profile that drove the shots; `null` on a fly run (it drives no box and resolves no default profile; a named scan-mode profile is refused) |
| `use_scan_mode` | the bound plan | Whether the run drove GEECS scan mode — the trigger profile's `use_scan_mode` flag: the devices were told the scan (`ScanStart`) and every shot (its number and save path, before the fire) and released at the close (`ScanEnd`), as under Master Control; each `primary` row then carries `announced_shot` |
| `native_image_save` | the bound plan | The run's LabVIEW-files **switch** — the preset's value, else the experiment default — not a record of what was written. It reaches a plugin-backed camera only as a strict full detector (a `.scalars` view, a gated run's stream and a gated batch's plugin-backed cameras write no native files whatever it says; a device without a file plugin always writes them, as a strict detector, a gated required device or a streamed one). Whether a device wrote is the presence of its `<det>-nonscalar_save_path` column, in `primary` (an earlier strict run), `shots` (gated) or its own `<det>_stream` (a shot-table run's device without a plugin, a gated run's streamed device) — a shot-table run's plugin-backed camera in its stream descriptor's configuration instead (`<det>-nonscalar_save_path` under the object `<det>`, a run-long constant: the stream is datum-only). The s-file never carries the column |
| `background_telemetry` | the bound plan | Whether the run read the background telemetry into its rows: the preset's value, else the experiment default; `False` also when the namespace carries no telemetry set at all (a run whose every candidate is its own device still says `True`, with nothing dropped and no extra column) |
| `background_dropped` | the bound plan (`background_wrapper`) | GEECS device names the background probe dropped for this run — no answer within its budget (a PV the gateway does not serve, a failed connect or describe); absent when the run read no background |
| `background_probe_error` | the bound plan (`background_wrapper`) | Present only when the probe itself failed (an unexpected error, logged at ERROR — never a member's failure, which is its own drop): `Type: message`, and the run opened with no background columns |
| `witness` | the bound plan (`registry.run_witness`) | The run's witness, **recorded only** (never waited on; it cannot refuse, fail or end a run): `null` when no witness is configured (none in `native_devices.yaml`); else `{device, domain, read}` — `read: true` when each strict `primary` row carries `witness_count`, `witness_stamp` and `witness_epoch` — and `read: false` with a `reason`: `down` (no answer within 5 s at the start; the run proceeds), a gated run (not read yet), a box-less profile (nothing fired for it to count), or `no witness for domain main` (`device: null`: one is configured, for another domain). A `note` (`no triggered device: the row may record the previous shot's count`) when the run has no triggered device, still `read: true`. `optimize` records it as the strict verbs do |
| `shots_per_step` | the bound plan | Rows per position (`1` for `count`, whose `num` is the shot count) |
| `description`, `background` | the client (`md`) | The preset's description (ScanInfo's `ScanStartInfo`) and background flag |
| `geecs` | the client (`md`) | Provenance only: `{preset, submission, operator, scan_log_variables}` — never a worker instruction; `ScanInfoCallback` copies `scan_log_variables` (the web scanner's shared list) into the ini's `[Scan Log]` section as a record |

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
1D regardless of axis count. It retains stock `plan_pattern` and the
pre-rebuild readers for historical runs. No existing event column changes meaning.

## Descriptor: configuration

Every `GeecsDetector` records its drain offset
(`<det>-drain_offset`, seconds — the calibrated edge-to-stamp latency, `0.0`
until measured); the `ShotControl` device records the standing
trigger state (`shot_control-state`) when it is read.

## Event stream `primary` of a shot-table run (`layout: "shot-table/1"`)

A strict `count` / `sweep`: **one event per fire**, refires included, and
no device data — every per-shot device (a GEECS detector, its `.scalars`
view, a native camera) records into its own `<device>_stream` (below).
What the row carries:

| Column | Epoch / type | Meaning |
|---|---|---|
| `shot_index` | integer | The fire's number in the run, from 1 (the s-file's row index is it) |
| `attempt` | integer | 1 for a shot's first fire, +1 for each refire of the same shot |
| `fire_stamp` | Unix s | The worker's clock at the SINGLESHOT put; under a box-less scan-mode profile (no put) at the announce. The unwitnessed row clock |
| `announce_stamp` | Unix s | Scan mode only: the worker's clock at the shot's announce |
| `witness_count`, `witness_stamp`, `witness_epoch` | count; Unix s (the count record's CA time); string | A run whose witness is read (start `witness.read`), as below |
| `<device>-baseline` | a frame count, or LabVIEW s | One per required device: its progress counter read **before** the fire — a plugin or native camera's frame count (`NumCaptured`, the fewest across its plugins), a stamped device's `acq_timestamp` (LabVIEW epoch, `NaN` before its first stamp). A record of row *k* is newer than baseline *k* |
| `complete` | boolean | Every required device reported for this fire (the plan's decision to stop refiring; which record is which shot's is the join's — a reader auditing the two flags *complete-mismatch*) |
| `missing` | string | The required devices that did not report, comma-joined ophyd names; `""` when complete |
| `extra_frames` | string | Gateway plugin cameras (`GeecsDetector`, not a native camera) that counted more than one frame since the previous fire, comma-joined `<name>=<frames>`; `""` when none. One fire may (a missed shot's late frame before the refire's); two in a row pause the run naming the camera (free-running: put it in trigger mode). Not in a fly run |
| `unwitnessed` | boolean | No witness read this run: the join assigns by fire interval |
| `witness_late` | boolean | The witness count stayed put while a device reported (a poll or USB stall) — never a stop |
| `interrupted` | boolean | A pause or suspension landed inside the fire's slot; the counters were re-read after the resume (a pause inside the row's own bundle drops it and reads it again — never a short row) |
| `bin_number`, the motors, `announced_shot` | as below | As in every strict row |
| the conditions | as below | Devices with no shot stamp (`CaSnapshotReadable`, setpoint devices) and the background telemetry, read per row |

**A fire nothing saw pauses the run.** A fire whose witness count and
every per-shot counter stayed put was not fired: the row is kept (flags
as found) and the run pauses with the reason in the log; the resume
fires again.  A pause never ends a scan and never loses a shot: a fire a
pause followed (nothing saw it, or a required device is down) costs no
refire, so the shot is still taken after the resume.  A shot whose refires are
exhausted keeps its rows, its last flagged `complete: false`, and the scan
continues; two such shots in a row pause the run, as does a required
device whose liveness reads down, or a gateway plugin camera counting
more than one frame on two fires in a row (`extra_frames`; the reason in
the log).

**A fly run** (`acquisition: fly`, start `clock: "soft-tick"`): one row
per tick of the worker's timer, `shot_period` apart on an absolute
schedule (a tick that would follow a late one by under half a period is
re-anchored a full period on, so no catch-up tick); the box is never
driven.  `witness_count` is the tick's number from 1, `witness_stamp` the
worker's clock at the tick (Unix s), `witness_epoch` the run's uid; no
`fire_stamp`, `unwitnessed`, `witness_late` or `extra_frames` column.  **One interval
rule, the plan's and the join's** (*m* the stamp budget, 30 ms):

| Device | Row *k*'s record | The plan calls it complete when |
|---|---|---|
| pressed (Software-source native, `synchronous`) | the last stamped in [t<sub>k</sub> − m, t<sub>k+1</sub> − m), however late in the period (earlier ones, a held tick's press landing after its wait, are extras) | its frame lands before t<sub>k+1</sub> − m |
| free-running (`asynchronous`) | the last at or before t<sub>k</sub>, *stale* when older than one period | it recorded in (t<sub>k−1</sub>, t<sub>k</sub>] |

So a required free-runner slower than the tick rate holds ticks.  The
window (t<sub>k−1</sub>, t<sub>k</sub>] never spans a move or a pause: at
the run's first tick, every step's first tick and the first tick after a
pause the counters are read again and the tick comes a full period
later.  An incomplete tick holds one more
(a new row, `attempt` + 1) up to the budget, then is flagged; two
exhausted in a row pause the run, the reason naming the device and
saying the box is not driven in fly mode (a device on a hardware trigger
line records only while the box already free-runs).  A named scan-mode
profile is refused.

**Every stamp column's epoch**: a device's `<device>-acq_timestamp` (an
event column, a `-baseline`) is in the **LabVIEW epoch** (seconds since
1904-01-01); `witness_stamp`, `fire_stamp` and `announce_stamp` are
**Unix seconds**; a stack's per-frame `…-frame_acq_timestamp` attribute is
Unix seconds.  `geecs_data_utils.shot_join.to_labview_epoch` is the one
conversion of a row clock.

## Event stream `primary` (a gated run's frames; earlier strict runs)

Every strict run before the shot table wrote this layout (no `layout`
key), `optimize` included.

| Column | Meaning |
|---|---|
| `<det>-<variable>` | The detector's DB-subscribed scalars, one column each (`safe_name`-mangled: `uc_amp4_ir_input-meancounts`) |
| `<det>-acq_timestamp` | The shot stamp: the join key for that detector's files and for cross-device alignment after the drain offset |
| `<det>` | A plugin-backed camera's frames: an external `STREAM:` key — the row's frame is the stream datum's index into `ScanNNN/<device>/<device>.h5` (`/entry/data/data`); `<det>-<variable>` for a second image variable. Absent from a partial row (see below) |
| `<det>-nonscalar_save_path` | The directory the detector's native files landed in this run — present only when the detector saved natively (`geecs_data_utils.tiled_schema.COMPANION_SUFFIXES` names the suffix) |
| `<device>-<variable>` | A scalar-only device's subscribed readbacks (`CaSnapshotReadable`); a native setpoint device's listed `readbacks` (`NativeSetpointDevice`) |
| `<device>-<settable>-position` / `-readback` | A settable child's readback when the DB subscribes it (`CaMotor` / `CaSettable`) or always, for a native setpoint device's settable (`NativeSetpoint`), and the scan motor's column |
| `bin_number` | The scan step the row belongs to, from 1 (the GEECS `per_step`; every row of a `count` is bin 1) — the s-file's `Bin #` |
| `announced_shot` | Scan mode only (`use_scan_mode`): the run-wide shot number announced for the row's shot, from 1 — every fire increments it, refires included, so a partial row keeps its number and the retake takes the next; a number is never re-sent. The join key for a device's shot-numbered files (`ScanNNN_<device>_SSS.<ext>`): the s-file's row index is not it — the s-file's `Announced Shot` |
| `witness_count` | A strict run with a witness (`witness.read`): the witness's hardware count as its record stood when the row was read (after every detector's frame) — the shot is `(witness_epoch, witness_count)`, never the count alone. `NaN` when the witness did not answer (a WARNING; the run continues). In a run with no triggered device (the start record's `note`) nothing waits for the shot, so a row may carry the previous shot's count while its steps still read +1. Audited against the previous row — row 1 against the baseline the stage reads (count and epoch; when it does not answer, the audit starts at row 2) — never failed: a step other than +1 (a doubled edge, a fire it did not see, a read before its update) is one WARNING naming the row, and the run's close logs one summary line (fires against what it counted: every row's fire from the baseline, else the fires after row 1) |
| `witness_stamp` | The count record's own CA timestamp (Unix seconds) — the IOC's clock at the poll that saw the edge, never the worker's read time; `NaN` with the count |
| `witness_epoch` | The witness's epoch string: it changes whenever the count restarts (an IOC restart); a change between rows is a WARNING. `""` when the witness did not answer |

A device listed as `X.scalars` (the scalars-only view every namespace
device carries; `save_images: false` in a preset) contributes the same
columns as `X` — for a detector `<det>-<variable>` and
`<det>-acq_timestamp` with no `-nonscalar_save_path`.

**Partial rows** (an earlier strict run). A shot on which a detector
produced no frame within the timeout is still a row: every other device's
columns are real, the frameless detector's `<det>-<variable>` and
`<det>-acq_timestamp` read `NaN`, and the row references **no** frames
for any plugin-backed camera (one datum per external key per event, or
none); the plan took one more shot for the step, and the s-file carries
the `NaN` holes.  A shot-table run writes no frame datums in `primary`:
a missed record is a gap in the device's own stream.

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
| `<det>-nonscalar_save_path` | A **native-saving required device**'s save directory (a device without a file plugin — a LabVIEW-native camera, a DAQ or wavefront sensor with its own file writer, a scope with every capture channel disabled): the same companion column a strict row carries, here a **run-long constant** — LabVIEW's saving is switched on at the run's first prepare and off at `unstage`, never per step. Its scalars and its own `<det>-acq_timestamp` ride in the row like any non-plugin device's (it may be the clock). Its files are named by that stamp and join by it (`geecs_data_utils.native_files`); a shot on which it dropped a frame is a row with no file — **no retake**. The stack check matches every row's stamp to a file in that directory at the stop (`geecs_data_utils.native_files.native_file_keys`) and appends a `native files check` line to `scan.log` — rows without a file and file stamps without a row counted apart (WARNING on either, never a failure). Absent for a plugin-backed camera (its stack is its record) and for a `.scalars` view |

An additive column convention, not a schema change: a reader that never
looked for the column in `shots` sees the rows it saw before.

**When the rows arrive**: the plan collects the
sampler about once a second **during** a batch, so `shots` pages arrive
mid-batch rather than one page at its end — a row goes out once every
plugin-backed camera of the step holds that shot's frame, and a row once
out is never withdrawn.  `seq_num` stays one sequence over the run.  A
batch interrupted by a **pause** keeps the shots every device reached and
the step continues after the resume, so one step may carry **more than
one stream datum per camera** in `primary` — contiguous, together
exactly the step's quota (the in-flight shot at the pause may be lost;
nothing before it is).  A batch that **fails** (a stalled camera, a
silent clock) records the shots it kept the same way, best effort, before
the run fails.  All additive: a reader that indexes datums by
their `indices` and rows by `seq_num` / stamp sees the same record.

The s-file of such a run is the `shots` rows with each stack's per-frame
columns joined on by offset-corrected stamp
(`geecs_data_utils.shot_join`): the attribute
`<ophyd>-hdf-<variable>-frame_acq_timestamp` becomes the column
`<ophyd>-acq_timestamp` and a subscribed scalar becomes
`<ophyd>-<scalar>` — the same spellings a strict row uses, so one header
map renames both.  One s-file row per `shots` row: a frame with no shot
inside the join window stays in the stack and in Tiled and is left out of
the s-file.  The same join adds a streamed device's columns
(`<name>_stream`, below) to the rows.

## Event stream `<name>_stream` (a shot-table run's per-shot device; a gated run's device not required)

Every per-shot device of a shot-table run, and a device listed
`non_essential` (gated), streams for the run into its own
`<ophyd name>_stream` (`geecs_data_utils.shot_join.device_stream`).
A shot-table run collects every stream once per fire (each collect a
contingency: a stream that fails to collect gaps, logged once, never
failing the run) and once more at the close; a gated run's stream is
collected at the close only.  Nothing in a stream is a placeholder: a fire a
device missed is simply absent from its stream.  Three shapes:

- a **plugin-backed** camera, or a **native camera** (its own HDF
  writer): a datum stream (its stack), no events; declared at its first
  frame (or at the close), so its descriptor reads the geometry the
  writer settled on;
- a **triggered device without a plugin** (a LabVIEW-native
  saver, a scalar device with a stamp, or a detector's `.scalars` view):
  **one event per stamp the device published** while the run was open,
  recorded the moment the stamp arrived, at the device's own rate:

| Column | Meaning |
|---|---|
| `<name>-acq_timestamp` | The device's own stamp — the event's shot id and the join key |
| `<name>-<variable>` | Its subscribed scalars, the latest cached values when the stamp arrived (the spellings a strict row uses) |
| `<name>-nonscalar_save_path` | A native saver listed itself (not its `.scalars` view): the directory its LabVIEW files landed in, a run-long constant (saving on at the run's unbounded prepare, off at `unstage`); the files are named with the event's stamp |

The descriptor's configuration carries the device's
`<name>-drain_offset` under the object `<name>` (the stream's collect
object is named after the device).  A device that published nothing has a
declared, empty stream (and a WARNING in the log).  The s-file joins these
events onto the rows as it joins frames — each row takes the nearest event
inside its own window, by offset-corrected stamp; a row with none reads
`NaN` in the device's columns (a device slower than the rep rate leaves
`NaN` on the rows it missed; an empty stream gives all-`NaN` columns from
the descriptor's numeric data keys), an event no row's window reaches
stays in the stream and in Tiled and is left out of the s-file, and a
string column (the save path) is never an s-file column.  The start
document's `per_shot_streams` (a gated run's: `non_essential`) names
which `<name>_stream`s these are (the offline re-export reads them from
there).  The
stack check matches a native saver's **events** (not the rows) to its
files: events without a file and file stamps without an event counted
apart, files stamped after the last event (saved between the stream's
close and the unstage) apart again — WARNING on the first two, never a
failure.

An additive stream convention, not a schema change: a reader that never
looked for these streams sees the runs it saw before.

## Background telemetry columns

Every subscribed scalar of the experiment whose device is **not** in the
run — not a detector, not the owner of a listed `.scalars` view, not a
streamed device, not a scan motor — rides in every row of the run's
row stream (`primary` for a strict run, `shots` for a gated one) under the
same `<device>-<variable>` keys a listed device would carry, read softly
from the gateway's monitor cache by the run's `BackgroundSnapshot`.  A detector
outside the run contributes its scalars and its `acq_timestamp` as its
monitor cache holds them (the last frame's), so a row's alignment to that
device is checkable.  A reading the gateway marks INVALID (a dead device's
stale readbacks) is `NaN`; a device the probe dropped (the start
document's `background_dropped`) has no columns in that run; the columns
carry their `Device Variable` headers in `geecs_scalar_headers`, so the
s-file has them like any other.  The run-level `baseline` stream (two
open/close rows) that preceded this is gone: a reader that looked for it
sees runs without it; nothing in the repo joined it.

## Legacy `Device Variable` headers

Every device carries `_column_headers` — event data-key → the GEECS
`Device Variable` header (`UC_Wavemeter Wavelength (nm)`); the
`scalar_headers` preprocessor merges the staged devices' maps into the
start document's `geecs_scalar_headers`, which the s-file callback
(`callbacks/sfile.py`, from the run's own rows at the stop document) and the
offline re-export (`geecs_data_utils.write_scalar_files_from_tiled`) read.
The witness's columns are headed `<device> count`, `<device> stamp` and
`<device> epoch`.
Both take their rows from `primary` when it has events and from `shots`
otherwise, and both run the same join, so a re-export checks the live path
rather than re-implementing it.


## Native optimization

`plan_name: optimize` opens one run on the shot-table layout: `primary` is
the shot table as for `count` / `sweep` (one row per fire, refires
included) and `bin_number` is the one-based adaptive iteration.  A plugin
camera an objective reads also records its gateway stamp in a companion
stream, `<name>_stamps_stream`: one event per stamp (`<name>-acq_timestamp`
and its scalars, like a `StampStream`), not listed in `per_shot_streams`
(the s-file reads the stack) — the plan's in-plan join reads it. Start metadata adds
`optimizer_config`, `max_iterations`, `optimization_variables`,
`optimization_objectives`, `optimization_move_targets`, and the resolved
`OptimizerConfig` as JSON text under `geecs.optimizer_json` (provenance only).
Bluesky forbids dots and slashes in document keys at any depth; event name
components use URI escaping with `~` as the escape marker (Tiled SQL forbids
`%`): `.`, `/`, `%`, `~`, `-` become `~2E`, `~2F`, `~25`, `~7E`, `~2D`.
Colons and underscores remain literal. The shared `optimization.events` codec
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

Acquisition is not rewindable (the strict slot): a pause inside a shot
costs no refire and the bin resumes at its next shot; the bin's complete
rows feed evaluation. Evaluation and Xopt tell are not replayed. Deferred pause lands at the next iteration checkpoint. Failed
measurement rows stay in the Xopt dump with `xopt_error: true` and are excluded
from model fitting. The dump is written before run close into the folder
from the emitted start document, including abort cleanup after an in-flight
calculation has finished.
