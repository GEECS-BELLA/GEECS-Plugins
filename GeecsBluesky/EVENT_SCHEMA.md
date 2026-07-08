# GEECS Bluesky Event Schema — v1

The contract every GEECS Bluesky run obeys, regardless of acquisition mode or
whether it was launched from the GUI, `BlueskyScanner`, or a custom notebook
plan.  This is the **canonical** copy; the design rationale lives in
`Planning/acquisition_modes/01_event_schema_contract.md`.

Consumers branch on **`geecs_event_schema`**, never on `acquisition_mode`, to
read data.  Mode is provenance, not shape.

## Start-document metadata

Always present:

| Key | Meaning |
|---|---|
| `geecs_event_schema` | Integer schema version. This document describes `1`. |
| `acquisition_mode` | `"strict_shot_control"` or `"free_run_time_sync"` |
| `plan_name` | `geecs_step_scan` or `geecs_free_run_step_scan` |
| `motor` | Scan-device ophyd name, or `null` for statistics collection (no scan variable) |
| `detectors` | Device ophyd names recorded in the run |
| `positions` | Scan positions (`[null]` for statistics collection) |
| `shots_per_step`, `num_points` | Loop dimensions |

Added by the run wrapper (`geecs_run_wrapper`) when a scan number is claimed:

| Key | Meaning |
|---|---|
| `experiment` | GEECS experiment name |
| `scan_number` | Day-scoped GEECS scan number |
| `scan_id` | Same value as `scan_number` (Bluesky-native display field; see note) |
| `scan_folder` | Absolute path of the claimed `scans/ScanNNN/` folder |
| `nonscalar_save_paths` | device → save dir map (when non-scalar saving is active) |
| `geecs_scalar_headers` | event data-key → legacy `Device Variable` header map (see note) |
| `bluesky_backend` | `true` |

**`geecs_scalar_headers` note:** maps each scalar event-stream data key
(`<ophyd>-<safe_var>`, e.g. `uc_wavemeter-wavelength_nm`) to its original GEECS
`Device Variable` header (`UC_Wavemeter Wavelength (nm)`). `safe_name()` mangling
is irreversible, so this map is the only way to recover legacy headers; it backs
the Tiled→s-file exporter (`geecs_data_utils.tiled_export`). Only true device
signals appear — derived companion columns (`-acq_timestamp`, `-shot_id`, …) are
excluded by construction.

Strict mode adds:

| Key | Meaning |
|---|---|
| `fires_own_shots` | `true` when the strict plan fires each shot (single-shot / `ARMED`) |

Free-run mode adds:

| Key | Meaning |
|---|---|
| `reference_device` | Pacemaker device name |
| `device_t0s` | device → t0 `acq_timestamp` captured by the t0-sync stage |
| `t0_sync_window_s` | Acceptance window used by the t0-sync stage |

ScanRequest runs (`GeecsSession.run`) may also add, for provenance:

| Key | Meaning |
|---|---|
| `applied_defaults` | Experiment-defaults fields that filled a silent request |
| `action_plans` | Assembled per-slot action execution order |
| `background_telemetry` | `{device: [variables]}` recorded as Tier-2 telemetry (M3c get-side; present only when telemetry ran) |

**`scan_id` note:** `scan_id` has no uniqueness contract in Bluesky — the
day-scoped number resets to 1 each day, which is fine (`uid` is the real key).
Never look a run up by `scan_id` alone; qualify with the day, or use
`scan_number` + the start-doc `time`.

## Event-stream columns

Row identity (every mode, every row — set by the plan via `ScanContext`):

- `bin_number` — 1-based step index (always `1` for statistics collection)
- `shot_index_in_bin` — 1-based shot index within the step
- `scan_event_index` — 1-based global row index

Scan device (when a motor is moved): its readback column(s).

**Per synchronous device** (triggered; has `acq_timestamp`) — every row:

| Column | Meaning |
|---|---|
| `<dev>-<variable>` | Data variables. Real values when `valid`; NaN when not. |
| `<dev>-acq_timestamp` | Raw device acquisition timestamp (back-dated to acquisition start). **The file-join key.** |
| `<dev>-t0_acq_timestamp` | The device's t0 (physical shot 1) timestamp, or NaN. |
| `<dev>-shot_id` | Derived physical trigger-opportunity number (dtype `number`, NaN when underivable). |
| `<dev>-shot_offset` | `device shot_id − row shot_id`. `0` = this cell belongs to this row's physical shot. |
| `<dev>-valid` | `shot_offset == 0` (boolean). Strict mode: constitutively `true`. |
| `<dev>-nonscalar_save_path` | Scanner-owned save directory (when non-scalar saving is active). |

In **strict** mode every device read follows its own awaited trigger, so
`shot_offset` is `0` by construction.  In **free-run** mode the row's shot ID
is the reference device's; contributors are labeled relative to it (a
late/long-exposure device lands at a negative offset carrying real,
truthfully-labeled data for realignment downstream by `shot_id`).

**Per snapshot device** (asynchronous, no `acq_timestamp`) — every row: its
data variables, sampled at row emission. No companion columns.

**Background-telemetry columns** (Tier 2, best-effort) — every live experiment
device with a `get='yes'` variable *not* in the save set is recorded as soft
snapshot columns read from the gateway monitor cache. They are distinguished
from Tier-1 save-set data by a **device-name prefix**: the ophyd device is
`telemetry_<device>`, so every column it contributes is keyed
`telemetry_<device>-<safe_var>`. Telemetry is **dtype-tolerant**: each column's
type is inferred from its PV — numeric variables stay numeric (float) so
downstream telemetry analysis keeps working, while enum/string/path variables
(e.g. `U_VisaPlungers` `DigitalOutput.Channel N`) are recorded as their
string/label value. A telemetry column set may therefore mix float and string
columns; do not assume every telemetry column is float. Typing is
**per-variable** — one non-numeric variable is captured as a string and never
drops the device's other (numeric) columns. Telemetry is read-only, sampled
once per row, **never waited on** — a value that cannot be read (a device that
went dead mid-scan) degrades to a dtype-appropriate null cell (NaN for a
numeric column, `""` for a string column), and a device unreachable at scan
start is dropped with a log line, never a dialog or abort. No telemetry
variable — and no telemetry device — is dropped for a *type* reason; only a
genuinely unreachable device degrades to a dropped device. Telemetry columns
are **not** added to
`geecs_scalar_headers` (they are Tier 2, not legacy s-file scalars). The
start-doc `background_telemetry` key (present only when telemetry ran) records
the `{device: [variables]}` actually selected. This is an additive
device-name convention, not a new schema field — it does not bump the version.

**Free-run tail flush:** after the last shot, free-run runs emit one final
event on a separate `flush` stream (one extra read of all devices) so a
contributor lagging at `shot_offset = -1` still records its final shot.

## Rules

1. **Stable keys.** A device configured into a run contributes its full column
   set to every event; missing data is NaN / `false`, never an omitted key
   (descriptors require a stable shape).
2. **`shot_id` is not a file-join key.** Join files to events by device
   `acq_timestamp`.  `shot_id` is matching machinery and diagnostics; jumps > 1
   across dead time are expected (it counts trigger opportunities, not rows).
3. **Additive changes don't bump the version** — readers ignore unknown
   columns/keys.  Only breaking changes (rename/remove/semantics) bump it.
4. **Version bumps never require a new Tiled catalog.** Runs are
   self-describing (own descriptors + metadata); versions coexist.
