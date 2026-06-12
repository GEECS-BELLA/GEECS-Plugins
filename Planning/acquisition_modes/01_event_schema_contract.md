# Event Schema Contract — v1

The contract every GEECS Bluesky run obeys, regardless of acquisition mode or
whether it was launched from the GUI, BlueskyScanner, or a custom notebook
plan. This draft graduates into `GeecsBluesky/EVENT_SCHEMA.md` when
implemented; that file becomes the canonical copy.

## Start-document metadata (required keys)

| Key | Meaning |
|---|---|
| `geecs_event_schema` | Integer schema version. This document describes `1`. |
| `acquisition_mode` | `"strict_shot_control"` or `"free_run_time_sync"` |
| `experiment` | GEECS experiment name (e.g. `"Undulator"`) |
| `scan_number` | Day-scoped GEECS scan number (claimed from the data server) |
| `scan_id` | Same value as `scan_number` (Bluesky-native display field) |
| `scan_folder` | Absolute path of the claimed `scans/ScanNNN/` folder |
| `rep_rate_hz` | External trigger rep rate |
| `plan_name`, `motor`, `detectors`, `positions`, `shots_per_step`, `num_points` | As today |
| `nonscalar_save_paths` | device → save dir map (when non-scalar saving active) |

Free-run mode additionally requires:

| Key | Meaning |
|---|---|
| `reference_device` | Pacemaker device name |
| `device_t0s` | device → t0 `acq_timestamp` map captured by the t0-sync stage |
| `t0_sync_window_s` | Acceptance window used by the t0-sync stage |

## Event stream columns

Row identity (every mode, every row — set by the plan via `ScanContext`):

- `bin_number` — 1-based motor step index
- `shot_index_in_bin` — 1-based shot index within the step
- `scan_event_index` — 1-based global row index

Motor: readback column(s), as today.

**Per sync device** (triggered, has `acq_timestamp`) — every row:

| Column | Meaning |
|---|---|
| `<dev>-<variable>` | Data variables. Real values when `valid`; NaN when not. |
| `<dev>-acq_timestamp` | Raw device acquisition timestamp (back-dated to acquisition start). **The file-join key.** |
| `<dev>-shot_id` | Device's derived physical trigger-opportunity number (see `02`). |
| `<dev>-shot_offset` | `device shot_id − row shot_id`. `0` = this cell's data belongs to this row's physical shot. |
| `<dev>-valid` | `shot_offset == 0`. In strict mode constitutively `True` / `0`. |
| `<dev>-nonscalar_save_path` | When non-scalar saving is active, as today. |

The row's own shot ID in free-run mode is the reference device's `shot_id`.
In strict mode every device read follows its own awaited trigger, so
`shot_offset` is 0 by construction.

**Per snapshot device** (async, no `acq_timestamp`) — every row: its data
variables, sampled at row emission. No companion columns.

## Rules

1. **Stable keys.** A device configured into a run contributes its full column
   set to every event. Missing data is NaN (numeric) / `""` (string) with
   `valid=False` — never an omitted key (descriptors require stable shape).
2. **Consumers branch on `geecs_event_schema`, never on `acquisition_mode`,**
   to read data. Mode is provenance, not shape.
3. **Additive changes don't bump the version** — readers ignore unknown
   columns/keys. Only breaking changes (rename/remove/semantics) bump it.
4. **Version bumps never require a new Tiled catalog.** Runs are
   self-describing; versions coexist.
5. **`shot_id` is not a join key for files.** Join files to events by device
   `acq_timestamp`.
6. Integer-typed companion columns must tolerate the missing case: `shot_id`
   and `shot_offset` are described as dtype `number` (NaN-able), not
   `integer`.
