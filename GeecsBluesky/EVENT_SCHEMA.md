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
| `bluesky_backend` | `true` |

Strict mode adds:

| Key | Meaning |
|---|---|
| `fires_own_shots` | `true` when the plan fires each shot (single-shot / `ARMED`); `false` for the free-running `trigger_and_read` fallback |

Free-run mode adds:

| Key | Meaning |
|---|---|
| `reference_device` | Pacemaker device name |
| `device_t0s` | device → t0 `acq_timestamp` captured by the t0-sync stage |
| `t0_sync_window_s` | Acceptance window used by the t0-sync stage |

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
