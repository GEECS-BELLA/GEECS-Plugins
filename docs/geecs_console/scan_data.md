# Scan Data

Every scan is recorded twice, deliberately:

1. **The classic GEECS scan folder** — so every existing analysis tool
   keeps working unchanged.
2. **A structured run in the Tiled catalog** — start/stop metadata plus
   one event row per shot, which is what the scan browser and modern
   analysis read.

The canonical, versioned column contract is
`GeecsBluesky/EVENT_SCHEMA.md`; this page is the consumer-oriented
summary. Column-name interpretation logic for analysis code lives in one
place: `geecs_data_utils.tiled_schema` — use it rather than parsing names
yourself.

## On disk

```
{base}/{experiment}/Y{YYYY}/{MM-Month}/{YY_MMDD}/
  ├── scans/ScanNNN/
  │   ├── ScanInfoScanNNN.ini      scan metadata (mode, parameter, shots)
  │   ├── ScanDataScanNNN.txt      s-file: per-shot scalar table (tab-separated)
  │   ├── scan.log                 the scan's own log (also triage input)
  │   └── <Device>/                natively saved per-shot files (images…)
  └── analysis/sNNN.txt            watched s-file copy for live analysis
```

The s-file is exported from the Tiled run after the scan completes, so
both records always agree. Native files are named
`<stem>_<acq_timestamp>.<ext>` — **files join to event rows by the
device's acquisition timestamp**, the single joining rule used everywhere
(`geecs_data_utils.native_files` owns the naming contract).

An aborted scan's folder is never deleted: it stays, possibly without
data, and the abort is logged loudly. Analysis tools must treat scan
folders as read-only artifacts — nothing on the analysis side ever
creates one.

## The event row (Tiled)

Each shot is one row. Columns come in families, all keyed by the ophyd
device name:

**Row identity** — `bin_number` (scan step), `shot_index_in_bin`,
`scan_event_index`, plus each scan axis's readback.

**Per synchronized device** (`<dev>` = reference or contributor):

| Column | Meaning |
|---|---|
| `<dev>-<variable>` | the recorded scalar(s) |
| `<dev>-acq_timestamp` | the device's own shot timestamp (the file-join key) |
| `<dev>-shot_id` | physical trigger-opportunity number, derived from its own timestamps |
| `<dev>-shot_offset` | its shot ID minus the row's (reference's) shot ID |
| `<dev>-valid` | `shot_offset == 0` — this cell belongs to this row's shot |
| `<dev>-save_path` | native file path, when saving images |

`valid`/`shot_offset` are the free-run honesty mechanism: a camera that
lagged one shot contributes its *real* latest frame labeled
`shot_offset = -1, valid = False`, and analysis can realign per device by
shifting on `shot_id`. Cross-device matching is shot-ID *equality* —
never assume consecutive rows are consecutive shots for every device.

**Snapshot devices** — value columns only, sampled at row time.

**Background telemetry** — `telemetry_<device>-<variable>` columns for
every live device outside the save sets. Dtype-tolerant (numeric stays
float; enums/strings are recorded as labels — don't assume float).
Additionally, every telemetry device records its own
`telemetry_<device>-acq_timestamp`: `0.0` means the device has never
acquired (it is not a triggered device — honest placeholder, not data).
Telemetry devices that *were* observed firing get the full
`shot_id`/`shot_offset`/`valid` companions, exactly like contributors —
so triggered diagnostics ride along with usable shot correlation even
when nobody put them in a save set. The start document's
`telemetry_shot_seeded` lists which devices got them. One rule for
analysis: **attribute telemetry samples by the `acq_timestamp` column,
never by a data column's own timestamp** (unchanged values are not
re-posted, so data-column timestamps are "time of last change").

**Run metadata** (the start document) — scan number and folder, mode,
save sets used, applied experiment defaults, action plans, device t0s,
`geecs_event_schema` version, and the scalar-header map used to produce
the legacy s-file names.

## Reading it back

- **Interactive**: the [Scan Browser](scan_browser.md).
- **Python**: `geecs_data_utils.tiled_catalog` — the `ScanCatalog`
  interface (`list_runs(experiment, day)` → `load_run(uid)`), plus
  `tiled_schema` for column semantics and `tiled_drift` for "did
  anything move during the scan" checks. All hermetically testable; no
  Bluesky import required on the analysis side.
- **Legacy**: `geecs_data_utils.ScanData.from_date(...)` reads the s-file
  exactly as before.
