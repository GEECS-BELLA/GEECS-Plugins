# GEECS-Data-Utils — Developer Context for Claude

Foundational data layer. Provides scan path navigation, scalar data loading,
data binning/aggregation, and a queryable Parquet-based scan metadata database.
Used by ImageAnalysis, ScanAnalysis, GeecsBluesky, GeecsScanner and the
Data Portal.

## Package Layout

```
geecs_data_utils/
  __init__.py                  # Public API: ScanTag, ScanPaths, ScanData, GeecsPathsConfig
  scan_paths.py                # ScanPaths: folder navigation + scan_info.ini parsing
  scan_data.py                 # ScanData: ScanPaths + scalar DataFrame loading + binning
  analysis_status.py           # read-only tolerant reader for scans/ScanNNN/
                               #   analysis_status/*.yaml (ScanAnalysis task_queue's
                               #   TaskStatus.to_dict() shape; contract pinned in
                               #   ScanAnalysis's suite, #682)
  shot_files.py                # completed-scan shot rows → native Path / stack ShotRef
  scalar_files.py              # shared s-file lock/merge and generated-scalar writes
  type_defs.py                 # ScanTag, ScanMode, ScanConfig, ECSDump Pydantic models
  geecs_paths_config.py        # GeecsPathsConfig: base path + experiment resolution
  config_base.py               # ConfigDirManager: generic config directory management
  config_roots.py              # Singleton instances for image/scan analysis config dirs
  utils.py                     # month_to_int, SysPath, ConfigurationError
  io/                          # generic path->ndarray readers (images, 1D,
                               #   IMAQ decode) + arrays.py: the three array
                               #   wire shapes devices push over TCP (nested
                               #   [x,y] pairs, CSV, LabVIEW flattened
                               #   waveform → volts), sniffed by payload;
                               #   pinned on tests/data/wire/ captures
                               #   + scan_stack.py: reader for
                               #   per-device image stacks in the areaDetector
                               #   NDFileHDF5 layout (/entry/data/data +
                               #   /entry/instrument/NDAttributes/<device>-hdf-
                               #   <variable>-frame_acq_timestamp, resolved by
                               #   timestamps_dataset — the bare acq_timestamp of
                               #   older stacks reads too; plus the device's
                               #   subscribed scalars as <device>-hdf-<variable>-
                               #   <scalar> since GeecsPvaGateway 0.9, read by
                               #   read_stack_attributes / parse_attribute_name;
                               #   frame_index_for_acq_timestamp joins a
                               #   shot to a frame WITHOUT reading it;
                               #   written by GeecsPvaGateway's file plugin,
                               #   #806/#829)
                               #   incl. ShotRef — a Path carrying a
                               #   frame index for per-shot pipelines,
                               #   and stack_content_kind: image /
                               #   lineout / waveform, from the plugin's
                               #   own wave_* declaration plus the frame
                               #   rank (see "Three kinds of stack")
                               #   + himg.py: the SDK-free codec for the HASO
                               #   .himg container (header bytes + uint16
                               #   frame; rebuilds byte-identically)
                               #   + himg_stack.py: the stack writer in io/ —
                               #   a HASO device folder's .himg files → its
                               #   capture stack (frames gzip+shuffle, stamps
                               #   from native names or the scan's rows, a
                               #   provenance group with each file's header
                               #   and SHA-256; verified after writing;
                               #   never creates a directory, never deletes)
                               #   + himg_compact.py: THE one deleter —
                               #   compact (verify every frame against the
                               #   stack AND the file, then delete the .himg,
                               #   leave himg_manifest.json) and restore
                               #   (rebuild them byte-identical); the guards
                               #   live here (scan may still be writing,
                               #   stack short of the folder, any mismatch)
                               #   + himg_worker.py: run any of those jobs in
                               #   a child interpreter, streaming progress,
                               #   logs, the report and errors back as events
  himg_cli.py                  # geecs-himg convert | verify | compact | restore: the .himg backlog command
  plotting_utils.py            # Simple matplotlib helpers for binned data
  scans_database/
    database.py                # ScanDatabase: filter + load Parquet dataset
    builder.py                 # ScanDatabaseBuilder: create/update Parquet dataset
    entries.py                 # ScanEntry, ScanMetadata: Pydantic models for Parquet rows
    filter_models.py           # FilterSpec, FilterArgs
    filters/                   # YAML filter preset files
```

## Core Abstractions

### Coordinate-aware samples (`frames`)

`Frame`, `Axis` and `ShotMeta` in `geecs_data_utils.frames` are the in-memory
sample vocabulary for the new analysis core. They do no I/O. A trace is 1D
with one coordinate axis; an image is 2D with axes in numpy `(y, x)` order.
Construction copies data to read-only float64 arrays so reuse of a live
source buffer cannot change in-flight analysis. Coordinate vectors are finite,
owned and read-only; invalid signal values remain visible to measures.
`crop` slices coordinates and data together, preserving calibrated/global
positions. `from_trace` / `as_trace` adapt the existing Nx2 reader convention
without resampling. No current consumer is switched by introducing these types.
Both pickle, and arrive **read-only again** (`__setstate__` re-freezes the
arrays; numpy pickles values, not flags) — the analysis core's process-pool
workers return measurements built on them.

### `ScanTag`

Pydantic model identifying a scan. Immutable and hashable.

```python
ScanTag(year=2024, month=1, day=15, number=42, experiment="Undulator")
```

### `ScanPaths`

Wraps and validates a scan folder path; provides access to metadata and
sub-paths.

```python
paths = ScanPaths(tag=ScanTag(...), base_directory="/data")
paths.get_folder()              # Path to Scan042/
paths.get_analysis_folder()     # Path to Scan042/analysis/
paths.load_scan_info()          # Dict parsed from scan_info.ini
paths.get_folders_and_files()   # Lists device folders and files
ScanPaths.get_latest_scan_tag(experiment, year, month, day, base_directory)
```

Class-level `paths_config: GeecsPathsConfig` — shared across all instances.
Call `ScanPaths.reload_paths_config()` if experiment changes at runtime.

### `ScanData`

Composes `ScanPaths` with scalar data loading and binning.

```python
# Factory methods (preferred)
sd = ScanData.from_date(year=2024, month=1, day=15, number=42,
                        experiment="Undulator",
                        load_scalars=True, source="sfile")

sd = ScanData.latest(experiment="Undulator", load_scalars=True)

# Access
sd.paths.get_folder()           # Path to scan folder
sd.data_frame                   # Optional scalar DataFrame
sd.list_columns()               # Flat list of column names
sd.find_cols("centroid", mode="contains")   # Search column names
```

`source="sfile"` reads the text scalar summary; `source="tdms"` reads the binary
TDMS file. Prefer `sfile` for speed unless you need waveform data.

### `GeecsPathsConfig`

Resolves the GEECS data root and experiment name. Reading order:
1. Explicit `set_base_path` argument
2. `GEECS_DATA_LOCAL_BASE_PATH` under `[Paths]` in `~/.config/geecs_python_api/config.ini`
3. Raises `ConfigurationError` — no implicit server defaults; base path must be explicit

Also provides optional paths for config repos and FROG DLL.

## GEECS Folder Convention

```
{base_path}/{experiment}/Y{YYYY}/{MM-Month}/{YY_MMDD}/scans/Scan{NNN}/
  └── Scan{NNN}.tdms
  └── ScanData_scan.txt      (s-file: scalar summary)
  └── scan_info.ini          (scan parameters)
  └── analysis/              (created by ScanAnalysis)
{base_path}/{experiment}/Y{YYYY}/{MM-Month}/{YY_MMDD}/
  └── ECS Live dumps/        (device state snapshots, one per scan)
```

`ScanPaths` validates this convention and raises if the path doesn't conform.

## Scalar output files

`scalar_files` owns generated-scalar normalization and persistence shared by
legacy and replacement analysis runners. `merge_sfile` holds the existing
`.txt.lock` exclusive sidecar across read/merge/write, preserves unrelated
columns/cells, and refreshes the caller through its returned DataFrame. Missing
update values retain existing s-file cells (`combine_first` compatibility);
`write_scalar_sidecar` writes the generated values themselves, including NaNs.
Both keep the last duplicate shot update and sort by shot identity. Neither
creates parent directories; destination naming and directory policy belong to
the source host. Case-insensitive key normalization keeps the first matching
column, including when other case variants or duplicate labels coexist.

## Binning System

The pure core lives in `geecs_data_utils/data/binning.py`
(analysis-tabs W1c): `bin_frame(frame, cfg) -> BinnedFrame` — frame in,
centers + asymmetric error bands out, counts as a separate series, no
instance state. Every web-endpoint number is reproducible in a notebook
by the same one call.

```python
from geecs_data_utils.data.binning import BinningConfig, bin_frame

config = BinningConfig(
    bin_col="Bin #",
    value_cols=["signal_x", "signal_y"],   # None → all numeric minus Shotnumber
    agg="median",           # "mean" or "median"
    err="iqr",              # "std", "stderr", "mad", "iqr", "percentile"
    percentiles=(0.25, 0.75),
)
result = bin_frame(frame, config)
result.frame    # MultiIndex columns: (col, {"center","err_low","err_high"})
result.counts   # per-bin sample counts (a Series, not a pseudo column)

binned = sd.bin(config)      # ScanData one-call API — legacy shape,
                             # counts re-attached as ("count", "center")
binned = sd.binned_scalars   # stateful compat property (same shape)
```

`BinningConfig` also carries the numeric-binning options (`bin_edges` /
`bin_width` / `quantile_bins`, `label`, `right`, `origin`); a
non-numeric bin column always bins by identity. Consumed today only by
`plotting_utils.plot_binned`-style callers that are handed the frame —
ScanAnalysis does its **own** binning and does not use this. Deliberate
deltas vs the pre-0.23.0 implementation (see CHANGELOG): `Shotnumber`
is excluded from the default value columns, the bin key is excluded
from the dropna row policy (NA-bin-labelled rows aggregate into their
own row), and all-NaN columns never cause drops.

## Parquet Scan Database

A Hive-partitioned Parquet dataset indexing all historical scans. Not used in
live analysis — primarily for offline search and meta-analysis.

### Schema

Partitioned by `year` and `month`:
```
parquet_root/year=2024/month=1/0.parquet
```

Each row is a `ScanEntry`: date, number, experiment, file paths, non-scalar
device list, scan metadata (parsed from scan_info.ini), ECS dump (JSON),
analysis presence flag, notes.

### Querying

```python
from geecs_data_utils.scans_database import ScanDatabase
from datetime import date

db = ScanDatabase("/data/Undulator/scan_database_parquet")
df = (db
      .with_date_range(date(2024, 1, 1), date(2024, 12, 31))
      .with_experiment("Undulator")
      .with_named_filter("my_filter", date(2024, 6, 15))  # YAML preset
      .load())
```

### Building / Updating

```python
from geecs_data_utils.scans_database import ScanDatabaseBuilder

ScanDatabaseBuilder.stream_to_parquet(
    data_root="/data",
    experiment="Undulator",
    output_path="/data/Undulator/scan_db",
    date_range=(date(2024, 1, 1), date.today()),
    mode="append",      # or "overwrite"
)
```

## Config Directory Management

`analysis_configs` owns read-only discovery of unique diagnostic stems under
`analyzers/`, YAML mapping reads, and recursive overrides. `read_diagnostic`
requires an explicit config root for stems (or accepts an explicit `Path`) and
returns the source path and a fresh raw document. It neither validates analysis
schemas nor imports numerical analyzers; consumers own typed validation and
default-root selection. No folders are created. ImageAnalysis's typed loader
delegates here; other consumers should share this boundary.

`ConfigDirManager` (config_base.py) manages a directory that can hold multiple
YAML config files. ImageAnalysis and ScanAnalysis now resolve through the
unified Scan/ImageAnalysis config tree.

```python
from geecs_data_utils.config_roots import image_analysis_config, scan_analysis_config

# These are pre-built singletons. Path resolved from:
# 1. SCAN_ANALYSIS_CONFIG_DIR
# 2. config.ini Paths.scan_analysis_configs_path
# 3. Raises ValueError when no base directory is available

cfg_path = scan_analysis_config.find_config(
    "UC_GaiaMode",
    patterns=["{name}.yaml", "{name}.yml"],
    missing_base_message="Set SCAN_ANALYSIS_CONFIG_DIR",
)
```

## Key Type Definitions

- **`ScanMode`** (Enum) — `STANDARD`, `NOSCAN`, `OPTIMIZATION`, `BACKGROUND`
- **`ScanConfig`** — dataclass: scan_mode, device_var, start, end, step,
  wait_time, shots_per_step, additional_description
- **`ECSDump`** / **`DeviceDump`** — Pydantic models for ECS live dump files

## Useful Utilities

```python
from geecs_data_utils.utils import month_to_int
month_to_int("January")  # → 1
month_to_int(3)           # → 3

from geecs_data_utils.utils import read_geecs_tdms
data = read_geecs_tdms(path)  # → dict[device][variable] → np.ndarray

from geecs_data_utils.plotting_utils import plot_binned, plot_binned_multi
```

## How Other Packages Use This

- **ImageAnalysis** — `ScanPaths` to locate device data folders per scan
- **ScanAnalysis** — `ScanData` for binning scalar data in summary plots;
  `ScanPaths` as the base for scan folder resolution
- **GeecsBluesky / GeecsScanner** — `ScanPaths` for scan folder
  resolution and post-scan file organization (`ScanConfig` / `ScanMode`
  remain here as legacy vocabulary; their engine consumer was deleted
  2026-08-20)

## `analysis_status/` reader (`analysis_status`)

The one *schema-light* read-side view of the per-task YAMLs
ScanAnalysis's task queue writes at
`scans/Scan<NNN>/analysis_status/<task_id>.yaml` — for consumers outside
ScanAnalysis (GEECS-MCP's `get_scan_analysis` today; #682).  Code that
needs the typed `TaskStatus` to drive the queue (`claim_is_active`) keeps
using `task_queue.read_statuses` — GEECS-MCP's `run_tools` does, behind
its `analysis-run` extra.  `read_analysis_statuses(scan_folder)` returns
`{task_id: AnalysisStatus}` in filename order.  Schema-light on purpose:
`TaskStatus.to_dict()` in `ScanAnalysis/scan_analysis/task_queue.py`
stays the authoritative shape; `STATUS_FIELDS` here names its keys and
every field is coerced tolerantly (odd types → `None`/`()`, a torn or
non-mapping file → one `unreadable` entry, unknown keys ignored,
`.claim`/`.tmp` siblings skipped, `*.yaml` only — what the queue's own
readers glob — timestamps tz-aware with naive → UTC).
A missing scan folder or status dir reads as empty; nothing is ever
created.  The writer/reader contract is pinned in ScanAnalysis's suite
(`tests/test_analysis_status_contract.py`, the package that can import
both sides) — when the writer grows a field, extend `STATUS_FIELDS` +
`AnalysisStatus` in the same PR.

## Tiled catalog layer (`tiled_catalog` / `tiled_schema` / `tiled_drift`)

The Tiled analogue of `ScanPaths`/`ScanData`: day → scan → data over the
Bluesky runs a GEECS scan records to the lab Tiled server.  Pure and
Qt-free by design — consumed by the Data Portal's scan browser today and
intended for ScanAnalysis Tiled readers later (ScanAnalysis depends on
this package and must never depend on GeecsBluesky or a GUI package).

- **`tiled_catalog`** — the `ScanCatalog` protocol (`probe` / `list_runs`
  / `load_run`), `RunSummary`/`RunDetail`/`CatalogStatus` dataclasses,
  `summary_from_metadata`, the offline `StubCatalog`, and
  `TiledScanCatalog`.  `tiled` is imported lazily inside methods behind
  the existing `tiled` extra (the `tiled_export` pattern).  Day listing is
  one metadata-only search on the `start.time` epoch range (+
  `start.experiment` when set), newest first; the event table is the
  primary stream's **scalar table only** (`read_primary_scalars`: the
  composite node's table part through `primary.base`, never
  `run["primary"].read()`, which downloads every camera stack and
  per-frame attribute array and outer-joins their dimensions — #834; see
  `GeecsBluesky/TILED_SETUP.md`).  Connection details are
  constructor args; `from_config()` reads `[tiled]` from
  `~/.config/geecs_python_api/config.ini` with `configparser` — **never
  import `geecs_bluesky` here** (it depends on us).  Catalog methods may
  block on the network: interactive callers must dispatch them off the
  GUI thread.  Since the portal arc phase 2 the module also owns the
  shared front-end helpers `resolve_scan_folder` (RunDetail → existing
  scan folder, strictly read-only — the scan-folder invariant's
  tree-untouched pin lives in this package's suite) and `metadata_rows`
  (pure RunDetail → display rows), consumed by the data portal (and by
  the Qt console's scan browser until its deletion, 2026-09-14); their daily-path fallback is
  `scan_paths.daily_scan_folder`, the offline-first (None, never raise,
  never create) module-level companion to
  `ScanPaths.get_daily_scan_folder`.
- **`folder_catalog`** — the same `ScanCatalog` protocol over the scan
  **folders** on the share, for every scan Tiled never saw (LabVIEW
  Master Control, experiments not on the Bluesky path, pre-Bluesky days).
  `FolderScanCatalog` lists a day's `scans/ScanNNN` from the `ScanInfo`
  inis and synthesizes the start-doc keys `tiled_schema` reads (`motors`
  = the s-file column that records the scan parameter, alias and all;
  `num_points`/`shots_per_step`; `scan_folder`; `time` = the first
  shot's LabVIEW `DateTime Timestamp`, clamped to the folder's day). A
  loaded run has **`data=None`** on purpose: its scalars are the s-file,
  which `scan_frame` already reads for a run-less scan — one s-file
  reader. Completion = a non-empty `ScanEndInfo`, else the analysis
  s-file's presence (Master Control leaves `ScanEndInfo` empty always).
  Uids are `folder:{experiment}:{YYYY-MM-DD}:{number}`, parsed back by
  `load_run`. `MergedScanCatalog(primary, folders)` lists the primary's
  runs plus every folder whose scan number no primary run claims, routes
  `load_run` by uid prefix, and degrades to folders alone when the
  primary is down (re-raising only when the folders are empty too).
  A **finished** scan's documents are cached per catalog (bounded LRU by
  folder) — nothing feeding them changes after the end, and the portal
  re-lists the day on every scan page — so a finished day costs one
  `os.scandir` and no file opens; unfinished scans are re-read each call.
  Never key this on the ini's mtime: Master Control finishes a scan by
  writing the analysis s-file, not by touching the ini.
  Read-only; the tree-untouched pin is in `tests/test_folder_catalog.py`.
- **`tiled_schema`** — event-schema column semantics, ONE module,
  version-tagged (`TARGET_SCHEMA_VERSION = 1`);
  `GeecsBluesky/EVENT_SCHEMA.md` is the contract.  Anything that
  interprets a column name (companion suffixes, `telemetry_` prefix,
  pinned/reference-timestamp selection, scan-variable readback detection,
  `geecs_scalar_headers` prettification, NOSCAN/1D/GRID/OPT
  classification) belongs here, not in consumers.  When the schema
  evolves, touch this file.
- **`shot_join`** — the one home of the rule that joins a run's per-frame
  stream columns onto its shot rows (`FrameColumns`,
  `join_frames_to_shots`, `row_windows`, `shot_clock_column`,
  `frame_columns_from_attributes`, `SHOTS_STREAM`) — and a non-essential
  device's **event** stream onto them the same way
  (`frame_columns_from_events`, `non_essential_stream`: one event per stamp
  a triggered device without a plugin published, NaN on the rows it
  missed).  Pure arithmetic over
  arrays — no I/O, no Bluesky, no pandas — because two callers must agree
  exactly: the worker's live s-file callback, reading the stacks off the
  share, and `tiled_export`'s offline re-export, reading the same columns
  back out of Tiled.  **One** `drain_offsets` map covers both sides of every
  comparison (a shot's cross-device stamps differ by a per-device
  constant) — correcting only one side shifts
  the s-file by a row, which is wrong data rather than missing data, so the
  two sides must never come from two places.  Each row's match window is
  **its own**: half the shot period, narrowed to half the distance to its
  closest neighbour, so one anomalous pair of row stamps tightens only those
  two rows.  Ownership is then resolved **globally** — nearest pair first,
  one frame per row and one row per frame — rather than leaving that
  invariant to window arithmetic, which two rows published a fraction of a
  millisecond apart defeated.  A frame no row claims is
  an **orphan**: it stays in the stack and in Tiled and is left out of the
  s-file — one s-file row per essential shot, always.
- **`tiled_export`** — the legacy scalar files of a Bluesky run, live
  (`write_scalar_files` from the documents, what the worker calls) or
  offline (`write_scalar_files_from_tiled`).  The rows are `primary`'s
  events when it has them and the per-shot `shots` events otherwise (a
  gated run), and `join_frame_columns` appends every datum-only stream's
  per-frame columns (and every non-essential event stream's, named by the
  start document's `non_essential`) before `geecs_scalar_headers` renames and orders them
  — one projection for both shapes of run.  `read_frame_columns` reads a
  stream's 1-D attribute arrays **by name** and never the frame stack
  itself (the #834/#836 lesson).
- **`tiled_drift`** — pure "moved during scan" telemetry drift analysis
  (plain float sequences in, dataclasses out; zero Qt, zero pandas):
  |last − first| > 3σ of in-scan spread, σ ≈ 0 guarded by a relative
  epsilon, NaN/string samples tolerated (telemetry is dtype-tolerant per
  the event schema).

Tests are hermetic (`tests/test_tiled_*.py`) — fake client objects that
quack like Tiled search results; no network, no real catalog.

Data Utils has no intra-repo dependencies. `io.scan_stack.LABVIEW_EPOCH_OFFSET`
is a file-format constant, deliberately independent of Core's wire-format
constant. GeecsBluesky (which already depends on both) pins their equality;
sharing this integer does not justify an access-library dependency here.

## Three kinds of stack (0.36.0)

Since GeecsPvaGateway 0.13 the file plugin writes array variables as well
as images, so a scan folder holds three kinds of stack through one layout:

| content | frames | the axis lives in |
|---|---|---|
| image | `(N, H, W)` | pixel indices |
| lineout (MagSpec spectra) | `(N, n, 2)` | column 0 of the data |
| waveform (scope traces) | `(N, n)` | the per-frame `wave_x0` / `wave_dx` |

`scan_stack.stack_content_kind` tells them apart, and it does **not** guess
from the rank — `(N, H, W)` pixels and `(N, n, 2)` rows are both rank 3.
The plugin declares it: it writes the `wave_*` attributes for an array
variable and never for an image one. Read an array stack's shot with
`read_1d_data(ShotRef(stack, index), Data1DConfig(data_type="pva_stack"))`,
which returns the same `Data1DResult` a native scope file does — so a 1D
analyzer reads a Bluesky scan without learning a new concept.

**Never hand a consumer a padded frame.** The gateway serves arrays at
native length (GeecsPvaGateway 0.14.0), but the MagSpec lineout stacks it
recorded 2026-09-19..24 are NaN-padded to a fixed row count so a run had
one shape; padding makes two shots
of different true lengths *same-shaped*, which silently defeats the
shape guard a per-shot averager relies on (`average_data` in ScanAnalysis'
`single_device_scan_analyzer`) and averages column 1 index-wise over axes
that do not line up. The reader trims to the true length — a waveform's
declared `wave_samples` first, the pad boundary only as fallback — and
refuses a frame whose padding and declaration disagree. New readers of
these stacks go through `read_1d_data`; do not add a second un-padding
rule anywhere else.

## HDF5 over SMB — the reader's half of the contract

The per-device frame stacks are written from Windows onto an SMB share and
read back from Linux.  Three rules hold on this side, and the writer's half
(never SWMR, flush per frame, `locking=False`) is in
`GeecsPvaGateway/CLAUDE.md`:

1. **Never read a stack while it is being written.**  Analysis reads after
   the plugin closed the file — the `finalized` root attribute is the flag,
   and the stop document precedes it.  Live use is the PVA stream, never
   the file.  This is a contract, not best-effort.
2. **Readers open lock-free**, through `scan_stack.open_stack`
   (`locking=False`): the HDF5 lock across SMB is the known failure mode.
   A service that reads these files (Tiled) also needs
   `HDF5_USE_FILE_LOCKING=FALSE` in its unit environment —
   `GeecsBluesky/TILED_SETUP.md`.
3. **One frame per chunk** is the written layout, because the two access
   patterns are per-shot random access (`read_shot`) and whole-stack
   reads; whole-frame chunks serve both.
4. **A run keeps one handle.** Each open is several protocol round trips
   over SMB (~2 ms of a 5 ms per-frame read), so a consumer reading every
   frame of one stack opens it once and reads through `read_frame(f, i)`
   — the read behind `read_shot`, same bounds check — rather than
   `read_shot` per frame. ScanAnalysis's `V2ShotSource` does this per run
   and per pool worker.

## Key Dependency

- `nptdms` — TDMS binary file reading
- `pyarrow` — Parquet I/O
- `duckdb` — available for ad-hoc Parquet queries but minimal use
- `pydantic >= 2.0` — all data models
- `tiled[client]` — optional `tiled` extra (`tiled_export`, `tiled_catalog`)

## Two-axis geometry (`scan_grid`, 0.34.0)

`grid_scan(frame, start, GridConfig, RowFilters)` is a pure calculation over the
unfiltered union frame. Native grid metadata and product Sweep specifications
supply planned coordinates, indexed by one-based acquisition bin (including snake
traversal). Relative coordinates are anchored to the first available bin readback.
Other trajectories use measured per-bin coordinates without inventing empty cells.
Repeated XY positions remain separate visits. More than two motors is rejected.

Filters affect statistics/membership only. The existing binning core computes
independent center/error choices; quantile interval width is calculated from the
quantiles themselves, not clipped offsets from a center outside the interval.
Finite selected-scalar counts control error visibility. Bin identity uses one
provider (native when available, otherwise s-file), matching Images; namespaces
are never reconciled. Results include geometry, sample counts and member shots.
`tiled_schema.shot_axis_for_frame` is the shared shot-identity resolver used by
Grid and re-exported by the portal figures module for Plot/Images and notebooks.
It preserves the union frame's suffixed s-file fallback after name collisions.

### Completed-scan input references

`shot_files.map_shot_files(directory, rows, device=..., file_tail=...)` owns
the mapping formerly inside SingleDeviceScanAnalyzer. It reads directories,
stats native files and reads stack timestamps, never frame arrays or outputs.
Only use HDF5 discovery after the scan closes. A partial timestamp join never
falls back to shot-number filenames; only a zero join can. Native direct
stat probes bypass stale SMB listings. `prefer_stack=True` tries capture
frames first, while `stacks_only=True` refuses native fallback with
`StackMappingUnavailable`. The ScanAnalysis adapter translates that into its
`DataUnavailableWarning`; queue/status policy remains outside data-utils.

## HASO `.himg` capture stacks (0.46.0)

The HASO wavefront sensor saves natively (one `.himg` per shot, ~24.6 MB
of 8-bit values in 16-bit words) and no file plugin writes a stack for it.
`io.himg_stack.convert_himg_folder(device_dir, rows=)` writes the same
`<device>/<device>.h5` the PVA file plugin would have — `/entry/data/data`
`(N, H, W)` uint16 one frame per chunk, gzip + shuffle (~5.8x), the stamp
attribute `<device>-hdf-himg-frame_acq_timestamp` in Unix s — so
`find_stack_file`, the shot mapper's `prefer_stack`, the portal's gallery
and the core's source read it with no HASO-specific code. Stamps come from
the native filename, or for legacy `ScanNNN_<device>_NNN.himg` names from
the scan's `acq_timestamp` column (`rows`; the s-file / ScanData table).
What makes it lossless is `/entry/instrument/himg/`: each file's `header`
bytes (everything before the pixels — `io.himg` rebuilds header + frame
into the original bytes), `source_name`, `source_size`, `source_sha256`.
`verify_himg_stack` rebuilds every frame and checks that hash, and
`convert_himg_folder` runs it before renaming the `.part` file into place,
so a stack under the reader's name is one whose every frame rebuilds its
source. Converting adds that one file and nothing else: no directory is
ever created, the `.himg` files are never touched. `geecs-himg convert |
verify` is the shell form for the backlog; ScanAnalysis's `himg_to_stack`
kind is the per-scan click in the Data Portal.

**Compaction and restore (0.48.0, `io.himg_compact`)** — the one place
this package deletes. `compact_himg_folder(device_dir)` runs the stack's
own audit (`verify_himg_stack(against_files=True)`: every frame rebuilt
and checked against the recorded SHA-256 *and* against the `.himg` still
on disk) and only then deletes the `.himg` files, leaving
`himg_manifest.json` (what went, when, the stack that holds it) beside
the stack; the stack itself — its per-shot header rows, the `haso`
measure's sensor header — is never rewritten. The guards live in the
function so a click and a shell command refuse the same things:
`HimgFolderActive` while any `.himg` is younger than `MIN_SOURCE_AGE_S`
(a minute) or there is no closed-run evidence
(`data.sfile.run_closed_evidence(scan_folder)`: the `ScanDataScanNNN.txt`
the stop document writes — `scan_data_txt_path_for`, the one
construction of that path — else the analysis s-file;
`require_closed=False` is the shell's escape hatch for a dead scan),
`HimgStackIncomplete` for a `.himg` the stack has no frame for,
`NoHimgStack` without a `.himg` stack, a `.part` file as another
writer's, and on any disagreement nothing is deleted —
`HimgVerificationFailed` when a frame does not rebuild its hash (the
stack is damaged), `HimgSourceChanged` when a file on disk differs from
its intact frame (the file changed after conversion): opposite
remedies, so two errors and two `HimgVerifyReport` lists (`mismatches`
vs `changed`). `restore_himg_folder` rebuilds each file (`.part` +
rename, hash-checked first), keeps a file already there when it matches
and stops when it does not, and removes the manifest; mtimes are not
restored, bytes are. Both touch only the one device folder. The
converter's side of the contract: `write_himg_stack(overwrite=True)`
refuses with `HimgSourcesDeleted` when the existing stack holds frames
whose files are gone — a compacted folder's stack is the only copy, and
the way to reconvert is restore first. `geecs-himg compact | restore`
are the shell forms; ScanAnalysis's `himg_compact` (destructive — the
portal asks for the scan number, and `create_scan_analyzer` refuses the
kind without `allow_destructive`) and `himg_restore` kinds are the
clicks. Every long loop takes a `progress(done, total,
phase)` callback, and `io.himg_worker.run_himg_job` runs any of the four
jobs in a child interpreter, relaying progress, log records, the report
dataclass and the package's own error classes back over a JSON-lines
event stream — how the portal runs them out of its own process. Source
files are read through `read_source_bytes` (`posix_fadvise DONTNEED`
after the read) so a 44 GB scan does not sit in the service cgroup's page
cache.
