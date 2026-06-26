# Bluesky External Assets and Analysis Roadmap

## Purpose

This document sketches the path for representing native GEECS nonscalar files
as Bluesky external assets while preserving the current GEECS scan-folder
layout. The immediate goal is not to rewrite detector file saving. The goal is
to make files already written by GEECS/LabVIEW devices visible to Bluesky,
databroker/Tiled, notebooks, and later analysis tools through the standard
external asset model.

The first implementation has now proven the path with native saved camera image
files. HASO, richer scope trace handling, post-run analysis logging, and
optimization should build on the same foundation later.

## Current State

As of 2026-06-25, GeecsBluesky can command native nonscalar saving by setting
device `localsavingpath` and `save` controls, emit Bluesky `Resource`/`Datum`
documents for supported native files, preserve the existing string save-path
fields for compatibility, and locally fill camera image assets from Tiled run
metadata.

Implemented pieces:

- `geecs_bluesky.assets` contains asset specs, registry definitions, document
  construction helpers, and a camera image handler.
- Asset behavior is keyed by GEECS database device type, not by individual
  device name.
- `GeecsDb.get_device_type(device_name)` resolves device type strings from the
  real GEECS database.
- The registry covers the known native nonscalar file conventions from the
  legacy file mover policy:
  - `"Point Grey Camera"` camera PNG files
  - `"FROG"` spatial and temporal image directories
  - TDMS pair devices such as `"PicoscopeV2"`,
    `"Thorlabs CCS175 Spectrometer"`, `"RohdeSchwarz_RTA4000"`, and
    `"ThorlabsWFS"`
  - `"MagSpecCamera"` and `"MagSpecStitcher"` image/text variants
- Sync native-file-saving detectors emit datum-id event fields plus
  `Resource`/`Datum` docs during acquisition.
- Existing `<device>-nonscalar_save_path` string fields are kept for now.
- Unregistered device types are ignored by the asset layer rather than breaking
  acquisition.
- Device-server save paths are translated from the scanner-local mount path to
  the configured LabVIEW/device-server base path before writing
  `localsavingpath`.
- Tiled ingestion is guarded so an external-asset persistence failure warns and
  disables that callback instead of aborting a scan.
- `tiled_external_asset_readback.ipynb` demonstrates the local-first archive
  readback path: user enters date, scan number, device, and shot number; the
  helper queries Tiled for the real event row, resolves the datum ID, and fills
  the camera image locally.
- `load_camera_image_from_tiled(...)` and
  `load_camera_image_from_tiled_run(...)` provide the notebook-proven library
  path for camera images.
- Local readback can map Windows/device-server roots such as `Z:/data` to local
  mounts such as `/Volumes/hdna2/data` via explicit `root_map` or the shared
  GEECS config.
- Strict shot-control fallback was removed in PR #441. `strict_shot_control`
  now means true plan-owned single-shot with a reachable `ARMED` shot-control
  state. Free-running behavior should be requested explicitly with
  `free_run_time_sync`.

Important discoveries:

- The database device type string for ordinary cameras is
  `"Point Grey Camera"`, with spaces. It is not `PointGreyCamera`.
- Direct LabVIEW/native saving writes raw filenames such as
  `<device_name>_<acq_timestamp>.png` inside the device directory.
- The legacy file mover later prepends `ScanXXX_` and may translate acquisition
  timestamps to shot-number-based filenames. That legacy post-processing name
  is not the filename emitted directly by the device server.
- Asset field names should not be normalized into a new semantic style. They
  should preserve the registry's native names, such as `interpSpec` and
  `interpDiv`, and then pass through the existing GeecsBluesky safe-name logic.
- On non-Windows scanner hosts, the path sent to device servers must use the
  configured `geecs_device_server_data_base_path` such as `Z:/data`, even when
  the scanner itself reads files through `/Volumes/hdna2/data`.

During a run, events include enough context to locate native files:

- scan folder metadata in the start document
- nonscalar save directory metadata
- per-event `acq_timestamp`
- detector/device identity
- datum IDs for registered external assets

The system is now partway through the desired shift:

```text
legacy compatibility:
  event data still contains save path + timestamp fields

current Bluesky direction:
  event data contains datum_id for registered native files
  Resource/Datum docs describe where the native file lives
  handlers know how to open that file when requested
```

The main remaining gap is Tiled-side support for these custom GEECS asset specs.
Until the lab Tiled server has matching readers/adapters, GEECS datum IDs are
stored as ordinary event metadata there rather than being fully fillable through
Tiled.

## Validation Snapshot

Hardware validation has covered:

- real database lookup for camera device type resolution
- NOSCAN and STANDARD scanner acquisition
- DG645 strict shot-control NOSCAN acquisition
- native saved camera assets with `Resource`/`Datum` documents
- emitted resource paths resolving to real PNG files on disk
- scanner-local to device-server path translation
- local Tiled readback of a camera asset from an archived Bluesky run

Known example for local readback validation:

```text
experiment: Undulator
date: 2026-06-24
scan: Scan006
device: UC_Amp2_IR_input
shot: 2
```

`UC_Amp2_IR_input` was used for the clean native-save hardware smoke test.
`UC_TopView` was reachable but had a physical camera/save issue during testing,
so it should not be treated as the validation source for this checkpoint.

On 2026-06-25, a GUI-launched strict-shot-control scan on Windows produced
matching event/file counts, but the requested shot count was observed as `n+1`
relative to the UI request. That appears separate from asset association and is
likely tied to the scanner's time-derived `shots_per_step` model; track it as a
scan-configuration semantics issue, not an external-assets blocker.

## Design Principle

Asset behavior should be keyed by **GEECS device type**, not by individual
device name.

For example, `UC_TopView` should not be special-cased. The database should tell
the backend that `UC_TopView` is a `"Point Grey Camera"` device, and the asset
system should look up the registered behavior for that type.

This keeps the model scalable:

```text
device name -> GeecsDb -> device type -> asset registry -> asset docs/handler
```

## Proposed Package Boundaries

```text
geecs_data_utils.io
  Generic file readers.
  No Bluesky.
  No ImageAnalyzer.
  No optimization.

geecs_bluesky.assets
  Asset spec names.
  Device-type asset registry.
  Resource/Datum document emission.
  Handler classes.

geecs_bluesky.analysis
  Bluesky-run-to-analysis adapters.
  Analysis result logging.
  Derived-run helpers.

image_analysis
  Per-image science logic.
  Feature extraction.

scan_analysis
  Legacy scan-folder analysis plus reusable aggregation/reporting pieces.
```

## Phase 1: External Asset Foundation

### Goals

- Define a small `geecs_bluesky.assets` layer.
- Add a device-type asset registry.
- Emit formal Bluesky external asset references for native GEECS files.
- Keep physical files in the normal `Scan###` directory tree.
- Preserve scan-folder compatibility with the legacy GEECS ecosystem.

Status: implemented in the current PR for the first supported device classes,
with compatibility fields preserved.

### MVP Target

Start with native camera PNG files written by GEECS camera devices. This is now
implemented for `"Point Grey Camera"` devices.

Direct native camera files:

```text
<scan_folder>/<device_name>/<device_name>_<acq_timestamp>.png
```

where `acq_timestamp` is formatted to three decimal places.

The legacy file mover may later rewrite or copy these into names such as
`ScanXXX_<device_name>_<shot_number>.png`. The asset registry should point at
the direct native save first, because that is the file available during Bluesky
acquisition.

The exact registry entry should be keyed by the database device type, for
example:

```text
device_type: Point Grey Camera
spec: GEECS_CAMERA_IMAGE
extension: .png
event field: image
handler: GeecsCameraImageHandler
```

Do not hardcode `UC_TopView` behavior except in tests or examples.

`"Point Grey Camera"` is the GEECS database device type used for all basic
cameras, regardless of actual manufacturer. Conceptually this is closer to a
generic NI-IMAQ camera type, but the asset registry should use the database
string as it exists today.

### Registry Shape

The registry should encode file conventions by device type:

```python
AssetDefinition(
    device_type="Point Grey Camera",
    spec="GEECS_CAMERA_IMAGE",
    event_field="image",
    extensions=(".png",),
    files_from_event=...,
    handler="GeecsCameraImageHandler",
)
```

The `files_from_event` logic should be able to use:

- device name
- device type
- scan number
- scan folder
- nonscalar save directory
- event timestamp / `acq_timestamp`
- detector event field names

Some device types emit multiple files for one event. The current PR handles the
known multi-file conventions from the scanner file mover policy, but new device
types should still be added explicitly and tested against real database device
type strings.

### Database Requirement

`GeecsDb` exposes a lightweight query for device type:

```python
GeecsDb.get_device_type("UC_TopView") -> "Point Grey Camera"
```

Use `GEECS-PythonAPI` only as a reference for table/column names if needed.
Do not import or depend on it from `GeecsBluesky`.

This should continue to be tested with the real lab database when available,
because stale assumptions about device type strings are easy to miss in pure
unit tests.

### Asset Document Shape

For the MVP, prefer standard `Resource`/`Datum` external asset docs unless a
specific databroker/Tiled constraint requires `StreamResource`/`StreamDatum`.

Conceptually:

```text
Resource
  uid: ...
  spec: GEECS_CAMERA_IMAGE
  root: /Volumes/hdna2/data/Undulator/Y2026/.../scans/Scan042
  resource_path: UC_Amp2_IR_input/UC_Amp2_IR_input_1234567890.123.png
  resource_kwargs: {}
  path_semantics: posix

Datum
  datum_id: ...
  resource: <resource uid>
  datum_kwargs: {}

Event
  data:
    uc_amp2_ir_input-image: <datum_id>
  filled:
    uc_amp2_ir_input-image: false
```

Use `root` + `resource_path` rather than storing only absolute strings in the
event. This gives future clients a clean place to handle mount differences
between lab machines, analysis workstations, and archives.

For the first implementation, choose `root` so a normal consumer with the
NetApp mapped can access the file directly from `root / resource_path`. The
exact split can be adjusted, but it should not require consumers to know GEECS
scan-folder conventions just to resolve the file.

### File Completion

The file may not exist at the exact instant the event document is assembled.
For the camera MVP this should not block the design, because the filename is
deterministic and files are expected to appear quickly.

Implementation options:

- emit deterministic asset docs during acquisition and let the handler retry
  briefly if the file is not visible yet
- defer asset association until after the run if a device type cannot provide
  deterministic per-event filenames

For cameras, prefer emitting asset docs during acquisition. This is the path
needed for live consumers, and the filename is deterministic from the scan
context, device name, and event `acq_timestamp`.

The registry should allow post-run association eventually for device types that
cannot provide deterministic per-event filenames.

## Phase 2: Reader and Handler Layer

Generic file readers have landed in `geecs_data_utils.io`, and
`geecs_bluesky.assets.handlers` contains a thin camera image handler.

Initial handler:

```text
GEECS_CAMERA_IMAGE
  -> geecs_data_utils.io image reader
  -> np.ndarray
```

Handlers should be thin I/O adapters:

```text
Resource + Datum -> loaded array/object
```

They should not run scientific analysis, make plots, decide pass/fail, or
aggregate scan results.

Status: implemented for camera images, including local Tiled event lookup and
local fill. Text-array and TDMS-oriented specs are represented in the registry,
but richer handler/readback behavior should be validated separately before
relying on them for analysis.

## Phase 3: Post-Run Analysis

After native files can be represented and filled as external assets, add a
post-run analysis layer that consumes Bluesky/Tiled runs.

Desired flow:

```text
raw run
  scalar data
  external asset references

post-run analysis job
  stream events
  fill assets event-by-event or in chunks
  call ImageAnalysis feature extractors
  persist derived results linked to the raw run
```

This should not route Bluesky runs through legacy scan-folder reconstruction
unless the input is truly a legacy scan. For Bluesky runs, the event/resource
model should already encode which file belongs to which shot.

Possible output forms:

- derived analysis run linked to the raw run UID
- analysis event stream
- Parquet/CSV summary indexed by run UID and scan number
- processed external assets such as plots, reports, or normalized arrays

The analysis product should record:

- raw run UID
- analysis code version
- analysis configuration
- input asset IDs or resource IDs
- derived scalar/table/asset outputs

## Phase 4: Optimization

Optimization should use a narrow feature-extraction contract, not the full
legacy `ScanAnalysis` orchestration.

Conceptual loop:

```text
Xopt asks for next point
Bluesky moves hardware and acquires shot(s)
asset handler fills required image/trace data
feature extractor computes objective/constraints
Xopt receives result
Bluesky records objective diagnostics
```

The optimizer usually needs a small, explicit result:

```text
objective
constraints
diagnostic scalars
optional references to raw/derived assets
```

Do not block Phase 1/2 on optimization design. The asset model is the data
access foundation optimization will need later.

## Deferred Scope

Do not include the following in the first external-assets implementation unless
a later PR explicitly scopes it:

- HASO `.himg` handling
- full scope trace interpretation beyond native TDMS asset references
- direct HDF5/Zarr/TIFF writing by Bluesky
- post-run analysis result persistence
- Xopt integration
- new user-facing config schemas
- splitting `geecs-bluesky` into a separate repo

## Remaining Work

- Add proper Tiled/databroker readers/adapters for custom GEECS asset specs so
  assets are fillable through Tiled instead of stored only as datum-id metadata.
- Decide whether Bluesky native-save runs need a finalization/mover step for
  legacy filename compatibility, or whether direct native filenames are the
  canonical Bluesky path.
- Add opt-in integration coverage for local Tiled camera fill using a known run
  such as 2026-06-24 Scan006 / `UC_Amp2_IR_input`.
- Investigate the GUI/requested-shot-count `n+1` behavior separately from
  external-asset readback.
- Expand handler validation beyond camera PNGs only when there is a concrete
  consumer for those file types.
- Design the post-run analysis output contract for derived scalars, tables, and
  processed assets linked to the raw run.

## Testing Strategy

Resolved choices:

- `root` should be the scan folder, so `root / resource_path` directly resolves
  to the native file for a consumer with the NetApp mounted.
- Event field names should follow the current GeecsBluesky safe-name convention:
  `<safe_device_name>-<safe_asset_field>`, for example `uc_topview-image`.
- Asset docs should be emitted during acquisition for deterministic native file
  conventions, especially camera images.

No-hardware tests should continue to cover:

1. Create a temporary scan-like folder with a synthetic camera PNG named using
   the GEECS convention.
2. Generate a synthetic Bluesky run containing an event with matching
   `acq_timestamp` and a `"Point Grey Camera"` device type.
3. Emit `Resource`/`Datum` docs for that image.
4. Fill the event through the `GEECS_CAMERA_IMAGE` handler.
5. Assert the filled value is the expected NumPy array.

This proves the path construction, asset docs, handler registration, and fill
behavior without requiring live hardware. A separate hardware harness or
notebook should validate that real devices produce the expected filenames and
that Tiled ingestion/readback works in the lab environment.

## Recommended Next PRs

1. Promote the notebook-proven Tiled camera readback path into a stable, small
   API surface for "date/scan/device/shot -> filled camera asset"; keep the
   notebook as a thin example.
2. Add opt-in integration coverage for that API using a known archived run and
   local root mapping.
3. Run a fresh strict `ARMED` native-save hardware check after PR #441 and
   confirm event count, file count, and asset fill agree.
4. Define the minimal post-run feature-extraction result schema for analysis and
   optimization consumers.
5. Implement a small post-run analysis runner that consumes Bluesky runs, fills
   image assets in chunks, and writes derived results linked to the raw run.
6. Add HASO/scope-specific handlers only after the camera/Tiled path is proven
   end-to-end.
