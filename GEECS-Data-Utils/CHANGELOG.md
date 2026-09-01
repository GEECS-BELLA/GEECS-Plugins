# Changelog — geecs-data-utils

All notable changes to this package will be documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.26.0] - 2026-09-01

### Added

- `analysis_status` — the one tolerant, read-only reader for a scan
  folder's `analysis_status/*.yaml` task files (#682):
  `read_analysis_statuses(scan_folder) -> dict[task_id, AnalysisStatus]`
  (+ `read_analysis_status`, `parse_status_timestamp`,
  `analysis_status_dir`, `STATUS_DIR_NAME`/`STATUS_FIELDS`/
  `STATUS_STATES`), exported at the package root as `AnalysisStatus` /
  `read_analysis_statuses`. Schema-light by design: the writer
  (ScanAnalysis `task_queue.TaskStatus.to_dict()`) stays authoritative;
  every field is coerced tolerantly (odd types degrade the field, a torn
  or non-mapping file degrades that one entry to `unreadable`, unknown
  keys are ignored, `.claim`/`.tmp` siblings are skipped, `.yml` is
  accepted, timestamps naive→UTC — and, beyond the consumer copy it
  replaces, an unquoted stamp YAML already parsed into a `datetime` is
  accepted too). A missing scan folder or status dir reads as empty;
  nothing is ever created (the scan-folder invariant). The writer/reader
  contract is pinned cross-package in ScanAnalysis's suite
  (`tests/test_analysis_status_contract.py`), which can import both
  sides. Promoted from GEECS-MCP's local `_read_task_statuses`.

## [0.25.0] - 2026-09-01

### Added

- `io.average_frames(frames, label=...)` — THE shared `nanmean` frame
  averager (per-bin averaged images for the portal's Images tab, W2a).
  Accepts a list of frames or an already-stacked array; NaN pixels are
  excluded per-position; empty input or inhomogeneous shapes degrade
  to `None` with a `label`-attributed warning instead of raising (the
  `np.asanyarray` hard-error on numpy 2.x). `ImageAnalyzerResult`'s
  internal averager delegates to it as of ImageAnalysis 1.13.0; the
  remaining divergent copies (ScanAnalysis `average_data`'s `np.mean`,
  the hand-rolled mean in `HIMG_with_average_saving`) converge in
  trailing PRs.

## [0.24.0] - 2026-08-30

### Added

- `tiled_schema.is_key_timestamp_column` — the `ts_` per-key
  event-recording timestamps Tiled adds when reading the primary
  stream (machinery, not measurements: front-ends hide them from pick
  lists by default), and `tiled_schema.timestamp_epoch` — the
  two-epoch naming convention (`ts_*` = Unix event times;
  anything named `acq_timestamp` — event companions and s-file headers
  alike — = LabVIEW wire epoch), so renderers can show timestamps as
  real datetimes instead of raw seconds. (Analysis-tabs W1e, owner
  feedback on the live Plot tab.)

## [0.23.0] - 2026-08-30

### Changed

Analysis-tabs wave W1c — the binning rewrite (the review's flagship
"improve"; the old property had no production consumers, so no
migration):

- **`data/binning.py`** — `bin_frame(frame, cfg) -> BinnedFrame`, the
  pure stateless core: same `BinningConfig` vocabulary (moved here,
  re-exported from `scan_data` so existing imports keep working), same
  `(col, {center, err_low, err_high})` output schema, minus the warts —
  no `id(df)` cache key, label-aligned error assignment (no positional
  `.values`), vectorized `mad` (the per-group Python `apply` is gone),
  one-shot output assembly (the pandas `PerformanceWarning` from
  fragmented inserts is gone — pinned live with `-W error`), counts as
  a separate series on `BinnedFrame` (no `("count","center")` shape
  special-case), and clean quantile extraction (the 50-line defensive
  unstack ladder is gone).
- **Deliberate semantic fixes**: default `value_cols` now excludes
  `Shotnumber` (as the config docstring always promised) while keeping
  the bin source column (its per-bin center is the natural X axis);
  the `dropna` row policy excludes the bin key (a grouping key is not
  a measurement — including it made `dropna="all"` vacuous) and the
  all-NaN-column guard now applies to both policies. Visible symptom
  of the dropna change: rows whose **bin column** is NaN now survive
  `dropna="any"` and aggregate into their own NA-labelled bin row
  (previously silently dropped) — filter upstream when unwanted.
- **Bin-cache invalidation on direct reassignment**: `data_frame` is
  now a property whose setter invalidates the binned-scalars cache, so
  `sd.data_frame = df` (a supported pattern) can never serve stale
  binned results (the old `id(df)` key recomputed on reassignment by
  accident; the first rewrite pass lost that).
- Empty binning results (all rows dropped, or an empty frame) keep the
  two-level column MultiIndex instead of degrading to flat columns.
- **`ScanData.bin(config)` exists** — the API the docs always
  advertised; `binned_scalars` stays as a compatibility wrapper
  delegating to `bin_frame` and re-attaching the legacy
  `("count","center")` column. 22 hand-computed unit tests pin every
  err mode (the numbers the LabVIEW-source comparison will check);
  the canonical-scan integration test passes on real data.
- Docs notebook `basic_usage.ipynb` refreshed onto `sd.bin(...)`.

## [0.22.0] - 2026-08-30

### Added

Analysis-tabs wave W1b: `data/row_filters.py` — GEECSplotter's
"outer OR, inner AND" filter model as Pydantic vocabulary
(`RowFilters` → named `FilterGroup`s of `FilterCondition`s, each a
`within`/`outside` inclusive bounds pair) with `filter_mask` (the
composable pass mask — `.sum()` is the live count) and `apply_filters`.
Deliberately NOT lowered onto `apply_row_filters` (a mask-returning,
OR-capable, explicit-NaN primitive can't be built on the AND-only
frame-returning kernel; the legacy tuple vocabulary stays for its
consumers — composing `RowFilters` into `DatasetBuilder` is future
work). NaN handling is an **explicit** `nan_policy` (`exclude` fails
the condition under both modes — the complement must not silently pass
NaN — `keep` passes); NaN bounds are refused at validation (±inf stays
legal for half-open ranges); datetime/timedelta columns and duplicated
labels are refused loudly per the `numeric_series` coercion doctrine.
Disabled and empty groups are ignored, and no active groups is the
identity, never an empty result. JSON round-trip pinned (the
endpoint/URL/config form).

## [0.21.0] - 2026-08-30

### Added

Analysis-tabs wave W1a (the substrate —
`Planning/data_portal/03_analysis_tabs_design.md`):

- `data/sfile.py` — THE s-file reader + path convention, one home:
  `read_sfile(path)` (tab-separated, headers verbatim) and
  `sfile_path_for_scan(scan_folder)` (`{day}/analysis/s{N}.txt`,
  unpadded). `ScanData.load_scalars` now delegates (pinned by a spy
  test); the ScanAnalysis duplicate reads converge in a trailing PR.
  **Behavior note**: the old path derived the s-file via
  `get_analysis_folder()`, which creates `analysis/ScanNNN/` — so every
  s-file read used to mkdir on the share. Creating analysis folders is
  permitted (only `scans/ScanNNN` creation is the hard invariant), but
  a *read* shouldn't write as a side effect: the delegated path is pure
  (pinned by a tree-untouched test).
- `scan_frame.py` — the union-with-provenance frame:
  `scan_frame(detail, scan_folder)` unions the Bluesky event table and
  the s-file with per-column provenance (`run`/`sfile`, `computed`
  reserved), outer-joined on shot identity — **`Shotnumber ==
  scan_event_index`, both 1-based**. No name reconciliation; exact
  collisions suffix the s-file column. Either provider may be absent;
  corrupt s-files degrade to run-only. Strictly read-only (tree
  untouched pinned).
- Canonical-scan registry gains `undulator_bluesky_1d` (2026-08-29
  Scan 1 — both providers); new integration tests exercise `read_sfile`
  on a real s-file and the full catalog→folder→union path against the
  live Tiled server (verified on lab data 2026-08-30).

## [0.20.2] - 2026-08-30

### Changed

- CLAUDE.md truth-up (docs-only): the Binning System section documented
  a `sd.bin(config)` method that does not exist (the real API is
  `set_binning_config` + the `binned_scalars` property) and claimed
  ScanAnalysis renderers consume it (they do their own binning; the
  property has no production consumers). Corrected, with a pointer to
  the planned pure `bin_frame` rewrite
  (`Planning/data_portal/03_analysis_tabs_design.md`).

## [0.20.1] - 2026-08-29

### Changed

- `tiled_schema.plottable_columns` docstring restored to the shared-rule
  claim — true again now that the console's B4 consumes the helper
  (docs-only).

## [0.20.0] - 2026-08-29

### Added

Consolidation of the shot↔file join machinery (2026-08-29 review — the
"created a shared seam but left sibling copies alive" cluster); one
implementation each, consumed by ScanAnalysis and the data portal:

- `native_files.probe_native_file` — THE exact-key native-file stat
  probe (direct stats bypass stale SMB listing caches; ±1 ms is `%.3f`
  rounding canonicalisation, never a tolerance window).
- `io.scan_stack.stack_frame_index_map` (**keep-first** on duplicate
  millisecond keys — the deterministic contract; the portal's private
  copy was keep-last, a real parity break) +
  `frame_index_for_timestamp` + `read_shot_for_acq_timestamp` (the
  single-shot join in ONE h5py open — the gallery's hot path was three
  opens per request).
- `tiled_schema.normalize_token` — the device↔column matching rule,
  public (ScanAnalysis's byte-identical private copy now delegates).
- `scan_paths.VENDOR_ONLY_EXTS` + `io.images.DISPLAYABLE_IMAGE_EXTS` —
  the one extension taxonomy (three uncoordinated consumer-local sets
  before). `_ACCEPTABLE_EXTS` gains `has` (the missing sibling of the
  already-accepted `himg`), so a `.has`-only HASO folder infers its real
  extension instead of defaulting to `png` — retires the portal's
  `_vendor_only` directory-scan workaround.
- `tiled_catalog.fmt_time_of_day` — the shared HH:MM run-list formatter
  (the portal's copy retired; the console's B1 swap rides the B4 rewire).

## [0.19.0] - 2026-08-29

### Changed

- `tiled_schema.numeric_series` vectorized (the per-value Python loop
  was 7–13× slower and ran on the full uncapped frame per portal page
  view) and hardened: pandas nullable dtypes (`Int64`/`Float64` with
  `pd.NA`) no longer raise `TypeError`, datetime/timedelta columns are
  no longer reported "plottable" as ~1e18 ns integers, and a duplicated
  column label returns `None` instead of raising.  The returned series
  is now always `float64`.
- `tiled_catalog.resolve_scan_folder`: a run whose start doc names no
  experiment no longer falls through to the daily re-base —
  `daily_scan_folder` would substitute the host config's default
  experiment, whose same-numbered `ScanNNN` is a different scan's data.
- `tiled_schema.plottable_columns` docstring corrected: the console B4
  has not yet been rewired onto the shared helpers (the "cannot drift"
  claim was aspirational); pinned end-to-end fallback test added for
  `resolve_scan_folder` through the real `daily_scan_folder`
  construction (2026-08-29 review findings).

## [0.18.0] - 2026-08-29

### Added

- `tiled_schema.device_acq_timestamp_column` — schema-safe matching of a
  device name / on-disk folder stem to its `-acq_timestamp` event column
  (runs of non-alphanumerics collapse to underscores, the ScanAnalysis
  matching rule).  Added on portal-arc phase-4 review: the portal was
  re-deriving `safe_name` with a spaces-only mangle that silently missed
  hyphenated diagnostic stems and degraded to ordinal file joins.

## [0.17.0] - 2026-08-29

### Changed

- `tiled_catalog.resolve_scan_folder`: a recorded `scan_folder` start-doc
  path that does not exist on THIS host no longer short-circuits to
  `None` — it falls through to the daily-path fallback, re-basing onto
  the local data root.  The recorded path is host-specific (the Linux
  worker records `/mnt/...`; clients mount the share at `/Volumes/...`
  or `Z:`), so the old behavior made every cross-host consumer (portal
  gallery, console Open button off the worker) fail spuriously.  Found
  live on the portal-arc phase-4 smoke against real runs.

## [0.16.0] - 2026-08-29

### Added

- `tiled_schema.plottable_columns` / `tiled_schema.numeric_series` — the
  ONE scalar-plot pick-list rule for front-ends over the catalog
  (console scan browser B4, data portal): machinery excluded via
  `data_columns`, plottability by tolerant coercion per the
  dtype-tolerant telemetry contract (numeric strings plot; all-NaN and
  non-numeric columns do not).  Added on portal-arc phase-3 review — the
  portal was re-deriving the semantics with a raw dtype check and
  diverging from the console.

## [0.15.0] - 2026-08-29

### Added

- `tiled_catalog.resolve_scan_folder` and `tiled_catalog.metadata_rows` —
  the console scan browser's RunDetail helpers moved down (portal arc
  phase 2) so the scan browser and the data portal share one
  implementation.  `resolve_scan_folder` is strictly read-only (repo
  scan-folder invariant; the tree-untouched pin now lives in this
  package's suite).
- `scan_paths.daily_scan_folder` — offline-first module-level companion
  to `ScanPaths.get_daily_scan_folder` (returns `None` instead of
  raising when the data root or experiment is unresolvable; never
  creates directories).  Re-based from the console's
  `ops_paths.todays_scan_folder`, which now delegates here.

## [0.14.0] - 2026-08-27

### Added

- **`io/scan_stack.py` — reader for per-device capture frame stacks**
  (`geecs-capture/*`, the contract in
  `GeecsBluesky/geecs_bluesky/capture/FORMAT.md`): `find_stack_file` /
  `is_stack_file` (schema-attribute dispatch, never extension),
  `read_stack_timestamps` (with LabVIEW-epoch conversion via the new
  `LABVIEW_EPOCH_OFFSET` constant), `read_shot` (one chunk read), and
  `ShotRef` — a `Path` subclass carrying a frame index that travels
  per-shot analysis pipelines (pickles correctly for process pools).
  Read-only by design: producing stacks is the capture daemon's job.

## [0.13.6] - 2026-08-20

### Changed

- Docs-only: `native_files.py` cross-reference updated to the optimization
  stack's new home (`geecs_bluesky.optimization.session_bridge`).

## [0.13.5] — 2026-07-25

### Fixed

- **`decode_imaq_image_string`: tail-anchored fallback for wrappers without the
  repeated-name payload anchor** (rolled forward from master, released there as
  0.12.1). Some cameras (observed live: `UC_Amp2_IR_input`) flatten frames
  without repeating the device-name string before the pixel block, so every
  frame failed with "payload not divisible by rows". Dispatch is structural:
  when the name never repeats, geometry and pixel type (IMAQ type code, header
  offset +36) come from the IMAQ struct and the pixel block is the last
  `rows * stride` bytes (stride 64-byte aligned); anchored wrappers keep the
  strict loud-error path. Verified against a captured live frame.

## [0.13.4] — 2026-07-16

### Changed

- **`tiled_export` uses the canonical config reader** (issue #527): the
  private `_read_tiled_config` duplicate is deleted in favor of
  `tiled_catalog.read_tiled_config`, so the package no longer carries two
  parsers of the same `[tiled]` section.  Pinned by an identity test in
  `tests/test_tiled_export.py`.

## [0.13.3] — 2026-07-13

### Fixed

- `scan_log_loader.HEADER_RE` now parses **Bluesky** scan.log lines: the
  Bluesky stack writes a `scan=ScanNNN` context token where the legacy
  engine wrote `shot=<n>`, so every line of a Bluesky scan.log failed the
  header regex and log triage reported zero entries on perfectly good logs
  (observed live 2026-07-13, Undulator Scans 1–3).  Both tokens are now
  accepted; the capture keeps its historical `shot` name so downstream
  consumers (GEECS-LogTriage) are unaffected.

## [0.13.2] — 2026-07-13

### Changed

- Docstring condensation (docs-only): `tiled_catalog`'s module docstring
  states the config/dependency-direction rule in one line (the rationale
  lives in this package's `CLAUDE.md`), and `tiled_drift`'s
  `RELATIVE_SIGMA_EPSILON` comment defers to the module docstring's σ ≈ 0
  explanation instead of repeating it.

## [0.13.1] — 2026-07-12

### Fixed

- `TiledScanCatalog.load_run` no longer raises on runs whose primary stream
  holds no event rows (aborted or legacy runs read back as a dimensionless
  xarray Dataset, where `to_dataframe` raises "no valid index for a
  0-dimensional object" — hit live in the scan browser's first session).
  Such runs now degrade to `data=None` with a log line, the same contract
  as a missing primary stream.

## [0.13.0] — 2026-07-12

### Added

- **Tiled scan-catalog layer** — the Tiled analogue of `ScanPaths`/`ScanData`
  (day → scan → data over Bluesky-recorded runs), pure and Qt-free, consumable
  by GUIs (the GEECS-Console scan browser) and batch analysis alike:
  - `geecs_data_utils.tiled_catalog` — the `ScanCatalog` protocol,
    `RunSummary`/`RunDetail`/`CatalogStatus` dataclasses,
    `summary_from_metadata`, the offline `StubCatalog`, and
    `TiledScanCatalog` (lazy `tiled` import behind the existing `tiled`
    extra; metadata-only day listing via a `start.time` range +
    `start.experiment` search; event table read with the repo-blessed
    `run["primary"].read()` pattern).  Connection details are constructor
    args; `TiledScanCatalog.from_config()` reads the `[tiled]` section of
    the shared `config.ini` directly with `configparser` — no
    `geecs_bluesky` import (dependency direction preserved).
  - `geecs_data_utils.tiled_schema` — event-schema v1 column semantics in
    ONE version-tagged module (`GeecsBluesky/EVENT_SCHEMA.md` is the
    contract): row-identity columns, per-device companion suffixes,
    `telemetry_` Tier-2 prefix, `geecs_scalar_headers` display-name
    prettification, reference-timestamp/pinned-column selection,
    scan-variable readback detection, scan-shape classification
    (NOSCAN/1D/GRID/OPT), planned shot totals.
  - `geecs_data_utils.tiled_drift` — pure "moved during scan" telemetry
    drift analysis: a column drifts when |last − first| exceeds 3σ of its
    in-scan spread, with a relative-epsilon guard for σ ≈ 0, NaN/string
    tolerance (dtype-tolerant telemetry), and significance-sorted results.
- Hermetic tests for all three modules (fake Tiled client objects — no
  network): `tests/test_tiled_catalog.py`, `tests/test_tiled_schema.py`,
  `tests/test_tiled_drift.py`.

## [0.12.0] — 2026-07-06

### Added

- **`geecs_data_utils.native_files` — THE native-file naming contract.**
  A GEECS device's natively saved per-shot file is named
  `{stem}_{acq_timestamp:.3f}{file_tail}` inside a per-device folder named by
  the same stem (`{device_name}{directory_suffix}`). This convention was
  previously implemented independently in three packages — GeecsBluesky's
  asset registry (producer), ScanAnalysis's `SingleDeviceScanAnalyzer`
  (reader), and GEECS-Scanner-GUI's optimization session bridge (waiter) —
  and had already drifted (only ScanAnalysis carried the ±1 ms
  rounding-boundary canonicalization). The new module is the single source
  of truth: filename/stem/path construction (`native_file_name`,
  `native_file_stem`, `native_file_path`, `native_file_name_from_key`,
  `render_timestamp`), integer-millisecond canonicalization for row↔file
  joins (`timestamp_key`, `timestamp_key_candidates` with the ±1 ms
  rounding-boundary candidates), the filename-timestamp extraction pattern
  (`filename_timestamp_regex`), and the legacy Master Control
  `Scan{NNN}_{device}_{shot:03d}{tail}` pattern (`legacy_filename_regex`).
  All exported from the package root; contract pinned by
  `tests/test_native_files.py`.

## [0.11.0] — 2026-06-30

### Changed
- `scan_analysis_config` now bootstraps from `SCAN_ANALYSIS_CONFIG_DIR` or
  `scan_analysis_configs_path` in the shared GEECS user config, making the
  unified Scan/ImageAnalysis config tree the canonical runtime config root.
- `image_analysis_config` remains available for legacy callers but now resolves
  from the same unified Scan/ImageAnalysis config root as `scan_analysis_config`;
  `IMAGE_ANALYSIS_CONFIG_DIR` is no longer an active discovery path.
- Moved the config-driven 1D file readers (`read_1d_data`, `Data1DConfig`,
  `Data1DType`, `Data1DResult`) into `geecs_data_utils.io.array1d` so Bluesky
  and ImageAnalysis can share line/scope/spectrum loaders without depending on
  the `image_analysis` package.

### Fixed

- Deployments configured only via the legacy `IMAGE_ANALYSIS_CONFIG_DIR`
  env var or `Paths.config_root` ini key keep working (PR #449 review #6):
  both config-root singletons fall back to the legacy sources when the
  unified scan-analysis root is unset, emitting a `DeprecationWarning`
  (and a log warning) naming the legacy source used and the migration
  target.

## [0.10.0] — 2026-06-23

### Added
- `geecs_data_utils.io.decode_imaq_image_string`: decodes an in-memory NI IMAQ
  "Flatten Image to String" payload (as received live over the device TCP stream
  and stored in `device.state["image"]`) to a 2-D NumPy array. Handles both
  device modes for any camera/ROI — compressed (embedded JFIF JPEG, 8-bit) and
  uncompressed raw (8/16-bit), deriving width, height, IMAQ border, row-stride
  padding and pixel depth from the message and cropping out the border. This is a
  wire-format decoder, distinct from the file readers like `read_imaq_image`.

## [0.9.1] — 2026-06-26

### Changed
- Dropped the Python 3.10 support claim; minimum is now `python >=3.11,<3.12`,
  matching the integrated monorepo environment (the root project and the
  GUI/PythonAPI/Bluesky packages all require >=3.11).

## [0.9.0] — 2026-06-18

### Added
- New `geecs_data_utils.io` subpackage owning generic `path -> numpy.ndarray`
  file readers, relocated from `image_analysis.utils`:
  - `geecs_data_utils.io.images.read_imaq_image` (format dispatcher),
    `read_imaq_png_image` (NI IMAQ 16-bit PNG), `read_tsv_file`,
    `load_image_from_h5`.
  - Gives ImageAnalysis, post-run analysis tools, and Bluesky external-asset
    handlers a shared reader foundation without depending on the
    `image_analysis` package.
- New hard dependencies for the readers: `pypng`, `imageio`, `h5py` (kept
  light; heavier image libs such as opencv / scikit-image deliberately stay
  out of this package).

### Notes
- These readers are consumer-only file loaders; they read existing files and
  never create scan folders (cross-package scan-folder invariant unaffected).

## [0.8.0] — 2026-06-15

### Added
- New `geecs_data_utils.tiled_export` module: reads a Bluesky scan back from a
  Tiled catalog and writes the legacy GEECS scalar files
  (`scans/ScanNNN/ScanDataScanNNN.txt` and the mutable `analysis/sNNN.txt`).
  - `write_scalar_files_from_tiled(uid, ...)` — fetch a run by uid and write
    both files; resolves Tiled connection from `~/.config/geecs_python_api/
    config.ini [tiled]` when not passed explicitly.
  - `build_legacy_scalar_dataframe(start_doc, primary_df)` — the pure transform
    (renames Bluesky `<ophyd>-<safe_var>` columns to legacy `Device Variable`
    via the run's `geecs_scalar_headers`, drops companion columns, emits
    `Bin #` / `scan` / `Shotnumber`), unit-testable without a live server.
  - Consumer-only: writes into an existing `scans/ScanNNN/` folder, never
    creates one (cross-package scan-folder invariant).
- New optional `tiled` extra (`pip install 'geecs-data-utils[tiled]'`); the
  module lazy-imports the Tiled client so the dependency is only needed for
  export.

## [0.7.0] — 2026-05-20

### Added
- New `geecs_data_utils.data` subpackage: shared tabular utilities used by
  both analysis and modeling layers.
  - `data.columns`: `find_cols`, `resolve_col`, `resolve_col_detailed`,
    `flatten_columns`, `ColumnMatchMode`, `ResolveColResult`.
  - `data.cleaning`: `RowFilterSpec`, `apply_row_filters`, `OutlierConfig`,
    `apply_outlier_config`, `sigma_clip_frame`, `sigma_nan_frame`.
  - `data.dataset`: `DatasetBuilder`, `DatasetFrame`, `LoadScansReport`
    for multi-scan scalar dataset assembly with filters / outliers /
    `dropna` and a visibility report for skipped scans.
- New `geecs_data_utils.analysis` subpackage:
  - `analysis.correlation`: `CorrelationReport` for target-vs-numeric
    correlation ranking (Pearson / Spearman / Kendall) with row filters,
    substring exclusions, and `top_n`.
- New optional `geecs_data_utils.modeling.ml` subpackage (install with the
  `ml` extra):
  - `MLDatasetBuilder` / `DatasetResult`: select target + features from a
    DataFrame for modeling, with optional `exclude_terms` for substring-
    based feature pruning (matching `CorrelationReport.exclude_terms`).
  - `RegressionTrainer` / `ModelArtifact`: linear / ridge / elastic-net
    fits with standard preprocessing, metrics, and optional CV scores.
  - `save_model_artifact` / `load_model_artifact`: joblib + JSON
    sidecars (`FeatureSchema`, `ModelMetadata`, `TrainingMetrics`).
    Metadata captures `sklearn_version`, `joblib_version`, `numpy_version`,
    `python_version`, and an `artifact_version` so loaders can warn on
    runtime mismatches.
  - `predict_from_scan`: inference helper that expects scan columns to
    match the training feature schema exactly.

### Changed
- `ScanData.find_cols` / `resolve_col` now delegate to
  `geecs_data_utils.data.columns` so single-scan and multi-scan code paths
  share semantics. Behavior is preserved.

### Removed
- Unused `ScanPaths.data_dict`, `ScanPaths.data_frame`, and
  `ScanPaths.get_device_data()`. No external callers within the monorepo.
## [0.6.4] — 2026-05-21

### Changed
- `ScanPaths` `read_mode` docstring tightened to document that
  `read_mode=False` (silent folder creation) is for scanner-side callers
  only — the GEECS scanner and BlueskyScanner, which legitimately bring new
  scan folders into existence. Analysis-side callers (ScanAnalysis,
  ImageAnalysis) must use the default `read_mode=True`. Behaviour is
  unchanged; the contract is now pinned by
  `tests/test_scan_paths_create_invariant.py`. Context: a sibling fix in
  `scan_analysis` 1.3.6 removed an analysis-side silent-create that
  converted transient SMB visibility blips into data loss.

## [0.6.3] — 2026-05-19

### Removed
- `EXPERIMENT_TO_SERVER_DICT` and the associated `_get_default_server_address` /
  `_is_default_server_address` helpers removed from `GeecsPathsConfig`. The dict
  was an implicit, hard-coded mapping of experiment names to server paths that
  silently overrode explicit config, caused confusion when paths differed between
  sites, and is now fully superseded by `GEECS_DATA_LOCAL_BASE_PATH` in
  `config.ini`. Any machine previously relying on the implicit `Z:/data` default
  should add `GEECS_DATA_LOCAL_BASE_PATH = Z:/data` to its config.

### Changed
- `ScanData.from_date` and `ScanData.latest`: `experiment` parameter is now
  `Optional[str]` (was `str`). Callers that pass `None` propagate to
  `ScanPaths.get_scan_tag`, which already handles `None` by falling back to
  `paths_config.experiment`; flat-layout sites can omit the experiment entirely.

## [0.6.2] — 2026-05-19

### Fixed
- `ScanPaths.get_daily_scan_folder`: skips the experiment path segment when
  `tag.experiment` is `None`, producing `{base}/Y{YYYY}/...` instead of
  crashing.

## [0.6.1] — 2026-05-19

### Changed
- `GeecsPathsConfig`: `GEECS_DATA_LOCAL_BASE_PATH` from `config.ini` is now
  tried **before** the experiment-to-server-address dict (`EXPERIMENT_TO_SERVER_DICT`),
  which becomes a fallback. This means analysis-only machines that define a
  local data root are no longer overridden by the `Z:/data` server default.
- `GeecsPathsConfig`: `experiment` is now optional — a `ConfigurationError` is
  only raised when `base_path` cannot be determined. Callers that need the
  experiment name (e.g. LiveWatch, GDoc integration) supply it at runtime via
  `ScanTag`; it no longer needs to be defined in `config.ini`.
- `_get_default_server_address` signature updated to accept `Optional[str]`.

## [0.6.0] — 2026-05-12

### Added
- `ScanPaths.build_device_file_map`, `ScanPaths.build_asset_filename`, and
  `ScanPaths.build_asset_path` accept an optional `device_file_stem` kwarg
  (default `None` → falls back to `device`). Use this when a device's data
  folder name differs from the token used inside per-shot filenames — e.g.,
  folder `U_BCaveMagSpec-interpSpec` containing files named
  `Scan042_U_BCaveMagSpec_001.csv`.
- `ScanData._append_expected_asset_columns`, `ScanData.set_data_frame`,
  `ScanData.load_scalars`, and `ScanData.from_date` accept an optional
  `stem_override: dict[str, str]` kwarg, mirroring the existing
  `ext_override` pattern. Maps device folder names to their in-filename
  stems so the `<device>_expected_path` DataFrame columns resolve to real
  files for affected devices. Without the override, those columns
  previously contained nonexistent paths for any device where the folder
  name and filename stem differ.

## [0.5.0] — 2026-05-08

### Added
- `timestamp_from_string(string)` and `timestamp_from_filename(file)` migrated
  from `geecs_python_api.tools.files.timestamping`. Both are exported from the
  package root. Eliminates the scanner's dependency on the now-deleted
  `GEECS-PythonAPI` timestamping module.
- `tests/test_utils.py` — first test suite for `geecs_data_utils`.

## [0.4.1] — 2026-05-07

### Changed
- `ScanConfig` migrated from `@dataclass` to `pydantic.BaseModel`.
  Construction syntax is unchanged (all fields use keyword arguments); the
  migration adds runtime validation and makes `ScanConfig` composable with other
  Pydantic models throughout the scanner engine.

## [0.4.0] — 2026-05-07

### Added
- `scan_log_loader` module providing `LogEntry`, `Severity`, `parse_lines`,
  `parse_scan_log`, and `load_scan_log`. Reads the per-scan log format
  written by `geecs_scanner.logging_setup.attach_scan_log` (multi-line
  tracebacks aggregated into the preceding record). Returned models are
  shared with the new `geecs-log-triage` subpackage and intended for any
  consumer needing to read scan logs (notebooks, plotting helpers,
  diagnostics tooling).

## [0.3.0] — 2026-05-06

### Added
- `GeecsPathsConfig` now reads an optional `wavekit_config_path` key from the
  `[Paths]` section of `config.ini` and exposes it as an attribute (consistent
  with the existing `frog_dll_path` / `frog_python32_path` pattern). Returns
  `None` if the key is absent or the path does not exist.

## [0.2.1] — current
<!-- Add entries here when changes are made -->
