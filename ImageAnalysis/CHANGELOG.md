# Changelog — image-analysis

All notable changes to this package will be documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [1.11.1] — 2026-07-16

### Changed

- Mechanical pre-commit normalization (repo-wide `pre-commit run --all-files`
  pass): ruff-format reformatting and unused-import removal (`ureg`) in
  `tests/algorithms/test_grenouille_analysis.py` and
  `tests/algorithms/test_qwlsi_analysis.py`. Tests only; no behavior changes.

## [1.11.0] — 2026-07-09

### Added

- `load_diagnostic()` now preserves the diagnostic YAML filename stem as a
  private `source_id` on `DiagnosticAnalysisConfig`. ScanAnalysis can use this
  source id as the analyzer id for direct single-diagnostic runs, matching the
  group-loader/LiveWatch naming contract without adding a YAML field.

### Changed

- `LineStitcher` now stitches and saves the available lineout segments when
  one or more sibling device files are missing, instead of failing the whole
  shot. Missing siblings are recorded in result metadata (`warnings`,
  `missing_sibling_devices`, `missing_sibling_files`) so ScanAnalysis/LiveWatch
  can surface the incomplete data product while preserving the useful stitched
  TSV.

## [1.10.0] — 2026-06-30

### Added
- `ImageAnalyzerResult.feature_scalars()` returns JSON-serializable scalar
  feature values for downstream analysis logging.

### Changed
- String-based camera/line config lookup now prefers the unified
  ScanAnalysis config root (`SCAN_ANALYSIS_CONFIG_DIR` /
  `scan_analysis_configs_path`); the legacy ImageAnalysis config root is no
  longer part of active config discovery.
- `image_analysis.data_1d_utils` now re-exports the shared 1D file readers from
  `geecs_data_utils.io.array1d`; existing imports keep working while Bluesky can
  use the same loaders without importing ImageAnalysis.

### Fixed

- The HASO test module's `importorskip` now passes `exc_type=ImportError`:
  the WaveKit SDK's Windows-only `_ctypes.FreeLibrary` import raises a bare
  ImportError elsewhere, which pytest >= 8 no longer auto-skips — collection
  errored on Linux/macOS instead of skipping.
- Widened the stray runtime `pytest` constraint from `^7.4` to `>=7.4` — it
  was the only pin holding the integrated monorepo env below pytest 8
  (moving it out of runtime deps entirely is deferred cleanup).
- The two Undulator demo `__main__` blocks (`hi_res_mag_cam_analyzer`,
  `BCaveMagSpecStitcher`) set the camera-config root on the instance the
  loader actually consults (`scan_analysis_config`), so the demos resolve
  their local config dirs again. (PR #449 review #6)

## [1.9.1] — 2026-06-25

### Changed
- Dropped the Python 3.7-3.10 / numpy 1.x support claim. The minimum is now
  `python >=3.11,<3.12`, matching the integrated monorepo environment (the root
  project and the GUI/PythonAPI/Bluesky packages all require >=3.11) and the
  numpy 2.0+ APIs the analyzers actually use (e.g. `np.trapezoid`). The
  per-Python version splits for numpy/scipy/pandas/Pint collapse to single
  modern pins.

### Fixed
- numpy 2.4 compatibility: `np.trapz` was removed in numpy 2.4.0, breaking
  `ict_algorithms.py` and `grenouille.py`. Switched both to `np.trapezoid`
  (available since numpy 2.0, so safe across all supported envs).
- numpy 2.4 compatibility for `qwlsi.py`'s Abel-transform path: pinned
  `pyabel >=0.9.1` (earlier PyAbel releases used the removed `np.trapz`); 0.9.1
  is numpy-2.4 compatible. `import abel` is kept inside `calculate_density` (its
  only consumer) so importing the module — and the wavefront paths that don't
  need an Abel transform — stays light.

## [1.9.0] — 2026-06-18

The generic file readers move to `geecs_data_utils.io.images`. ImageAnalysis
keeps a deprecation shim at the old import location for one release.

(Version jumps 1.7.0 → 1.9.0: the `pyproject.toml` version was never bumped
to the already-documented 1.8.0, so this release advances past it rather than
colliding with that changelog entry.)

### Changed
- `read_imaq_image`, `read_imaq_png_image`, `read_tsv_file`, and
  `load_image_from_h5` moved out of `image_analysis.utils` into
  `geecs_data_utils.io.images`. These were never image-*analysis* logic —
  they are lower-level `path -> numpy.ndarray` readers — and relocating them
  lets Bluesky asset handlers and other consumers load files without
  depending on the `image_analysis` package.
- Internal callers (`base.py`, `processing/array2d/background.py`,
  `analyzers/density_from_phase_analysis.py`) now import the readers from
  `geecs_data_utils.io.images` directly.

### Deprecated
- Importing the four readers from `image_analysis.utils` still works via a
  module-level `__getattr__` shim, but now emits a `DeprecationWarning`
  pointing at the new location. The shim will be removed in a future release;
  update imports to `geecs_data_utils.io.images`.

### Notes
- `image_analysis.utils` retains its non-reader members (`ROI`, `NotAPath`,
  `extract_shot_number`, `ensure_float64_processing`).

## [1.8.0] — 2026-06-01

Single source of truth for output naming (issue #412). All scalar-key
prefix/suffix infrastructure and the per-image-config ``name`` field
move out of ImageAnalysis. The analyzer hierarchy gains an
``output_name`` kwarg that the diagnostic factory wires from
``DiagnosticAnalysisConfig.effective_output_name``. Companion to
ScanAnalysis 1.13.0.

### Changed (breaking)
- ``DiagnosticAnalysisConfig``: new ``output_name: Optional[str]``
  field replaces the briefly-named ``metric_prefix`` (the latter
  never shipped). Reads as "the stem string used for all outputs
  this analyzer produces" — drives both s-file column prefixes AND
  per-analyzer output directory names. ``effective_output_name``
  property: ``output_name`` if set, else ``name``.
  ``metric_suffix`` keeps its name (narrow — scalar-key-only).
- ``StandardAnalyzer``, ``Standard1DAnalyzer``, ``BeamAnalyzer``,
  ``LineAnalyzer``: add ``output_name`` keyword-only kwarg. Replaces
  the previous ``camera_name`` property (read by
  ``SingleDeviceScanAnalyzer`` for output dir labels and by MagSpec
  for per-file output paths).
- ``BeamAnalyzer``, ``LineAnalyzer``, ``Standard1DAnalyzer``,
  ``StandardAnalyzer``, ``GrenouilleAnalyzer``,
  ``DownrampPhaseAnalyzer``, ``FrogSpectralPhaseAnalyzer``,
  ``LineStitcher``, ``BCaveMagOpt``, ``ICT1DAnalyzer`` — all emit
  **bare scalar keys** (``"x_fwhm"``, ``"image_total"``,
  ``"temporal_fwhm"``, ``"charge_pC"`` etc.). Namespacing with the
  ``output_name`` prefix moves to ``SingleDeviceScanAnalyzer`` in
  ScanAnalysis.

### Removed (breaking)
- ``CameraConfig.name`` and ``Line1DConfig.name`` are gone from the
  documented schema. Analyzer identity now flows through the
  ``output_name`` constructor kwarg (set by the diagnostic factory
  from ``effective_output_name``); the per-image-config ``name``
  field that was previously coupled to it has nowhere left to live.
  ``extra="allow"`` on both config classes means legacy YAMLs that
  still carry ``name:`` inside ``image:`` still load (the key lands
  in ``model_extra``), but nothing reads it for behavior.
- ``DiagnosticAnalysisConfig._inject_name_into_image`` validator
  (used to default ``image.name`` from top-level ``name``). No
  longer needed — there's no ``image.name`` field to default.
- Loader's filename-stem ``name`` injection.
- ``StandardAnalyzer.camera_name`` property; renamed to
  ``output_name``.
- ``Standard1DAnalyzer.camera_name`` compatibility property; renamed
  to ``output_name``.
- ``flatten_beam_stats`` and ``compute_beam_slopes``: ``prefix`` and
  ``suffix`` kwargs.
- ``LineBasicStats.to_dict``: ``prefix`` and ``suffix`` kwargs.
- All ``name_suffix`` constructor kwargs (and the
  ``camera_config.name`` mutation pattern that backed them) on
  ``StandardAnalyzer``, ``BeamAnalyzer``, ``DownrampPhaseAnalyzer``,
  ``GrenouilleAnalyzer``.
- All previous ``metric_prefix`` / ``metric_suffix`` constructor
  kwargs on ``LineAnalyzer``, ``FrogSpectralPhaseAnalyzer``,
  ``LineStitcher``, ``BCaveMagOpt``.
- ``StandardAnalyzer.apply_metric_suffix`` utility method.
- ``StandardAnalyzer.camera_config_name`` shadow attribute.
- Private ``_normalize_metric_suffix`` helper in
  ``frog_spectral_phase_analyzer``.

### Fixed
- ``GrenouilleAnalyzer``, ``ICT1DAnalyzer``,
  ``DownrampPhaseAnalyzer`` were still self-prefixing their scalar
  keys with the camera name (e.g. ``"UC_FROG_temporal_fwhm"``). With
  ScanAnalysis also prefixing, this produced doubly-prefixed columns
  (``"UC_FROG_UC_FROG_temporal_fwhm"``). All three now emit bare
  keys and rely on ScanAnalysis to namespace them once, end-to-end.
- ``DownrampPhaseAnalyzer`` was calling
  ``compute_beam_slopes(phase_array, prefix=self.camera_name)``;
  ``compute_beam_slopes`` had its ``prefix`` kwarg removed earlier
  in #412 (so this call would have failed at runtime). Now passes
  no prefix and reads the bare-key result.

### Migration
- Production YAMLs require **no changes for typical use** — the
  bit-identical contract holds when the diagnostic config has no
  ``output_name`` / ``metric_suffix`` fields: ScanAnalysis defaults
  the prefix to ``diag.name`` and the suffix to ``""``, producing
  the same s-file column names as before (``UC_TopView_x_fwhm``).
- YAMLs that used the in-development ``metric_prefix:`` key (never
  shipped) should rename it to ``output_name:``.
- Standalone notebook ``CameraConfig(name="foo", ...)`` usage stops
  validating ``name`` against a typed field — it lands in
  ``model_extra`` and is silently ignored. Notebook code that read
  ``cfg.name`` should switch to tracking the identifier itself.
- Notebook code that passed ``metric_suffix=`` / ``metric_prefix=`` /
  ``name_suffix=`` to an analyzer constructor needs to drop the
  kwarg and either live with bare keys or rename the dict keys
  themselves after analysis.

### Test count
- 213 tests pass (was 220; 6 tests of removed kwargs deleted, 1
  ``name``-assertion test rewritten as an ``output_name``-assertion
  test).

## [1.7.0] — 2026-05-30

### Added
- 2D image background subtraction now supports `method: edge`, which
  subtracts the mean border-pixel value from each image. The border
  thickness is configurable via `edge_width` and defaults to 1 pixel.
- `load_diagnostic` gains an optional `overrides: dict` keyword. The
  override dict is deep-merged into the on-disk YAML before Pydantic
  validation: nested mappings merge key-by-key, scalars and lists
  replace wholesale, and the merged result re-validates so override
  typos or type mismatches surface with the same error path as a bad
  YAML on disk. Generic — works for any field on the diagnostic, not
  optimization-specific. First consumer is Scanner-GUI 0.26.0's
  `MultiDeviceScanEvaluator`, which uses it to flip `scan.mode` per
  optimization run without forking the diagnostic.
- Private `_deep_merge(base, overlay) -> dict` helper next to
  `load_diagnostic`. Recursive dict merge; returns a new dict (never
  mutates inputs). Visible only to the loader module.

## [1.6.0] — 2026-05-28

Add the `FrogSpectralPhaseAnalyzer` and the auxiliary-column loading
support it needed. Originally branched before PR-E; rebased onto the
post-PR-E surface and **simplified to fit the atomic load+analyze
contract** — aux columns now flow through the local `auxiliary_data`
dict rather than through analyzer-instance state.

### Added
- `algorithms.polynomial_fit` provides reusable weighted polynomial
  fitting with finite-value filtering, optional threshold masking, and
  sign canonicalization.
- `Data1DConfig.auxiliary_columns: Dict[str, int]` — declarative
  mapping from name → column index for sidecar columns loaded
  alongside the primary `x` / `y` data. Row-aligned with the primary
  Nx2 line data.
- `Data1DResult.auxiliary_column_data: dict[str, np.ndarray]` —
  loader output carrying the named aux columns. Lives at the loader
  boundary; consumed by the analyzer and discarded.
- `analyzers.frog_spectral_phase_analyzer.FrogSpectralPhaseAnalyzer`
  fits retrieved FROG spectral phase lineouts and reports physical
  dispersion terms (`GD`, `GDD`, `TOD`, ...). Uses the `weights`
  aux column for intensity-weighted polynomial fitting when configured.
- `docs/image_analysis/examples/pulse_duration_jitter_analysis.ipynb`
  demonstrates loading a Scan010 retrieved FROG lineout and running
  the new analyzer.

### Changed
- `Standard1DAnalyzer.analyze_image_file` is now the canonical atomic
  load+analyze entry point. It reads the file, stashes descriptive
  metadata (units, labels) on `self.data_metadata`, and routes any
  loaded aux columns through `auxiliary_data["_aux_columns"]` to
  `analyze_image`. Per-shot data (the loaded arrays) no longer
  travels through analyzer-instance state between separate pipeline
  phases — this is the correctness property the post-PR-E
  `SingleDeviceScanAnalyzer` contract enforces.
- Subclasses that need ROI-filtered aux columns should call
  `_preprocess_line_data` directly so the line and aux arrays come
  back from the same call boundary (see `FrogSpectralPhaseAnalyzer`
  for the pattern).
- `FrogSpectralPhaseAnalyzer.analyze_image` now emits the **fit**
  curve as `result.line_data` — fixed-length (`fit_num_points`,
  default 300), wavelength-sorted Nx2. Raw scattered phase samples
  and the intensity-weight curve move to `result.render_data`
  (`raw_wavelength_nm`, `raw_spectral_phase`,
  `fit_normalized_reference`). This makes every shot in a scan
  emit identically-shaped `line_data`, so `Array1DScanAnalyzer`'s
  waterfall and per-bin averaging aggregate cleanly across shots —
  fixing the inhomogeneous-shape crash hit on 500-shot scans where
  per-shot ROI-edge wobble produced variable raw lengths.
- `FrogSpectralPhaseAnalyzer.__init__` takes a typed `Line1DConfig`
  (matches the post-PR-E `Standard1DAnalyzer` contract). String-by-
  name resolution moved to the loader — call
  `image_analysis.config.load_line_config(name)` first.

### Removed
- `ImageAnalyzerResult.line_auxiliary_column_data` field. Aux columns
  no longer escape the analyzer — they're consumed and discarded
  inside the analyze call. ScanAnalysis never had a use for them; no
  downstream consumer breaks.
- `Standard1DAnalyzer._loaded_auxiliary_column_data` instance state
  and the `_use_loaded_auxiliary_columns` flag that controlled when
  `analyze_image` read from it. Both were the pre-PR-E shuttle for
  the now-deleted load-all pipeline.

### Fixed
- `data_1d_utils` now skips a detected CSV/TSV header row after
  parsing column metadata, allowing `read_1d_data` to load Grenouille
  retrieved lineout TSVs written with named column headers.
- `_interpolation_enabled` no longer reads the deleted
  `InterpolationConfig.enabled` field. Post-PR-F semantics:
  `pipeline.steps` is the single source of truth — if
  `INTERPOLATION` is in the step list and the sub-config is present,
  interpolation runs.
- `ImageAnalyzerResult.average` no longer crashes on inhomogeneous-
  shape arrays. The previous code called `np.nanmean(values, axis=0)`
  directly on `line_data`, `processed_image`, and every NDArray
  render_data field; on numpy 2.x this raises a hard `ValueError`
  through `np.asanyarray` when shots have different shapes. Now: a
  new module-level `_safe_nanmean_arrays` helper detects mismatched
  shapes, logs a warning naming the offending field, and returns
  `None` so the caller omits the key from the averaged result. List-
  valued render_data fields get the same length check. Surfaces on
  parameter scans with FROG: `raw_wavelength_nm`,
  `raw_spectral_phase`, and `fit_normalized_reference` are variable-
  length per shot by design (they track the raw ROI'd data); they're
  now dropped from the per-bin average with a warning while
  fixed-length fields (`line_data`, `fit_omega_detuning_rad_per_fs`)
  average as expected.

## [1.5.0] — 2026-05-27

Loader API consolidation (PR-E). Companion to ScanAnalysis 1.7.0.
Collapses the public surface for going from disk to a configured
analyzer into a single `image_analysis.config` namespace, switches the
diagnostic `image:` schema to a type-discriminated payload, and
finishes the `offline_analyzers/` → `analyzers/` rename.

Net code delta (this PR): roughly +1750 / −2200 LoC across the two
packages despite adding the new public-config module and ~370 LoC of
new tests — the consolidation paid off.

### Added
- `image_analysis.config` public namespace as the single entry point.
  Exports: `load_diagnostic`, `load_camera_config`, `load_line_config`,
  `create_image_analyzer`, and all sub-models (`CameraConfig`,
  `Line1DConfig`, `ROIConfig`, `BackgroundConfig`,
  `DiagnosticAnalysisConfig`, `ImageAnalyzerSpec`, `ImageKind`,
  `ScanType`).
- `create_image_analyzer(DiagnosticAnalysisConfig) -> ImageAnalyzer`
  factory — the Mode 2 (config-driven) entry point.
- Type-discriminated `image:` payload in the diagnostic schema:
  `Array2DImageConfig` (carries `camera_config_name`) and
  `Array1DImageConfig` (carries `line_config_name`), discriminated by
  the analyzer's class path. The legacy `image_kind` / `scan_type`
  fields are gone.

### Changed
- `offline_analyzers/` renamed to `analyzers/`. The `offline_`
  qualifier was a holdover from a never-built online counterpart.
- `BackgroundManager` (class) collapsed to
  `apply_background(image, config, *, cache=None)` (function). The
  path-keyed cache moves onto the analyzer instance (`self._bg_cache`),
  so the manager class no longer needs to exist.
- Configs consolidated under `image_analysis.config/` only — no
  parallel `processing/` config tree.

### Removed
- Polymorphic `camera_config_name=` / `line_config_name=` constructor
  kwargs on analyzers. Constructors now take typed `CameraConfig` /
  `Line1DConfig` models only. The string-name → file → model load lives
  at the loader/factory layer (`load_camera_config`, `load_diagnostic`,
  `create_image_analyzer`).
- `image_analyzer` alias registry. Diagnostic YAMLs use full class
  paths (e.g. `image_analysis.analyzers.beam_analyzer.BeamAnalyzer`).
  The bare-string form defaults to camera + array2d; verbose-dict form
  with explicit `class_path` / `kwargs` handles 1D and no-image cases.

### Breaking
- `from image_analysis.offline_analyzers import …` →
  `from image_analysis.analyzers import …`.
- `BackgroundManager.apply(...)` → `apply_background(image, config,
  cache=...)`.
- Analyzer constructors no longer accept string config names — load via
  `load_camera_config("X")` / `load_diagnostic(...)` first, then pass
  the resulting typed model into the constructor (or use the Mode 2
  `create_image_analyzer` factory).
- Diagnostic YAMLs with `image_kind:` or `scan_type:` fields no longer
  validate; the type discriminator is now the analyzer's `class_path`.

## [1.4.0] — 2026-05-24

Companion release to the ScanAnalysis 1.6.0 unified-configs cutover.
Removes a chunk of dead and now-misleading background-handling
surface area.

### Removed
- `BackgroundConfig.background_scan_number` field. The scan-as-background
  workflow is now expressed via the diagnostic config's
  `scan.background_source.scan_number` directive, handled in
  ScanAnalysis. (The `SingleDeviceScanAnalyzer._generate_scan_background`
  consumer is removed in ScanAnalysis 1.6.0.)
- `image_analysis.processing.array2d.compute_background` — the function
  had no live callers (BackgroundManager covers `from_file`/`constant`
  directly) and referenced `config.level` / `config.percentile` fields
  that don't exist on `BackgroundConfig`, so it was unreachable.
- `BackgroundMethod.PERCENTILE_DATASET` and `BackgroundMethod.MEDIAN`.
  Both enum values were only reachable through the deleted
  `compute_background`. The aggregation helpers
  `_compute_percentile_background` and `_compute_median_background`
  stay — they're used by `compute_and_cache_scan_background`.
- `_compute_constant_background` helper and its `TestConstantBackground`
  test class.

### Breaking
- `BackgroundConfig` no longer accepts `background_scan_number`.
- `BackgroundMethod` no longer includes `PERCENTILE_DATASET` or `MEDIAN`.
- `compute_background` is no longer exported from
  `image_analysis.processing.array2d`.

## [1.3.1] — 2026-05-22

### Removed
- `ImageAnalyzer.analyze_image_batch` default removed from the base
  class. The only override (`StandardAnalyzer.analyze_image_batch`) was
  deleted in 1.3.0 along with the dynamic-background subsystem, and the
  only call site (`SingleDeviceScanAnalyzer._run_batch_analysis`) is
  deleted in the companion ScanAnalysis 1.4.0 release. No analyzers
  outside of the deleted `StandardAnalyzer` override implemented this
  hook.

## [1.3.0] — 2026-05-22

### Removed
- Dynamic background subsystem deleted from this package as part of the
  shot-by-shot scan analyzer refactor. The motivation was that the
  load-all-images-first pipeline this feature required forced an
  organizational anti-pattern (shared analyzer-instance state shuttling
  per-shot data between the load and analyze phases) that caused real bugs
  (the FROG aux-columns regression in particular). Deletions:
  - `StandardAnalyzer.analyze_image_batch` override (the dynamic-bg hook
    consumed by `Array{1,2}DScanAnalyzer._run_batch_analysis`)
  - `BackgroundManager.generate_dynamic_background` method
  - `DynamicComputationConfig` Pydantic model
  - `BackgroundConfig.dynamic_computation` field

  Static background methods (`from_file`, `constant`, `none`) are
  unchanged. The scan-as-background workflow (`background_scan_number`)
  is preserved — it operates on a separate scan's data and caches a
  `.npy` average, so it never coupled to the current scan's load-all
  pipeline. A replacement `DynamicBackground` analyzer that uses the
  per-scan analysis output tree will land in a follow-up PR.

  Affected configs (in `GEECS-Plugins-Configs`): YAMLs that referenced
  `dynamic_computation` need that key removed. `background_scan_number`
  keys remain valid.

## [1.2.1] — 2026-05-21

### Fixed
- Offline analyzers that write output subfolders inside `scans/ScanNNN/` no
  longer auto-create the scan folder itself. Affected sites:
  `LineStitcher._save_stitched_output` (creates `<scan_dir>/<self.name>/`)
  and `MagSpecManualCalibAnalyzer._save_calibrated_outputs` (creates
  `<scan_dir>/<camera>-interp/` and `<scan_dir>/<camera>-interpSpec/`). All
  three previously used `mkdir(parents=True, exist_ok=True)`, which silently
  re-created the scan folder if it was briefly invisible on the share. On a
  new NetApp/domain this masked transient SMB visibility blips as permanent
  data loss: the analyzer planted an empty `ScanNNN/` directory entry over
  the real scan contents. Each site now verifies the scan folder is visible
  before creating its output subfolder; if it isn't, it raises
  `FileNotFoundError`. The task queue logs the failure and records
  `state="failed"` when the scan folder/status directory is still writable.
  If the entire scan folder is absent, status writes are skipped rather than
  recreating the folder; the user can rerun/relaunch once the share recovers.
  This pairs with the
  `scan_analysis` 1.3.6 fix for the same anti-pattern in `task_queue`.
- `processing.array1d.background.save_background_to_file` no longer creates
  parent directories. Same anti-pattern: `mkdir(parents=True, exist_ok=True)`
  could silently materialise a scan folder if a caller passed a destination
  inside `scans/ScanNNN/`. Callers must ensure the parent directory exists
  before calling; otherwise the function now raises `FileNotFoundError`.
  (The function isn't currently called from anywhere in the codebase; this
  is a defensive consistency fix to prevent future regressions.)

## [1.2.0] — 2026-05-20

### Added
- `LineAnalyzer` now accepts a `metric_prefix` constructor arg that overrides
  the `line_config.name`-derived prefix on scalar metric keys. Lets one
  `Line1DConfig` be reused across multiple analyzer instances that report
  under different names — e.g. a stitcher that loads from a per-device
  config but emits metrics under a composite name.
- 1D background subtraction `FROM_FILE` now supports every format the line
  loader understands (npy, csv, tsv, tek_scope_hdf5, tdms_scope). The
  background file is read via `read_1d_data` using the same `data_loading`
  config the line itself uses. Previously only `.npy` / `.npz` were
  accepted.

### Fixed
- `StandardAnalyzer.analyze_image_batch` no longer raises `AttributeError`
  when the camera config has no `background:` section (in which case
  `background_manager` is `None`).

### Changed (breaking — internal/config-level)
- `LineStitcher` constructor: `output_folder` and `output_label` kwargs are
  replaced by a single `name` kwarg that drives the metric prefix, the
  output subdirectory, and the output filename label. Existing YAML configs
  must update from `output_folder`/`output_label` to `name`.
- `compute_background` (in `processing/array1d/background.py`) now accepts
  a `data_loading: Optional[Data1DConfig]` argument, required for the
  `FROM_FILE` method. Pipeline callers already pass it through; direct
  callers of this helper must update.

## [1.1.5] — 2026-05-19

### Added
- `FrogRetrievalResult.tw_per_joule` property — peak power per unit energy
  (TW/J), computed as `1000 / (sum(temporal_intensity) * dt)` with `dt` in
  femtoseconds. Matches the LabVIEW Grenouille analysis scalar.
- `GrenouilleAnalyzer` now emits `{camera_name}_tw_per_joule` in its scalar
  results.

## [1.1.4] — 2026-05-12

### Changed
- `data_1d_utils._read_csv` (and `_read_tsv` via delegation) now use
  `numpy.loadtxt` instead of `numpy.genfromtxt`. NumPy 1.23+ ships a C-coded
  parser for `loadtxt` that is ~10–20× faster on clean numeric tables.
  Profiling against `Standard1DAnalyzer` / `LineAnalyzer` showed ~90% of
  per-shot wallclock was inside `genfromtxt`; this swap removes that
  bottleneck for multi-scan processing of 1D text-format data (notably
  interpolated-spectrum files).

  **Behavior note:** `loadtxt` is stricter on malformed rows — it raises
  `ValueError` rather than substituting NaN. For clean processed spectra
  this is the desired behavior; loud failures are preferred over silent
  NaNs. Files with intentionally missing/non-numeric values that previously
  parsed via `genfromtxt`'s coercion will now fail to load.
### Fixed
- `BackgroundConfig` now accepts `method="from_file"` with `background_scan_number`
  set and no explicit `file_path`.  Previously the `@field_validator("file_path")`
  ran before `background_scan_number` was validated (Pydantic processes fields
  top-to-bottom), so it always raised an error when only a scan number was given.
  Replaced with a `@model_validator(mode="after")` that checks both fields after
  full model construction.

## [1.1.3] — 2026-05-06

### Fixed
- `GrenouilleAnalyzer.analyze_image` no longer raises `AttributeError` when
  `auxiliary_data` is `None` (i.e. when called via `analyze_image_file` without
  passing auxiliary data). The TSV lineout export is now correctly skipped in
  that case.

### Added
- `tests/conftest.py` — session-scoped autouse fixture that initialises the
  image-analysis config base directory; no-op on CI.
- `tests/analyzers/test_grenouille_analyzer.py` — integration tests for
  `GrenouilleAnalyzer` covering scalar keys, physical plausibility of FWHM and
  FROG error, result structure, and sidecar TSV creation. Config is frozen in
  code to avoid YAML drift.
- `tests/analyzers/test_haso_analyzer.py` — integration tests for
  `HASOHimgHasProcessor` covering result structure and all five sidecar file
  outputs. Entire module skips when WaveKit SDK is unavailable.

## [1.1.2] — 2026-05-06

### Fixed
- `BeamAnalyzer` now reports `x_CoM`, `y_CoM`, and `peak_location` in the
  full-sensor coordinate system rather than in the ROI-local frame.
  `beam_profile_stats` gains an `roi_offset` parameter; `BeamAnalyzer`
  passes `(roi.x_min, roi.y_min)` automatically when an ROI is configured.
  Width stats (`rms`, `fwhm`) are unaffected by the offset.
- `apply_roi_cropping` no longer raises `ValueError` when the configured ROI
  extends beyond the actual image dimensions. Boundaries are now clamped to
  the image size with a `WARNING` log message. If clamping leaves a zero-area
  ROI the full image is returned.

## [1.1.1] — 2026-05-06

### Removed
- `lcls-tools` dependency (closes #231). The package was not used in any active
  code paths; the one internal helper (`gaussian_fit_beam_size`) has been
  rewritten using `scipy.optimize.curve_fit`, which is already a dependency.
  `image_analysis/algorithms/lcls_tools_gauss_fit.py` (a thin wrapper that was
  never imported) has been deleted.

## [1.1.0] — current
