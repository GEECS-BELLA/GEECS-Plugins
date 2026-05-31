# Changelog — scan-analysis

All notable changes to this package will be documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [1.9.0] — 2026-05-28

Sign-aware default colormap for 1D waterfall plots.

### Added
- `Line1DRendererConfig.colormap_mode` now supports an `"auto"` value
  (also added to `BaseRendererConfig`'s Literal). `"auto"` inspects
  the data range:
  - If the data crosses zero (`min < 0 < max`), use an **asymmetric
    diverging** colormap via `matplotlib.colors.TwoSlopeNorm(vmin,
    vcenter=0, vmax)` with `RdBu_r` by default. The negative and
    positive ranges each occupy the appropriate half of the colormap,
    zero gets the neutral midpoint color, and asymmetric data (e.g.
    `[-0.026, +0.85]`) doesn't waste color resolution on empty range.
  - If the data is all non-negative, keep the legacy sequential
    behavior (vmin=0, plasma).
  - If the data is all-negative, use sequential with `vmin = data.min()`
    so nothing is clipped.
- `Line1DRendererConfig` defaults `colormap_mode="auto"`. Explicit
  `"sequential"` / `"diverging"` / `"custom"` continue to behave as
  before — opt-in to the legacy floor-at-zero by setting
  `colormap_mode: "sequential"` in the renderer kwargs.

### Changed
- `Line1DRenderer._get_colormap_params_1d` returns a 4-tuple
  `(vmin, vmax, cmap, norm)`. The `norm` is non-`None` only for the
  auto-diverging case (when it's a `TwoSlopeNorm`); the waterfall
  call site routes it to `pcolormesh(norm=...)` and skips
  `vmin`/`vmax`. All other modes keep `norm=None` and the legacy
  `vmin`/`vmax` path.
- `Line1DRendererConfig.cmap` default is now `None` (was `"plasma"`).
  This lets the renderer pick a mode-appropriate default
  (`RdBu_r` for auto-diverging, `plasma` for sequential and the
  non-sign-crossing auto branch) while still respecting any explicit
  user choice. Existing YAMLs that hard-coded `cmap: "plasma"` keep
  that exact behavior.

Surfaces in production: FROG spectral-phase scans where the fit
(after `phi0` subtraction) crosses zero — previously the negative
region was floored to the bottom of the plasma colormap, hiding a
real physical effect (the polynomial coefficients sweeping through
zero along the scan parameter). Now visible at a glance.

## [1.8.2] — 2026-05-28

Companion to ImageAnalysis 1.6.0 (FROG spectral phase analyzer). Hardens
the 1D averaging and waterfall paths against inhomogeneous per-shot
lineouts that analyzer-specific ROI/weight masking can produce.

### Fixed
- `SingleDeviceScanAnalyzer.average_data` no longer raises `ValueError`
  when per-shot lineouts have inhomogeneous shapes. It detects the
  mismatch, logs a warning, and returns `None`.
  `Array1DScanAnalyzer._postprocess_noscan` skips the averaged-line
  figure on `None` while still saving per-shot scalars and the
  waterfall summary. This unblocks 1D analyzers whose per-shot
  ROI/weight masking yields variable sample counts — most
  immediately `FrogSpectralPhaseAnalyzer`, where each shot's valid
  retrieved-spectrum window differs.
- `Line1DRenderer._create_waterfall_plot` skips with a warning when
  per-shot y-arrays have inhomogeneous shapes, instead of crashing
  on `np.array(data_arrays)` (hard `ValueError` on numpy 2.x).
  Mirrors the `average_data` fix and prevents `run_analysis` from
  dying mid-postprocess after per-shot scalars + data files have
  already been written. The warning points at the two real fixes
  (interpolation onto a common grid via the pipeline, or having the
  analyzer emit fixed-length summary `line_data`) so the user knows
  how to actually get a waterfall.
  
## [1.8.2] — 2026-05-22

### Added
- LiveWatch GUI: facility dropdown now includes `PWlaserData`, `p2`, and
  `ControlRoom` alongside the existing `Undulator`, `Thomson`, and `(none)`
  options.

## [1.8.1] — 2026-05-28

ConfigFileGUI quality-of-life: validated dropdowns for `Literal[...]`
fields and a collapsible image section in the scan-analyzer editor.

### Added
- `LiteralFieldWidget` in `field_widgets.py`. `Literal["a", "b", ...]`
  fields on Pydantic models are now rendered as a `QComboBox`
  populated with the literal arguments, instead of falling through to
  the string-input fallback. Dispatched before the `Enum` case in
  `create_field_widget` and handles bare `Literal[...]` as well as the
  inner type of `Optional[Literal[...]]`. First concrete user is
  `FromCurrentScanSpec.method` (`Literal["median", "percentile"]`) in
  the scan field's `background_source.from_current_scan` editor.
- Collapse toggle on the **Image** group in `scan_analyzer_editor.py`.
  A `QToolButton` (▼ / ▶) in the type-discriminator row hides the
  embedded `ConfigEditorPanel` while leaving the type combo visible,
  so users can fold the long camera/line form away without losing
  sight of the current image kind. The toggle is orthogonal to the
  `(none)` discriminator hide — both states are tracked separately
  and ANDed in `_refresh_image_panel_visibility`.

## [1.8.0] — 2026-05-27

ConfigFileGUI rewrite against the PR-E unified-diagnostic surface
(PR-F), plus LiveWatchGUI hardening, plus a single-source-of-truth
refactor for what the per-shot pipeline runs.

### Changed
- **`pipeline.steps` is now the single source of truth for which
  processing steps run.** The earlier behavior — "any sub-config
  present implies its step is on" — is gone. If a step should run, it
  must appear in `pipeline.steps`; if it should not, omit the step from
  the list (the presence of a `BackgroundConfig` / `ROIConfig` /
  similar no longer auto-enables it). Makes the on-disk config faithful
  to what actually executes.
- `ConfigFileGUI` collapsed to a single-surface scan-config editor.
  The separate experiment editor is gone; the tree view, groups
  editor, and `scan_analyzer_editor` are rewritten against
  `DiagnosticAnalysisConfig`. `scan_config_io` rewritten for the
  unified-diagnostic layout.

### Added
- `LiveWatchGUI`: group list dedups on insert (configs that cross
  multiple namespaces no longer surface duplicate rows), with a new
  namespace-filter dropdown to narrow the visible groups.
- Extended round-trip tests for unified-diagnostic and group YAMLs
  (`tests/test_config_gui_roundtrip.py`), marked `gui` for headless CI.

### Fixed
- `ConfigFileGUI`: no longer crashes when switching image type to
  `"line"`.
- `ConfigFileGUI`: absent sub-configs are no longer silently
  pre-populated with defaults on load. Sections present on disk show
  populated; sections absent on disk show empty. Saving back produces
  a minimal diff.
- `ConfigFileGUI`: the misleading `"Enabled"` label on
  `OptionalFieldWidget` was removed. Presence-vs-absence is now
  expressed by adding/removing the section, not by a checkbox inside
  it.

### Breaking
- `pipeline.steps` semantics change: configs that previously relied on
  auto-enable from sub-config presence must add explicit `steps:`
  entries to keep those steps running.

## [1.7.0] — 2026-05-27

Loader API consolidation (PR-E). Companion to ImageAnalysis 1.5.0 and
Scanner-GUI 0.22.0. Adds the `create_scan_analyzer` factory as the
public production entry point and pins the dependency direction
(ScanAnalysis depends on ImageAnalysis, never the reverse) by
re-validating `scan:` at this layer.

### Added
- `scan_analysis.config.create_scan_analyzer(DiagnosticAnalysisConfig,
  *, id, priority) -> ScanAnalyzer` — the Mode 2 (config-driven)
  factory for going from a loaded diagnostic config to a ready-to-run
  scan analyzer. Re-exported from `scan_analysis.config`.
- `load_analysis_group(...)` now returns a `LoadedAnalysisGroup`
  carrying resolved diagnostic configs ready to hand to
  `create_scan_analyzer` (one call per analyzer in the group).

### Changed
- `scan:` field validation now happens in ScanAnalysis at scan-analyzer
  build time rather than at YAML-load time in ImageAnalysis. The
  ImageAnalysis-side `scan` field is weakly typed (`Optional[Dict[str,
  Any]]`); `create_scan_analyzer` calls
  `ScanRuntimeConfig.model_validate(diag.scan or {})` itself. This
  breaks what would otherwise be a circular dependency between the two
  packages and keeps ImageAnalysis ignorant of scan-runtime concerns.
- `ScatterPlotterAnalysis` is now a standalone utility class, not part
  of the config-driven scan pipeline. It is not wired into the unified
  diagnostic schema and is unaffected by the loader API.

### Breaking
- Any caller still constructing scan analyzers via the
  pre-PR-E factory path needs to migrate to `create_scan_analyzer` (or
  the `task_queue.load_analyzers_from_config` wrapper, which is
  unchanged in signature). Direct construction (`Mode 1`) still works
  unchanged for exploration code that prefers it.

## [1.6.0] — 2026-05-24

Cutover to the unified diagnostic-config schema. The split
`scan_analysis_configs/library/` and `experiments/` layers are gone;
analyzers live one-file-per-diagnostic under `analyzers/<namespace>/`
and are assembled into groups under `groups/<namespace>/`. The old
loader/factory/models are deleted.

### Added
- `scan_analysis.config.load_analysis_group` and
  `create_diagnostic_analyzer` re-exported from
  `scan_analysis.config` as the public API for loading and
  instantiating analyzers from disk.
- `scan.background_source` directive is the sole way to express
  cross-scan-dark and dynamic-from-current-scan backgrounds in
  the unified schema.

### Changed
- `scan_analysis.task_queue.load_analyzers_from_config` now consumes
  the unified analysis-group layout via
  `analysis_group_loader.load_analysis_group` and
  `diagnostic_factory.create_diagnostic_analyzer`. The function
  signature is unchanged (still takes a group name + optional
  `config_dir`), but it no longer accepts the old experiment-wrapper
  YAMLs.
- `LiveWatchGUI`: replaced the broken `_try_list_experiments` (which
  imported the deleted `config_loader`) with `_try_list_groups` built
  on `discover_groups`. The combo box now populates from
  `<config_dir>/groups/` and falls back to an empty list (instead of
  hard-coding `"Undulator"`) when no groups are found.

### Removed
- `scan_analysis.config.config_loader` and all of its public symbols
  (`ScanAnalyzerInfo`, `ScanAnalyzersConfig`, `ExperimentAnalysisConfig`,
  `load_experiment_analyzers`, `list_available_configs`,
  `set_analyzer_attributes`, etc.).
- All Array2D / Array1D / Array1DLineout fields from
  `analyzer_config_models.py`; only `PlotParameterConfig` and
  `ScatterAnalyzerConfig` remain.
- `analyzer_factory.create_analyzer` now only dispatches scatter
  configs. Image-analyzer-driven analyzers go through
  `create_diagnostic_analyzer`.
- The `BackgroundConfig.background_scan_number`-driven path
  (`SingleDeviceScanAnalyzer._generate_scan_background`). Use
  `scan.background_source.scan_number` instead.

### Breaking
- Experiment-wrapper YAMLs (`experiments/<name>.yaml`) and the library
  shorthand entries are no longer recognised. Callers must point at
  the new `analyzers/` + `groups/` layout.
- The `analyzer_config_models` / `analyzer_factory` public surface
  shrinks to scatter-only.

## [1.5.1] — 2026-05-22

### Removed
- `ConfigFileGUI`: removed the `_connect_dynamic_computation_defaults`
  auto-population helper. The widget set it wired up no longer exists
  in `BackgroundConfig` (deleted in ImageAnalysis 1.3.0). Updated
  docstring example in `SectionWidget.show_errors` accordingly.

## [1.5.0] — 2026-05-22

### Changed
- `SingleDeviceScanAnalyzer._process_all_shots` now dispatches to one of
  two mode-specific pipelines:
  - `per_shot` mode: fuses load+analyze into a single
    `analyze_image_file(path, aux)` task per shot. Per-shot data never
    travels through analyzer-instance state between separate pipeline
    phases — this is the correctness property the refactor enforces.
  - `per_bin` mode: streams bin-by-bin. For each bin, only that bin's
    files are loaded (in parallel), averaged, analyzed, and released
    before the next bin starts. Memory stays bounded by one bin's
    image count.
- The bin-grouping logic (previously embedded in
  `_prepare_per_bin_units`) is now in `_group_files_by_bin`. The
  shared post-task bookkeeping (store result, log scalars, queue
  s-file updates) is in `_consume_result`.

### Removed
- `_load_all_data`: there is no longer a phase that materialises all
  shots upfront. Both modes do their own loading inline.
- `_prepare_per_shot_units` / `_prepare_per_bin_units` /
  `_analyze_units`: replaced by `_analyze_per_shot` and
  `_analyze_per_bin_streaming`, which embed the unit-construction logic
  directly into the per-shot/per-bin loops.
- `self.raw_data`: no longer populated anywhere; dropped from `cleanup`.

### Behavior notes
- `per_bin` mode used to process all bins in parallel (after a single
  load-all). It now processes bins serially, parallelising only within
  a bin. For typical scans (~20 bins of ~20–50 shots) this trades a
  small wall-clock cost for bounded memory and no shared-state
  contamination across bins. Public API is unchanged: callers still see
  `analyzer.run_analysis(scan_tag)` returning a list of artifact paths
  and `analyzer.results[shot_num|bin_num]` populated as before.

## [1.4.0] — 2026-05-22

### Removed
- `SingleDeviceScanAnalyzer._run_batch_analysis` and the
  `stateful_results` plumbing through `_prepare_per_*_units` are gone.
  These were the call sites for `ImageAnalyzer.analyze_image_batch`,
  whose only real consumer was the now-deleted dynamic-background
  subsystem. The load-all path itself is still in place pending the
  per-shot fusion (commit 3 of the shot-by-shot refactor).

### Changed
- `_resolve_background_paths` is now called once at the start of
  `_process_all_shots` rather than from inside the deleted
  `_run_batch_analysis`. Behavior for the preserved
  `background_scan_number` ("scan-as-background") feature is unchanged
  — that path was always orthogonal to the load-all pipeline.
- `_resolve_background_paths` simplified: dropped the
  `dynamic_computation.auto_save_path` placeholder-resolution branch
  (no longer applicable).

## [1.3.6] — 2026-05-21

### Fixed
- `task_queue.init_status_for_scan` and `task_queue.update_status` now refuse
  to create a missing scan folder. The previous `mkdir(parents=True,
  exist_ok=True)` would silently bring `ScanNNN/` into existence with only
  `analysis_status/` inside it. Under normal conditions this was harmless,
  but on a new NetApp/domain it allowed a transient SMB visibility blip
  (triggered, in one reported incident, by an Explorer double-click during
  a scan write) to be converted into permanent data loss: the analysis stack
  planted an empty `Scan015/` directory over the real one during the window
  that the real folder was briefly invisible. Both functions now log an
  error and return early when `scan_folder` is not a visible directory; the
  LiveWatch loop continues processing other work and can pick the scan up on
  a later pass or after relaunch if the folder reappears. The `parents=True`
  flag is dropped everywhere — only `analysis_status/` is ever auto-created,
  and only inside an already-visible scan folder. Invariant pinned by new
  tests in `tests/test_task_queue.py::TestScanFolderCreationInvariant`.
## [1.3.6] — 2026-05-20

### Changed
- `Array1DScanAnalyzer._postprocess_scan` now bypasses scan-parameter
  binning and falls through to `_postprocess_noscan` when
  `waterfall_sort_key` is set in `renderer_kwargs`. Previously the sort
  key only took effect on noscan data; for parameter scans the waterfall
  was always binned by the scan parameter, so the kwarg had no observable
  effect. With this change you can render every shot of a parameter scan
  as a waterfall row ordered by any s-file column (e.g. a downstream
  diagnostic) — at the cost of the per-bin averaged spectra, which are
  not produced in this mode. Behavior with no sort key is unchanged.

### Fixed
- Waterfall sort-key filtering in `_postprocess_noscan` is now
  NaN-aware. Previously a single missing or NaN value in the chosen
  auxiliary column poisoned `mean()` / `std()`, making the
  sigma-based outlier filter reject every shot ("kept 0 of N"). Now
  shots with a non-finite sort value are dropped up front with a
  warning; if none remain, rendering is skipped with a clear log line
  instead of producing an empty figure. Also drops the
  `else float(shot_num)` fallback in the lookup: mixing shot indices
  with real sort-key values would silently corrupt the same
  mean/std-based filter on partially-populated columns.

## [1.3.5] — 2026-05-20

### Fixed
- `SingleDeviceScanAnalyzer._run_analysis_core` no longer swallows
  unexpected exceptions. Previously the outer `except Exception` printed
  a local traceback, logged a warning, and returned `None`. Because
  `task_queue.run_worklist` treats a `None` return as "no display
  files" and unconditionally writes `state="done"`, real analysis
  failures were being marked as completed successfully. Exceptions now
  propagate to `run_worklist`, which catches them and writes
  `state="failed"` with the error message (matching the existing
  `DataUnavailableWarning` → `state="no_data"` handling). The unused
  `PRINT_TRACEBACK` module-level flag and `traceback` import are removed.

## [1.3.4] — 2026-05-20

### Fixed
- `SingleDeviceScanAnalyzer.__init__` now initializes `self.stateful_results
  = {}`. Previously it was only set in `cleanup()` or after a successful
  `analyze_image_batch()` call. When `_run_batch_analysis` silently swallowed
  an exception from the underlying ImageAnalyzer (e.g. a `BeamAnalyzer` whose
  `camera_config.background` is null, leading to a `NoneType` access on its
  background manager), `_prepare_per_shot_units` would then raise
  `AttributeError: ... has no attribute 'stateful_results'`.

## [1.3.3] — 2026-05-19

### Changed
- `LiveWatchConfig.experiment` is now `Optional[str]` (was `str = ""`).
  The `"(none)"` → `None` translation moved from `to_scan_tag()` into
  `_build_config()` in `live_watch_window.py`, so the dataclass always holds
  a clean `None` rather than a sentinel string.

## [1.3.2] — 2026-05-19

### Fixed
- LiveWatch GUI: added `(none)` facility option for data roots with no
  experiment subdirectory (e.g. `N:\data\Y2026\...` instead of
  `N:\data\Undulator\Y2026\...`).
- `LiveWatchConfig.to_scan_tag()`: no longer falls back to `analyzer_group`
  as the path experiment when facility is unset — passes `None` instead, so
  the path builder can omit the experiment segment cleanly.

## [1.3.0] — 2026-05-06

### Added
- `ScatterPlotterAnalysis` now supports a configurable `x_column` parameter:
  any s-file column can be used as the x-axis. For 1D scans the per-bin
  statistic of the column is used; for noscans the per-shot values are used.
  Falls back to the scan parameter (1D scans) or shot number (noscans) when
  `x_column` is not set.
- `PlotParameter` gains an optional `y_range: tuple[float, float]` field to
  fix the y-axis limits for a series. When omitted matplotlib auto-ranges.
- Output PNG is now saved to `<scan_folder>/analysis/scatter_plots/<filename>.png`
  (previously saved outside the scan folder tree).
- New Pydantic config models `PlotParameterConfig` and `ScatterAnalyzerConfig`
  (`type: "scatter"`) allow scatter plots to be declared in experiment YAML
  configs alongside `array2d` / `array1d` analyzers.
- `create_analyzer()` factory handles `ScatterAnalyzerConfig` and converts
  config objects to `PlotParameter` / `ScatterPlotterAnalysis` instances.
- Unit tests for `PlotParameterConfig` and `ScatterAnalyzerConfig` (no
  external data required).
- Integration tests for `ScatterPlotterAnalysis` and the factory round-trip
  against Undulator 2026-05-05 Scan018 (`@pytest.mark.integration`).

## [1.3.1] — 2026-05-12

### Fixed
- **ConfigFileGUI**: `Name` field in the General Settings panel is now editable.
  Previously it was frozen to the YAML filename stem, making it impossible to set
  a `name` that differs from the config file name. The field is now populated from
  the loaded config's `name` value and saved back normally.
- **ConfigFileGUI**: Float fields now support scientific notation (e.g. `1e-13`).
  `QDoubleSpinBox` with `decimals=6` silently rounded sub-micro values to zero.
  Replaced with `ScientificDoubleSpinBox`, a thin subclass that overrides
  `validate` / `valueFromText` / `textFromValue` to accept and display scientific
  notation, with precision set to 15 significant digits.

### Added
- Unit tests for `ScientificDoubleSpinBox` (validate, text↔value conversion,
  parametrized round-trip).
- Round-trip tests for `config_io` save/load cycle covering `name` field
  independence from filename and survival of scientific-notation floats.

## [1.1.1] — current
<!-- Add entries here when changes are made -->
