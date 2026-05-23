# Changelog — scan-analysis

All notable changes to this package will be documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

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
